#!/usr/bin/env python3
"""Generated coefficient header for the transcendental functions in ``include/ulpgrad``.

One file, one block per function family. ``exp`` and ``exp2`` are **copied
verbatim** out of the file this script writes, not refitted: their bounds were
measured against those exact bit patterns, and a fit that moved by one ULP would
invalidate ``data/error-bounds.json`` for the two functions that already ship.
Everything else is fitted here, at 40 decimal digits, and then rendered to
float32 and scored as a *pipeline* -- the quantised coefficients are what the
library computes with, so the residual that matters is the one of the assembled
function, not of the 40-digit polynomial.

``rsqrt``
    The Newton iteration converges in three steps from a bit-trick estimate. The
    estimate is ``magic - (bits >> 1)``: ``magic`` is searched here (not copied
    from a folklore constant) to minimise the maximum ULP distance of the whole
    three-step pipeline over ``M`` in ``[1, 4)``, which is the interval the
    function reduces every input to. Because the final ``2^-k`` scaling is an
    exact exponent-field move, that sweep covers every normal float32 and every
    subnormal one: the bound is a complete enumeration, not a sample.

Writes ``include/ulpgrad/detail/coefficients.h``. Run from the repository root::

    python experiments/m4_transcendental/generate_coefficients.py
"""

import argparse
import os
import re
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'experiments', 'm0_exp'))
sys.path.insert(0, os.path.join(REPO, 'experiments', 'm3_fit'))

import mpmath                       # noqa: E402  (path has to be set first)
import fit_coeffs as fc             # noqa: E402
import gen_reference as gr          # noqa: E402

OUT_PATH = os.path.join(REPO, 'include', 'ulpgrad', 'detail', 'coefficients.h')
MP_DPS = 40

F32 = np.float32


# --------------------------------------------------------------------------- #
# float32 arithmetic model
# --------------------------------------------------------------------------- #


def fma(a, b, c):
    """float32 fused multiply-add: the exact product and sum, rounded once.

    numpy has no float32 fma, and the point of the model is to reproduce what
    ``ULPGRAD_FMA`` does, so the product and the add are formed in binary64 --
    exact for two float32 operands -- and the result is rounded to float32 once.
    """
    return F32(np.float64(a) * np.float64(b) + np.float64(c))


def bits_of(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def ulp_keys(bits):
    """Monotone integer image of a float32 pattern (same metric as the library)."""
    b = np.asarray(bits, dtype=np.uint32)
    signed = b.astype(np.int32).astype(np.int64)
    negative = (b & np.uint32(0x80000000)) != 0
    return np.where(negative, -(b & np.uint32(0x7FFFFFFF)).astype(np.int64), signed)


def ulp_distance(got_bits, ref_bits):
    return np.abs(ulp_keys(got_bits) - ulp_keys(ref_bits))


def score(got, reference_bits_array):
    """(max ULP, points at max, total, exactly rounded count) of a model run.

    `reference_bits_array` is already a uint32 bit-pattern array, which is what
    the metric consumes: converting it to float32 first would reinterpret the
    patterns as *values* and report nonsense.
    """
    distance = ulp_distance(bits_of(got), np.asarray(reference_bits_array, dtype=np.uint32))
    peak = int(distance.max())
    return peak, int(np.count_nonzero(distance == peak)), int(distance.size), \
        int(np.count_nonzero(distance == 0))


# --------------------------------------------------------------------------- #
# error-free helpers
# --------------------------------------------------------------------------- #


def float_literal(value):
    """C++ float literal that round-trips a float32 (%.9g plus a '.' guard)."""
    text = '%.9g' % float(F32(value))
    if ('e' not in text) and ('E' not in text) and ('.' not in text):
        text += '.0'
    return text + 'f'


def hex_of(value):
    return gr.float_to_bits(float(F32(value)))


# --------------------------------------------------------------------------- #
# Remez front end
# --------------------------------------------------------------------------- #


def remez_fit(target, degree, lo, hi, name, verbose=True):
    """Remez exchange on a relative-error metric; returns a small result record.

    The fit machinery is `experiments/m0_exp/fit_coeffs.py`, the one that produced
    the validated `exp` coefficients: it reads the fitted function from a module
    global, so the target is swapped in and swapped back. The residual reported is
    the maximum relative error of the *40-digit* polynomial on a 20001-point grid;
    the float32 rendering and the pipeline model below are what decide whether the
    quantised coefficients still hold the bound.
    """
    original = fc.q_func
    fc.q_func = target
    try:
        coefficients, iterations, converged = fc.remez(degree, lo, hi, max_iter=14,
                                                       verbose=False)
        grid = [lo + (hi - lo) * i / 20000 for i in range(20001)]
        residual = max(abs(error) for error in fc._eval_rel_error(coefficients, grid))
    finally:
        fc.q_func = original
    rendered = fc.to_float32(coefficients)
    if verbose:
        print('%-6s degree %d: %d Remez iteration(s), %s, relative residual %.3e'
              % (name, degree, iterations, 'converged' if converged else 'no convergence',
                 float(residual)))
    return {
        'coefficients': rendered,
        'residual': float(residual),
        'iterations': iterations,
        'converged': converged,
    }


def reference_bits(values, target):
    """Correctly rounded float32 results, or the canonical NaN, for a grid."""
    out = np.empty(len(values), dtype=np.uint32)
    for i, value in enumerate(values):
        result = target(mpmath.mpf(float(value)))
        out[i] = F32_QNAN if mpmath.isnan(result) else gr.round_to_f32_bits(result)
    return out


F32_QNAN = 0x7FC00000


# --------------------------------------------------------------------------- #
# rsqrt
# --------------------------------------------------------------------------- #

# The magic constant is searched, not remembered. The window is wide enough to
# contain the published constants (0x5F3759DF, 0x5F375A86) and narrow enough
# that a brute force over the objective is cheap.
RSQRT_MAGIC_LO = 0x5F370000
RSQRT_MAGIC_HI = 0x5F380000
RSQRT_STEPS = 3


def rsqrt_pipeline(m_values, magic, steps=RSQRT_STEPS):
    """1/sqrt(M) for M in [1, 4), exactly as include/ulpgrad/rsqrt.h computes it."""
    m = np.asarray(m_values, dtype=np.float32)
    y = (np.uint32(magic) - (m.view(np.uint32) >> np.uint32(1))).view(np.float32)
    half = F32(0.5) * m                     # exact: a power-of-two scale
    for _ in range(steps):
        square = F32(y * y)
        correction = fma(-half, square, F32(0.5))
        y = fma(y, correction, y)
    return y


def rsqrt_reference(m):
    """Correctly rounded 1/sqrt(M) via binary64, then a single rounding.

    Binary64 sqrt is correctly rounded and the reciprocal adds at most half an
    ULP of binary64, so the double rounding is 2^-53 away from any float32
    boundary the sweep visits; the committed golden reference, built in mpmath,
    is what the declared bound is finally measured against.
    """
    return (1.0 / np.sqrt(np.asarray(m, dtype=np.float64))).astype(np.float32)


def mantissa_grid(per_binade):
    """M over [1, 4): both exponent fields, `per_binade` mantissas each."""
    out = []
    for field in (127, 128):
        frac = np.linspace(0, (1 << 23) - 1, per_binade).astype(np.uint32)
        out.append(((np.uint32(field) << np.uint32(23)) | frac).view(np.float32))
    return np.concatenate(out)


def fit_rsqrt(verbose=True):
    """Search the magic constant, then verify on the complete [1, 4) sweep."""
    coarse = mantissa_grid(4000)
    coarse_ref = rsqrt_reference(coarse)

    best = None
    for magic in range(RSQRT_MAGIC_LO, RSQRT_MAGIC_HI):
        got = rsqrt_pipeline(coarse, magic)
        distance = ulp_distance(bits_of(got), bits_of(coarse_ref))
        key = (int(distance.max()), -int(np.count_nonzero(distance == 0)))
        if best is None or key < best[0]:
            best = (key, magic)
    magic = best[1]

    # Complete sweep: 2 * 2^23 patterns covers every M the reduction can produce.
    worst = 0
    exact = 0
    total = 0
    for field in (127, 128):
        step = 1 << 20
        for start in range(0, 1 << 23, step):
            frac = np.arange(start, start + step, dtype=np.uint32)
            m = ((np.uint32(field) << np.uint32(23)) | frac).view(np.float32)
            distance = ulp_distance(bits_of(rsqrt_pipeline(m, magic)), bits_of(rsqrt_reference(m)))
            worst = max(worst, int(distance.max()))
            exact += int(np.count_nonzero(distance == 0))
            total += int(distance.size)

    if verbose:
        print('rsqrt: magic 0x%08X over [0x%08X, 0x%08X), %d Newton steps'
              % (magic, RSQRT_MAGIC_LO, RSQRT_MAGIC_HI, RSQRT_STEPS))
        print('rsqrt: complete [1, 4) sweep, %d points, max %d ULP, %.2f %% exactly rounded'
              % (total, worst, 100.0 * exact / total))

    meta = {
        'magic': magic,
        'steps': RSQRT_STEPS,
        'window': (RSQRT_MAGIC_LO, RSQRT_MAGIC_HI),
        'sweep_points': total,
        'sweep_max_ulp': worst,
        'sweep_exact_pct': 100.0 * exact / total,
    }
    return meta


# --------------------------------------------------------------------------- #
# log, log2, log1p
#
# One reduction, one series, two fits. x = m * 2^e with m in [sqrt(1/2), sqrt(2))
# makes (m - 1) exact by Sterbenz, and s = (m - 1)/(m + 1) lands in
# [-0.17157, 0.17157]. Then
#
#     log(m)  = 2 atanh(s)       = 2s + 2s*z*R(z),  z = s^2
#     log2(m) = 2 atanh(s)/ln2   = 2s*log2(e) + 2s*z*R2(z)
#
# `R` and `R2` are fitted separately: folding the 1/ln2 into `R` afterwards would
# cost a product on a value that is already the result, and this file exists to
# keep that kind of rounding out of the hot path.
#
# The denominator is the one quantity in the pipeline that is silently wrong:
# `2 + t` rounds, so `s` is the ratio of an exact numerator to a rounded
# denominator, and the series cannot see it. The residual is exact and free --
# d_lo = (2 - h) + t is the two-sum leftover of that very addition -- so the half
# ULP is recoverable for the price of 1/(2m), and 1/(2 + 2t) as a degree-2 fit
# buys that without a second division.
# --------------------------------------------------------------------------- #

SQRT2 = mpmath.sqrt(2)
S_MAX = (SQRT2 - 1) / (SQRT2 + 1)          # |s| at the edge of the reduction
Z_MAX = S_MAX ** 2
LOG_DEGREE = 4                              # R through z^4, i.e. the series to s^9
LOG2_E = 1 / mpmath.log(2)

# The range of t = m - 1 under the reduction, which is also the range the
# denominator reciprocal is fitted on.
T_LO = float(mpmath.sqrt(mpmath.mpf(1) / 2) - 1)
T_HI = float(SQRT2 - 1)
INV_TWO_M_DEGREE = 2


def inv_two_m(t):
    """1 / (2 + 2t) = 1/(2m): the factor the denominator correction multiplies."""
    return 1 / (2 + 2 * t)



def r_log_tail(z):
    """R(z) = (atanh(s)/s - 1)/z, the tail of log(m) after its leading 2s."""
    if z == 0:
        return mpmath.mpf(1) / 3
    s = mpmath.sqrt(z)
    return (mpmath.atanh(s) / s - 1) / z


def r_log2_tail(z):
    """R2(z) = (log2(m)/(2s) - log2(e))/z, the tail after the leading 2s*log2(e)."""
    if z == 0:
        return LOG2_E / 3
    s = mpmath.sqrt(z)
    value = mpmath.log((1 + s) / (1 - s)) / (2 * s) / mpmath.log(2)
    return (value - LOG2_E) / z


def ln2_parts():
    """ln2 as ln2_hi + ln2_lo, with ln2_hi keeping 12 significand bits.

    Twelve bits is what makes `e * ln2_hi` exact for every |e| <= 149: the
    product needs 8 + 12 bits, which fits in 24. The residual is one float32, and
    it is folded into the series correction rather than added on top of the
    result, where it would cost a rounding at the result's scale.
    """
    raw = gr.float_to_bits(float(mpmath.mpf(float(mpmath.log(2)))))
    high = struct.unpack('<f', struct.pack('<I', raw & 0xFFFFF000))[0]
    low = float(F32(float(mpmath.log(2)) - mpmath.mpf(high)))
    return F32(high), F32(low)


def reduce_for_log(x):
    """x = m * 2^e with m in [sqrt(1/2), sqrt(2)); mirrors detail::reduce_for_log."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    fraction = u & np.uint32(0x007FFFFF)
    exponent = (u >> np.uint32(23)).astype(np.int32) - 127
    high = fraction > np.uint32(0x3504F3)      # m > sqrt(2): halve it, exact
    exponent = exponent + high.astype(np.int32)
    field = np.where(high, np.uint32(126), np.uint32(127)).astype(np.uint32)
    return exponent.astype(np.float32), ((field << np.uint32(23)) | fraction).view(np.float32)


def horner(coefficients, z):
    value = coefficients[-1]
    for index in range(len(coefficients) - 2, -1, -1):
        value = fma(value, z, coefficients[index])
    return value


def atanh_argument(t, inv_two_m_coeffs):
    """(s, residual) for t = m - 1, mirroring `detail::atanh_argument`.

    `h = 2 + t` rounds; `d_lo` is what it dropped, exactly, and the logarithm
    pays `-t*d_lo/(2m)` for it, in the natural-log scale. The reciprocal comes
    from a quadratic, so the correction costs no second division.
    """
    h = F32(F32(2.0) + t)                       # rounded
    d_lo = F32(F32(F32(2.0) - h) + t)           # exact: the two-sum residual
    s = F32(t / h)                              # correctly rounded
    reciprocal = horner(inv_two_m_coeffs, t)
    dropped = fma(t, d_lo, F32(0.0))            # rounded product, as ULPGRAD_FMA(.., 0)
    residual = fma(dropped, F32(-reciprocal), F32(0.0))
    return s, residual


def log_pipeline(x, tail, ln2_hi, ln2_lo, inv_two_m_coeffs):
    exponent, m = reduce_for_log(x)
    s, residual = atanh_argument(F32(m - F32(1.0)), inv_two_m_coeffs)
    z = F32(s * s)
    two_s = F32(s + s)                          # exact
    shift = F32(exponent * ln2_lo)              # fma(e, ln2_lo, 0): rounded, not fused
    correction = F32(F32(two_s * z) * horner(tail, z))   # fma(two_s*z, tail, 0)
    log_m = F32(two_s + F32(F32(shift + residual) + correction))
    return fma(exponent, ln2_hi, log_m)


def log2_pipeline(x, tail, log2_e, inv_two_m_coeffs):
    exponent, m = reduce_for_log(x)
    s, residual = atanh_argument(F32(m - F32(1.0)), inv_two_m_coeffs)
    z = F32(s * s)
    two_s = F32(s + s)
    leading = F32(two_s * log2_e)
    product_residual = fma(two_s, log2_e, -leading)     # the tail of that product
    scaled = F32(residual * log2_e)                     # fma(residual, log2_e, 0)
    delta = fma(F32(two_s * z), horner(tail, z), F32(product_residual + scaled))
    return F32(exponent + F32(leading + delta))


def log1p_pipeline(x, tail, ln2_hi, ln2_lo, inv_two_m_coeffs):
    x = np.asarray(x, dtype=np.float32)
    magnitude = x.view(np.uint32) & np.uint32(0x7FFFFFFF)
    small = magnitude <= np.uint32(0x3ED413CD)   # |x| <= sqrt(2) - 1

    # Short branch: t = x and the denominator is 2 + x, so the same correction
    # applies with no reduction at all.
    s, residual = atanh_argument(x, inv_two_m_coeffs)
    z = F32(s * s)
    two_s = F32(s + s)
    short_form = F32(two_s + F32(F32(F32(two_s * z) * horner(tail, z)) + residual))

    # |x| > sqrt(2) - 1: log(1+x) through the same reduction. The rounding of
    # 1+x is recovered exactly by Knuth's two-sum and folded into the series
    # correction, which is what keeps the branch at the same bound as log.
    h = F32(F32(1.0) + x)
    bb = F32(h - x)
    low = F32(F32(x - F32(h - bb)) + F32(F32(1.0) - bb))
    exponent, m = reduce_for_log(h)
    s2, residual2 = atanh_argument(F32(m - F32(1.0)), inv_two_m_coeffs)
    z2 = F32(s2 * s2)
    two_s2 = F32(s2 + s2)
    extra = fma(exponent, ln2_lo, F32(low / h))
    long_form = fma(exponent, ln2_hi,
                    F32(two_s2 + F32(F32(F32(F32(two_s2 * z2) * horner(tail, z2)) +
                                         F32(extra + residual2)))))
    return np.where(small, short_form, long_form)


def log_model_grid(seed=0x10AD0001, count=1200):
    """Stratified x for the model check: every exponent, both sides of 1."""
    rng = np.random.default_rng(seed)
    fields = np.arange(1, 255, 2, dtype=np.uint32)
    fraction = rng.integers(0, 1 << 23, size=fields.size).astype(np.uint32)
    structural = ((fields << np.uint32(23)) | fraction).view(np.float32)
    random = np.exp(rng.uniform(-30.0, 30.0, count)).astype(np.float32)
    around_one = np.arange(np.uint32(0x3F800000) - 64, np.uint32(0x3F800000) + 64,
                           dtype=np.uint32).view(np.float32)
    return np.concatenate([structural, random, around_one])


def fit_log(verbose=True):
    """Fit R and R2, then check both pipelines against mpmath."""
    log_fit = remez_fit(r_log_tail, LOG_DEGREE, mpmath.mpf(0), Z_MAX, 'log')
    log2_fit = remez_fit(r_log2_tail, LOG_DEGREE, mpmath.mpf(0), Z_MAX, 'log2')
    inv_fit = remez_fit(inv_two_m, INV_TWO_M_DEGREE, mpmath.mpf(T_LO), mpmath.mpf(T_HI),
                        'inv2m')
    tail = log_fit['coefficients']
    tail2 = log2_fit['coefficients']
    inv_two_m_coeffs = inv_fit['coefficients']
    ln2_hi, ln2_lo = ln2_parts()
    log2_e = F32(float(LOG2_E))
    series_max = F32(float(SQRT2 - 1))

    xs = log_model_grid()
    reference = reference_bits(xs, mpmath.log)
    peak, at_peak, total, exact = score(log_pipeline(xs, tail, ln2_hi, ln2_lo, inv_two_m_coeffs),
                                        reference)
    if verbose:
        print('log:  model on %d points: max %d ULP (%d at max), %.2f %% exactly rounded'
              % (total, peak, at_peak, 100.0 * exact / total))

    reference2 = reference_bits(xs, lambda value: mpmath.log(value) / mpmath.log(2))
    peak2, at_peak2, total2, exact2 = score(
        log2_pipeline(xs, tail2, log2_e, inv_two_m_coeffs), reference2)
    if verbose:
        print('log2: model on %d points: max %d ULP (%d at max), %.2f %% exactly rounded'
              % (total2, peak2, at_peak2, 100.0 * exact2 / total2))

    positive = np.concatenate([
        xs[xs > np.float32(0.5)],
        np.exp(np.random.default_rng(7).uniform(-1.0, 4.0, 400)).astype(F32),
    ])
    reference3 = reference_bits(positive, mpmath.log1p)
    peak3, at_peak3, total3, exact3 = score(
        log1p_pipeline(positive, tail, ln2_hi, ln2_lo, inv_two_m_coeffs), reference3)
    if verbose:
        print('log1p: model on %d points: max %d ULP (%d at max), %.2f %% exactly rounded'
              % (total3, peak3, at_peak3, 100.0 * exact3 / total3))

    return {
        'tail': tail,
        'tail2': tail2,
        'log_fit': log_fit,
        'log2_fit': log2_fit,
        'inv_two_m': inv_two_m_coeffs,
        'inv_two_m_fit': inv_fit,
        'ln2_hi': ln2_hi,
        'ln2_lo': ln2_lo,
        'log2_e': log2_e,
        'series_max': series_max,
        'model': (peak, at_peak, total, 100.0 * exact / total),
        'model2': (peak2, at_peak2, total2, 100.0 * exact2 / total2),
        'model1p': (peak3, at_peak3, total3, 100.0 * exact3 / total3),
    }


# --------------------------------------------------------------------------- #
# tanh
#
# Two branches, and the measurement picks them. Below `POLY_MAX` the function is
# an odd polynomial: tanh(x) = x + x^3*R(x^2), with the leading `x` exact and the
# correction at most 12 % of the result, so the roundings of R cost nothing.
# Above it, tanh(x) = (u - 1)/(u + 1) with u = 2^(x*2*log2(e)): the saturation to
# +-1 comes out of the division by itself, and above `SATURATION` no float32
# input can round to anything but +-1, so the branch is a comparison rather than
# an approximation.
# --------------------------------------------------------------------------- #

POLY_MAX = 0.5             # branch, in |x|; also the fit interval for R
POLY_MAX_Y = POLY_MAX ** 2
TANH_DEGREE = 4
TWO_LOG2_E = 2 / mpmath.log(2)


def tanh_tail(y):
    """R(y) = (tanh(sqrt(y)) - sqrt(y)) / sqrt(y)^3, with -1/3 filled in at 0."""
    if y == 0:
        return mpmath.mpf(-1) / 3
    x = mpmath.sqrt(y)
    return (mpmath.tanh(x) - x) / (x ** 3)


def tanh_saturation_bits():
    """Smallest float32 x whose correctly rounded tanh(x) is 1.0."""
    def rounds_to_one(bits):
        value = gr.round_to_f32_bits(mpmath.tanh(mpmath.mpf(gr.bits_to_float(bits))))
        return value == 0x3F800000

    return gr._bisect_bits(0x41000000, 0x41200000, rounds_to_one)   # 8.0 .. 10.0


def tanh_pipeline(x, tail, two_log2_e, saturation_bits, poly_max=POLY_MAX):
    """tanh as include/ulpgrad/tanh.h computes it."""
    x = np.asarray(x, dtype=np.float32)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        y = F32(x * x)
        poly = fma(F32(x * y), horner(tail, y), x)
        w = F32(x * two_log2_e)
        exponent = np.rint(w)
        reduced = F32(w - exponent)
        polynomial = fma(horner(exp2_coefficients(), reduced), reduced, F32(1.0))
        u = np.ldexp(polynomial.astype(np.float64),
                     exponent.astype(np.int32)).astype(np.float32)
        upper = F32((u - F32(1.0)) / (u + F32(1.0)))
    magnitude = x.view(np.uint32) & np.uint32(0x7FFFFFFF)
    upper = np.where(magnitude >= np.uint32(saturation_bits),
                     np.where((x.view(np.uint32) & np.uint32(0x80000000)) != 0,
                              F32(-1.0), F32(1.0)), upper)
    return np.where(np.abs(x) < F32(poly_max), poly, upper)


def exp2_coefficients():
    """The exp2 coefficients the library ships, read back from its own header.

    tanh is built on `exp2`, so the model has to use exactly the bit patterns the
    library computes with; re-fitting them here would model a function that does
    not exist.
    """
    text = read_block(OUT_PATH, 'exp2_coeffs')
    values = []
    for index in range(7):
        match = re.search(r'constexpr float c%d = ([^;]+);' % index, text)
        if match is None:
            raise RuntimeError('exp2_coeffs has no c%d' % index)
        values.append(F32(float(match.group(1).rstrip('f'))))
    return np.array(values, dtype=np.float32)


def fit_tanh(verbose=True):
    """Fit R, find the saturation threshold, and check the pipeline."""
    fit = remez_fit(tanh_tail, TANH_DEGREE, mpmath.mpf(0), mpmath.mpf(POLY_MAX_Y), 'tanh')
    tail = fit['coefficients']
    two_log2_e = F32(float(TWO_LOG2_E))
    saturation = tanh_saturation_bits()

    rng = np.random.default_rng(0x7A4E)
    xs = np.concatenate([
        rng.uniform(-1.5, 1.5, 900).astype(np.float32),
        rng.uniform(-9.5, 9.5, 900).astype(np.float32),
        np.linspace(-1.0, 1.0, 400).astype(np.float32),
        np.array([0.0, -0.0, 0.5, -0.5, 0.4999999, 1e-30, 1e-38, 9.0109, -9.0109,
                  100.0, -100.0, 1e30, 3.0, -3.0], dtype=np.float32)])
    reference = reference_bits(xs, mpmath.tanh)
    peak, at_peak, total, exact = score(tanh_pipeline(xs, tail, two_log2_e, saturation), reference)
    if verbose:
        print('tanh: degree %d on |x| <= %g, saturation 0x%08X (%r)'
              % (TANH_DEGREE, POLY_MAX, saturation, gr.bits_to_float(saturation)))
        print('tanh: model on %d points: max %d ULP (%d at max), %.2f %% exactly rounded'
              % (total, peak, at_peak, 100.0 * exact / total))

    return {
        'tail': tail,
        'fit': fit,
        'two_log2_e': two_log2_e,
        'saturation': saturation,
        'poly_max': F32(POLY_MAX),
        'model': (peak, at_peak, total, 100.0 * exact / total),
    }


# --------------------------------------------------------------------------- #
# block emission
# --------------------------------------------------------------------------- #

# Only the coefficient namespaces (``*_coeffs``): the enclosing
# ``namespace ulpgrad`` / ``namespace detail`` also sit at column zero and would
# otherwise swallow the whole file.
BLOCK_RE = re.compile(
    r'((?:^///[^\n]*\n)*^namespace (\w+_coeffs) \{\n.*?^\}  // namespace \2\n)',
    re.M | re.S,
)

PROVENANCE_HEADER = """#pragma once

/// @file
/// Minimax coefficients for the library polynomials. **Generated file.**
///
/// Regenerate with `python experiments/m3_fit/generate_coefficients.py` for the
/// `exp` and `exp2` blocks and with
/// `python experiments/m4_transcendental/generate_coefficients.py` for the rest.
/// Every number below is a float32 rendered from a 40-digit fit, and the bit
/// pattern next to it is what the library actually computes with.
///
/// The `exp` and `exp2` blocks are re-emitted verbatim by both scripts rather
/// than refitted: their measured bounds belong to those exact bit patterns.
///
/// Change a coefficient and the declared ULP bound stops applying: rerun
/// `tools/measure_bounds.cpp` before touching the table it feeds."""


def read_block(path, name):
    """Return the exact source text of a generated block, doc comment included."""
    with open(path, 'r', encoding='ascii') as handle:
        text = handle.read()
    for match in BLOCK_RE.finditer(text):
        if match.group(2) == name:
            return match.group(1)
    raise RuntimeError('%s has no block named %s' % (path, name))


def emit_rsqrt(lines, meta):
    low, high = meta['window']
    lines.append('/// rsqrt: the bit-trick estimate `magic - (bits >> 1)` followed by three')
    lines.append('/// Newton steps in the `y + y*t` form, `t = 0.5 - M*y^2/2`. That form is not')
    lines.append('/// cosmetic: the textbook `y * (1.5 - M*y^2/2)` rounds a quantity that is')
    lines.append('/// already near 1, which costs a whole ULP on the last iteration, while `t` is')
    lines.append('/// near zero when the iterate is accurate, so the rounding costs nothing.')
    lines.append('///')
    lines.append('/// Provenance: `magic` minimises the maximum ULP distance of the whole')
    lines.append('/// three-step pipeline over `M` in `[1, 4)`, searched exhaustively over')
    lines.append('/// `[0x%08X, 0x%08X)` on a 8000-point mantissa grid, then verified on the'
                 % (low, high))
    lines.append('/// complete interval: %d points, max %d ULP, %.2f %% exactly rounded.'
                 % (meta['sweep_points'], meta['sweep_max_ulp'], meta['sweep_exact_pct']))
    lines.append('/// The final `2^-k` scaling is an exact exponent-field move, so the sweep')
    lines.append('/// covers every normal and every subnormal input.')
    lines.append('namespace rsqrt_coeffs {')
    lines.append('')
    lines.append('constexpr std::uint32_t magic = 0x%08Xu;' % meta['magic'])
    lines.append('constexpr int newton_steps = %d;' % meta['steps'])
    lines.append('')
    lines.append('}  // namespace rsqrt_coeffs')
    lines.append('')


def emit_log(lines, meta):
    peak, at_peak, total, exact_percent = meta['model']
    peak2, at_peak2, total2, exact_percent2 = meta['model2']
    peak3, at_peak3, total3, exact_percent3 = meta['model1p']
    lines.append('/// log, log2, log1p share one reduction: x = m * 2^e with m in')
    lines.append('/// [sqrt(1/2), sqrt(2)), so `m - 1` is exact by Sterbenz and')
    lines.append('/// s = (m - 1)/(m + 1) lands in [-0.17157, 0.17157]. Then')
    lines.append('/// log(m) = 2 atanh(s) = 2s + 2s*z*R(z) with z = s^2, and the series is')
    lines.append('/// evaluated with its leading 2s kept out of the polynomial: the correction is')
    lines.append('/// O(s^3), so the roundings it pays for are invisible next to 2s.')
    lines.append('///')
    lines.append('/// R is fitted by `experiments/m4_transcendental/generate_coefficients.py`:')
    lines.append('/// Remez exchange on a relative-error metric, %d iteration(s), %s, relative'
                 % (meta['log_fit']['iterations'],
                    'converged' if meta['log_fit']['converged'] else 'no convergence'))
    lines.append('/// residual %.2e at 40 decimal digits, degree %d in z (the series through'
                 % (meta['log_fit']['residual'], LOG_DEGREE))
    lines.append('/// s^9). Modelled over the assembled pipeline on %d points: max %d ULP'
                 % (total, peak))
    lines.append('/// (%.2f %% exactly rounded). The declared bound is the golden measurement,'
                 % exact_percent)
    lines.append('/// not this model.')
    lines.append('///')
    lines.append('/// The denominator of `s` is the one quantity that is silently wrong:')
    lines.append('/// `h = 2 + t` rounds, and `s` is an exact numerator over a rounded')
    lines.append('/// denominator, which the series cannot see because the series is a')
    lines.append('/// function of `s`. Its residual `d_lo = (2 - h) + t` is exact and costs one')
    lines.append('/// two-sum, and `log(m)` pays `-t*d_lo/(2m)` for it. `inv_two_m` is')
    lines.append('/// `1/(2 + 2t)` over the whole reduction interval, degree %d, relative'
                 % INV_TWO_M_DEGREE)
    lines.append('/// residual %.2e: the correction it scales is under 4e-8, so its own'
                 % meta['inv_two_m_fit']['residual'])
    lines.append('/// rounding lands two orders of magnitude below one ULP of the result.')
    lines.append('namespace log_coeffs {')
    lines.append('')
    for index, value in enumerate(meta['tail']):
        lines.append('constexpr float r%d = %s;  // 0x%08X'
                     % (index, float_literal(value), hex_of(value)))
    lines.append('')
    for index, value in enumerate(meta['inv_two_m']):
        lines.append('constexpr float inv_two_m_%d = %s;  // 0x%08X'
                     % (index, float_literal(value), hex_of(value)))
    lines.append('')
    lines.append('constexpr float ln2_hi = %s;  // 0x%08X, 12 significand bits'
                 % (float_literal(meta['ln2_hi']), hex_of(meta['ln2_hi'])))
    lines.append('constexpr float ln2_lo = %s;  // 0x%08X'
                 % (float_literal(meta['ln2_lo']), hex_of(meta['ln2_lo'])))
    lines.append('constexpr float series_max = %s;  // 0x%08X, sqrt(2) - 1'
                 % (float_literal(meta['series_max']), hex_of(meta['series_max'])))
    lines.append('')
    lines.append('}  // namespace log_coeffs')
    lines.append('')
    lines.append('/// log2: the same series with log2(e) instead of 1 in front of the leading')
    lines.append('/// term, fitted separately rather than obtained by scaling a float. R2 is a')
    lines.append('/// Remez fit on the same interval, relative residual %.2e, and the pipeline'
                 % meta['log2_fit']['residual'])
    lines.append('/// models %d ULP on %d points (%.2f %% exactly rounded).'
                 % (peak2, total2, exact_percent2))
    lines.append('namespace log2_coeffs {')
    lines.append('')
    for index, value in enumerate(meta['tail2']):
        lines.append('constexpr float r%d = %s;  // 0x%08X'
                     % (index, float_literal(value), hex_of(value)))
    lines.append('')
    lines.append('constexpr float log2_e = %s;  // 0x%08X'
                 % (float_literal(meta['log2_e']), hex_of(meta['log2_e'])))
    lines.append('')
    lines.append('}  // namespace log2_coeffs')
    lines.append('')
    lines.append('/// log1p: the same coefficients as log, which is why the header shares them')
    lines.append('/// instead of fitting a second series. The short branch covers')
    lines.append('/// |x| <= series_max, where s = x/(2+x) keeps |s| inside the fitted interval;')
    lines.append('/// modelled %d ULP on %d points (%.2f %% exactly rounded).'
                 % (peak3, total3, exact_percent3))
    lines.append('')


def emit_tanh(lines, meta):
    peak, at_peak, total, exact_percent = meta['model']
    lines.append('/// tanh: two branches, both with generated constants.')
    lines.append('///')
    lines.append('/// Below poly_max the function is an odd polynomial,')
    lines.append('/// tanh(x) = x + x^3*R(x^2): the leading x is exact and the correction is at')
    lines.append('/// most 12 % of the result, so the roundings R pays for are invisible. R is a')
    lines.append('/// Remez fit on y = x^2 in [0, poly_max^2], %d iteration(s), %s, relative'
                 % (meta['fit']['iterations'],
                    'converged' if meta['fit']['converged'] else 'no convergence'))
    lines.append('/// residual %.2e at 40 decimal digits, degree %d.'
                 % (meta['fit']['residual'], TANH_DEGREE))
    lines.append('///')
    lines.append('/// Above it, tanh(x) = (u-1)/(u+1) with u = 2^(x*two_log2_e); the argument is')
    lines.append('/// a single product, because folding `2` into the constant of `log2(e) is one')
    lines.append('/// rounding instead of two. `saturation` is the smallest float32 whose')
    lines.append('/// correctly rounded tanh is 1.0, found by bisection over the bit patterns;')
    lines.append('/// past it the answer is a sign, not an approximation.')
    lines.append('///')
    lines.append('/// Modelled over the assembled pipeline on %d points: max %d ULP (%.2f %%'
                 % (total, peak, exact_percent))
    lines.append('/// exactly rounded). The declared bound is the golden measurement.')
    lines.append('namespace tanh_coeffs {')
    lines.append('')
    for index, value in enumerate(meta['tail']):
        lines.append('constexpr float r%d = %s;  // 0x%08X'
                     % (index, float_literal(value), hex_of(value)))
    lines.append('')
    lines.append('constexpr float two_log2_e = %s;  // 0x%08X, 2*log2(e)'
                 % (float_literal(meta['two_log2_e']), hex_of(meta['two_log2_e'])))
    lines.append('constexpr float poly_max = %s;  // 0x%08X'
                 % (float_literal(meta['poly_max']), hex_of(meta['poly_max'])))
    lines.append('constexpr float saturation = %s;  // 0x%08X'
                 % (float_literal(gr.bits_to_float(meta['saturation'])), meta['saturation']))
    lines.append('')
    lines.append('}  // namespace tanh_coeffs')
    lines.append('')


def emit(path, blocks):
    """Re-emit the file: preserved blocks first, generated ones after them."""
    lines = PROVENANCE_HEADER.split('\n')
    lines.append('')
    lines.append('namespace ulpgrad {')
    lines.append('namespace detail {')
    lines.append('')
    body = []
    for text in blocks:
        body.append(text.rstrip('\n'))
    lines.append('\n\n'.join(body))
    lines.append('')
    lines.append('}  // namespace detail')
    lines.append('}  // namespace ulpgrad')
    lines.append('')
    text = '\n'.join(lines)
    text = text.replace('#pragma once\n\n', '#pragma once\n\n#include <cstdint>\n\n', 1)
    with open(path, 'w', encoding='ascii', newline='\n') as handle:
        handle.write(text)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=OUT_PATH)
    parser.add_argument('--sections', default='rsqrt,log,tanh',
                        help='comma-separated subset of rsqrt, log, tanh')
    args = parser.parse_args(argv)

    mpmath.mp.dps = MP_DPS
    wanted = [name.strip() for name in args.sections.split(',') if name.strip()]

    preserved = [read_block(args.out, 'exp_coeffs'), read_block(args.out, 'exp2_coeffs')]
    generated = []

    if 'rsqrt' in wanted:
        meta = fit_rsqrt()
        block = []
        emit_rsqrt(block, meta)
        generated.append('\n'.join(block))

    if 'log' in wanted:
        meta = fit_log()
        block = []
        emit_log(block, meta)
        generated.append('\n'.join(block))

    if 'tanh' in wanted:
        meta = fit_tanh()
        block = []
        emit_tanh(block, meta)
        generated.append('\n'.join(block))

    emit(args.out, preserved + generated)
    print('wrote %s' % os.path.relpath(args.out, REPO))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
