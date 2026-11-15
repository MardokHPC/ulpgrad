#!/usr/bin/env python3
"""Minimax coefficients for Q(r) = (exp(r) - 1) / r on |r| <= ln2/2.

Primary path: Remez exchange (extremal points, <= 10 iterations) in mpmath at
40 decimal digits, on a relative-error metric. Fallback if Remez fails to
converge: weighted least squares (weight 1/|Q(r)|) on a dense grid, verified
afterwards on a 200k-point grid.

The mpmath coefficients are then rendered to float32 and refined with a Ziv-style
coordinate descent: each coefficient is moved by +-2 ULP, and the move is kept
if it lowers the measured maximum ULP error of the *whole* exp pipeline on a
tuning grid. Coefficients that survive rounding are the only ones that matter:
the fit is never reported at 40 digits and then shipped at 24 bits.

Writes ``coeffs.h`` next to this script. Run from the repository directory::

    python experiments/m0_exp/fit_coeffs.py
"""

import argparse
import os
import sys

import mpmath
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gen_reference as gr

MP_DPS = 40
LN2 = None            # set in main(), after mp.dps is configured
HI = None             # ln2/2


# --------------------------------------------------------------------------- #
# the fitted function
# --------------------------------------------------------------------------- #


def q_func(r):
    """(exp(r) - 1) / r, with the removable singularity at r = 0 filled in."""
    if r == 0:
        return mpmath.mpf(1)
    return mpmath.expm1(r) / r


# --------------------------------------------------------------------------- #
# Remez exchange
# --------------------------------------------------------------------------- #


def _chebyshev_extrema(degree, lo, hi):
    m = degree + 2                              # one more point than coefficients
    mid = (lo + hi) / 2
    half = (hi - lo) / 2
    return [mid + half * mpmath.cos(mpmath.pi * j / (m - 1)) for j in range(m)]


def _solve_system(degree, nodes):
    """Solve for the coefficients and the levelled relative error E."""
    n = degree + 1
    mat = mpmath.matrix(n + 1, n + 1)
    rhs = mpmath.matrix(n + 1, 1)
    for j, x in enumerate(nodes):
        fx = q_func(x)
        for i in range(n):
            mat[j, i] = x ** i / fx
        mat[j, n] = mpmath.mpf((-1) ** j)
        rhs[j] = 1
    sol = mpmath.lu_solve(mat, rhs)
    coeffs = [sol[i] for i in range(n)]
    return coeffs, sol[n]


def _eval_rel_error(coeffs, xs):
    """Relative error of the polynomial against Q on the points xs."""
    out = []
    for x in xs:
        acc = mpmath.mpf(0)
        for c in reversed(coeffs):
            acc = acc * x + c
        out.append(acc / q_func(x) - 1)
    return out


def _local_abs_maxima(xs, es):
    """Indices of the local maxima of |e| on an ordered grid, endpoints included."""
    if not xs:
        return []
    mags = [abs(e) for e in es]
    idx = [0] if mags[0] >= mags[1] else []
    for i in range(1, len(xs) - 1):
        if mags[i] > mags[i - 1] and mags[i] >= mags[i + 1]:
            idx.append(i)
    if mags[-1] >= mags[-2]:
        idx.append(len(xs) - 1)
    return idx


def _alternates(points):
    signs = [1 if e > 0 else -1 for _, e in points]
    return all(signs[i] != signs[i + 1] for i in range(len(signs) - 1))


def _pick_extrema(cands, m):
    """Select m extremal points with alternating signs, or None."""
    if len(cands) == m and _alternates(cands):
        return cands
    best = None
    for start in range(0, len(cands) - m + 1):
        window = cands[start:start + m]
        if not _alternates(window):
            continue
        score = min(abs(e) for _, e in window)
        if best is None or score > best[0]:
            best = (score, window)
    return None if best is None else best[1]


def remez(degree, lo, hi, grid=8000, max_iter=10, verbose=True):
    """Remez exchange on the relative error; returns (coeffs, iterations, converged)."""
    nodes = _chebyshev_extrema(degree, lo, hi)
    xt = [lo + (hi - lo) * i / (grid - 1) for i in range(grid)]
    best = None
    prev_max = None
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        coeffs, level = _solve_system(degree, nodes)
        es = _eval_rel_error(coeffs, xt)
        peak = max(abs(e) for e in es)
        if verbose:
            print('    remez iter %2d: level %.6e   max|e| on grid %.6e' % (it, float(abs(level)),
                                                                           float(peak)))
        if best is None or peak < best[0]:
            best = (peak, coeffs)
        if prev_max is not None and peak >= prev_max * (1 - 1e-12):
            converged = True
            break
        prev_max = peak

        idx = _local_abs_maxima(xt, es)
        cands = [(xt[i], es[i]) for i in idx]
        picked = _pick_extrema(cands, degree + 2)
        if picked is None:
            if verbose:
                print('    remez iter %2d: no alternating set of %d extrema, stopping'
                      % (it, degree + 2))
            break
        nodes = [x for x, _ in picked]

    return best[1], it, converged


# --------------------------------------------------------------------------- #
# fallback: weighted least squares
# --------------------------------------------------------------------------- #


def fit_lstsq(degree, lo, hi, grid=20000):
    """Weighted least squares with weight 1/|Q|, on a Chebyshev basis."""
    ts = np.linspace(-1.0, 1.0, grid)
    mid = float((lo + hi) / 2)
    half = float((hi - lo) / 2)
    rs = mid + half * ts
    fs = np.array([float(q_func(mpmath.mpf(r))) for r in rs])
    vander = np.polynomial.chebyshev.chebvander(ts, degree)
    weights = 1.0 / np.abs(fs)
    a = vander * weights[:, None]
    b = weights
    cheb, *_ = np.linalg.lstsq(a, b, rcond=None)

    # Chebyshev in t -> monomial in t -> monomial in r (t = (r - mid) / half).
    mono_t = np.polynomial.chebyshev.cheb2poly(cheb)
    shift = np.polynomial.polynomial.Polynomial([-mid / half, 1.0 / half])
    acc = np.polynomial.polynomial.Polynomial([0.0])
    for j, bj in enumerate(mono_t):
        acc = acc + bj * shift ** j
    coeffs = [mpmath.mpf(float(c)) for c in acc.coef]
    while len(coeffs) < degree + 1:
        coeffs.append(mpmath.mpf(0))
    return coeffs[:degree + 1]


# --------------------------------------------------------------------------- #
# float32 rendering and Ziv refinement
# --------------------------------------------------------------------------- #


def to_float32(coeffs):
    return np.array([np.float32(float(c)) for c in coeffs], dtype=np.float32)


def ulp_shift(value, steps):
    """Move a positive float32 by `steps` ULP."""
    bits = gr.float_to_bits(float(np.float32(value)))
    return np.float32(gr.bits_to_float(bits + steps))


class Pipeline:
    """binary64 model of m0_exp(): every float32 step rounded as float32.

    Products are formed exactly in binary64 (24x24 bits fit) and the sum is
    rounded once to float32, which is what fmaf does; plain float32 adds are
    formed in binary64 and rounded once, which is what the hardware does.
    """

    def __init__(self, coeffs, inv_ln2, ln2_hi, ln2_lo, ovf, unf):
        self.c = np.array(coeffs, dtype=np.float32)
        self.inv_ln2 = np.float32(inv_ln2)
        self.ln2_hi = np.float32(ln2_hi)
        self.ln2_lo = np.float32(ln2_lo)
        self.ovf = np.float32(ovf)
        self.unf = np.float32(unf)

    def reduction(self, x):
        product = np.float32(np.float64(x) * np.float64(self.inv_ln2))
        kf = np.float32(np.rint(product))
        r = np.float32(np.float64(x) - np.float64(kf) * np.float64(self.ln2_hi))
        r = np.float32(np.float64(r) - np.float64(kf) * np.float64(self.ln2_lo))
        return kf, r

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        kf, r = self.reduction(x)
        top = len(self.c) - 1
        p = np.float32(self.c[top])
        for i in range(top - 1, -1, -1):
            p = np.float32(np.float64(p) * np.float64(r) + np.float64(self.c[i]))
        y = np.float32(np.float64(p) * np.float64(r) + 1.0)
        out = np.ldexp(y.astype(np.float64), kf.astype(np.int32)).astype(np.float32)
        out = np.where(x >= self.ovf, np.float32(np.inf), out)
        out = np.where(x <= self.unf, np.float32(0.0), out)
        return out


def _ulp_keys_from_bits(bits):
    """Map uint32 float patterns to a monotonically ordered int64 key."""
    b = np.asarray(bits, dtype=np.uint32)
    signed = b.astype(np.int32).astype(np.int64)
    neg = (b & np.uint32(0x80000000)) != 0
    return np.where(neg, -(b & np.uint32(0x7FFFFFFF)).astype(np.int64), signed)


def score(pipeline, xs, ref_bits):
    """Maximum ULP distance of the modelled pipeline against the reference."""
    got = pipeline(xs)
    dist = np.abs(_ulp_keys_from_bits(got.view(np.uint32))
                  - _ulp_keys_from_bits(ref_bits))
    dist = dist[np.isfinite(got)]
    peak = int(dist.max()) if dist.size else 0
    return peak, int(np.count_nonzero(dist == peak))


def ziv_refine(pipeline, xs, ref_bits, passes=3, reach=2, verbose=True):
    """Coordinate descent over +-reach ULP of each coefficient.

    Returns (coefficients, (max_ulp, points_at_max), passes_used).
    """
    best_coeffs = pipeline.c.copy()
    best_score = score(pipeline, xs, ref_bits)
    if verbose:
        print('    ziv start: max %d ulp (%d points at max)' % best_score)
    used = 0
    for p in range(passes):
        improved = False
        for j in range(len(best_coeffs)):
            for step in (1, -1, 2, -2):
                if abs(step) > reach:
                    continue
                cand = best_coeffs.copy()
                cand[j] = ulp_shift(cand[j], step)
                cand_pipeline = Pipeline(cand, pipeline.inv_ln2, pipeline.ln2_hi,
                                         pipeline.ln2_lo, pipeline.ovf, pipeline.unf)
                s = score(cand_pipeline, xs, ref_bits)
                if s < best_score:
                    best_score = s
                    best_coeffs = cand
                    pipeline.c = cand
                    improved = True
                    if verbose:
                        print('    ziv pass %d: c%d %+d ulp -> max %d ulp (%d points)'
                              % (p + 1, j, step, s[0], s[1]))
        used = p + 1
        if not improved:
            break
    pipeline.c = best_coeffs
    return best_coeffs, best_score, used


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def _float_literal(value):
    """C++ float literal that round-trips a float32 (%.9g needs a '.' guard)."""
    text = '%.9g' % float(value)
    if ('e' not in text) and ('E' not in text) and ('.' not in text):
        text += '.0'
    return text + 'f'


def emit_header(path, coeffs, meta):
    lines = []
    lines.append('#pragma once')
    lines.append('')
    lines.append('// Generated by experiments/m0_exp/fit_coeffs.py -- do not edit by hand.')
    lines.append('//')
    lines.append('// Minimax fit of Q(r) = (exp(r) - 1) / r on |r| <= ln2/2, degree %d, relative'
                 % meta['degree'])
    lines.append('// error metric. %s' % meta['method'])
    lines.append('// mpmath (%d decimal digits) max relative residual of Q on a %d-point grid: %.3e'
                 % (MP_DPS, meta['verify_grid'], meta['residual']))
    lines.append('// float32 rendering: coefficients rounded to nearest, then %d pass(es) of Ziv'
                 % meta['ziv_passes'])
    lines.append('// coordinate descent over +-2 ULP per coefficient, objective = maximum ULP')
    lines.append('// distance of the whole exp pipeline on a %d-point tuning grid (seed 0x%08X).'
                 % (meta['tuning_points'], meta['tuning_seed']))
    lines.append('// Modelled max ULP on that tuning grid: %d (not the reported number: the'
                 % meta['tuning_ulp'])
    lines.append('// reported bound comes from measure.cpp on the full sample).')
    lines.append('//')
    lines.append('// exp(x) = 2^k * (1 + r * Q(r)), Q evaluated as an explicit fmaf chain.')
    lines.append('')
    for i, c in enumerate(coeffs):
        value = float(np.float32(c))
        lines.append('static constexpr float c%d = %s;  // 0x%08X'
                     % (i, _float_literal(value), gr.float_to_bits(value)))
    lines.append('')
    lines.append('// Argument reduction constants: ln2_hi keeps 12 significand bits.')
    lines.append('static constexpr float inv_ln2 = %s;  // 0x%08X'
                 % (_float_literal(meta['inv_ln2']), gr.float_to_bits(meta['inv_ln2'])))
    lines.append('static constexpr float ln2_hi = %s;  // 0x%08X'
                 % (_float_literal(meta['ln2_hi']), gr.float_to_bits(meta['ln2_hi'])))
    lines.append('static constexpr float ln2_lo = %s;  // 0x%08X'
                 % (_float_literal(meta['ln2_lo']), gr.float_to_bits(meta['ln2_lo'])))
    lines.append('')
    lines.append('// Smallest x with round_to_nearest(exp(x)) == +inf, and largest x whose')
    lines.append('// rounded exp is +0. Computed by bisection over float32 patterns.')
    lines.append('static constexpr float overflow_threshold = %s;  // 0x%08X'
                 % (_float_literal(meta['ovf']), gr.float_to_bits(meta['ovf'])))
    lines.append('static constexpr float underflow_threshold = %s;  // 0x%08X'
                 % (_float_literal(meta['unf']), gr.float_to_bits(meta['unf'])))
    lines.append('')
    with open(path, 'w', encoding='ascii', newline='\n') as fh:
        fh.write('\n'.join(lines))


def main(argv=None):
    global LN2, HI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--degree', type=int, default=6)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--tuning-points', type=int, default=25000)
    parser.add_argument('--tuning-seed', type=lambda s: int(s, 0), default=0xC0FFEE01)
    parser.add_argument('--no-ziv', action='store_true')
    parser.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      'coeffs.h'))
    args = parser.parse_args(argv)

    mpmath.mp.dps = MP_DPS
    LN2 = mpmath.log(2)
    HI = LN2 / 2
    lo, hi = -HI, HI

    print('fitting Q(r) on [%.17g, %.17g], degree %d, mpmath dps %d'
          % (float(lo), float(hi), args.degree, mpmath.mp.dps))

    residual = None
    converged = False
    method = ''
    coeffs = None
    try:
        coeffs, iters, converged = remez(args.degree, lo, hi, max_iter=args.iterations)
        method = ('Remez exchange on Chebyshev nodes, %d iteration(s), %s.'
                  % (iters, 'converged' if converged else 'stopped without convergence'))
        verify = [lo + (hi - lo) * i / 200000 for i in range(200001)]
        es = _eval_rel_error(coeffs, verify)
        residual = max(abs(e) for e in es)
        print('remez: %d iterations, converged=%s, max relative residual %.6e'
              % (iters, converged, float(residual)))
        if float(residual) > 1e-6:
            raise RuntimeError('remez residual %.3e too large, falling back' % float(residual))
    except Exception as exc:                          # noqa: BLE001  (documented fallback)
        print('remez failed (%s); falling back to weighted least squares' % exc)
        coeffs = fit_lstsq(args.degree, lo, hi)
        verify = [lo + (hi - lo) * i / 200000 for i in range(200001)]
        es = _eval_rel_error(coeffs, verify)
        residual = max(abs(e) for e in es)
        method = 'Weighted least squares (weight 1/|Q|) on a dense grid; Remez not converged.'
        print('lstsq: max relative residual %.6e' % float(residual))

    print('residual of the mpmath polynomial: %.6e (relative, %d-point grid)'
          % (float(residual), 200001))

    # ---- float32 rendering ------------------------------------------------
    c32 = to_float32(coeffs)

    ln2_f = float(LN2)
    ln2_hi = gr.bits_to_float(gr.float_to_bits(np.float32(ln2_f)) & 0xFFFFF000)
    ln2_lo = float(np.float32(ln2_f - ln2_hi))
    inv_ln2 = float(np.float32(1.0 / ln2_f))
    ovf = gr.bits_to_float(gr.first_overflow_bits())
    unf = gr.bits_to_float(gr.last_underflow_bits())
    print('ln2_hi = %.17g  (k*ln2_hi exact for |k| <= 256)' % ln2_hi)
    print('ln2_lo = %.17g  residual of the pair: %.3e'
          % (ln2_lo, abs((ln2_hi + ln2_lo) - ln2_f)))
    print('overflow threshold  = %.17g' % ovf)
    print('underflow threshold = %.17g' % unf)

    pipeline = Pipeline(c32, inv_ln2, ln2_hi, ln2_lo, ovf, unf)

    # ---- tuning grid (different seed from the measurement sample) ---------
    n_uni = int(args.tuning_points * 0.8)
    xs_uni = gr.uniform_sample(n_uni, args.tuning_seed)
    xs_uni_f = xs_uni.view(np.float32)
    rng = np.random.default_rng(args.tuning_seed)
    ks = rng.integers(-140, 120, size=args.tuning_points - n_uni)
    us = rng.uniform(-1.0, 1.0, size=args.tuning_points - n_uni)
    xs_r = (ks.astype(np.float64) * ln2_f + us * 0.5 * ln2_f).astype(np.float32)
    xs = np.concatenate([xs_uni_f, xs_r]).astype(np.float32)
    ref = gr.reference_bits(xs.view(np.uint32), 'tuning')

    before = score(pipeline, xs, ref)
    print('tuning grid: %d points, max %d ulp before Ziv (%d points at max)'
          % (xs.size, before[0], before[1]))

    ziv_passes = 0
    final_score = before
    if not args.no_ziv:
        final_coeffs, final_score, ziv_passes = ziv_refine(pipeline, xs, ref)
        print('tuning grid: max %d ulp after Ziv (%d points at max)'
              % (final_score[0], final_score[1]))
        c32 = final_coeffs

    meta = {
        'degree': args.degree,
        'method': method,
        'verify_grid': 200001,
        'residual': float(residual),
        'ziv_passes': ziv_passes,
        'tuning_points': xs.size,
        'tuning_seed': args.tuning_seed,
        'tuning_ulp': final_score[0],
        'inv_ln2': inv_ln2,
        'ln2_hi': ln2_hi,
        'ln2_lo': ln2_lo,
        'ovf': ovf,
        'unf': unf,
    }
    emit_header(args.out, c32, meta)
    print('wrote %s' % args.out)
    print('coefficients (float32, decimal / hex):')
    for i, c in enumerate(c32):
        print('  c%d = %.9gf   %s' % (i, float(c), float(np.float32(c)).hex()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
