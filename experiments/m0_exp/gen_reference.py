#!/usr/bin/env python3
"""Correctly rounded float32 ``exp`` reference plus a stratified input sample.

Outputs, both binary so the C++ side never parses text:

  ref.bin        8-byte little-endian records: uint32 x_bits, uint32 ref_bits
  manifest.txt   one line per sample category: "<category> <first_index> <count>"

The reference is ``mp.exp(x)`` at 40 decimal digits, rounded to float32 by an
explicit round-to-nearest-even implementation (:func:`round_to_f32_bits`).
Going through ``float()`` would double round (53-bit significand, then 24-bit),
which disagrees with direct rounding on exact ties.

Sample categories, in this order:

  uniform      1e6 points uniform in [-30, 30], deterministic LCG
  structural   all 254 normal exponents x 4 mantissa quarters, both signs
  borders      k*ln2 and (k+-1/2)*ln2 for every k in range, +-2 ULP each;
               overflow / subnormal / underflow thresholds +-2 ULP
  subnormal    5000 deterministic subnormal arguments, both signs
  specials     NaN with payload, +-inf, +-0, FLT_MAX, smallest subnormal

Run from the repository directory (never from a directory containing a stray
``numbers.py``, which shadows the standard library and breaks ``import mpmath``)::

    python experiments/m0_exp/gen_reference.py
"""

import argparse
import os
import struct
import sys
import time

import mpmath
import numpy as np

MP_DPS = 40

MASK64 = 0xFFFFFFFFFFFFFFFF
LCG_MUL = 6364136223846793005
LCG_INC = 1442695040888963407

F32_INF = 0x7F800000
F32_MAX = 0x7F7FFFFF
F32_MIN_SUB = 0x00000001


# --------------------------------------------------------------------------- #
# bits <-> float
# --------------------------------------------------------------------------- #


def bits_to_float(bits):
    """Reinterpret a uint32 bit pattern as a Python float (exact)."""
    return struct.unpack('<f', struct.pack('<I', int(bits)))[0]


def float_to_bits(value):
    """Reinterpret a Python float as a uint32 bit pattern (exact)."""
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]


def is_nan_bits(bits):
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def is_inf_bits(bits):
    return (bits & 0x7FFFFFFF) == F32_INF


# --------------------------------------------------------------------------- #
# deterministic LCG (Knuth MMIX constants); top 32 bits of each draw
# --------------------------------------------------------------------------- #


def lcg_u32(n, seed):
    state = seed & MASK64

    def draw():
        nonlocal state
        for _ in range(n):
            state = (state * LCG_MUL + LCG_INC) & MASK64
            yield state >> 32

    return np.fromiter(draw(), dtype=np.uint32, count=n)


# --------------------------------------------------------------------------- #
# round to nearest even, float32 -- the reference arithmetic
# --------------------------------------------------------------------------- #


def _round_half_even(x):
    """Round a non-negative mpf to the nearest integer, ties to even."""
    floor = mpmath.floor(x)
    frac = x - floor
    n = int(floor)
    half = mpmath.mpf(1) / 2
    if frac > half:
        return n + 1
    if frac < half:
        return n
    return n if (n % 2 == 0) else n + 1


def round_to_f32_bits(value):
    """Round an exact mpf to the nearest float32 (RNE); return the raw bits.

    Implemented on the significand, not via ``float()``: a double round
    through binary64 would differ from direct rounding on ties.
    """
    if mpmath.isnan(value):
        raise ValueError('NaN has no correctly rounded bit pattern')
    if mpmath.isinf(value):
        return F32_INF if value > 0 else (F32_INF | 0x80000000)
    if value == 0:
        return 0
    sign = 0x80000000 if value < 0 else 0
    magnitude = abs(value)
    mant, expo = mpmath.frexp(magnitude)       # magnitude = mant * 2**expo
    exp = expo - 1                             # 2**exp <= magnitude < 2**(exp+1)
    if exp < -126:
        # Subnormal: magnitude = q * 2**-149 with q in [0, 2**23).
        q = _round_half_even(mpmath.ldexp(mant, expo + 149))
        if q >= (1 << 23):
            return sign | 0x00800000           # rounded up to the smallest normal
        return sign | q
    q = _round_half_even(mpmath.ldexp(mant, 24))   # 24-bit significand
    if q == (1 << 24):
        q >>= 1
        exp += 1
    if exp > 127:
        return sign | F32_INF                  # overflow to infinity
    return sign | ((exp + 127) << 23) | (q & 0x7FFFFF)


# --------------------------------------------------------------------------- #
# self-test of the rounding routine
# --------------------------------------------------------------------------- #


def selftest(verbose=True):
    def check(name, ok):
        if not ok:
            raise AssertionError('selftest failed: ' + name)
        if verbose:
            print('  ok  ' + name)

    mp = mpmath

    # 1. Rounding a float32 value must return the same bits (idempotence).
    draws = lcg_u32(200000, 0xA5A5A5A5)
    bad = 0
    for raw in draws:
        b = int(raw)
        if is_nan_bits(b):
            continue
        if round_to_f32_bits(mp.mpf(bits_to_float(b))) != b:
            bad += 1
    check('idempotence on 200k random patterns (%d mismatches)' % bad, bad == 0)

    # 2. Exact midpoints between adjacent floats must round to the even one.
    bad = 0
    ties = 0
    for raw in lcg_u32(20000, 0x1234ABCD):
        b = (int(raw) & 0x007FFFFF) | 0x3F000000   # normal, exponent 126
        if b >= 0x7F7FFFFF:
            continue
        v0 = mp.mpf(bits_to_float(b))
        v1 = mp.mpf(bits_to_float(b + 1))
        mid = (v0 + v1) / 2
        even = b if (b & 1) == 0 else b + 1
        if round_to_f32_bits(mid) != even:
            bad += 1
        # A hair below the midpoint rounds down, a hair above rounds up.
        lo = mid - mp.mpf(2) ** (-80) * v0
        hi = mid + mp.mpf(2) ** (-80) * v0
        if round_to_f32_bits(lo) != b:
            bad += 1
        if round_to_f32_bits(hi) != b + 1:
            bad += 1
        ties += 1
    check('ties-to-even on %d midpoints' % ties, bad == 0)

    # 3. Directed overflow / underflow boundaries.
    p128 = mp.mpf(2) ** 128
    half_ulp_max = mp.mpf(2) ** 103            # half the distance FLT_MAX -> 2**128
    check('2**128 - 2**103 - eps -> FLT_MAX',
          round_to_f32_bits(p128 - half_ulp_max - 1) == F32_MAX)
    check('2**128 - 2**103 (tie) -> +inf',
          round_to_f32_bits(p128 - half_ulp_max) == F32_INF)
    check('2**128 - 2**103 + eps -> +inf',
          round_to_f32_bits(p128 - half_ulp_max + 1) == F32_INF)
    check('2**-149 -> smallest subnormal',
          round_to_f32_bits(mp.mpf(2) ** -149) == F32_MIN_SUB)
    check('2**-150 (tie) -> +0', round_to_f32_bits(mp.mpf(2) ** -150) == 0)
    check('2**-150 + eps -> smallest subnormal',
          round_to_f32_bits(mp.mpf(2) ** -150 + mp.mpf(2) ** -200) == F32_MIN_SUB)
    check('-(2**-149) -> -smallest subnormal',
          round_to_f32_bits(-(mp.mpf(2) ** -149)) == (0x80000000 | F32_MIN_SUB))

    # 4. Agreement with numpy on float64 inputs (single rounding, no ties).
    doubles = np.random.default_rng(7).standard_normal(50000) * 1e3
    bad = 0
    for d in doubles:
        if round_to_f32_bits(mp.mpf(float(d))) != float_to_bits(np.float32(d)):
            bad += 1
    check('agreement with numpy on 50k float64 inputs (%d mismatches)' % bad, bad == 0)


# --------------------------------------------------------------------------- #
# sample construction
# --------------------------------------------------------------------------- #


def uniform_sample(n, seed):
    """n points uniform in [-30, 30]; deterministic LCG, no numpy RNG."""
    u = (lcg_u32(n, seed) >> np.uint32(8)).astype(np.float64) / float(1 << 24)
    x = (-30.0 + 60.0 * u).astype(np.float32)
    return x.view(np.uint32)


def structural_sample():
    """254 normal exponents x 4 mantissa quarters (0, 1/4, 1/2, 3/4), both signs."""
    exps = np.arange(1, 255, dtype=np.uint32)
    fracs = np.array([0x000000, 0x200000, 0x400000, 0x600000], dtype=np.uint32)
    pos = ((exps[:, None] << np.uint32(23)) | fracs[None, :]).ravel()
    neg = pos | np.uint32(0x80000000)
    return np.concatenate([pos, neg]).astype(np.uint32)


def _bisect_bits(lo, hi, pred):
    """Smallest b in [lo, hi] where the monotone predicate pred(bits) holds."""
    while lo < hi:
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _neighbours(bits, reach=2):
    """bits + d for d in [-reach, +reach], dropping patterns that are not finite."""
    out = []
    for d in range(-reach, reach + 1):
        b = bits + d
        if b < 0 or b > 0xFFFFFFFF:
            continue
        if is_nan_bits(b) or is_inf_bits(b):
            continue
        out.append(b)
    return out


def first_overflow_bits():
    """Smallest float32 x whose correctly rounded exp(x) is +inf."""
    def rounds_to_inf(bits):
        return round_to_f32_bits(mpmath.exp(mpmath.mpf(bits_to_float(bits)))) == F32_INF

    return _bisect_bits(0x42000000, 0x43000000, rounds_to_inf)   # 32.0 .. 128.0


def last_underflow_bits():
    """Largest float32 x whose correctly rounded exp(x) is +0."""
    def rounds_to_zero(bits):
        return round_to_f32_bits(mpmath.exp(mpmath.mpf(bits_to_float(bits)))) == 0

    # For negative x, larger bit patterns mean smaller values, so the predicate
    # is monotone in the raw bits; the first pattern that rounds to zero is the
    # last x that still does.
    return _bisect_bits(0xC2C00000, 0xC3000000, rounds_to_zero)  # -96.0 .. -128.0


def border_sample():
    """Reduction boundaries, thresholds and their +-2 ULP neighbourhoods."""
    mp = mpmath
    ln2 = mp.log(2)
    out = []

    # k*ln2 (r == 0) and (k +- 1/2)*ln2 (|r| == ln2/2, the worst case of the fit).
    for k in range(-150, 129):
        for off in (mp.mpf(0), mp.mpf(1) / 2, mp.mpf(-1) / 2):
            v = (k + off) * ln2
            out.extend(_neighbours(round_to_f32_bits(v)))

    # Overflow: first float32 whose correctly rounded exp is +inf.
    out.extend(_neighbours(first_overflow_bits(), reach=4))

    # Underflow to zero.
    out.extend(_neighbours(last_underflow_bits(), reach=4))

    # Subnormal onset: first float32 whose exp is subnormal (2**-126).
    def is_subnormal_result(bits):
        v = round_to_f32_bits(mp.exp(mp.mpf(bits_to_float(bits))))
        return (v & 0x7F800000) == 0 and (v & 0x007FFFFF) != 0

    sub = _bisect_bits(0xC2A00000, 0xC2C00000, is_subnormal_result)  # -80.0 .. -96.0
    out.extend(_neighbours(sub, reach=4))

    # The decimal thresholds named in the plan, plus exact ln(2**-150) and ln(2**-126).
    for lit in (88.72, 88.72283935546875, -103.97, -103.97207708399179,
                -87.33654475055918):
        try:
            out.extend(_neighbours(round_to_f32_bits(mp.mpf(lit)), reach=4))
        except ValueError:
            pass

    seen = set()
    uniq = []
    for b in out:
        if b not in seen:
            seen.add(b)
            uniq.append(b)
    return np.array(uniq, dtype=np.uint32)


def subnormal_sample(n, seed):
    """n deterministic subnormal arguments, alternating signs."""
    raw = lcg_u32(n, seed)
    mag = (raw % np.uint32(0x7FFFFF)).astype(np.uint32) + np.uint32(1)
    sign = np.where((raw & np.uint32(1)) != 0, np.uint32(0x80000000), np.uint32(0)).astype(
        np.uint32)
    return (mag | sign).astype(np.uint32)


def special_sample():
    bits = np.array([
        0x7FC00001,   # quiet NaN, non-zero payload
        0x7F800001,   # signalling NaN, non-zero payload
        0xFFC00000,   # negative quiet NaN
        0x7F800000,   # +inf
        0xFF800000,   # -inf
        0x00000000,   # +0
        0x80000000,   # -0
        F32_MAX,      # largest finite
        F32_MIN_SUB,  # smallest positive subnormal
        0x80000001,   # smallest negative subnormal
    ], dtype=np.uint32)
    return bits


# --------------------------------------------------------------------------- #
# reference
# --------------------------------------------------------------------------- #


def reference_bits(xbits, label, progress_every=250000):
    """Correctly rounded float32 exp for every input pattern in xbits."""
    ref = np.empty(xbits.size, dtype=np.uint32)
    start = time.perf_counter()
    for i in range(xbits.size):
        b = int(xbits[i])
        if is_nan_bits(b):
            # Contract: NaN propagates with its payload intact (checked bit-exactly
            # by the measurement program, kept out of the ULP statistics).
            ref[i] = np.uint32(b)
            continue
        if is_inf_bits(b):
            ref[i] = np.uint32(F32_INF if b == F32_INF else 0)
            continue
        value = mpmath.mpf(bits_to_float(b))     # exact: float32 -> binary64 -> mpf
        ref[i] = np.uint32(round_to_f32_bits(mpmath.exp(value)))
        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.perf_counter() - start
            rate = (i + 1) / elapsed
            print('  %-11s %8d/%d  %.0f pts/s  eta %.0f s'
                  % (label, i + 1, xbits.size, rate,
                     (xbits.size - i - 1) / rate))
            sys.stdout.flush()
    return ref


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uniform', type=int, default=1000000,
                        help='number of uniform points in [-30, 30] (default 1e6)')
    parser.add_argument('--seed', type=int, default=0x5EED0001)
    parser.add_argument('--out-dir', default=os.path.dirname(os.path.abspath(__file__)),
                        help='where to write ref.bin and manifest.txt (default: this directory)')
    parser.add_argument('--skip-selftest', action='store_true')
    args = parser.parse_args(argv)

    mpmath.mp.dps = MP_DPS
    print('mpmath dps = %d' % mpmath.mp.dps)

    if not args.skip_selftest:
        print('self-test of the float32 rounding routine')
        selftest()

    categories = [
        ('uniform', uniform_sample(args.uniform, args.seed)),
        ('structural', structural_sample()),
        ('borders', border_sample()),
        ('subnormal', subnormal_sample(5000, args.seed ^ 0x9E3779B9)),
        ('specials', special_sample()),
    ]

    xbits = np.concatenate([c[1] for c in categories]).astype(np.uint32)
    print('sample points: %d' % xbits.size)

    ref = np.empty(xbits.size, dtype=np.uint32)
    offset = 0
    for name, block in categories:
        ref[offset:offset + block.size] = reference_bits(block, name)
        offset += block.size

    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    records = np.empty((xbits.size, 2), dtype='<u4')
    records[:, 0] = xbits
    records[:, 1] = ref
    records.tofile(out + '/ref.bin')

    with open(out + '/manifest.txt', 'w', encoding='ascii', newline='\n') as fh:
        offset = 0
        for name, block in categories:
            fh.write('%s %d %d\n' % (name, offset, block.size))
            offset += block.size

    nans = sum(1 for b in xbits if is_nan_bits(int(b)))
    print('wrote %s/ref.bin (%d records, %d bytes, %d NaN inputs)'
          % (out, xbits.size, xbits.size * 8, nans))
    print('wrote %s/manifest.txt' % out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
