#!/usr/bin/env python3
"""Generated coefficient header for the polynomials in ``include/ulpgrad``.

Two blocks, one file:

``exp``
    The degree-6 minimax fit of ``Q(r) = (exp(r) - 1) / r`` on ``|r| <= ln2/2``
    produced by ``experiments/m0_exp/fit_coeffs.py`` (Remez exchange, float32
    rendering, Ziv refinement). It is **not refitted**: the hex patterns of
    ``experiments/m0_exp/coeffs.h`` are parsed and re-emitted, so the 1 ULP
    bound measured in ``experiments/m0_exp`` and ``experiments/m0_gpu``
    transfers to the library by construction.

``exp2``
    A new degree-6 minimax fit of ``Q(r) = (2**r - 1) / r`` on ``|r| <= 1/2``,
    with the same machinery: Remez exchange on a relative-error metric at 40
    decimal digits, float32 rendering, then Ziv coordinate descent over +-2 ULP
    per coefficient on the ULP distance of the whole ``exp2`` pipeline. The
    interval is the same ``r`` range ``exp`` fits, because ``2**r = exp(r ln2)``
    and ``r ln2`` also spans ``[-ln2/2, +ln2/2]``.

Writes ``include/ulpgrad/detail/coefficients.h``. Run from the repository root::

    python experiments/m3_fit/generate_coefficients.py
"""

import argparse
import os
import re
import sys

import mpmath
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'experiments', 'm0_exp'))
sys.path.insert(0, HERE)

import fit_coeffs as fc        # noqa: E402  (path has to be set first)
import gen_golden as gg        # noqa: E402
import gen_reference as gr     # noqa: E402

MP_DPS = 40
M0_COEFFS = os.path.join(REPO, 'experiments', 'm0_exp', 'coeffs.h')
OUT_PATH = os.path.join(REPO, 'include', 'ulpgrad', 'detail', 'coefficients.h')


# --------------------------------------------------------------------------- #
# the exp2 fit
# --------------------------------------------------------------------------- #


def q2(r):
    """Q(r) = (2**r - 1) / r, with the removable singularity filled in."""
    if r == 0:
        return mpmath.log(2)
    return mpmath.expm1(r * mpmath.log(2)) / r


def exp2_model(coefficients, overflow, underflow):
    """binary64 model of the exp2() pipeline, every step rounded to float32.

    A product plus add is formed exactly in binary64 and rounded once, which is
    what ``fmaf`` does; a plain float32 add is formed in binary64 and rounded
    once, which is what the hardware does; ``y * 2**k`` is exact in binary64 for
    a 24-bit significand, so the final cast rounds exactly once even on the
    subnormal grid -- the same single rounding the integer shift performs.
    """
    coeffs = np.asarray(coefficients, dtype=np.float32)

    def evaluate(xs):
        x = np.asarray(xs, dtype=np.float32)
        k = np.rint(x).astype(np.float32)                 # exact integer, |k| <= 151
        r = (x - k).astype(np.float32)                    # exact by Sterbenz
        p = coeffs[-1]
        for i in range(len(coeffs) - 2, -1, -1):
            p = np.float32(np.float64(p) * np.float64(r) + np.float64(coeffs[i]))
        y = np.float32(np.float64(p) * np.float64(r) + 1.0)
        with np.errstate(over='ignore'):                  # the guard may saturate below
            out = np.ldexp(y.astype(np.float64), k.astype(np.int32)).astype(np.float32)
        out = np.where(x >= np.float32(overflow), np.float32(np.inf), out)
        out = np.where(x <= np.float32(underflow), np.float32(0.0), out)
        return out

    return evaluate


def exp2_tuning_grid(seed, points):
    """Half dense uniform, half straddling the reduction boundaries."""
    dense = gg.uniform_bits(points // 2, seed, -20.0, 20.0).view(np.float32)
    rng = np.random.default_rng(seed)
    ks = rng.integers(-150, 129, size=points - dense.size)
    offsets = rng.uniform(-0.5, 0.5, size=ks.size)
    straddling = (ks.astype(np.float64) + offsets).astype(np.float32)
    return np.concatenate([dense, straddling]).astype(np.float32)


def fit_exp2(degree, seed, tuning_points, verbose=True):
    """Remez + float32 rendering + Ziv refinement; returns (float32 coeffs, meta)."""
    lo, hi = mpmath.mpf(-0.5), mpmath.mpf(0.5)
    original = fc.q_func
    fc.q_func = q2                                   # the fit routines read it globally
    try:
        coefficients, iterations, converged = fc.remez(degree, lo, hi, max_iter=12,
                                                      verbose=verbose)
        grid = [lo + (hi - lo) * i / 200000 for i in range(200001)]
        residual = max(abs(e) for e in fc._eval_rel_error(coefficients, grid))
    finally:
        fc.q_func = original

    rendered = fc.to_float32(coefficients)
    pipeline = exp2_model(rendered, 128.0, -150.0)

    xs = exp2_tuning_grid(seed, tuning_points)
    ref = np.array([gg.reference_one(bits, gg.ref_exp2) for bits in xs.view(np.uint32)],
                   dtype=np.uint32)
    before = fc.score(pipeline, xs, ref)
    if verbose:
        print('    tuning grid: %d points, max %d ulp before Ziv (%d points at max)'
              % (xs.size, before[0], before[1]))

    best = rendered.copy()
    best_score = before
    passes = 0
    for iteration in range(3):
        improved = False
        for index in range(best.size):
            for step in (1, -1, 2, -2):
                candidate = best.copy()
                candidate[index] = fc.ulp_shift(candidate[index], step)
                score = fc.score(exp2_model(candidate, 128.0, -150.0), xs, ref)
                if score < best_score:
                    best = candidate
                    best_score = score
                    improved = True
                    if verbose:
                        print('    ziv pass %d: c%d %+d ulp -> max %d ulp'
                              % (iteration + 1, index, step, score[0]))
        passes = iteration + 1
        if not improved:
            break
    if verbose:
        print('    tuning grid: max %d ulp after Ziv' % best_score[0])

    meta = {
        'degree': degree,
        'iterations': iterations,
        'converged': converged,
        'residual': float(residual),
        'ziv_passes': passes,
        'tuning_points': int(xs.size),
        'tuning_seed': seed,
        'tuning_ulp': best_score[0],
        'overflow': gr_float(gg.exp2_overflow_bits()),
        'underflow': gr_float(gg.exp2_underflow_bits()),
    }
    return best, meta


# --------------------------------------------------------------------------- #
# reading the validated exp coefficients
# --------------------------------------------------------------------------- #


def gr_float(bits):
    return gr.bits_to_float(int(bits))


def read_m0_coefficients(path):
    """Parse ``static constexpr float <name> = <literal>;  // 0xXXXXXXXX``."""
    pattern = re.compile(
        r'static\s+constexpr\s+float\s+(\w+)\s*=\s*([^;]+);\s*//\s*0x([0-9A-Fa-f]{8})')
    found = {}
    with open(path, 'r', encoding='ascii') as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                literal_text = match.group(2).strip().rstrip('f')
                found[match.group(1)] = (float(literal_text), int(match.group(3), 16))
    if not found:
        raise RuntimeError('no coefficients parsed from %s' % path)
    return found


# --------------------------------------------------------------------------- #
# emission
# --------------------------------------------------------------------------- #


def literal(value):
    """C++ float literal that round-trips a float32 (%.9g plus a '.' guard)."""
    text = '%.9g' % float(np.float32(value))
    if ('e' not in text) and ('E' not in text) and ('.' not in text):
        text += '.0'
    return text + 'f'


def emit_block(lines, namespace, values):
    """Emit one namespace of named constants, each with its hex pattern."""
    lines.append('namespace %s {' % namespace)
    lines.append('')
    for name, value in values:
        bits = gr.float_to_bits(float(np.float32(value)))
        if gr.bits_to_float(bits) != float(np.float32(value)):
            raise RuntimeError('%s does not round-trip: %r' % (name, value))
        lines.append('constexpr float %s = %s;  // 0x%08X' % (name, literal(value), bits))
    lines.append('')
    lines.append('}  // namespace %s' % namespace)
    lines.append('')


def emit_header(path, exp_values, exp2_values, exp2_meta):
    lines = []
    lines.append('#pragma once')
    lines.append('')
    lines.append('/// @file')
    lines.append('/// Minimax coefficients for the library polynomials. **Generated file.**')
    lines.append('///')
    lines.append('/// Regenerate with `python experiments/m3_fit/generate_coefficients.py`;')
    lines.append('/// every number below is a float32 rendered from a 40-digit minimax fit, and')
    lines.append('/// the hex pattern next to it is what the library actually computes with.')
    lines.append('/// Change a coefficient and the declared ULP bound stops applying: rerun')
    lines.append('/// `tools/measure_bounds.cpp` before touching the table it feeds.')
    lines.append('')
    lines.append('namespace ulpgrad {')
    lines.append('namespace detail {')
    lines.append('')
    lines.append('/// exp(x) = 2^k * (1 + r * Q(r)), Q a degree-6 minimax fit of')
    lines.append('/// (exp(r) - 1) / r on |r| <= ln2/2.')
    lines.append('///')
    lines.append('/// Provenance: the Remez exchange of `experiments/m0_exp/fit_coeffs.py`')
    lines.append('/// (Ziv-refined on a 25000-point tuning grid), copied bit for bit from the')
    lines.append('/// hex patterns of `experiments/m0_exp/coeffs.h`. Not refitted here, so the')
    lines.append('/// measured bound of that fit transfers unchanged.')
    emit_block(lines, 'exp_coeffs', exp_values)
    lines.append('/// exp2(x) = 2^k * (1 + r * Q(r)), Q a degree-%d minimax fit of'
                 % exp2_meta['degree'])
    lines.append('/// (2^r - 1) / r on |r| <= 1/2, fitted by this script: Remez exchange on a')
    lines.append('/// relative-error metric, %d iteration(s), %s.' %
                 (exp2_meta['iterations'],
                  'converged' if exp2_meta['converged'] else 'stopped without convergence'))
    lines.append('/// mpmath (%d decimal digits) max relative residual of Q on a 200001-point'
                 % MP_DPS)
    lines.append('/// grid: %.3e; then %d pass(es) of Ziv coordinate descent over +-2 ULP on a'
                 % (exp2_meta['residual'], exp2_meta['ziv_passes']))
    lines.append('/// %d-point tuning grid (seed 0x%08X), objective = maximum ULP distance of the'
                 % (exp2_meta['tuning_points'], exp2_meta['tuning_seed']))
    lines.append('/// whole exp2 pipeline. Modelled max ULP on that grid: %d, not the declared'
                 % exp2_meta['tuning_ulp'])
    lines.append('/// bound: the declared bound comes from the golden reference measurement.')
    emit_block(lines, 'exp2_coeffs', exp2_values)
    lines.append('}  // namespace detail')
    lines.append('}  // namespace ulpgrad')
    lines.append('')
    with open(path, 'w', encoding='ascii', newline='\n') as handle:
        handle.write('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=lambda text: int(text, 0), default=0xC0FFEE02)
    parser.add_argument('--tuning-points', type=int, default=6000)
    parser.add_argument('--degree', type=int, default=6)
    parser.add_argument('--out', default=OUT_PATH)
    args = parser.parse_args(argv)

    mpmath.mp.dps = MP_DPS

    # ---- exp: copied, never refitted --------------------------------------
    m0 = read_m0_coefficients(M0_COEFFS)
    exp_order = ['c%d' % i for i in range(args.degree + 1)]
    exp_order += ['inv_ln2', 'ln2_hi', 'ln2_lo', 'overflow_threshold', 'underflow_threshold']
    missing = [name for name in exp_order if name not in m0]
    if missing:
        raise RuntimeError('missing in %s: %s' % (M0_COEFFS, ', '.join(missing)))
    exp_values = [(name, gr_float(m0[name][1])) for name in exp_order]
    degree = len([name for name in exp_order if name.startswith('c')]) - 1
    print('exp: degree %d coefficients read from %s' % (degree, os.path.relpath(M0_COEFFS, REPO)))

    # ---- exp2: fitted here -------------------------------------------------
    print('fitting Q(r) = (2^r - 1) / r on [-0.5, 0.5], degree %d, mpmath dps %d'
          % (args.degree, mpmath.mp.dps))
    exp2_coefficients, meta = fit_exp2(args.degree, args.seed, args.tuning_points)
    exp2_values = [('c%d' % i, float(exp2_coefficients[i])) for i in range(exp2_coefficients.size)]
    exp2_values.append(('overflow_threshold', meta['overflow']))
    exp2_values.append(('underflow_threshold', meta['underflow']))
    print('exp2: residual %.3e, overflow threshold %r, underflow threshold %r'
          % (meta['residual'], meta['overflow'], meta['underflow']))

    emit_header(args.out, exp_values, exp2_values, meta)
    print('wrote %s' % os.path.relpath(args.out, REPO))
    for name, value in exp_values:
        print('  exp.%-20s %s' % (name, literal(value)))
    for name, value in exp2_values:
        print('  exp2.%-19s %s' % (name, literal(value)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
