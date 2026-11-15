# M0 spike — the measured ULP bound of a float-only `exp`

**Question.** The project target is `<= 1.0 ulp` for an "accurate" variant, and the plan behind it
admitted the target might not survive contact with a real float32 pipeline. This spike answers with
one number, before any library code exists.

**Answer.** On 1,009,863 sample points, against a correctly rounded `mpmath` reference at
40 decimal digits:

| metric | value |
|---|---|
| **max ULP distance** | **1** |
| points with ULP > 1 | **0** |
| points with ULP = 0 (exactly rounded) | 948,634 (93.94 %) |
| points with ULP = 1 | 61,229 (6.06 %) |
| p50 / p99 / p999 | 0 / 1 / 1 |
| where the maximum is | `x = 18.2770233` (`0x41923758`), `k = 26`, `r = 0.255196631` |
| NaN inputs | 3, bit-exact failures **0** |
| reference +-inf cases | 498, mismatches **0** |

`Accurate <= 1.0 ulp` is reachable with **degree 6**, which is the plan's own target. Degree 7
is not needed and does not improve the bound. Degree 5 also stays at 1 ULP on this sample,
which says something uncomfortable and useful: **the polynomial degree is not what limits the
error any more** — the final rounding and the binary32 coefficient rounding are.

Everything below comes from an execution, not from a plan. The machine-readable form of every
run quoted here is in `report.json`, `report-strict.json`, `report-avx2.json`,
`report-deg5.json` and `report-deg7.json`.

---

## 1. Environment

| item | value |
|---|---|
| OS | Microsoft Windows version 10.0.19045 (22H2), x64 |
| CPU | Intel Core i7-8700K @ 3.70 GHz |
| compiler | MSVC 19.44.35229 (VS 2022 BuildTools 14.44.35207), `cl /O2 /fp:precise /std:c++17 /W4 /WX` |
| Python | 3.14.8 (Anaconda), numpy 2.5.4, mpmath 1.4.1 |
| reference precision | `mpmath.mp.dps = 40` |

The targets quoted throughout come from the project plan, which is kept out of the repository on
purpose (it is a working document: it carries coordination notes and local paths, and
`.gitignore` lists it). Every number in this report, by contrast, is produced by code that is
tracked: `fit_coeffs.py`, `gen_reference.py` and `measure.cpp`, plus the seed and sample manifest
they record. No path that identifies the author's machine appears in the tree — the only absolute
paths are the two standard MSVC install locations in `run.bat`, and they are toolchain locations,
not personal ones.

No GPU was involved. The spike is host-only on purpose (see §9).

## 2. What is in this directory

| file | role |
|---|---|
| `fit_coeffs.py` | Remez exchange in mpmath for `Q(r) = (exp(r)-1)/r` on `\|r\| <= ln2/2`, relative-error metric; float32 rendering + Ziv refinement; writes `coeffs.h` |
| `gen_reference.py` | correctly rounded float32 `exp` reference + stratified sample, binary output `ref.bin` (8-byte records: `uint32 x_bits`, `uint32 ref_bits`) and `manifest.txt` |
| `measure.cpp` | host C++17 implementation of `exp` and the measurement harness |
| `run.bat` | activates `vcvars64`, builds, runs, leaves `report*.json` |
| `coeffs.h` | **generated** coefficient file (do not edit by hand) |
| `ref.bin` | **generated** 8 MB reference sample (not committed: regenerate with `gen_reference.py`) |
| `report*.json` | measured results of the runs quoted here |

The implementation under test, verbatim:

```cpp
const float kf = std::rint(x * inv_ln2);
const int k = static_cast<int>(kf);
float r = std::fma(-kf, ln2_hi, x);
r = std::fma(-kf, ln2_lo, r);
float p = c6;                       // Q(r) = (exp(r)-1)/r, degree 6
p = std::fma(p, r, c5);  ...  p = std::fma(p, r, c0);
const float y = std::fma(p, r, 1.0f);   // exp(r)
// scale by 2^k through the exponent field; subnormal tail: one exact scale,
// one rounded scale
```

No `a*b+c` anywhere: every product-plus-add is an explicit `fmaf`/`std::fma`.

## 3. Method

**Reference.** `mp.exp(mp.mpf(x))` with `x` the exact float32 value (float32 → binary64 →
`mpf` is lossless), then rounded to float32 by an explicit round-to-nearest-even on the
significand. The rounding routine is *not* `float()`: going through binary64 would double
round (53 bits, then 24) and disagree with direct rounding on exact ties. It is self-tested
on every run:

```
  ok  idempotence on 200k random patterns (0 mismatches)
  ok  ties-to-even on 20000 midpoints
  ok  2**128 - 2**103 - eps -> FLT_MAX
  ok  2**128 - 2**103 (tie) -> +inf
  ok  2**-149 -> smallest subnormal
  ok  2**-150 (tie) -> +0
  ok  agreement with numpy on 50k float64 inputs (0 mismatches)
```

**Sample** (`manifest.txt` records the boundaries):

| category | count | what |
|---|---|---|
| `uniform` | 1,000,000 | uniform in [-30, 30], deterministic 64-bit LCG (Knuth MMIX constants), seed `0x5EED0001` |
| `structural` | 2,032 | all 254 normal exponents × 4 mantissa quarters (0, 1/4, 1/2, 3/4), both signs |
| `borders` | 2,824 | `k*ln2` and `(k ± 1/2)*ln2` for every `k` in [-150, 128], each ±2 ULP; overflow threshold `88.72283935546875`, underflow threshold `-103.97208404541016`, subnormal onset, each ±4 ULP |
| `subnormal` | 5,000 | deterministic subnormal arguments, both signs |
| `specials` | 10 | NaN with payload (quiet and signalling), ±inf, ±0, FLT_MAX, smallest subnormal |

**Metric.** `ulp_distance` on the monotone integer image of the bit pattern
(`i < 0 ? -(i & 0x7fffffff) : i`), which is the mapping the plan asked to verify separately.
It is asserted on every run before any number is printed:

```
  ok  monotone mapping and ulp_distance
```

with hand-computed pins: `+1.0` vs its successor = 1, `-1.0` vs its successor toward -inf = 1,
`1.0` vs `2.0` = 2^23, `-0` vs `+0` = 0, `+1.0` vs `-1.0` = 2,130,706,432, `+inf` vs `-inf`
= 2·2,139,095,040, smallest subnormal vs `+0` = 1. NaN inputs are excluded from the ULP
statistics and checked bit-for-bit instead (NaN in → NaN out with the same payload: 0 failures).

## 4. The numbers

```
points measured (NaN excluded) : 1009863 of 1009866
max ulp                        : 1
  at x                         : 18.2770233 (0x41923758)
  k = 26, r = 0.255196631
p50 / p99 / p999 ulp           : 0 / 1 / 1
points with ulp > 1            : 0
points with ulp > 0            : 61229 (6.06%)
points exactly rounded         : 948634 (93.94%)
output bit hash (FNV-1a)       : 0xCF0FDE5FDCA81271

specials
  NaN inputs                   : 3, bit-exact failures 0
  reference +-inf              : 498, mismatches 0
  other mismatches             : 0
```

The distribution is bimodal and trivial: 0 or 1 ULP, nothing else. `p99 = 1` means the 99th
percentile of the error is one ULP (6.06 % of the sample is there); `p999 = 1` means there is no
tail at all — there is nothing between "one ULP" and "the maximum".

Per category (identical numbers, so the bound does not hide in a corner of the sample):

```
  category          count  max ulp      p99 worst x
  uniform         1000000        1        1   18.2770233
  structural         2032        1        0   5.96046448e-08
  borders            2824        1        1   -88.7228241
  subnormal          5000        0        0   -
  specials             10        0        0   -
```

## 5. Where the maximum is, and why

`x = 18.2770233` (`0x41923758`) → `k = 26`, `r = 0.255196631`. Two things matter about this
point and neither is surprising after the fact:

1. **`r` is far from zero but not at the boundary.** `|r| = 0.255` sits inside
   `[-ln2/2, +ln2/2]` where the fit is least accurate, and it is *not* the extreme value
   `ln2/2 = 0.3466` — the maximum of the error is not at the maximum of `|r|`, because the
   dominant term is the rounding of `1 + r*Q(r)`, which is worst when the mantissa of the
   result lands near a rounding boundary.
2. **`k = 26` is unremarkable**, which is the point: the error does not grow with `|x|`.
   The bucket table in §6 shows the same 1 ULP at `2^4` and at `2^6` and at `2^-1`.

There is **no degradation with the argument**: bucket 133 (`|x| ∈ [64, 128)`, i.e. the edge of
the overflow range) shows the same 1 ULP as bucket 126 (`|x| ∈ [0.5, 1)`). This is exactly the
property the project sells, and it survives the measurement.

### Per-bucket table (worst point per exponent bucket of `x`)

16 of the 256 non-empty buckets reached 1 ULP. The other 240 buckets — including every
`|x| ∈ (2^-126, 2^-24)` and every `|x| >= 2^7`, where `exp` saturates to 1, `+inf` or `+0` —
were exactly rounded on all points.

| bucket | `\|x\|` range | points | max ULP | share at 1 ULP | worst `x` |
|---:|---|---:|---:|---:|---:|
| 103 | [2^-24, 2^-23) | 8 | 1 | 12.5 % | 5.96046448e-08 |
| 119 | [2^-8, 2^-7) | 159 | 1 | 0.6 % | -0.00638723373 |
| 120 | [2^-7, 2^-6) | 291 | 1 | 0.7 % | 0.0133645535 |
| 121 | [2^-6, 2^-5) | 548 | 1 | 1.6 % | -0.0241005421 |
| 122 | [2^-5, 2^-4) | 1,046 | 1 | 1.6 % | -0.0498962402 |
| 123 | [2^-4, 2^-3) | 2,124 | 1 | 2.7 % | -0.0803411007 |
| 124 | [2^-3, 2^-2) | 4,181 | 1 | 5.1 % | -0.125555992 |
| 125 | [2^-2, 2^-1) | 8,259 | 1 | 7.2 % | -0.315892696 |
| 126 | [2^-1, 2^0) | 16,606 | 1 | 3.8 % | 0.97928524 |
| 127 | [2^0, 2^1) | 33,328 | 1 | 6.1 % | 1.27526164 |
| 128 | [2^1, 2^2) | 66,389 | 1 | 6.5 % | 2.95028329 |
| 129 | [2^2, 2^3) | 133,813 | 1 | 6.6 % | -5.79544163 |
| 130 | [2^3, 2^4) | 266,830 | 1 | 5.7 % | -13.5778675 |
| 131 | [2^4, 2^5) | 467,327 | 1 | 6.2 % | 18.2770233 |
| 132 | [2^5, 2^6) | 928 | 1 | 5.8 % | -62.7298164 |
| 133 | [2^6, 2^7) | 989 | 1 | 6.2 % | -88.7228241 |

The buckets are not equally populated because the uniform sample in [-30, 30] puts most points
in [16, 32), while the `structural` category contributes exactly 8 points per exponent.

## 6. Degree 5, 6 and 7 — the degree is not the binding constraint

All three were fitted with the same machinery and measured on the same sample
(`report-deg5.json`, `report.json`, `report-deg7.json`):

| degree | mpmath residual of `Q` | max ULP | points at 1 ULP | exactly rounded | output hash |
|---:|---|---:|---:|---:|---|
| 5 | 1.072e-08 | 1 | 65,158 (6.45 %) | 93.55 % | `0xCEF4C702C255A2E2` |
| **6** | **2.322e-10** | **1** | **61,229 (6.06 %)** | **93.94 %** | `0xCF0FDE5FDCA81271` |
| 7 | 4.470e-12 | 1 | 61,237 (6.06 %) | 93.94 % | `0xB38E7032A5891C17` |

Going from degree 5 to 6 improves the analytical residual 46× and moves 0.39 percentage points
of the sample from "1 ULP" to "exactly rounded" (6.45 % → 6.06 %), with the bound unchanged.
Going from 6 to 7 improves the residual another 52× and changes nothing measurable — 8 points
out of a million in the *wrong* direction, i.e. inside the noise that more Ziv passes would
absorb. **The plan was right**: the 0.5 ULP class needs a double-float accumulator, not a higher
degree. Degree 6 is the right stopping point: it uses the
remaining error budget from the polynomial side, so the declared 1 ULP is not carried by a
single term, and it costs one `fmaf` less per call than degree 7 for no measurable gain.

### Error budget, and why it matches

For `|r| <= ln2/2 = 0.3466`, `1 <= exp(r) <= 1.4143`:

| term | bound | in ULP of the result |
|---|---|---|
| analytical polynomial error (2.322e-10 relative on `Q`) | `\|r\|·δQ/exp(r) <= 1.1e-10` | 0.001 |
| float32 rounding of the coefficients | `<= 2^-24` on `Q`, weighted by `\|r·Q\|/exp(r) <= 0.29` | `<= 0.30` |
| argument reduction, two fma roundings | `<= 2^-24·max(\|r1\|,\|r\|)` absolute | `<= 0.19` |
| final `fmaf(p, r, 1.0f)` | one correct rounding | 0.5 |
| **total** | | **<= ~1.0** |

The measured maximum is 1.0 ULP. The budget is dominated by the last two lines, which is why
no polynomial can improve it, and why the sum lands *just* at the target instead of comfortably
inside it. If a future sample finds 2 ULP somewhere, this table says where to look: the
reduction, not the fit.

## 7. The control: MSVC `expf`, and what it costs

Same sample, same reference, same metric:

| | max ULP | p99 | p999 | > 1 ULP | ns/call |
|---|---:|---:|---:|---:|---:|
| `m0_exp`, SSE2 build (`fmaf` is a library call) | 1 | 1 | 1 | 0 | 15.09 |
| `m0_exp`, `/fp:strict`, SSE2 | 1 | 1 | 1 | 0 | 15.13 |
| `m0_exp`, `/arch:AVX2` (hardware FMA) | 1 | 1 | 1 | 0 | 7.50 |
| MSVC `std::expf` | 1 | 0 | 0 | 0 | 3.14 |

Values from `report.json`, `report-strict.json`, `report-avx2.json`. Across all recorded runs the
same code varied between 15.0 and 17.1 ns/call (SSE2) and 7.24 and 7.50 ns (AVX2), and `expf`
between 3.03 and 5.31 ns — the last one only on a loaded machine, which is why the two `-deg5`/
`-deg7` artifacts show `expf` at 4.90 and 5.31 ns. **The timing is the least trustworthy number
here**: one scalar call in a loop, no unrolling, ±10 % run to run. The ULP numbers, by contrast,
were bit-identical across all five runs.

Read this honestly:

- **MSVC's `expf` is as tight as ours** on this sample (max 1 ULP, and it is exactly rounded
  much more often: p99 = 0). The spike does **not** show a tighter bound than the platform
  libm. It shows that a 30-line float-only implementation written from a minimax fit reaches
  the same bound class, which is the claim needed to justify the library.
- **We are 2.3× slower than `expf` with hardware FMA and 5× slower without.** The default
  MSVC target on x64 is SSE2, where `fmaf` compiles to a helper call instead of one `vfmadd`
  instruction; that is the whole difference between 16 ns and 7 ns. The honest conclusion is
  that the CPU timing of this spike is not a number to publish — the project's speed claim is
  about GPU throughput (M8) and about *not degrading with the argument*. Do not quote 7.25 ns
  anywhere: it measures one scalar call in a loop, not a kernel.

The speed comparison is a warning for M3/M8, not a result: `Accurate` needs to beat something,
and on the CPU it does not beat `expf`. On the GPU the relevant baseline is `__expf`
(`2 + floor(1.16·|x|)` ULP, ~37 ULP at `|x| = 30`), and there the constant 1 ULP bound is the
product.

## 8. Verdict on the targets

| target | measured | verdict |
|---|---|---|
| `<= 1.0 ulp` for the accurate path | max 1 ULP, 0 points above | **met**, with degree 6, with no margin to spare |
| "degree 6 vs degree 7" | degree 6 and 7 both give max 1 ULP | **degree 6**, as planned |
| "0.55 ULP needs double-float" | 93.94 % exactly rounded at degree 6 already; the remaining 6.06 % cannot be fixed by degree or coefficients | **confirmed**: the 0.5 ULP class needs a double-float accumulator |
| bound independent of compiler FP flags | `/fp:precise`, `/fp:strict` and `/arch:AVX2` produce the **same FNV-1a hash** `0xCF0FDE5FDCA81271` over 1,009,866 result bit patterns | **supported** for FMA-on/off at the code level |
| "the number in the README is a test that can fail" | the harness recomputes everything from `ref.bin` and exits non-zero on any assertion failure | **prototype exists** |

The one number that would have to change if a denser sweep finds a 2: nothing in the design,
only the declared bound. `1.0` is declared here as *measured on 1,009,863 points*, not as
*proven*.

## 9. What this spike does not prove

1. **It is a sample, not a proof.** 1M uniform points in [-30, 30] is about 11,500 points per
   ln2-wide reduction interval, i.e. one point every 6e-5 in `r`; a pathological `x` whose
   mantissa sits exactly on a rounding tie could still produce 2 ULP. The honest declaration is
   "max 1 ULP measured on 1,009,863 points", and the CI sweep (M3) is what turns this into a
   table per bucket of `x`.
2. **No FTZ in this spike, and no GPU.** All of this ran on x86-64 with subnormals honoured.
   CUDA's default `-ftz=true` flushes subnormal *inputs and outputs*, which changes the
   behaviour of `exp` for `x <= -87.34` (subnormal results) and for subnormal inputs. The
   `subnormal` category of this sample is exactly what would break under FTZ, and it is
   measured here only with FTZ off. **Resolved later, not here:** with the final scaling moved
   to integer bit operations, `experiments/m0_gpu/REPORT.md` measures 1 ULP with `-ftz=true`
   as well — and 1 ULP is exactly what `libdevice expf` does *not* achieve.
3. **No host/GPU parity in this spike.** The hash above proves bit-identical output across three
   x86 builds (software FMA, hardware FMA, `/fp:strict`). **Resolved later, not here:**
   `experiments/m0_gpu/REPORT.md` compares the device against this exact host implementation and
   finds 0 differing bit patterns over 1,009,866 points.
4. **The subnormal *output* path is exercised but not audited.** The two-step scaling
   (`y * 2^(k+64)` then `* 2^-64`) rounds once, which is correct, but it is only tested through
   the 5,000-point subnormal category and the 498 points that overflow to `+inf`, not
   exhaustively over the 2^23 subnormal outputs reachable. That scaling was later replaced by an
   integer shift, for the reason the GPU spike found: `mul.f32` is flushed to zero under FTZ.
5. **No throughput claim.** See §7: the CPU number is not the GPU number, and one scalar call
   in a loop is not a kernel.
6. **Not vectorised, not inlined into anything, no `-fmad=false` variant.** The implementation
   always uses explicit `fmaf`; a build where the compiler is allowed to contract (or where FMA
   does not exist at all, as on a GPU without `fma`) would need a second measurement.

## 10. What went wrong on the way

Kept because it is the useful part:

1. **The degree-7 Horner chain skipped `c5`** and measured 34 ULP (uniform max 34, p99 29) — a
   plausible-looking number that would have been reported as "degree 7 is worse" if the
   polynomial residual (4.5e-12, the *mpmath* residual, computed in 40 digits where the chain
   had no bug) had not made it impossible. The residual is what caught it, not the measurement:
   **compare the measured bound against the budget before believing an outlier.** With the
   chain fixed, degree 7 gives 1 ULP.
2. **`%.9g` emitted `1f`** for the literal `1.0f`, which does not compile (`error C3688:
   invalid literal suffix 'f'`, message localized by the toolchain). Small, but it is the kind of
   thing that only shows up on the first build of a generated header.
3. **`"timing_checksum": nan`** in the JSON: summing `exp` over a sample that contains NaN and
   `+inf` produces a NaN, and `NaN` is not valid JSON. Removed from the JSON; the accumulator
   is printed to stdout instead, where it does its only job (stopping the optimiser from
   deleting the timed loop).
4. **`gen_reference.py` wrote relative to the current directory.** `--out-dir` defaulted to
   `experiments/m0_exp`, which is correct from the repository root and wrong from inside
   `experiments/m0_exp`, where `run.bat` puts you — and it never showed, because the sample file
   already existed and the generator was skipped. It surfaced only when the last verification
   deleted `ref.bin` and asked the whole pipeline to rebuild itself:
   `FileNotFoundError: 'experiments/m0_exp/ref.bin'`. Now the default is the script's own
   directory. **A build step that only runs when a file is missing has to be tested by deleting
   the file.**

## 11. Reproducing

```
cd experiments/m0_exp
run.bat                  # canonical: /O2 /fp:precise, degree 6 -> report.json
run.bat strict           # /fp:strict      -> report-strict.json
run.bat precise avx2     # /arch:AVX2      -> report-avx2.json
run.bat precise none 5   # degree 5        -> report-deg5.json
run.bat precise none 7   # degree 7        -> report-deg7.json
```

`run.bat` regenerates `coeffs.h` from the fit every time (a stale generated header is exactly
the kind of thing that makes a measurement lie) and regenerates the 8 MB `ref.bin` only when it
is missing — delete it to force a full reference rebuild.

Two measurements, because the difference matters when reproducing: a **warm** run, with `ref.bin`
already present, takes 26.7 s end to end on the machine in §1 (of which ~20 s is the Remez fit and
its 200k-point verification grid). A **cold** run, after deleting `ref.bin`, takes 75-90 s: the
reference generation alone runs at about 20k correctly rounded 40-digit evaluations per second.
