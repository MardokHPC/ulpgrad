# M0 GPU spike — does a float-only `exp` have a reason to exist on a device?

**Question.** The host spike (`experiments/m0_exp/REPORT.md`) proved the degree-6 float32 `exp`
reaches a 1 ULP bound, and then reported the uncomfortable part: MSVC's `expf` is also 1 ULP
there and is 2–5× faster. If the same thing happens on the GPU, the `Accurate` positioning of
the project changes before M3–M6 get built. This is the measurement. The first version of this
report left three questions open; §7–§9 close them with numbers.

**Answer, one line.** On an RTX 2060 SUPER the degree-6 `exp` **is free where the kernel is
memory-bound and 1.55× slower where the working set is in L2 and the schedule is instruction
issue**, and what it buys is **1 ULP against 2 ULP for libdevice `expf`** and **63 ULP for
`__expf`**, now **with the CUDA default `-ftz=true`** as well: the 2^k scaling is built on the
bits instead of with a float multiply, so the 8,388,582 ULP of the previous version is gone
(max 1 ULP, 0 of 1,009,866 parity differences, in all three builds). The 1.55× is **not** the
Horner chain — Estrin makes it 8% *slower* (§8) — it is the instruction count per element, of
which the polynomial is only 20% (§7).

Everything below is an execution of `exp_bench.cu` through `run.bat`, plus `profile.bat` for the
Nsight Compute counters. The machine-readable form of the benchmark is `report.json` (three
configurations merged), of the profiler `stalls.csv`.

---

## 1. Environment

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 2060 SUPER, 8192 MiB, sm_75, 34 SMs, L2 4096 KB |
| driver | 610.88 (CUDA driver API 13.3) |
| toolkit | CUDA 12.6 (`nvcc` V12.6), runtime 12.6 |
| host compiler | MSVC 14.44.35207 (VS 2022 BuildTools) |
| OS | Windows 10.0.19045 (22H2), x64 |
| device flags | `-arch=sm_75 -std=c++17 -O3 -Xcompiler /O2,/W4,/EHsc`, 256 threads/block |
| profiler | Nsight Compute 2023.1.0 (`…/Nsight Compute 2023.1.0/target/windows-desktop-win7-x64/ncu.exe`) |

**Why not CUDA 12.1.** The toolchain on this machine is MSVC 14.44, and the CUDA 12.1 headers do
not survive it at all:

```
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\include\yvals_core.h(902): error:
  static assertion failed with "error STL1002: Unexpected compiler version, expected CUDA 12.4 or newer."
```

`--allow-unsupported-compiler` (the usual workaround for a *newer* MSVC) does not reach that
guard, because the guard lives in the MSVC STL, not in nvcc. 12.6 compiles the same file with no
diagnostics. This is recorded rather than worked around: anybody rebuilding this on CUDA 12.1
will hit the same wall.

## 2. What is in this directory

| file | role |
|---|---|
| `exp_bench.cu` | the whole spike as one translation unit: kernels, host parity reference, ULP analysis, timing harness, occupancy probe, JSON writer, JSON merger |
| `run.bat` | builds three configurations, runs them, merges the parts into `report.json` |
| `profile.bat` | profiles one warm launch per variant of the L2 sweep with Nsight Compute, writes `stalls.csv` |
| `stalls.csv` | the profiler rows quoted in §7, one block per variant (not committed: `profile.bat` regenerates it) |
| `fit_fast_coeffs.py` | degree-3/4 coefficients for the `Fast` candidates, from the `../m0_exp` fit machinery (see §10.7 for why it exists) |
| `coeffs_fast3.h`, `coeffs_fast4.h` | **generated** coefficient headers (do not edit by hand) |
| `REPORT.md` | this file |
| `report.json` | measured results of the three configurations quoted here |
| `run-ftz-true.log`, `run-ftz-false.log`, `run-fmad-false.log` | stdout of each run (not committed: `*.log` is ignored) |

Nothing here is library code and nothing here is in CMake. It is evidence.

## 3. What is measured, and why those nine

`N = 16,777,216` floats (64 MB in, 64 MB out). Nine variants, selected by a `switch`, each one
instantiated for a scalar and a `float4` kernel so the compiler cannot collapse them:

| variant | what it is |
|---|---|
| `ulp_exp` | ours: two-part Cody-Waite reduction, degree-6 `Q(r)` from `../m0_exp/coeffs.h`, 2^k through the exponent field **and through an integer shift in the subnormal range**, every product-plus-add an explicit `__fmaf_rn` |
| `ulp_exp_estrin` | the same polynomial and coefficients, evaluated with Estrin's scheme (§8) |
| `libdevice_expf` | `expf`, the accurate one (documented max 2 ULP) |
| `fast___expf` | `__expf`, the fast one |
| `sfu_ex2_approx` | `ex2.approx.f32` (SFU) applied to `x * inv_ln2`, i.e. `__expf` with the wrapper removed |
| `fast_p3` | `Fast` candidate 1: degree-3 minimax `Q(r)`, same reduction and scaling (§9) |
| `fast_p4` | `Fast` candidate 2: degree-4 |
| `fast_sfu` | `Fast` candidate 3: `ex2.approx.f32` on the **reduced** residual, so the scaling stays integer and exact |
| `copy` | reads and writes, does no arithmetic: the **bandwidth floor** |

`sfu_ex2_approx` is not `__exp2f` because **CUDA has no `__exp2f` intrinsic** (checked in the
12.1/12.4/12.6 headers); `ex2.approx.f32` is the instruction `__expf` is built from, so this
variant is the true floor below `__expf`. The measurement confirms it: `__expf` and
`sfu_ex2_approx` produce the **identical FNV-1a hash** `0xD1FDEAA0194C61A8` over 16,777,216
results in [-30, 30).

**Timing.** `cudaEvent` around each launch, 10 warm-up launches, 50 iterations, **median** with
p10/p90 per iteration, and **two independent rounds per variant of which the faster median is
reported** (the reason is §10.5: the GPU's clock state at the start of a compute-heavy phase is
not the steady state, and one round penalises whichever variant is measured first). Three sweeps:

1. **DRAM** — `N = 16 Mi`, one launch per timed interval.
2. **DRAM, reverse order** — the same sweep with the variant order reversed, as a control for
   clock drift inside one process.
3. **Latency-bound, working set in L2** — `N = 256 Ki` (1 MB in + 1 MB out, inside the 4 MB L2),
   grid = 8 blocks/SM and each thread walking the buffer 128 times, so the memory wall is gone and
   the number is the per-element cost of the function. This is *not* "compute-bound" in the
   throughput sense; §7 shows what it is bound by.

A 0.3 s warm-up burst runs before any timing; without it the first variant in the process came out
14.5 % slow, which is what §10.5 is about.

`ns/elem`, `Melem/s` and `GB/s` are computed from the median; `GB/s` assumes 8 bytes of traffic per
element (4 in + 4 out) and `% copy` is the ratio to the floor of the *same* vectorisation group. A
`% copy` above 100 is run-to-run noise, not a faster copy: the program prints that line under the
L2 table for exactly this reason, and it applies to `libdevice_expf` at 100.1 % and `__expf` at
101.1 % in §4.

## 4. Throughput

Canonical build (`-ftz=true -fmad=true`, CUDA defaults). Verbatim from `run-ftz-true.log`:

```
=== throughput, DRAM (N = 16777216 floats = 67 MB in + 67 MB out, 256 threads/block) ===
variant          vec    median ms     p10 ms     p90 ms    ns/elem   Melem/s     GB/s   % copy
ulp_exp          1         0.3585     0.3580     0.3634     0.0214   46798.9    374.4    98.8%
ulp_exp_estrin   1         0.3602     0.3586     0.3666     0.0215   46582.7    372.7    98.3%
libdevice_expf   1         0.3548     0.3533     0.3641     0.0212   47280.0    378.2    99.8%
fast___expf      1         0.3540     0.3530     0.3583     0.0211   47391.1    379.1   100.0%
sfu_ex2_approx   1         0.3543     0.3531     0.3586     0.0211   47352.6    378.8   100.0%
fast_p3          1         0.3587     0.3573     0.3640     0.0214   46773.8    374.2    98.7%
fast_p4          1         0.3587     0.3580     0.3626     0.0214   46773.8    374.2    98.7%
fast_sfu         1         0.3603     0.3587     0.3648     0.0215   46566.1    372.5    98.3%
copy             1         0.3541     0.3531     0.3638     0.0211   47374.0    379.0   100.0%
ulp_exp          f4        0.3557     0.3541     0.3609     0.0212   47160.9    377.3    99.7%
ulp_exp_estrin   f4        0.3597     0.3543     0.3748     0.0214   46636.5    373.1    98.6%
libdevice_expf   f4        0.3546     0.3532     0.3645     0.0211   47318.4    378.5   100.0%
fast___expf      f4        0.3545     0.3539     0.3564     0.0211   47331.2    378.6   100.0%
sfu_ex2_approx   f4        0.3545     0.3540     0.3565     0.0211   47331.2    378.6   100.0%
fast_p3          f4        0.3547     0.3540     0.3586     0.0211   47305.6    378.4   100.0%
fast_p4          f4        0.3544     0.3538     0.3592     0.0211   47344.0    378.8   100.0%
fast_sfu         f4        0.3543     0.3531     0.3575     0.0211   47352.6    378.8   100.1%
copy             f4        0.3545     0.3536     0.3577     0.0211   47322.7    378.6   100.0%
floor: 379.0 GB/s scalar, 378.6 GB/s float4 (8 B/element)
```

- **At this size nothing is compute-bound.** Nine variants, scalar and `float4`, land between
  0.3540 and 0.3603 ms. `ulp_exp` is **+1.2 % over the copy floor** (0.3585 vs 0.3541 ms) and
  **+1.0 % over `libdevice expf`** (0.3585 vs 0.3548), and **+0.3 % with `float4`** (0.3557 vs
  0.3545 ms).
- **`float4` does not help** — 378.6 GB/s for `copy` against 379.0 GB/s scalar is noise. The scalar
  kernel already issues one 4-byte load and one 4-byte store per thread with full coalescing, so
  this GPU saturates at 256 threads/block either way. The vectorised numbers are reported because
  they were cheap to add, but the comparable-with-the-library number is the scalar one.
- The floor itself, ~379 GB/s of read+write traffic, is the honest ceiling for this access pattern
  on this card; that is the number every variant is being measured against.

The latency-bound picture is completely different. Same run:

```
=== throughput, latency-bound, working set in L2 (N = 262144 floats = 2 MB, 128 repeats, full grid) ===
variant          vec    median ms     p10 ms     p90 ms    ns/elem   Melem/s     GB/s   % copy
ulp_exp          1         0.2287     0.2275     0.2300     0.0068  146715.5   1173.7    64.5%
ulp_exp_estrin   1         0.2471     0.2458     0.2495     0.0074  135790.7   1086.3    59.7%
libdevice_expf   1         0.1475     0.1454     0.1501     0.0044  227555.5   1820.4   100.1%
fast___expf      1         0.1460     0.1453     0.1475     0.0044  229799.7   1838.4   101.1%
sfu_ex2_approx   1         0.1533     0.1518     0.1554     0.0046  218818.0   1750.5    96.2%
fast_p3          1         0.2151     0.2122     0.2305     0.0064  156014.9   1248.1    68.6%
fast_p4          1         0.2211     0.2193     0.2268     0.0066  151747.6   1214.0    66.7%
fast_sfu         1         0.2268     0.2260     0.2276     0.0068  147915.9   1183.3    65.0%
copy             1         0.1476     0.1461     0.1508     0.0044  227407.5   1819.3   100.0%
floor: 1819.3 GB/s scalar (8 B/element)
note: a % copy above 100 is run-to-run noise, not a faster copy
```

- `libdevice_expf` (100.1 %) and `__expf` (101.1 %) are indistinguishable from the copy floor; the
  numbers above 100 % are the harness's own noise band (§4 stability below), not a copy that runs
  faster than itself.
- `ulp_exp` is at **64.5 % of the floor: 1.55× `copy`, 1.55× `libdevice expf`, 1.57× `__expf`**
  (0.2287 against 0.1476 / 0.1475 / 0.1460 ms). The three builds of the same source give 1.549,
  1.553 and 1.551 × copy, and earlier runs of the same binary 1.51–1.54, so **1.55 ± 0.03** is the
  number to carry.
- The previous version of this file reported 1.65× with `ulp_exp` at 0.2429 ms and `copy` at
  0.1473 ms. The current `ulp_exp` is 0.2287 ms against a 0.1476 ms floor. That is a cross-report
  comparison, not a controlled one, but the direction is worth recording: **the integer scaling
  that fixes `-ftz` did not cost time in this ring, it saved about 6 %** (fewer instructions per
  element, §7).

So the throughput answer depends entirely on which regime a real kernel is in, and it is not
"depends" as an escape: the two regimes are 16 Mi / DRAM and ≤ 4 Mi / L2, and both are measured.

Stability, from the reverse-order control in the same process (`report.json`,
`throughput_dram_reverse_order`), verbatim:

```
stability: median ms, forward vs reverse order (same process)
variant          vec      forward    reverse    delta
ulp_exp          1         0.3585     0.3589     0.1%
ulp_exp_estrin   1         0.3602     0.3601    -0.0%
libdevice_expf   1         0.3548     0.3545    -0.1%
fast___expf      1         0.3540     0.3540     0.0%
sfu_ex2_approx   1         0.3543     0.3543    -0.0%
fast_p3          1         0.3587     0.3641     1.5%
fast_p4          1         0.3587     0.3653     1.8%
fast_sfu         1         0.3603     0.3668     1.8%
copy             1         0.3541     0.3598     1.6%
ulp_exp          f4        0.3557     0.3623     1.8%
ulp_exp_estrin   f4        0.3597     0.3557    -1.1%
libdevice_expf   f4        0.3546     0.3548     0.1%
fast___expf      f4        0.3545     0.3561     0.5%
sfu_ex2_approx   f4        0.3545     0.3546     0.0%
fast_p3          f4        0.3547     0.3546    -0.0%
fast_p4          f4        0.3544     0.3543    -0.0%
fast_sfu         f4        0.3543     0.3548     0.1%
copy             f4        0.3545     0.3551     0.2%
worst forward/reverse disagreement: 1.8%
```

1.8 % worst case in this run; 0.2 % in the cleanest run of the session and 2.9 % in the worst. That
band is what "1.55× vs 1.57×" has to be read against, and it is why §9 does not call a 1–3 %
difference between two variants a win.

The FNV-1a hash of the whole output buffer is printed for every row and is identical between the
scalar and `float4` kernels of the same variant, which is the proof that the vectorised kernel is
the same function and that nothing was optimised away. The five hashes that matter:

```
ulp_exp          0x789AE7F068186E4D     (unchanged from the previous version)
ulp_exp_estrin   0x1CC61326A14D2AC0
fast_p3          0x1CFA081BBB7F2DB7
fast_p4          0xD406F9D5F6DBD633
fast_sfu         0x89FA0BA01A192BFA
```

`ulp_exp` hashing the same over the 16 Mi benchmark input is expected: that input is in
[-30, 30), where every result is normal, so the integer scaling produces the same bits as the
float scaling it replaced. Only the subnormal tail changed — and that is the 255 points of §6.

## 5. ULP on the device

The reference sample is `../m0_exp/ref.bin`: 1,009,866 records of `(x_bits, ref_bits)` with a
correctly rounded 40-digit reference. 1,009,863 points carry a ULP distance (the NaN inputs are
excluded and checked bit-for-bit instead).

**With `-ftz=true`** (the CUDA default — this is the case that was broken before), verbatim from
`run-ftz-true.log`:

```
=== ULP on the reference sample (1009866 points) ===
function             points  max ulp    p50    p99   p999  > 1 ulp     == 0
ulp_exp             1009863        1      0      1      1        0   93.94%
ulp_exp_estrin      1009863        1      0      1      1        0   92.00%
libdevice_expf      1009863  8388582      0      1      1      913   69.89%
fast___expf         1009863  8388518      3     20     26   673681   13.87%
sfu_ex2_approx      1009863       63      3     20     25   673491   13.89%
fast_p3             1009863       70     12     60     69   936385    2.98%
fast_p4             1009863        3      1      2      2    74641   49.42%
fast_sfu            1009863        2      0      1      2     1101   69.16%
```

**With `-ftz=false`** (subnormals honoured), same source, verbatim from `run-ftz-false.log`:

```
function             points  max ulp    p50    p99   p999  > 1 ulp     == 0
ulp_exp             1009863        1      0      1      1        0   93.94%
ulp_exp_estrin      1009863        1      0      1      1        0   92.00%
libdevice_expf      1009863        2      0      1      1      687   69.92%
fast___expf         1009863       63      3     20     25   673491   13.89%
sfu_ex2_approx      1009863       63      3     20     25   673491   13.89%
fast_p3             1009863       70     12     60     69   936385    2.98%
fast_p4             1009863        3      1      2      2    74641   49.42%
fast_sfu            1009863        2      0      1      2     1101   69.16%
```

Read the two tables together:

- **`ulp_exp` is max 1 ULP with either value of `-ftz`**, 0 points above 1 ULP, and the same
  93.94 % exactly rounded. The declared bound no longer depends on a compiler flag. The worst
  point is still `x = 18.2770233` with `k = 26`, `r = 0.255196631` — the *identical* worst point
  the host spike found, so the arithmetic is unchanged and only the scaling path was rebuilt.
- **The row that changed is `libdevice_expf`.** With `-ftz=true` it carries the same subnormal
  flush as before (max 8,388,582, 913 points above 1 ULP against 687 with `-ftz=false`): it is
  still the library that has the problem this report is about. The difference between the two
  builds is *entirely* the subnormal tail; p50/p99/p999 do not move, which is why a report that
  printed only percentiles would have missed it twice.
- **`fast_p3`, `fast_p4` and `fast_sfu` are identical in both builds, digit for digit.** That is
  not a coincidence: they are built on the same integer scaling, so they cannot be affected by
  FTZ. The property generalises — a variant that scales through the exponent field inherits the
  FTZ-independence for free.
- `__expf` and `sfu_ex2_approx` degrade exactly as documented, and the bucket table below shows
  where. `__expf` follows `2 + floor(1.16·|x|)`: 57 ULP in `[32, 64)` and 63 at the top of the
  range, and it is *worse* with FTZ on (8,388,518) because its subnormal results are flushed too.

```
max ulp per |x| bucket (the argument-degradation evidence), -ftz=true build
function           |x| < 1 1 <= |x| < 2 2 <= |x| < 4 4 <= |x| < 8 8 <= |x| < 16 16 <= |x| < 32 32 <= |x| < 64 |x| >= 64
ulp_exp                  1         1         1         1         1         1         1         1
ulp_exp_estrin           1         1         1         1         1         1         1         1
libdevice_expf           2         2         2         2         2         2         1   8388582
fast___expf              2         3         4         7        14        28        57   8388518
sfu_ex2_approx           2         3         4         7        14        28        57        63
fast_p3                 70        70        70        70        70        70        70        70
fast_p4                  3         3         3         3         3         3         3         3
fast_sfu                 2         2         2         2         2         2         2         1
```

(`ulp_exp` and `ulp_exp_estrin` are 1 in every bucket with `-ftz=false` too; `libdevice` recovers
its documented 2 and the last bucket drops from 8,388,582 to 1.)

## 6. Parity host/device, bit-exact, and the subnormal path exercised

The host side of the parity test is `host_exp()` in `exp_bench.cu`, the same formula as
`experiments/m0_exp/measure.cpp`. Before comparing anything, the program hashes its own host
output over the whole sample and compares it with the hash `measure.cpp` printed:

```
host reference hash        : 0xCF0FDE5FDCA81271 (measure.cpp printed 0xCF0FDE5FDCA81271) match
```

So the device is being compared with the *audited* host implementation, not with a re-derivation.
Then, in **all three builds** (`-ftz=true`, `-ftz=false`, `-ftz=true -fmad=false`):

```
=== parity host vs device (ulp_exp on 1009866 points) ===
differing bit patterns     : 0 of 1009866
```

**0 differences.** The host path scales a subnormal result with `(y * 2^(k+64)) * 2^-64`, and the
device now rounds the significand onto the subnormal grid with integer shifts; those two are the
same rounding of the same exact value (the first multiply is exact — the intermediate is normal —
so the host also rounds once), and the measurement confirms it rather than the argument.

255 points is thin evidence for a branch with several edge cases, so the harness also walks the
whole subnormal-output range and compares bit for bit:

```
=== subnormal-output sweep (ulp_exp, 65536 points in [-104, -86]) ===
bit differences vs the host path : 0 of 65536
device outputs subnormal         : 60567
device outputs +0                : 102
rounded up into 0x00800000       : 0
largest subnormal 0x007FFFFF     : 0
boundary 2^-126: the two closest float32 arguments sit at -180 and +76 ulp of
  2^-126 (1 ulp = 2^-24 relative), and the window that lands on 0x00800000 or
  0x007FFFFF is +-0.5 ulp: no float32 argument reaches it
```

- **60,567 of the 65,536 results are subnormal and 102 are exactly `+0`**, all bit-identical to
  the host. So the branch is exercised at every shift `s = 1 … 24`, not just at the 255 sample
  points, and `x` below the underflow threshold gives exact `+0` rather than a wrong subnormal.
- **The two encoding edges `0x00800000` (an `m` that rounds up into the smallest normal) and
  `0x007FFFFF` are unreachable from a float32 argument**, and that is measured, not assumed: the
  two float32 arguments closest to `-126·ln2` put `exp(x)` 180 and 76 ULP away from `2^-126`, and
  the window that would round onto those encodings is ±0.5 ULP. The carry is still implemented —
  it is one `++m` — but no test can produce it, so no test claims to.
- **Subnormal *inputs* are harmless, verified rather than deduced.** With `-ftz=true` every
  arithmetic operand is flushed, so `x` subnormal gives `kf = 0`, `r = 0`, `y = 1.0f` and the
  result `1.0f`; the host computes `y = 1 + x`, which rounds to `1.0f` as well. The 5,000 subnormal
  arguments of the sample are inside the 0-difference count above.

## 7. Why the L2 sweep is 1.55× and not 1.00×: instruction issue, not the chain

Two independent measurements, because the first version of this report guessed "8 dependent FMAs
in a Horner chain" and left it at that. `profile.bat` profiles one warm launch of each L2 kernel
and `stalls.csv` holds the raw rows. The counters are readable on this machine without admin; one
metric is not — `smsp__warp_issue_stalled_fixed_latency_*` reports `n/a` on sm_75, so the
dependency stalls have to be read from `smsp__warp_issue_stalled_wait_*`.

Per element of each L2 kernel, from `stalls.csv` (34 SMs × 4 sub-partitions, divided by the
33,554,432 element-visits):

| variant | instr/elem | fma pipe | alu pipe | xu pipe | issue_active | cycles/elem | × copy (cycles) | × copy (timed) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ulp_exp` | 44.75 | 16.77 | 12.84 | 2.01 | 80.3 % | 1.782 | 1.58 | 1.55 |
| `ulp_exp_estrin` | 48.75 | 20.04 | 13.57 | 2.01 | 80.8 % | 1.924 | 1.70 | 1.67 |
| `libdevice_expf` | 18.89 | 8.65 | 4.30 | 1.01 | 54.2 % | 1.181 | 1.05 | 1.00 |
| `fast___expf` | 11.36 | 2.72 | 2.70 | 1.01 | 32.9 % | 1.137 | 1.01 | 0.99 |
| `sfu_ex2_approx` | 16.41 | 6.01 | 5.07 | 1.00 | 46.1 % | 1.171 | 1.04 | 1.04 |
| `fast_p3` | 41.75 | 14.77 | 11.84 | 2.01 | 81.7 % | 1.639 | 1.45 | 1.46 |
| `fast_p4` | 42.75 | 15.77 | 11.84 | 2.01 | 80.8 % | 1.678 | 1.49 | 1.50 |
| `fast_sfu` | 43.41 | 15.27 | 12.80 | 3.00 | 80.5 % | 1.733 | 1.53 | 1.54 |
| `copy` | 9.36 | 1.79 | 2.63 | 0.01 | 27.3 % | 1.130 | 1.00 | 1.00 |

The model that fits every row to within 6 %:

> `cycles/element ≈ instructions/element / (32 × issue_active)`

predicted vs measured — 1.742/1.782, 1.886/1.924, 1.089/1.181, 1.079/1.137, 1.112/1.171,
1.596/1.639, 1.653/1.678, 1.686/1.733, 1.071/1.130. Same sign, same size for `copy` as for us, so
the residual is constant overhead, not a property of the arithmetic.

That is the whole explanation:

- **Every variant of ours saturates the issue stage at ~80 %** (`issue_active` 80.3–81.7 %). At
  4 warp-instructions per cycle and 32 elements per warp, an instruction costs 1/128 cycle per
  element, and there is no way to issue more.
- **`libdevice_expf` (18.89) and `__expf` (11.36) fit under the L2 floor** and are bandwidth-bound
  at 1.13–1.18 cycles/element; their arithmetic is irrelevant, which is exactly why they look
  "free".
- **We do not fit.** 44.75 instructions/element need 1.78 cycles/element against a 1.13 floor.
  To reach the floor we would need ≈ 29 instructions/element, i.e. to delete a third of them.
- **`warps_active` is 90 % and the theoretical occupancy is 100 %** (24 registers/thread, 4
  blocks/SM, 32 warps/SM for `ulp_exp`; every variant in the table is 4 blocks / 32 warps). So the
  long-standing guess that "a chain with few warps is latency-exposed" does not apply here: there
  are already as many warps as the SM can hold, and the stall named `wait` (40 % for `ulp_exp`,
  13 % for `libdevice`, 11 % for `copy`) is the fixed-latency dependency stall you cannot remove
  by adding warps — the SM still has to issue the next instruction of that chain.
- **The polynomial is 9 of the 44.75 instructions** (9 `FFMA`; the fma pipe also carries ~6.8
  `IMAD` for address and exponent arithmetic). Deleting it entirely would leave ~35.75
  instructions per element = 1.39 cycles = **1.24× copy**, and the measured `fast_p3` with three
  `FFMA` removed is 1.46× copy. **Four fifths of the gap is the scaffolding — guard branches,
  `rintf`, the reduction and the exponent-field scaling — not the polynomial.**

Static SASS, as a cross-check (`cuobjdump -sass exp_bench.exe`; inner loop body divided by its
unroll factor, which the `LDG` count gives: 3 for our polynomials, 7 for `copy`/libdevice/`__expf`,
1 for the two SFU kernels):

| variant | instr/elem (static) | FFMA | FMUL/FADD | FRND/MUFU | LDG/STG | int + branch |
|---|---:|---:|---:|---:|---:|---:|
| `ulp_exp` | 64.3 | 9 | 1 | 1 | 2 | 51.3 |
| `ulp_exp_estrin` | 68.3 | 9 | 3 | 1 | 2 | 53.3 |
| `libdevice_expf` | 16.6 | 4 | 3 | 1 | 2 | 6.6 |
| `fast___expf` | 9.0 | 0 | 1 | 1 | 2 | 5.0 |
| `sfu_ex2_approx` | 23 | 0 | 3 | 1 | 2 | 17.0 |
| `fast_p3` | 61.3 | 6 | 1 | 1 | 2 | 51.3 |
| `fast_p4` | 62.3 | 7 | 1 | 1 | 2 | 51.3 |
| `fast_sfu` | 70 | 2 | 4 | 1 | 2 | 61.0 |
| `copy` | 7.0 | 0 | 0 | 0 | 2 | 5.0 |

The static column overestimates our variants by 40 % (64.3 vs 44.75 executed) and the library ones
by ~10 %: the guard's early-exit paths are compiled inline in the unrolled body and are jumped
over for in-range input. That is why the table that decides anything is the profiler one, and why
the static count alone would have produced the wrong *ranking* (`fast_sfu` at 70 static
instructions is faster than `ulp_exp` at 64.3).

For the record, the file's own `cudaOccupancyMaxActiveBlocksPerMultiprocessor` probe:

```
theoretical occupancy of the L2 probe (256 threads/block, 1024 threads/SM max)
variant            regs   blocks   warps  occupancy
ulp_exp              24        4      32     100.0%
ulp_exp_estrin       26        4      32     100.0%
libdevice_expf       40        4      32     100.0%
fast___expf          38        4      32     100.0%
sfu_ex2_approx       14        4      32     100.0%
fast_p3              24        4      32     100.0%
fast_p4              24        4      32     100.0%
fast_sfu             14        4      32     100.0%
copy                 41        4      32     100.0%
```

None of these variants is register-limited, so "more work per thread" cannot buy warps; what it
could buy is instruction-level parallelism inside a warp (a `float4` version of this probe, not
measured here — see §12).

## 8. Estrin on the degree 6: measured, and slower

`ulp_exp_estrin` is the same polynomial, the same coefficients, the same reduction and the same
scaling, evaluated as

```
p = ((c0 + c1 r) + r^2 (c2 + c3 r)) + r^4 ((c4 + c5 r) + r^2 c6)
```

with `__fmaf_rn` for each product-plus-add and two extra multiplies for `r^2` and `r^4`. Chain
length 4 instead of 8, at +2 `FMUL` and +4 instructions per element.

| | `ulp_exp` | `ulp_exp_estrin` | delta |
|---|---:|---:|---:|
| L2 median ms | 0.2287 | 0.2471 | **+8.0 %** |
| L2 ns/elem | 0.0068 | 0.0074 | +8.0 % |
| × copy | 1.55 | 1.67 | |
| executed instructions/elem | 44.75 | 48.75 | +8.9 % |
| max ULP, `-ftz` on / off | 1 / 1 | 1 / 1 | unchanged |
| points > 1 ULP | 0 | 0 | unchanged |
| exactly rounded | 93.94 % | 92.00 % | −1.9 pp |

The three builds agree: 1.674, 1.677 and 1.670 × copy against 1.549, 1.553 and 1.551. **Estrin does
not improve anything; it costs 8 %**, and the +4 instructions per element explain the 8 % (§7's
model predicts 48.75/44.75 = 1.089). The accuracy is unchanged — 1 ULP max, still 0 points above
it — but slightly more of the sample lands at 1 ULP instead of 0 (92.00 % vs 93.94 % exactly
rounded), because the shorter chain rounds in different places.

So the lever the first version of this report pointed at ("a 2–3× shorter chain could plausibly
close most of the gap at zero accuracy cost") **does not exist on this GPU**: the ring is not
chain-limited, it is issue-limited, and every instruction you remove helps while every
rearrangement of the same instructions does not.

## 9. `Fast`: three candidates, measured

The question was whether a `Fast` exists with a **flat** bound under 3 ULP that is **faster than
`ulp_exp` and close to `__expf`**. Three candidates, all built on the same reduction and the same
integer scaling so that only the residual evaluation differs:

- **`fast_p3`** — degree-3 minimax fit of `Q(r) = (exp(r)-1)/r`, generated by
  `fit_fast_coeffs.py --degree 3`. Remez residual `1.499256e-05` relative on a 200,001-point grid,
  converged in 4 iterations.
- **`fast_p4`** — degree 4, same machinery. Remez residual `4.329841e-07`; the Ziv pass over ±2 ULP
  found nothing to improve (the modelled bound is 3 ULP on the tuning grid either way).
- **`fast_sfu`** — `ex2.approx.f32` applied to `r * log2(e)` where `r` is the **reduced** residual
  (`|r| <= ln2/2`), then the integer scaling. This is the interesting one: `__expf` is the same
  instruction applied to the *whole* argument, and its 63 ULP comes from the single multiply
  carrying the magnitude; on a reduced residual only the SFU's own error is left.
  A refinement was looked for and not found: the cheap correction `fma(ex2(r·log2e), d, …)` can
  only fix the reduction error `d = r - k·ln2`, not the SFU's own 2^-22-class error, which is the
  dominant term. There is nothing economical to correct, so no refinement is claimed.

| candidate | max ULP `-ftz=true` | max ULP `-ftz=false` | > 1 ULP (both) | p50 / p99 / p999 | L2 ns/elem | % of floor | × `ulp_exp` |
|---|---:|---:|---:|---|---:|---:|---:|
| `fast_p3` | **70** | **70** | 936,385 | 12 / 60 / 69 | 0.0064 | 68.6 | 0.94 |
| `fast_p4` | **3** | **3** | 74,641 | 1 / 2 / 2 | 0.0066 | 66.7 | 0.97 |
| `fast_sfu` | **2** | **2** | 1,101 | 0 / 1 / 2 | 0.0068 | 65.0 | 0.99 |
| `ulp_exp` (reference) | 1 | 1 | 0 | 0 / 1 / 1 | 0.0068 | 64.5 | 1.00 |
| `__expf` (target) | 8,388,518 | 63 | 673,681 / 673,491 | 3 / 20 / 25 | 0.0044 | 101.1 | 0.64 |

- **`fast_p3` is dead on arrival** and the measurement says why: 1.5e-05 relative on `Q` is 70 ULP
  of the result, flat across every `|x|` bucket (70 in all eight). Degree 3 cannot represent
  `exp` on `|r| <= ln2/2` to better than that; the minimax residual bounds it from below, so no
  coefficient tuning rescues it.
- **`fast_p4` is a 3 ULP function**, not under 3: max 3, p99 2, 7.4 % of the sample above 1 ULP,
  flat (3 in every bucket). It buys 3 % of time.
- **`fast_sfu` is the only candidate that meets the accuracy half of the question**: max 2 ULP,
  flat (2 in every bucket, 1 for `|x| >= 64`), 0.11 % of the sample above 1 ULP, and 69.16 %
  exactly rounded against 93.94 % for `ulp_exp`. It is **1 % faster than `ulp_exp`**, inside the
  0.2–2.9 % noise band of this harness — call it equal.
- **None of them is close to `__expf`.** The best candidate is 1.54× copy against 1.01× for
  `__expf`, and §7 says why that gap cannot close: `fast_sfu` still executes 43.41 instructions
  per element (the guard, the reduction and the scaling are 41 of them), while `__expf` executes
  11.36 because it has none of those.

**Answer to the question as asked:** a flat sub-3-ULP `Fast` exists (`fast_sfu`, 2 ULP), it is not
measurably faster than `ulp_exp`, and it is 1.5× away from `__expf` rather than close to it. The
`Fast` tier as a *speed* tier is not reachable with this scaffolding; as an *accuracy* tier
(2 ULP instead of 1, at the same speed) it is.

## 10. What went wrong on the way

Kept because it is the part that makes the rest credible.

1. **`-Xcompiler /O2 /W4 /EHsc` is not a list.** Space-separated options get re-split by `nvcc`
   and `/EHsc` arrives as the token `/E`, so the build died with
   `nvcc fatal: Don't know what to do with 'C:/EHsc'`. `-Xcompiler` wants
   `-Xcompiler /O2,/W4,/EHsc`.
2. **CUDA 12.1 cannot compile this file at all** with MSVC 14.44 (§1). The failure is a
   `static_assert` in the MSVC STL, i.e. it happens *after* nvcc accepts the compiler, so the
   documented `--allow-unsupported-compiler` workaround is not enough.
3. **`__exp2f` does not exist.** The variant was written as `__exp2f(x * 1.442695f)` and the
   compiler said `identifier "__exp2f" is undefined`. `ex2.approx.f32` is the real instruction, and
   using it directly produced a better finding than the original plan: `__expf` and
   `ex2.approx(x·log2e)` produce the same hash, so `__expf` *is* that instruction plus a multiply.
4. **The first L2 probe measured the wrong thing.** With 256 blocks and one pass over 256 Ki
   elements the kernel takes ~9 µs, the event overhead is a comparable fraction, and the run
   reported `ulp_exp` at **1.04× copy** with `p10 = 0.0062 ms` and `p90 = 0.0156 ms`. That number
   would have produced the opposite verdict ("even compute-bound, our exp is free"). The fix was to
   keep the small buffer but fill the GPU and walk it 128 times, which moved the arithmetic to the
   front: 1.55× copy with a p10/p90 spread under 1 %.
5. **Clock ramp, which cost 14.5 %.** In the first full runs the first variant measured in the
   process — always `ulp_exp` scalar, because of the loop order — came out at 0.4207 ms while the
   same kernel later in the same process measured 0.3597 ms, and the reverse-order control
   reproduced the effect in reverse (`delta = -14.5 %` in the printed stability table). It is the
   GPU boost clock ramping, not the function; 10 warm-up launches (~4 ms) are not enough. The fix
   that held was **two rounds per variant, report the faster median**, plus a 0.3 s busy loop of
   `copy` before the first measurement. The committed run has a 1.8 % forward/reverse
   disagreement; the best run of the session had 0.2 % and the worst 2.9 %, and that is the band
   §9 has to be read in.
6. **The JSON merger emitted invalid JSON.** It stripped the outer braces of each part before
   joining them into an array, and `json.load` caught it (`Expecting ',' delimiter: line 4`) —
   which is why the report is generated by a program that re-reads its own output, not by
   copy-pasting stdout.
7. **`fit_coeffs.py --degree 3` fits correctly and then throws the fit away.** The Remez exchange
   converges to 1.499e-05 (degree 3) and 4.330e-07 (degree 4), and `main()` then rejects anything
   above 1e-6 as "remez failed" and falls back to a weighted-least-squares path that returns a
   polynomial with a **0.183** relative residual: the Chebyshev-in-`t` → monomial-in-`r` step in
   `fit_lstsq()` is not a composition, and at degree 6 nobody noticed because the fallback never
   runs. `fit_fast_coeffs.py` is the same pipeline without that guard; it prints the guard it is
   bypassing on every run. **A fallback that never runs in the canonical configuration is untested
   code, and this one was wrong.**
8. **`ncu` reports the requested metric as `n/a`.** `smsp__warp_issue_stalled_fixed_latency_*` does
   not exist on sm_75; the fixed-latency dependency stalls are counted as
   `smsp__warp_issue_stalled_wait_*`. The counter list was rewritten around `wait`, and the probe
   prints the `n/a` row rather than pretending it is zero.
9. **`--launch-count 9` does not mean "nine kernels".** It means nine launches *matching the
   filter*, and the first nine matches were nine replays of `ulp_exp`. The probe now indexes
   variants by launch position — the L2 sweep runs 50 launches per variant, so variant `v` is
   `--launch-skip 50v+25` — and every row of `stalls.csv` names its kernel.
10. **The boundary encodings are unreachable, which the first draft of the test assumed they were
    not.** The subnormal sweep reports 0 occurrences of `0x00800000` and `0x007FFFFF`, and the
    reason is geometric: consecutive float32 arguments near `-87.336` move the result by ~128 ULP,
    while the window that rounds onto those encodings is ±0.5 ULP. The test now prints that
    measurement next to the zeros instead of leaving them unexplained.

## 11. Verdict

**Our `exp` has a reason to exist on GPU as an accuracy product, and no reason as a speed
product.** With the numbers above:

- **Where it is free.** In any kernel that is memory-bound — the normal case for a full-tensor op
  on 64 MB of data — `ulp_exp` costs **0.3585 ms against 0.3541 ms for the copy floor and 0.3548
  ms for `libdevice expf`**, i.e. **+1.2 % over the floor and +1.0 % over both libraries, +0.3 %
  with `float4`** (0.3557 vs 0.3545 ms). Choosing it costs nothing that can be measured outside
  the 1.8 % forward/reverse band.
- **Where it costs.** In a kernel whose working set is already in L2, `ulp_exp` is **1.55× `copy`
  and 1.55× `libdevice expf`** (0.2287 vs 0.1476 / 0.1475 ms), 64.5 % of the floor. The cause is
  measured: **44.75 executed instructions per element at 80 % issue occupancy**, of which the
  degree-6 polynomial is 9. Estrin (shorter chain, 4 more instructions) is **8 % slower**, which
  rules out the chain as the mechanism; `copy` at 9.36 instructions is not issue-bound at all.
- **What it buys.** On 1,009,863 points: **max 1 ULP against 2 ULP for `libdevice expf` and 63 ULP
  for `__expf`**, with the bound flat from `|x| < 1` to `|x| >= 64`, **0 of 1,009,866 parity
  differences against the audited host path**, **0 of 65,536 in a dense subnormal sweep** (60,567
  subnormal results and 102 exact zeros included), and — new — **with `-ftz=true` as well as
  `-ftz=false`**. `libdevice expf` fails the same test with `-ftz=true`: 8,388,582 ULP.
- **The `Fast` tier.** A flat sub-3-ULP variant exists (`fast_sfu`, 2 ULP, 0.11 % of the sample
  above 1 ULP) and is the same speed as `ulp_exp`, so it is a *cheaper contract at the same price*
  rather than a faster function. `fast_p4` is 3 ULP and 3 % faster; `fast_p3` is 70 ULP and is not
  a candidate. None of the three approaches `__expf`, and §7 explains why none can while it keeps
  the guard + reduction + scaling: that scaffolding is ~41 of the 43–45 instructions per element.

So the project's position is unchanged in direction and now measured on both axes: **`Accurate` is
worth building where 2 ULP is not good enough and where the kernel is not instruction-bound; it is
not worth building as a faster `expf`.** If M3–M6 are aimed at "fastest exp", this spike says stop.
If they are aimed at "the tightest measured bound on a GPU, at no cost unless the kernel is
issue-bound", the numbers support that — and §12 names the one lever that could move the second
case.

## 12. What this does not prove, and what is left open

1. **The 1.55× is fixable, but not by the polynomial.** The scaffolding is ~41 instructions of the
   ~45 per element: three guard branches (NaN, overflow, underflow), `rintf`, the two-part
   reduction, and the exponent-field scaling with its subnormal branch. A branchless guard
   (clamping `x` instead of returning early, as libdevice does), or folding the guard into the
   scaling, is the only route to `__expf`-class time — and each of those changes what the function
   does at the edges, which is the part this report is careful about. **Not measured here.**
2. **Instruction-level parallelism inside a warp is untested for this ring.** With 32 warps/SM and
   100 % occupancy there is nothing left to add in warps, but the probe runs the *scalar* kernel;
   a `float4` version of it would interleave four independent chains per thread and could hide the
   `wait` stalls (40 % for `ulp_exp`). That is a different experiment from the `float4` DRAM row in
   §4, which only proves vectorisation does not help a bandwidth-bound kernel.
3. **One GPU, one size, one access pattern.** sm_75, 34 SMs, 4 MB of L2, 64 FP32 lanes per SM. A
   card with a wider issue stage or more L2 bandwidth moves both the floor and the issue ceiling;
   the model in §7 is a model, and the claim that the issue ceiling stays at 80 % on other
   architectures is not measured.
4. **The sample is a sample.** Max 1 ULP *measured on 1,009,863 points plus 65,536 subnormal-range
   points*, not proven. The CI sweep (M3) is what turns this into a table per bucket.
5. **No fused/production kernel was timed.** `out[i] = f(in[i])` is the cleanest comparison between
   implementations and the worst case for showing any speed difference; a softmax-style kernel that
   feeds the result straight into a reduction has different (better) odds for an issue-bound
   function, and worse ones for a bandwidth-bound one.
6. **`-ftz=false` is a supported build, not a tested contract.** The bound is 1 ULP in both builds
   on this sample; nobody has run the library at `-ftz=false` inside a real pipeline, where
   subnormal values arrive from upstream and the FTZ flush in *other* code changes the answer
   before `exp` sees it.
7. **Degree 4 is the smallest flat 3-ULP fit measured, not the smallest possible.** A degree-4 fit
   on a different reduction interval, or a degree-4 plus one refinement step on `2^r`, was not
   tried.

## 13. Reproducing

```
cd experiments/m0_exp && run.bat     # once, generates ref.bin (not committed)
cd experiments/m0_gpu && run.bat     # builds three configs, measures, writes report.json
```

A full `run.bat` takes **28.6 s** on the machine in §1, measured end to end by timing it
(16:13:24.79 → 16:13:53.41): three `nvcc` invocations plus three executions of the nine-variant
harness. Compilation dominates; the reference-sample ULP analysis (1 M points, double precision,
host side) is a fraction of a second.

The profiler is separate and slower:

```
cd experiments/m0_gpu && profile.bat  # writes stalls.csv, one warm launch per L2 variant
```

`run.bat` regenerates nothing that is generated: the coefficient headers are produced once, by
`python experiments/m0_exp/fit_coeffs.py` for the degree-6 header and by
`python experiments/m0_gpu/fit_fast_coeffs.py --degree N --out experiments/m0_gpu/coeffs_fastN.h`
for degrees 3 and 4, and all three are committed.

`report.json` is the merge of the three configurations and contains, per configuration, the three
throughput series, the ULP table with the per-bucket maxima, the parity counts, the subnormal sweep
counters and the occupancy table.
