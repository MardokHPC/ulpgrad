# ulpgrad

[![CI](https://github.com/MardokHPC/ulpgrad/actions/workflows/ci.yml/badge.svg)](https://github.com/MardokHPC/ulpgrad/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/MardokHPC/ulpgrad)](https://github.com/MardokHPC/ulpgrad/releases)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](#prerequisites)
[![Header-only](https://img.shields.io/badge/header--only-yes-brightgreen.svg)](#integration)

> Every function declares its maximum error in ULP, the interval where that bound holds, and what it
> does with subnormals, infinities and NaN. The error table is generated from a dense sweep against a
> high-precision reference and regenerated in CI: if the number in this README does not match the
> code, the build fails.

`ulpgrad` is a header-only C++17 library of approximate `float32` math functions. One header compiles
on the CPU and on the GPU, it has no link dependencies, and the accuracy claim is a test rather than
a sentence in a README.

> **Status: v0.1.0.** `exp` and `exp2` are in `ulpgrad/exp.h`, `rsqrt` in
> `ulpgrad/rsqrt.h`, `log`, `log2`, `log1p` in `ulpgrad/log.h`, and `tanh` in `ulpgrad/tanh.h`, with
> the declared bounds in `ulpgrad/tables.h` and the table in
> [`docs/error-tables.md`](docs/error-tables.md), both generated from one measurement against a
> committed golden reference. `erf`, `expm1` and `sigmoid` are not in this release.

## When to use it

| You want | Use |
|---|---|
| IEEE correctness and published derivations, GPU only | `expf` / `logf` from libdevice |
| Maximum speed, error that grows with the argument | `__expf` (`2 + floor(1.16·\|x\|)` ULP, up to 57 in `[32, 64)`) |
| **A bound that still holds under CUDA's default `-ftz=true`, identical on CPU and GPU** | **ulpgrad** |
| Correctly rounded (0.5 ULP) everywhere | `double` + cast, or MPFR on the CPU |
| A complete vectorised CPU/GPU math library | SLEEF / LLVM libmvec |

`ulpgrad` does not try to replace libdevice. It fills the gap between "`__expf` in a hot loop and no
idea what it costs you" and "adopt an entire math library for two functions": a small header with a
bound you can state in a review, that CI refuses to let drift, and that survives the compiler flags CUDA
turns on by default — which is where `libdevice`'s own `expf` does not.

## Prerequisites

| Requirement | Detail |
|---|---|
| Compiler | C++17: MSVC 2019+, GCC 9+, Clang 10+ |
| Build system | CMake 3.18+ |
| GPU (optional) | CUDA 11.8+ on Linux; **CUDA 12.4+ on Windows**, because MSVC 14.44's STL headers refuse older CUDA with `STL1002` and `--allow-unsupported-compiler` does not bypass a guard that lives in the STL. The tests never need a GPU |
| Dependencies | none at link time; GoogleTest is fetched only when `ULPGRAD_BUILD_TESTS=ON` |

## Quickstart

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DULPGRAD_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Minimal example

```cpp
#include <ulpgrad/exp.h>
#include <ulpgrad/log.h>
#include <ulpgrad/rsqrt.h>
#include <ulpgrad/tanh.h>

float y = ulpgrad::exp(x);      // one function, one bound: 1 ULP flat, host and device
float z = ulpgrad::exp2(k);     // the binary one: reduction is exact, so the same bound for less work
float r = ulpgrad::rsqrt(x);    // no division, no SFU: three Newton steps on an even exponent
float l = ulpgrad::log1p(x);    // the one to use for log(1 + x): 1e-30 comes back exact
float t = ulpgrad::tanh(x);     // polynomial below 0.5, one exp2 and one division above it
```

## Integration

### Option A: add_subdirectory

```cmake
add_subdirectory(third_party/ulpgrad)
target_link_libraries(my_target PRIVATE ulpgrad::ulpgrad)
```

### Option B: FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(ulpgrad
    GIT_REPOSITORY https://github.com/MardokHPC/ulpgrad.git
    GIT_TAG v0.1.0)
FetchContent_MakeAvailable(ulpgrad)
target_link_libraries(my_target PRIVATE ulpgrad::ulpgrad)
```

### Option C: installed package

```bash
cmake --install build --prefix /some/prefix
```

```cmake
find_package(ulpgrad CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE ulpgrad::ulpgrad)
```

## API reference

| Header | Symbols | Status |
|---|---|---|
| `ulpgrad/config.h` | `ULPGRAD_HD`, `ULPGRAD_FMA`, `ULPGRAD_FORCE_INLINE` | done |
| `ulpgrad/detail/bits.h` | `ulp_distance`, `scale_by_pow2`, bit casts | done |
| `ulpgrad/detail/coefficients.h` | minimax coefficients, hex pattern next to each | generated |
| `ulpgrad/tables.h` | `exp_max_ulp`, `exp2_max_ulp`, `rsqrt_max_ulp`, `log_max_ulp`, `log2_max_ulp`, `log1p_max_ulp`, `tanh_max_ulp`, per-bucket `ErrorBound` rows | generated |
| `ulpgrad/exp.h` | `exp`, `exp2` | done; `expm1` not shipped |
| `ulpgrad/log.h` | `log`, `log2`, `log1p` — one reduction, one atanh series, two fits | done |
| `ulpgrad/rsqrt.h` | `rsqrt` (bit-trick estimate + three Newton steps, no SFU) | done |
| `ulpgrad/tanh.h` | `tanh` — odd polynomial below 0.5, `(u-1)/(u+1)` above | done; `sigmoid` not shipped |
| `ulpgrad/detail/poly.h` | FMA-only polynomial evaluation | not built: a 7-line Horner chain in each function beats a header |
| `ulpgrad/pow.h` | `pow`, with the full IEEE special-case table | M6 |
| `ulpgrad/erf.h` | `erf`, `erfc` | v1.1 |

## Error table

The shipped table is [`docs/error-tables.md`](docs/error-tables.md), generated from
`data/error-bounds.json`, which is a real run of the headers in this repository against the golden
reference in `data/reference/`. `ulpgrad::exp_max_ulp`, `ulpgrad::exp2_max_ulp` and
`ulpgrad::rsqrt_max_ulp` in `include/ulpgrad/tables.h` are that same number, and
`tests/test_table_freshness.cpp` fails if the header and the JSON disagree.

| Function | Points | Max ULP | Exactly rounded | Worst bucket |
|---|---|---:|---:|---:|
| `exp` | 9,331 | **1** | 94.81 % | 1 |
| `exp2` | 7,917 | **1** | 94.16 % | 1 |
| `log` | 7,969 | **1** | 94.47 % | 1 |
| `log1p` | 7,005 | **1** | 92.93 % | 1 |
| `log2` | 7,969 | **1** | 93.84 % | 1 |
| `rsqrt` | 7,829 | **1** | 88.22 % | 1 |
| `tanh` | 7,230 | **2** | 88.02 % | 2 |

Side by side with the incumbent (`libdevice` figures are the ones NVIDIA documents in the CUDA Math
API for each function; check them against your toolkit's table, they are not measured here):

| Function | ulpgrad | `libdevice` | |
|---|---:|---:|---|
| `exp` | 1 | 2 | better |
| `exp2` | 1 | 2 | better |
| `rsqrt` | 1 | 2 | better |
| `log` | 1 | 1 | equal |
| `log2` | 1 | 1 | equal |
| `log1p` | 1 | 1 | equal |
| `tanh` | 2 | 2 | equal |

Three better, four equal, none worse. And `exp`/`exp2` keep their bound under `-ftz=true`, where the
documented figure for `libdevice` is not a number at all: it is 8,388,582 ULP measured on the same
sample, because FTZ flushes the subnormal results.

`rsqrt` is the one function whose bound is not a sample: the reduction maps every input to a
mantissa in `[1, 4)` and the final scaling is an exact exponent-field move, so the error was
enumerated over all `2^24` mantissas — 16,777,216 points, max 1 ULP, 87.01 % exactly rounded. The
golden row above is the same claim measured through the committed reference, specials included.

`log`, `log2` and `log1p` are 1 ULP, and one of the three sources that used to cost the logs a second
ULP is gone. The series argument `s = (m-1)/(m+1)` has an exact numerator and a **rounded**
denominator; its residual is the two-sum leftover of that very addition, so `log(m)` pays back
`-t*d_lo/(2m)` into the series correction with a degree-2 reciprocal — four operations, no second
division. What is left is the division's own half ULP, which a compensated quotient would recover, and
the rounding at the result's scale. Measured: at most 1 ULP in every bucket, and `log` went from
93.44 % to 94.47 % exactly rounded, `log2` from 92.50 % to 93.84 %.

`tanh` declares 2 ULP, for a different reason, and its worst input in the table names it:
`x = -0.5068`, one ULP above the branch. Above the threshold `tanh` is built on this library's
`exp2`, and `tanh` keeps `(1-t^2)/2` of the exponential's 1 ULP — 0.39 at that point — on top of
its own division rounding. The polynomial branch measures 1 ULP on its own. A 1 ULP `tanh` needs a
compensated exponential, not a better polynomial.

That is the measurement a consumer runs. The larger sample below is the M0 spike that justified the
design, on 1,009,863 points against the same kind of reference, on the host and on an RTX 2060 SUPER
(sm_75). Method, per-bucket breakdown and reproduction commands:
[`experiments/m0_exp/REPORT.md`](experiments/m0_exp/REPORT.md) and
[`experiments/m0_gpu/REPORT.md`](experiments/m0_gpu/REPORT.md).

| Function | Domain | Max ULP, `-ftz=false` | Max ULP, `-ftz=true` (CUDA default) | Exactly rounded |
|---|---|---|---|---|
| `exp` | `[-30, 30]` plus structural and border points | **1** | **1** | 93.94 % |
| `libdevice expf` | same sample | 2 | **8,388,582** | 69.89 % |
| `__expf` | same sample | 63 | **8,388,518** | 13.87 % |

Two things in that table are the whole project:

- **The bound is flat.** Every bucket of `|x|`, from `2^-24` to the overflow edge, sits at 1 ULP.
  `__expf` walks 2, 3, 4, 7, 14, 28, 57 as `|x|` doubles. This is the property being sold.
- **It survives the flag CUDA turns on by default.** `-ftz=true` flushes subnormal results, and both
  `libdevice expf` and `__expf` collapse to 8.4×10⁶ ULP at `x ≈ -87.34`. `ulpgrad` does not, because
  the final scaling is built on integer bit operations instead of a `mul.f32` that FTZ can flush.

## Benchmarks

RTX 2060 SUPER (sm_75), CUDA 12.6, driver 13.3, 16 Mi floats, 10 warmup + 50 measured iterations,
median with p10/p90, two rounds per variant. The floor is a copy kernel moving the same bytes.

<!-- BEGIN benchmarks (generated by tools/gen_table.py from data/benchmarks.json; do not edit) -->
| Kernel shape | ulpgrad | `libdevice expf` | `__expf` |
|---|---|---|---|
| DRAM-bound (67 MB in, 67 MB out), scalar | 98.8 % | 99.8 % | 100.0 % |
| DRAM-bound, `float4` | 99.7 % | 100.0 % | 100.0 % |
| L2-resident scalar ring, 128 repeats | 64.5 % | 100.1 % | 101.1 % |
<!-- END benchmarks -->

Read it as: **in a kernel that waits on memory — the normal case — the function costs 1.2 %**, and
that disappears entirely once the loop is vectorised. The 1.55× exists only in an artificial scalar
ring that re-reads 2 MB from L2 with no instruction-level parallelism to hide the dependency chain.

There is no speed claim here, and there will not be one. `__expf` is two instructions; no polynomial
beats that, and the profiling says the ring is instruction-issue bound, not chain bound — 44.75
instructions per element of which the polynomial is nine. Estrin made it 8 % **slower**. The claim is
accuracy at no measurable cost, not speed.

## What we got wrong

Kept because it is the part that shows the work is measured rather than asserted.

- **The headers did not compile for the device, and nothing noticed for two milestones.** `__fmaf_rn`,
  `__uint_as_float` and `__float_as_uint` are `__device__`-only, and the macro that selected them
  tested `__CUDACC__` — which is defined for the whole nvcc invocation, including the host pass. Every
  `__host__ __device__` function therefore called a device-only intrinsic from host code, eighteen
  errors deep. The switch is `__CUDA_ARCH__`, which exists only in the device pass. The reason it
  survived so long is structural: the suite is host-only by design, so it can never see this. There
  is now a device check in CI that compiles the real headers and compares bits.
- **The parity harness itself was contracting.** The first version of the device check combined the
  two functions as `exp(x) + 3.0f * exp2(x)`, nvcc fused it into an fma, and the harness reported a
  1 ULP parity failure at `x = -12.5` that belonged to the test, not to the library. The rule the
  project imposes on its own code — never write `a * b + c` on floats — applies to the tests too.
- **The library itself was contractible, one level down.** `log2`'s assembly needed the rounded value
  of `two_s * log2_e` twice: once as the leading term and once to recover that product's own rounding
  error. nvcc folded the product into the following addition anyway, which adds the recovered error a
  second time; MSVC with no FMA instruction could not, and the device came back 1 ULP off at
  `x = 88.7`. Writing the product as an `ULPGRAD_FMA` with a zero addend pins it. The lesson is
  narrower than "never contract": it is that an explicitly rounded intermediate has to *say* so,
  because "the compiler is allowed to" is not a bound.
- **The degree-7 Horner chain skipped a coefficient** and measured 34 ULP. It looked plausible — the
  obvious conclusion was "degree 7 is worse". What caught it was the mpmath residual, which said
  4.5e-12, i.e. three orders of magnitude smaller than 34 ULP could allow. Comparing the measurement
  against the error budget before believing an outlier is now a rule.
- **A build step that only runs when a file is missing was never tested**, because the file was always
  there. Deleting `ref.bin` surfaced a path bug in the reference generator immediately. Deleting the
  input is now part of verifying a pipeline.

Full write-ups: [`m0_exp/REPORT.md` §10](experiments/m0_exp/REPORT.md) and
[`m0_gpu/REPORT.md`](experiments/m0_gpu/REPORT.md).

## Quality

- **8 host suites, 61 cases**, runnable without a GPU in well under a second.
- **A device check that compiles the real headers** (`tests/device/exp_smoke.cu`) and answers two
  questions separately on **every point of the seven golden files** — 55,998 points, 7 functions,
  391,986 comparisons. *Parity*: the device result equals the host result bit for bit, so a differing
  bit fails the build. *Bound*: the device result stays within the number declared in `tables.h`,
  measured against the same mpmath reference the host suite uses. NaN patterns are compared bit for
  bit, never on the ULP metric. It exits 77 without a GPU, so CI compiles it anywhere, and the
  reference directory is a command-line argument rather than a compiled-in path.
- Warnings are fatal everywhere: `-Wall -Wextra -Wpedantic -Werror` for GCC/Clang, `/W4 /WX` for MSVC.
- CI builds and runs the host suites on Ubuntu 22.04 (GCC) and Windows 2022 (MSVC), and compiles the
  device check on Ubuntu.
- **The declared bound is a test.** `tests/test_exp.cpp` reads the committed golden reference and fails
  if the measured maximum exceeds the number in `tables.h`; `tests/test_table_freshness.cpp` fails if
  the header and `data/error-bounds.json` disagree. Editing either by hand breaks the build.
- The reference is a correctly rounded round-to-nearest-even computation in 40-digit `mpmath`, with an
  explicit single rounding to `float32` — not a cast through `double`, which would double round.
- Everything is regenerable from committed scripts: `experiments/m0_exp/run.bat`,
  `experiments/m0_gpu/run.bat`, `experiments/m3_fit/`, `tools/gen_table.py`.

## Limitations

- **The bound is measured, not proven.** `1 ULP` holds on 1,009,863 sample points, roughly one point
  every 6e-5 in the reduced argument. A pathological input sitting on a rounding tie could still reach
  2 ULP. Nothing here is an exhaustive verification, and the README will never claim otherwise.
- **`float` only.** No `double`, no `half`, no vector types. `double` alters the accuracy story and
  doubles the surface for no benefit to the ML kernels this targets.
- **No `erf` in v1.0.** It needs a rational minimax over several ranges; if time is short it ships in
  v1.1 and is declared missing rather than approximated badly.
- **The CPU is not where this wins.** Platform `expf` already achieves 1 ULP on this sample and is
  faster. The product is the flat bound, the identical result on both targets, and a bound that holds
  under `-ftz=true` where libdevice's own `expf` does not.
- **It is not faster than `__expf`, and it will not be.** Two SFU instructions cannot be beaten by a
  polynomial. If your kernel is arithmetic-bound on `exp`, `__expf` with 63 ULP is the honest choice.
- **`log`, `log2` and `log1p` are 1 ULP, not 0.5.** The honest reason is in `## Error table`: the
  atanh series argument is a correctly rounded division, and its own half ULP is not recovered —
  doing so needs a compensated quotient, i.e. a second division in the hottest function of the
  header. The other half ULP, the one the rounded denominator used to cost, is recovered and paid
  back for four operations.
- **`tanh` is 2 ULP above 0.5, and `sigmoid` is not shipped.** The 2 comes from the `exp2` it is
  built on, not from its polynomial, which is at 1 ULP. `0.5 * (1 + tanh(x/2))` would add another
  cancellation on top of that, so `sigmoid` needs its own fit and its own bound rather than a wrapper
  -- and until then it is declared missing, which is cheaper to defend than a wrapper with no number.
- **Nothing is exhaustive.** The sample is stratified, not complete; the only exhaustive parts are the
  `rsqrt` mantissa sweep, the subnormal-output sweep and the reasoning about the `2^-126` boundary. A
  different GPU, a different CUDA version or a different compiler may move the numbers, and the
  harness exists so that finding out costs one command instead of an argument.

## Repository structure

```text
ulpgrad/
├── include/ulpgrad/
│   ├── config.h              # ULPGRAD_HD, ULPGRAD_FMA, ULPGRAD_FORCE_INLINE
│   ├── exp.h                 # exp, exp2
│   ├── log.h                 # log, log2, log1p
│   ├── rsqrt.h               # rsqrt
│   ├── tables.h              # generated: declared bounds, readable from code
│   ├── tanh.h                # tanh
│   └── detail/
│       ├── bits.h            # ulp_distance, scale_by_pow2, bit casts, significant_bits
│       └── coefficients.h    # generated: minimax coefficients
├── cmake/
│   └── ulpgrad-config.cmake.in
├── data/
│   ├── error-bounds.json     # generated: the measured numbers the table is built from
│   └── reference/            # golden float32 reference: one CSV per function
├── docs/
│   └── error-tables.md       # generated: the table a human reads
├── experiments/
│   ├── m0_exp/               # host measurement: fit, reference, harness, report
│   ├── m0_gpu/               # device measurement: throughput, ULP, parity, SASS, report
│   ├── m3_fit/               # one-off tools: golden reference, exp2 coefficient fit
│   └── m4_transcendental/    # coefficient and magic-constant search for rsqrt and the logs
├── tests/
│   ├── test_main.cpp
│   ├── test_config.cpp
│   ├── test_bits.cpp
│   ├── test_exp.cpp
│   ├── test_log.cpp
│   ├── test_rsqrt.cpp
│   ├── test_tanh.cpp
│   ├── test_special_cases.cpp
│   ├── test_table_freshness.cpp
│   └── device/
│       ├── exp_smoke.cu      # compiles the real headers for sm_75, golden parity + bound
│       └── run.bat
├── tools/
│   ├── measure_bounds.cpp    # runs the library, writes data/error-bounds.json
│   └── gen_table.py          # JSON -> tables.h + docs/error-tables.md
├── experiments/find_toolchain.bat
├── .github/workflows/ci.yml
├── CMakeLists.txt
├── README.md  CHANGELOG.md  CONTRIBUTING.md  LICENSE
```

## License

MIT — see [LICENSE](LICENSE).
