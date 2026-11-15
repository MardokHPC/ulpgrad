# Contributing to ulpgrad

## Build and test

### Windows (cmd)

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DULPGRAD_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

### Linux

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DULPGRAD_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

The configure step fetches GoogleTest once, at a pinned tag, into `build/_deps`.

## Code rules

These are not negotiable; each one is load-bearing.

- **`ULPGRAD_FMA` everywhere, never `a * b + c` on `float`.** Implicit contraction is a
  compiler decision the declared bound cannot see, and only an explicit fused multiply-add keeps
  host and device bit-identical.
- **No CUDA header in a public header.** `ULPGRAD_HD` handles the annotation. The moment a public
  header includes `<cuda_runtime.h>`, the whole test suite needs a GPU to compile.
- **`noexcept` on every public math function.** Invalid input is a NaN, per IEEE-754. No
  exceptions, no error codes, no `errno` in `include/`.
- **Warnings are errors.** `-Wall -Wextra -Wpedantic -Werror`, `/W4 /WX`. Suppressions belong in
  CMake, never in source.
- **A ULP number is written only after a measurement exists.** If the measurement is worse than the
  plan, the plan gets corrected. Never the sample.

## Pull request checklist

- [ ] `cmake --build build` is clean with warnings as errors, on your platform.
- [ ] `ctest --test-dir build --output-on-failure` passes; no test was weakened to make it pass.
- [ ] Any new numeric claim has a test that regenerates it and fails on divergence.
- [ ] No absolute paths, no local plan documents, no binaries in the diff.

## Reporting a problem

Include the GPU model, driver and CUDA version, the compiler and its version, the CMake flags, and a
minimal reproducer. For a wrong result, include the exact input bits and the value you expected
against a reference you can name.
