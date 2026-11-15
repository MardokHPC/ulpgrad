# Changelog

## [0.1.0] - 2026-11-15

- `exp`, `exp2` (1 ULP), `rsqrt` (1 ULP), `log`, `log2`, `log1p` (1 ULP), `tanh` (2 ULP).
  Header-only, no CUDA header in the public API. Bounds live in `ulpgrad/tables.h`, generated from
  `data/error-bounds.json` and checked by CI.
- Golden `float32` reference (mpmath, 40 digits) under `data/reference/`, with host test suites and a
  device parity check over all golden points.
- CI: host suites on Ubuntu and Windows, device compile on Ubuntu.
