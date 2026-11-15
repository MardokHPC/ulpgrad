#pragma once

#include <cmath>

#ifdef __CUDACC__
#define ULPGRAD_HD __host__ __device__
#else
#define ULPGRAD_HD
#endif

// Single-rounding fused multiply-add: a * b + c rounded once.
// __CUDA_ARCH__ (device pass only), not __CUDACC__ (the whole nvcc invocation).
#ifdef __CUDA_ARCH__
#define ULPGRAD_FMA(a, b, c) __fmaf_rn((a), (b), (c))
#else
#define ULPGRAD_FMA(a, b, c) std::fmaf((a), (b), (c))
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define ULPGRAD_FORCE_INLINE __forceinline
#else
#define ULPGRAD_FORCE_INLINE inline __attribute__((always_inline))
#endif
