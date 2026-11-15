#pragma once

#include <ulpgrad/config.h>
#include <ulpgrad/detail/bits.h>
#include <ulpgrad/detail/coefficients.h>

namespace ulpgrad {

namespace detail {

// rint on the device, std::rint on the host.
[[nodiscard]] ULPGRAD_HD inline float round_to_nearest_even(float x) noexcept {
#ifdef __CUDA_ARCH__
    return rintf(x);
#else
    return std::rint(x);
#endif
}

}  // namespace detail

// e^x, rounded to nearest; +inf above the overflow threshold, +0 below underflow.
[[nodiscard]] ULPGRAD_HD inline float exp(float x) noexcept {
    const std::uint32_t bits = detail::float_to_bit_pattern(x);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        return x;  // NaN
    }
    if (x >= detail::exp_coeffs::overflow_threshold) {
        return detail::bit_pattern_to_float(0x7F800000u);
    }
    if (x <= detail::exp_coeffs::underflow_threshold) {
        return 0.0f;  // never -0
    }

    // Cody-Waite: kf*ln2_hi is exact, the second fma recovers the tail of ln2.
    const float kf = detail::round_to_nearest_even(x * detail::exp_coeffs::inv_ln2);
    float r = ULPGRAD_FMA(-kf, detail::exp_coeffs::ln2_hi, x);
    r = ULPGRAD_FMA(-kf, detail::exp_coeffs::ln2_lo, r);

    // exp(r) = 1 + r*Q(r), Q a degree-6 Horner chain.
    float p = detail::exp_coeffs::c6;
    p = ULPGRAD_FMA(p, r, detail::exp_coeffs::c5);
    p = ULPGRAD_FMA(p, r, detail::exp_coeffs::c4);
    p = ULPGRAD_FMA(p, r, detail::exp_coeffs::c3);
    p = ULPGRAD_FMA(p, r, detail::exp_coeffs::c2);
    p = ULPGRAD_FMA(p, r, detail::exp_coeffs::c1);
    p = ULPGRAD_FMA(p, r, detail::exp_coeffs::c0);
    const float y = ULPGRAD_FMA(p, r, 1.0f);

    return detail::scale_by_pow2(y, static_cast<int>(kf));
}

// 2^x, rounded to nearest; +inf at or above 128, +0 at or below -150.
[[nodiscard]] ULPGRAD_HD inline float exp2(float x) noexcept {
    const std::uint32_t bits = detail::float_to_bit_pattern(x);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        return x;  // NaN
    }
    if (x >= detail::exp2_coeffs::overflow_threshold) {
        return detail::bit_pattern_to_float(0x7F800000u);
    }
    if (x <= detail::exp2_coeffs::underflow_threshold) {
        return 0.0f;  // never -0
    }

    const float kf = detail::round_to_nearest_even(x);
    const float r = x - kf;  // exact

    // 2^r = 1 + r*Q(r), Q degree 6.
    float p = detail::exp2_coeffs::c6;
    p = ULPGRAD_FMA(p, r, detail::exp2_coeffs::c5);
    p = ULPGRAD_FMA(p, r, detail::exp2_coeffs::c4);
    p = ULPGRAD_FMA(p, r, detail::exp2_coeffs::c3);
    p = ULPGRAD_FMA(p, r, detail::exp2_coeffs::c2);
    p = ULPGRAD_FMA(p, r, detail::exp2_coeffs::c1);
    p = ULPGRAD_FMA(p, r, detail::exp2_coeffs::c0);
    const float y = ULPGRAD_FMA(p, r, 1.0f);

    return detail::scale_by_pow2(y, static_cast<int>(kf));
}

}  // namespace ulpgrad
