#pragma once

#include <ulpgrad/config.h>
#include <ulpgrad/exp.h>
#include <ulpgrad/detail/bits.h>
#include <ulpgrad/detail/coefficients.h>

#include <cstdint>

namespace ulpgrad {

// tanh(x), rounded to nearest; +-1 for |x| at or above the saturation threshold.
[[nodiscard]] ULPGRAD_HD inline float tanh(float x) noexcept {
    const std::uint32_t bits = detail::float_to_bit_pattern(x);
    const std::uint32_t magnitude_bits = bits & 0x7FFFFFFFu;
    if (magnitude_bits > 0x7F800000u) {
        return x;  // NaN
    }
    const float magnitude = detail::bit_pattern_to_float(magnitude_bits);
    if (magnitude_bits == 0u) {
        return x;  // +-0 in, +-0 out
    }
    if (magnitude >= detail::tanh_coeffs::saturation) {
        return detail::bit_pattern_to_float((bits & 0x80000000u) | 0x3F800000u);  // +-1
    }
    if (magnitude < detail::tanh_coeffs::poly_max) {
        // tanh(x) = x + x^3*R(x^2), one rounding in the final fma.
        const float square = x * x;
        float tail = detail::tanh_coeffs::r4;
        tail = ULPGRAD_FMA(tail, square, detail::tanh_coeffs::r3);
        tail = ULPGRAD_FMA(tail, square, detail::tanh_coeffs::r2);
        tail = ULPGRAD_FMA(tail, square, detail::tanh_coeffs::r1);
        tail = ULPGRAD_FMA(tail, square, detail::tanh_coeffs::r0);
        return ULPGRAD_FMA(x * square, tail, x);
    }
    // (u-1)/(u+1), u = e^(2x).
    const float u = exp2(x * detail::tanh_coeffs::two_log2_e);
    return (u - 1.0f) / (u + 1.0f);
}

}  // namespace ulpgrad
