#pragma once

#include <ulpgrad/config.h>
#include <ulpgrad/detail/bits.h>
#include <ulpgrad/detail/coefficients.h>

#include <cstdint>

namespace ulpgrad {

// 1/sqrt(x), rounded to nearest; +inf at +0, -inf at -0, NaN for x < 0.
[[nodiscard]] ULPGRAD_HD inline float rsqrt(float x) noexcept {
    const std::uint32_t bits = detail::float_to_bit_pattern(x);
    const std::uint32_t magnitude = bits & 0x7FFFFFFFu;
    if (magnitude > 0x7F800000u) {
        return x;  // NaN
    }
    if (magnitude == 0u) {
        return detail::bit_pattern_to_float((bits & 0x80000000u) | 0x7F800000u);  // 1/sqrt(+-0)
    }
    if ((bits & 0x80000000u) != 0u) {
        return detail::bit_pattern_to_float(0x7FC00000u);  // sqrt of a negative
    }
    if (magnitude == 0x7F800000u) {
        return 0.0f;  // 1/sqrt(+inf)
    }

    // x = M * 2^(2k), M in [1, 4): the power of two leaves exactly, so Newton sees only M.
    std::uint32_t fraction = magnitude & 0x007FFFFFu;
    int exponent = 0;
    if (magnitude < 0x00800000u) {
        const int width = detail::significant_bits(magnitude);
        fraction = (magnitude << (24 - width)) & 0x007FFFFFu;
        exponent = width - 150;
    } else {
        exponent = static_cast<int>(magnitude >> 23) - 127;
    }
    const int odd = exponent & 1;
    const int half_exponent = (exponent - odd) / 2;
    const std::uint32_t m_bits = (static_cast<std::uint32_t>(127 + odd) << 23) | fraction;
    const float m = detail::bit_pattern_to_float(m_bits);

    // y <- y + y*(0.5 - (M/2)*y^2), three times.
    const float half = 0.5f * m;  // exact
    float y = detail::bit_pattern_to_float(detail::rsqrt_coeffs::magic - (m_bits >> 1));
    for (int step = 0; step < detail::rsqrt_coeffs::newton_steps; ++step) {
        const float square = y * y;
        const float correction = ULPGRAD_FMA(-half, square, 0.5f);
        y = ULPGRAD_FMA(y, correction, y);
    }

    return detail::scale_by_pow2(y, -half_exponent);
}

}  // namespace ulpgrad
