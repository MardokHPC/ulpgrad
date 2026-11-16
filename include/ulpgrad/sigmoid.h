#pragma once

#include <ulpgrad/config.h>
#include <ulpgrad/exp.h>
#include <ulpgrad/detail/bits.h>

#include <cstdint>

namespace ulpgrad {

// 1/(1+e^-x), rounded to nearest; 0 at -inf, 1 at +inf.
[[nodiscard]] ULPGRAD_HD inline float sigmoid(float x) noexcept {
    const std::uint32_t bits = detail::float_to_bit_pattern(x);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        return x;  // NaN
    }
    // Two branches so no exponential overflows. On the negative side e^-x
    // overflows below x = -88; that would return 0 where the true value is a
    // subnormal, so the branch uses e^x, which stays in (0, 1].
    if (x >= 0.0f) {
        return 1.0f / (1.0f + ulpgrad::exp(-x));
    }
    const float e = ulpgrad::exp(x);  // in (0, 1]
    return e / (1.0f + e);
}

}  // namespace ulpgrad
