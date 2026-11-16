#pragma once

#include <ulpgrad/config.h>
#include <ulpgrad/exp.h>
#include <ulpgrad/detail/bits.h>

#include <cstdint>

namespace ulpgrad {

// x * sigmoid(x), rounded to nearest; -0 at -inf, +inf at +inf.
[[nodiscard]] ULPGRAD_HD inline float silu(float x) noexcept {
    const std::uint32_t bits = detail::float_to_bit_pattern(x);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        return x;  // NaN
    }
    if (bits == 0xFF800000u) {
        return detail::bit_pattern_to_float(0x80000000u);  // -inf * 0 would be NaN
    }
    // x / (1 + e^-x): on this side e^-x never exceeds e^88, so there is no
    // subnormal intermediate to lose bits in.
    if (x >= -88.0f) {
        return x / (1.0f + ulpgrad::exp(-x));
    }
    // Below -88, e^x is subnormal and 1 + e^x is exactly 1, so silu = x*e^x.
    // Multiplying the subnormal e^x by |x| would amplify its rounding by |x|
    // (tens of ULP); apply |x| before the power of two instead, so the subnormal
    // grid is rounded once.
    return -detail::exp_times(x, -x);
}

}  // namespace ulpgrad
