#pragma once

#include <ulpgrad/config.h>
#include <ulpgrad/detail/bits.h>
#include <ulpgrad/detail/coefficients.h>

#include <cstdint>

namespace ulpgrad {

namespace detail {

// x = m * 2^e, m in [sqrt(1/2), sqrt(2)), both exact.
struct LogReduction {
    float exponent;
    float mantissa;
};

// Split a positive, finite, non-zero float into mantissa and exponent.
[[nodiscard]] ULPGRAD_HD inline LogReduction reduce_for_log(float x) noexcept {
    const std::uint32_t bits = float_to_bit_pattern(x);
    std::uint32_t fraction = bits & 0x007FFFFFu;
    int exponent = 0;
    if (bits < 0x00800000u) {
        // Subnormal: renormalise with integer shifts, not a float scale (FTZ).
        const int width = significant_bits(bits);
        fraction = (bits << (24 - width)) & 0x007FFFFFu;
        exponent = width - 150;
    } else {
        exponent = static_cast<int>(bits >> 23) - 127;
    }
    const bool high = fraction > 0x3504F3u;  // m > sqrt(2)
    const std::uint32_t m_bits = ((high ? 126u : 127u) << 23) | fraction;
    LogReduction reduction;
    reduction.exponent = static_cast<float>(exponent + (high ? 1 : 0));
    reduction.mantissa = bit_pattern_to_float(m_bits);
    return reduction;
}

// The atanh argument s = t/(2+t), plus the correction the rounded denominator owes.
struct LogArgument {
    float s;
    float residual;
};

// s = t / (2 + t), and the correction for the rounding of 2 + t.
[[nodiscard]] ULPGRAD_HD inline LogArgument atanh_argument(float t) noexcept {
    const float h = 2.0f + t;           // rounded
    const float d_lo = (2.0f - h) + t;  // exact: what the rounding dropped
    const float s = t / h;              // correctly rounded
    float reciprocal = log_coeffs::inv_two_m_2;  // 1/(2 + 2t)
    reciprocal = ULPGRAD_FMA(reciprocal, t, log_coeffs::inv_two_m_1);
    reciprocal = ULPGRAD_FMA(reciprocal, t, log_coeffs::inv_two_m_0);
    // fma with a zero addend: a bare product feeding an addition is a contraction
    // candidate, and the fused form would skip the rounding this correction is built from.
    const float dropped = ULPGRAD_FMA(t, d_lo, 0.0f);
    LogArgument argument;
    argument.s = s;
    argument.residual = ULPGRAD_FMA(dropped, -reciprocal, 0.0f);
    return argument;
}

// 2 atanh(s) = 2s + 2s*z*R(z), plus an additive correction folded into the tail.
[[nodiscard]] ULPGRAD_HD inline float atanh_series(float s, float extra) noexcept {
    const float z = s * s;
    float tail = log_coeffs::r4;
    tail = ULPGRAD_FMA(tail, z, log_coeffs::r3);
    tail = ULPGRAD_FMA(tail, z, log_coeffs::r2);
    tail = ULPGRAD_FMA(tail, z, log_coeffs::r1);
    tail = ULPGRAD_FMA(tail, z, log_coeffs::r0);
    const float two_s = s + s;  // exact
    const float correction = ULPGRAD_FMA(two_s * z, tail, 0.0f);
    return two_s + (extra + correction);
}

}  // namespace detail

// ln(x), rounded to nearest; -inf at +-0, +inf at +inf, NaN for x < 0.
[[nodiscard]] ULPGRAD_HD inline float log(float x) noexcept {
    const std::uint32_t bits = detail::float_to_bit_pattern(x);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        return x;  // NaN
    }
    if (bits == 0u || bits == 0x80000000u) {
        return detail::bit_pattern_to_float(0xFF800000u);  // log(+-0) = -inf
    }
    if ((bits & 0x80000000u) != 0u) {
        return detail::bit_pattern_to_float(0x7FC00000u);  // log of a negative
    }
    if (bits == 0x7F800000u) {
        return x;  // log(+inf) = +inf
    }

    const detail::LogReduction reduction = detail::reduce_for_log(x);
    const detail::LogArgument argument = detail::atanh_argument(reduction.mantissa - 1.0f);
    const float shift = ULPGRAD_FMA(reduction.exponent, detail::log_coeffs::ln2_lo, 0.0f);
    const float log_m = detail::atanh_series(argument.s, shift + argument.residual);
    return ULPGRAD_FMA(reduction.exponent, detail::log_coeffs::ln2_hi, log_m);
}

// log2(x), rounded to nearest; exact integer for powers of two.
[[nodiscard]] ULPGRAD_HD inline float log2(float x) noexcept {
    const std::uint32_t bits = detail::float_to_bit_pattern(x);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        return x;  // NaN
    }
    if (bits == 0u || bits == 0x80000000u) {
        return detail::bit_pattern_to_float(0xFF800000u);
    }
    if ((bits & 0x80000000u) != 0u) {
        return detail::bit_pattern_to_float(0x7FC00000u);
    }
    if (bits == 0x7F800000u) {
        return x;
    }

    const detail::LogReduction reduction = detail::reduce_for_log(x);
    const detail::LogArgument argument = detail::atanh_argument(reduction.mantissa - 1.0f);
    const float s = argument.s;
    const float z = s * s;
    float tail = detail::log2_coeffs::r4;
    tail = ULPGRAD_FMA(tail, z, detail::log2_coeffs::r3);
    tail = ULPGRAD_FMA(tail, z, detail::log2_coeffs::r2);
    tail = ULPGRAD_FMA(tail, z, detail::log2_coeffs::r1);
    tail = ULPGRAD_FMA(tail, z, detail::log2_coeffs::r0);
    const float two_s = s + s;  // exact
    // Leading product split into value + fused residual, so its own rounding is not lost.
    const float leading = ULPGRAD_FMA(two_s, detail::log2_coeffs::log2_e, 0.0f);
    const float residual = ULPGRAD_FMA(two_s, detail::log2_coeffs::log2_e, -leading);
    const float scaled = ULPGRAD_FMA(argument.residual, detail::log2_coeffs::log2_e, 0.0f);
    const float delta = ULPGRAD_FMA(two_s * z, tail, residual + scaled);
    return reduction.exponent + (leading + delta);
}

// ln(1 + x), accurate for x near zero; returns x itself for |x| <= 2^-25.
[[nodiscard]] ULPGRAD_HD inline float log1p(float x) noexcept {
    const std::uint32_t bits = detail::float_to_bit_pattern(x);
    const std::uint32_t magnitude = bits & 0x7FFFFFFFu;
    if (magnitude > 0x7F800000u) {
        return x;  // NaN
    }
    if (magnitude == 0x7F800000u) {
        return (bits & 0x80000000u) != 0u ? detail::bit_pattern_to_float(0x7FC00000u) : x;
    }
    if (x == -1.0f) {
        return detail::bit_pattern_to_float(0xFF800000u);
    }
    if (x < -1.0f) {
        return detail::bit_pattern_to_float(0x7FC00000u);
    }
    if (magnitude <= 0x33000000u) {
        return x;  // |x| <= 2^-25: the answer rounds to x
    }
    if (magnitude <= 0x3ED413CDu) {
        // |x| <= sqrt(2)-1: s = x/(2+x) stays in the fitted interval.
        const detail::LogArgument argument = detail::atanh_argument(x);
        return detail::atanh_series(argument.s, argument.residual);
    }

    // |x| > sqrt(2)-1: recover the rounding of 1 + x with Knuth's two-sum.
    const float h = 1.0f + x;
    const float part = h - x;                            // exact
    const float low = (x - (h - part)) + (1.0f - part);  // exact residual
    const detail::LogReduction reduction = detail::reduce_for_log(h);
    const detail::LogArgument argument = detail::atanh_argument(reduction.mantissa - 1.0f);
    const float shift = ULPGRAD_FMA(reduction.exponent, detail::log_coeffs::ln2_lo, low / h);
    const float log_m = detail::atanh_series(argument.s, shift + argument.residual);
    return ULPGRAD_FMA(reduction.exponent, detail::log_coeffs::ln2_hi, log_m);
}

}  // namespace ulpgrad
