#pragma once

#include <ulpgrad/config.h>

#include <cstdint>
#include <cstring>

namespace ulpgrad {
namespace detail {

// Reinterpret a bit pattern as the float it encodes.
[[nodiscard]] ULPGRAD_HD inline float bit_pattern_to_float(std::uint32_t bits) noexcept {
#ifdef __CUDA_ARCH__
    return __uint_as_float(bits);
#else
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof value);
    return value;
#endif
}

// Reinterpret a float as its bit pattern.
[[nodiscard]] ULPGRAD_HD inline std::uint32_t float_to_bit_pattern(float value) noexcept {
#ifdef __CUDA_ARCH__
    return __float_as_uint(value);
#else
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof bits);
    return bits;
#endif
}

// Position of the leading one in a non-zero magnitude field, in [1, 23].
[[nodiscard]] ULPGRAD_HD constexpr int significant_bits(std::uint32_t magnitude) noexcept {
    int bits = 1;
    std::uint32_t field = magnitude;
    if (field >= 0x001000u) {  // 2^12
        field >>= 12;
        bits += 12;
    }
    if (field >= 0x000040u) {  // 2^6
        field >>= 6;
        bits += 6;
    }
    if (field >= 0x000008u) {  // 2^3
        field >>= 3;
        bits += 3;
    }
    if (field >= 0x000004u) {  // 2^2
        field >>= 2;
        bits += 2;
    }
    if (field >= 0x000002u) {  // 2^1
        field >>= 1;
        bits += 1;
    }
    return bits;
}

// Position of a float32 pattern on the total order of float32 values.
[[nodiscard]] ULPGRAD_HD constexpr std::int64_t monotone_key(std::uint32_t bits) noexcept {
    return (bits & 0x80000000u) != 0u
               ? -static_cast<std::int64_t>(bits & 0x7FFFFFFFu)
               : static_cast<std::int64_t>(bits);
}

// Number of representable float32 values strictly between a and b.
[[nodiscard]] ULPGRAD_HD inline unsigned ulp_distance(float a, float b) noexcept {
    const std::int64_t difference =
        monotone_key(float_to_bit_pattern(a)) - monotone_key(float_to_bit_pattern(b));
    return static_cast<unsigned>(difference < 0 ? -difference : difference);
}

// y * 2^k, applied to the exponent field, rounding once on the subnormal grid.
[[nodiscard]] ULPGRAD_HD inline float scale_by_pow2(float y, int k) noexcept {
    const std::uint32_t iy = float_to_bit_pattern(y);
    const int e = static_cast<int>(iy >> 23) + k - 127;
    if (e > 127) {
        return bit_pattern_to_float(0x7F800000u);  // overflow: +inf
    }
    if (e >= -126) {
        return bit_pattern_to_float(iy + (static_cast<std::uint32_t>(k) << 23));
    }
    const int shift = -126 - e;  // 1 .. 25
    if (shift > 24) {
        return 0.0f;  // at most half the smallest subnormal
    }
    std::uint32_t m = (iy & 0x007FFFFFu) | 0x00800000u;  // explicit leading bit
    const std::uint32_t remainder = m & ((1u << shift) - 1u);
    const std::uint32_t half = 1u << (shift - 1);
    m >>= shift;
    if (remainder > half || (remainder == half && (m & 1u) != 0u)) {
        ++m;
    }
    return bit_pattern_to_float(m);
}

}  // namespace detail
}  // namespace ulpgrad
