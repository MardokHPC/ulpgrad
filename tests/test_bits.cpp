#include <ulpgrad/detail/bits.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

// Compile-time contracts: every one of these is a property the functions
// below rely on, not a restatement of the signature.
static_assert(noexcept(ulpgrad::detail::ulp_distance(0.0f, 0.0f)),
              "the ULP metric must be callable from a noexcept context");
static_assert(noexcept(ulpgrad::detail::scale_by_pow2(1.0f, 0)),
              "the scaling step must be callable from a noexcept context");
static_assert(std::is_same<decltype(ulpgrad::detail::ulp_distance(0.0f, 0.0f)), unsigned>::value,
              "a distance of +inf to -inf does not fit in int, so the result is unsigned");

constexpr std::uint32_t kPosZero = 0x00000000u;
constexpr std::uint32_t kNegZero = 0x80000000u;
constexpr std::uint32_t kPosInf = 0x7F800000u;
constexpr std::uint32_t kNegInf = 0xFF800000u;
constexpr std::uint32_t kMinSubnormal = 0x00000001u;

using ulpgrad::detail::bit_pattern_to_float;
using ulpgrad::detail::float_to_bit_pattern;
using ulpgrad::detail::scale_by_pow2;
using ulpgrad::detail::ulp_distance;

// bit_pattern_to_float() == reinterpretation, sign and payload included.
float from_bits(std::uint32_t bits) { return bit_pattern_to_float(bits); }

}  // namespace

// ----------------------------------------------------------------------------- //
// Bits
// ----------------------------------------------------------------------------- //

TEST(Bits, BitPatternRoundTripPreservesEveryPayload) {
    // The round trip has to be exact for values no arithmetic touches: a
    // negative zero, a quiet NaN with a non-zero payload and a signalling NaN
    // all survive only if the conversion is a pure reinterpretation.
    const std::uint32_t patterns[] = {kPosZero,       kNegZero,       0x7FC00001u,
                                      0x7F800001u,    0xFFC00000u,    kPosInf,
                                      kNegInf,        kMinSubnormal,  0x7F7FFFFFu};
    for (const std::uint32_t bits : patterns) {
        EXPECT_EQ(float_to_bit_pattern(from_bits(bits)), bits) << "pattern 0x" << std::hex << bits;
    }
}

// ----------------------------------------------------------------------------- //
// Bits
// ----------------------------------------------------------------------------- //

TEST(Bits, UlpDistanceBetweenNeighboursIsOneOnBothSidesOfZero) {
    // +1.0 is 0x3F800000 and its successor is 0x3F800001: one step in the
    // encoding, one ULP.
    EXPECT_EQ(ulp_distance(1.0f, from_bits(0x3F800001u)), 1u);

    // -1.0 is 0xBF800000; the successor *toward -inf* is 0xBF800001, i.e. a
    // larger magnitude. That is the direction the mapping has to turn around.
    EXPECT_EQ(ulp_distance(-1.0f, from_bits(0xBF800001u)), 1u);
    EXPECT_LT(from_bits(0xBF800001u), -1.0f);
}

TEST(Bits, UlpDistanceIsTwoToTheTwentyThirdBetweenOneAndTwo) {
    // 1.0 -> 2.0 crosses the exponent boundary 127 -> 128: 2^23 float32
    // patterns live between the two, all of them with the same ULP width.
    EXPECT_EQ(ulp_distance(1.0f, 2.0f), 8388608u);
    EXPECT_EQ(ulp_distance(2.0f, 1.0f), 8388608u) << "the metric must be symmetric";
}

TEST(Bits, UlpDistanceTreatsTheTwoZerosAsTheSameValue) {
    EXPECT_EQ(ulp_distance(0.0f, -0.0f), 0u);
    EXPECT_EQ(ulp_distance(-0.0f, 0.0f), 0u);
}

TEST(Bits, UlpDistanceAcrossZeroIsTheSumOfBothMagnitudes) {
    // 1.0 is 2^30 - 2^23 patterns above +0, so 1.0 to -1.0 is 2 * 1065353216.
    EXPECT_EQ(ulp_distance(1.0f, -1.0f), 2130706432u);
}

TEST(Bits, UlpDistanceBetweenTheInfinitiesExceedsIntRange) {
    // The one case that forces an unsigned return type: 2 * 2139095040.
    EXPECT_EQ(ulp_distance(from_bits(kPosInf), from_bits(kNegInf)), 4278190080u);
    EXPECT_GT(ulp_distance(from_bits(kPosInf), from_bits(kNegInf)),
              static_cast<unsigned>(std::numeric_limits<int>::max()));
}

TEST(Bits, UlpDistanceFromZeroToTheSmallestSubnormalIsOne) {
    EXPECT_EQ(ulp_distance(from_bits(kMinSubnormal), 0.0f), 1u);
    EXPECT_EQ(ulp_distance(from_bits(0x807FFFFFu), -0.0f), 8388607u)
        << "the largest subnormal magnitude is 2^23 - 1 patterns above zero";
}

// ----------------------------------------------------------------------------- //
// Bits
// ----------------------------------------------------------------------------- //

TEST(Bits, ScaleByPow2IsExactForANormalResult) {
    // k = 0: the identity, and the case the reduction hits for |x| < ln2/2.
    EXPECT_EQ(scale_by_pow2(1.0f, 0), 1.0f);
    EXPECT_EQ(float_to_bit_pattern(scale_by_pow2(1.5f, 0)), 0x3FC00000u);

    // Exponent field arithmetic, no multiply: 1.5 * 2^-4.
    EXPECT_EQ(float_to_bit_pattern(scale_by_pow2(1.5f, -4)), 0x3DC00000u);
    EXPECT_EQ(scale_by_pow2(1.5f, -4), 0.09375f);
}

TEST(Bits, ScaleByPow2AtTheTopOfTheExponentRange) {
    // e = 127 is the largest normal exponent: 1.0 * 2^127.
    EXPECT_EQ(float_to_bit_pattern(scale_by_pow2(1.0f, 127)), 0x7F000000u);

    // The largest value the reduction can hand over, scaled by the largest
    // useful k, lands exactly on FLT_MAX rather than overflowing.
    const float largest_below_two = from_bits(0x3FFFFFFFu);
    EXPECT_EQ(float_to_bit_pattern(scale_by_pow2(largest_below_two, 127)), 0x7F7FFFFFu);

    // One step further is a real overflow: e = 128 has no encoding.
    EXPECT_EQ(scale_by_pow2(1.0f, 128), std::numeric_limits<float>::infinity());
    EXPECT_EQ(scale_by_pow2(largest_below_two, 128), std::numeric_limits<float>::infinity());
}

TEST(Bits, ScaleByPow2ShiftsOneBitAndRoundsTiesToEven) {
    // s = 1, exact: 1.0 * 2^-127 is the largest power of two below the smallest
    // normal.
    EXPECT_EQ(float_to_bit_pattern(scale_by_pow2(1.0f, -127)), 0x00400000u);

    // s = 1, exact tie: 1 + 2^-23 at 2^-127 sits exactly half a subnormal ULP
    // above 0x00400000, whose significand 0x400000 is even, so it stays.
    EXPECT_EQ(float_to_bit_pattern(scale_by_pow2(from_bits(0x3F800001u), -127)), 0x00400000u);

    // The same tie one step further: 1 + 3*2^-23 leaves an odd 0x400001, so
    // the tie rounds up to 0x400002.
    EXPECT_EQ(float_to_bit_pattern(scale_by_pow2(from_bits(0x3F800003u), -127)), 0x00400002u);
}

TEST(Bits, ScaleByPow2CarriesIntoTheSmallestNormalEncoding) {
    // s = 1 and a significand of all ones: 0xFFFFFF >> 1 is 0x7FFFFF with a
    // remainder of exactly half, and the odd bit forces the round up. The
    // result 0x00800000 is 2^-126, the smallest normal, which is exactly the
    // next encoding after the largest subnormal, so no extra carry case exists.
    const float largest_below_two = from_bits(0x3FFFFFFFu);
    EXPECT_EQ(float_to_bit_pattern(scale_by_pow2(largest_below_two, -127)), 0x00800000u);
    EXPECT_EQ(scale_by_pow2(largest_below_two, -127), from_bits(0x00800000u));
}

TEST(Bits, ScaleByPow2AtTheSmallestSubnormalShift) {
    // s = 24: the worst subnormal shift the reduction can reach. 1.5 * 2^-150
    // is three quarters of the smallest subnormal, which rounds to 2^-149.
    EXPECT_EQ(float_to_bit_pattern(scale_by_pow2(1.5f, -150)), 0x00000001u);

    // The same shift with a value below the rounding threshold rounds to zero.
    EXPECT_EQ(float_to_bit_pattern(scale_by_pow2(1.0f, -150)), 0x00000000u);

    // s = 25: at most half of the smallest subnormal is representable, and
    // half is even, so the result is exactly +0 and not the smallest subnormal.
    EXPECT_EQ(float_to_bit_pattern(scale_by_pow2(1.0f, -151)), 0x00000000u);
    EXPECT_EQ(float_to_bit_pattern(scale_by_pow2(from_bits(0x3FFFFFFFu), -151)), 0x00000000u);
}
