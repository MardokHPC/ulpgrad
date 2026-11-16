#include <ulpgrad/exp.h>
#include <ulpgrad/log.h>
#include <ulpgrad/rsqrt.h>
#include <ulpgrad/sigmoid.h>
#include <ulpgrad/silu.h>
#include <ulpgrad/tables.h>
#include <ulpgrad/tanh.h>
#include <ulpgrad/detail/bits.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr std::uint32_t kPosInf = 0x7F800000u;
constexpr std::uint32_t kNegInf = 0xFF800000u;
constexpr std::uint32_t kPosZero = 0x00000000u;
constexpr std::uint32_t kNegZero = 0x80000000u;
constexpr std::uint32_t kOne = 0x3F800000u;
constexpr std::uint32_t kLargestFinite = 0x7F7FFFFFu;
constexpr std::uint32_t kMinSubnormal = 0x00000001u;
constexpr std::uint32_t kMinNormal = 0x00800000u;
constexpr std::uint32_t kQuietNan = 0x7FC00000u;

// Contracts the callers rely on and the compiler can prove.
static_assert(noexcept(ulpgrad::exp(0.0f)), "exp cannot fail, so it cannot throw");
static_assert(noexcept(ulpgrad::exp2(0.0f)), "exp2 cannot fail, so it cannot throw");
static_assert(noexcept(ulpgrad::rsqrt(0.0f)), "rsqrt cannot fail, so it cannot throw");
static_assert(noexcept(ulpgrad::log(1.0f)), "log cannot fail, so it cannot throw");
static_assert(noexcept(ulpgrad::log2(1.0f)), "log2 cannot fail, so it cannot throw");
static_assert(noexcept(ulpgrad::log1p(1.0f)), "log1p cannot fail, so it cannot throw");
static_assert(noexcept(ulpgrad::tanh(1.0f)), "tanh cannot fail, so it cannot throw");
static_assert(noexcept(ulpgrad::sigmoid(1.0f)), "sigmoid cannot fail, so it cannot throw");
static_assert(noexcept(ulpgrad::silu(1.0f)), "silu cannot fail, so it cannot throw");

float from_bits(std::uint32_t bits) { return ulpgrad::detail::bit_pattern_to_float(bits); }

std::uint32_t bits_of(float value) { return ulpgrad::detail::float_to_bit_pattern(value); }

// Compares bit patterns, not values: -0 and +0 compare equal but are different
// results, and the payload of a NaN is only visible in the pattern.
void expect_bits(float actual, std::uint32_t expected) {
    EXPECT_EQ(bits_of(actual), expected)
        << "got 0x" << std::hex << bits_of(actual) << " (" << actual << "), expected 0x"
        << expected << " (" << from_bits(expected) << ")" << std::dec;
}

}  // namespace

// ----------------------------------------------------------------------------- //
// SpecialCases
// ----------------------------------------------------------------------------- //

TEST(SpecialCases, NanReturnsItsOwnPayloadFromBothFunctions) {
    // A NaN argument is returned unchanged, bit for bit: sign, quiet/signalling
    // bit and payload all survive, which is more than "the result is a NaN".
    const std::uint32_t patterns[] = {
        0x7FC00000u,  // quiet NaN, zero payload
        0x7FC00001u,  // quiet NaN, non-zero payload
        0x7F800001u,  // signalling NaN
        0xFFC00000u,  // negative quiet NaN
        0xFFA12345u,  // negative quiet NaN with a payload
    };
    for (const std::uint32_t bits : patterns) {
        SCOPED_TRACE("pattern 0x" + std::to_string(bits));
        expect_bits(ulpgrad::exp(from_bits(bits)), bits);
        expect_bits(ulpgrad::exp2(from_bits(bits)), bits);
        expect_bits(ulpgrad::rsqrt(from_bits(bits)), bits);
        expect_bits(ulpgrad::log(from_bits(bits)), bits);
        expect_bits(ulpgrad::log2(from_bits(bits)), bits);
        expect_bits(ulpgrad::log1p(from_bits(bits)), bits);
        expect_bits(ulpgrad::tanh(from_bits(bits)), bits);
        expect_bits(ulpgrad::sigmoid(from_bits(bits)), bits);
        expect_bits(ulpgrad::silu(from_bits(bits)), bits);
    }
}

TEST(SpecialCases, InfinitiesSaturateInTheCorrectDirection) {
    expect_bits(ulpgrad::exp(from_bits(kPosInf)), kPosInf);
    expect_bits(ulpgrad::exp(from_bits(kNegInf)), kPosZero);  // +0, never -0
    expect_bits(ulpgrad::exp2(from_bits(kPosInf)), kPosInf);
    expect_bits(ulpgrad::exp2(from_bits(kNegInf)), kPosZero);
}

TEST(SpecialCases, BothZerosReturnExactlyOne) {
    // exp(+-0) is exactly 1.0, and the sign of the zero does not leak: the
    // result has to be 0x3F800000 from either argument.
    expect_bits(ulpgrad::exp(0.0f), kOne);
    expect_bits(ulpgrad::exp(from_bits(kNegZero)), kOne);
    expect_bits(ulpgrad::exp2(0.0f), kOne);
    expect_bits(ulpgrad::exp2(from_bits(kNegZero)), kOne);
}

TEST(SpecialCases, LargestFiniteArgumentOverflows) {
    expect_bits(ulpgrad::exp(from_bits(kLargestFinite)), kPosInf);
    expect_bits(ulpgrad::exp2(from_bits(kLargestFinite)), kPosInf);
}

TEST(SpecialCases, OverflowThresholdIsExactOnBothSides) {
    // 0x42B17218 is the smallest float32 whose correctly rounded exp is +inf,
    // computed by bisection against the 40-digit reference. The guard therefore
    // cannot be off by one: the neighbour below still has a finite result.
    expect_bits(ulpgrad::exp(from_bits(0x42B17218u)), kPosInf);
    expect_bits(ulpgrad::exp(from_bits(0x42B17217u)), 0x7F7FFF84u);
    expect_bits(ulpgrad::exp(from_bits(0x42B17219u)), kPosInf);

    // 128.0 is the smallest float32 whose rounded 2^x is +inf, and no float32
    // exists between 128.0 and the exact real threshold, so the guard is exact.
    expect_bits(ulpgrad::exp2(128.0f), kPosInf);
    expect_bits(ulpgrad::exp2(from_bits(0x42FFFFFFu)), 0x7F7FFFA7u);
}

TEST(SpecialCases, UnderflowThresholdIsExactOnBothSides) {
    // 0xC2CFF1B5 is the largest float32 whose correctly rounded exp is +0; the
    // neighbour above it lands on the smallest subnormal, i.e. the rounding
    // boundary between "zero" and "one unit in the last place" is here.
    expect_bits(ulpgrad::exp(from_bits(0xC2CFF1B5u)), kPosZero);
    expect_bits(ulpgrad::exp(from_bits(0xC2CFF1B4u)), kMinSubnormal);
    expect_bits(ulpgrad::exp(from_bits(0xC2CFF1B6u)), kPosZero);

    // -150 is exactly half the smallest subnormal, which rounds to zero, so
    // -150 itself is the threshold and the neighbour above is not.
    expect_bits(ulpgrad::exp2(-150.0f), kPosZero);
    expect_bits(ulpgrad::exp2(from_bits(0xC315FFFFu)), kMinSubnormal);
}

TEST(SpecialCases, SubnormalResultsStartAtTheRightArgument) {
    // The first argument whose exp is subnormal still produces a large
    // subnormal; the transition into 0x00800000 is not reachable from a float32
    // argument, which is why the carry is implemented but untested here.
    expect_bits(ulpgrad::exp(from_bits(0xC2AEAC50u)), 0x007FFFE6u);

    // 2^-126 is exactly representable and is the last normal result of exp2;
    // one ULP further down the argument leaves the normal range by about 90
    // subnormal ULP, because one float32 step of the argument is a 1.19e-7
    // relative step of the result and the subnormal grid is 2^-149 wide.
    expect_bits(ulpgrad::exp2(-126.0f), kMinNormal);
    expect_bits(ulpgrad::exp2(from_bits(0xC2FC0001u)), 0x007FFFD4u);
}

TEST(SpecialCases, SubnormalArgumentsReturnExactlyOne) {
    // exp(x) for |x| < 2^-126 is within one ULP of 1.0 from both sides, so the
    // correct answer is 1.0 and the polynomial path never sees the subnormal.
    expect_bits(ulpgrad::exp(from_bits(kMinSubnormal)), kOne);
    expect_bits(ulpgrad::exp(from_bits(0x807FFFFFu)), kOne);
    expect_bits(ulpgrad::exp2(from_bits(kMinSubnormal)), kOne);
    expect_bits(ulpgrad::exp2(from_bits(0x807FFFFFu)), kOne);
}

// ----------------------------------------------------------------------------- //
// Rsqrt
// ----------------------------------------------------------------------------- //

TEST(Rsqrt, ZerosAndInfinitiesFollowTheDivisionContract) {
    // 1/sqrt(x) is a division, and a division by a signed zero has a sign:
    // 1/sqrt(-0) is -inf, not +inf. Getting this from 1.0f/sqrtf(x) would be
    // correct on both targets, but the function never divides, so the sign is
    // carried in the bit pattern explicitly.
    expect_bits(ulpgrad::rsqrt(from_bits(kPosZero)), kPosInf);
    expect_bits(ulpgrad::rsqrt(from_bits(kNegZero)), kNegInf);
    expect_bits(ulpgrad::rsqrt(from_bits(kPosInf)), kPosZero);
    expect_bits(ulpgrad::rsqrt(from_bits(kNegInf)), kQuietNan);
}

TEST(Rsqrt, NegativeArgumentReturnsTheCanonicalQuietNan) {
    // The pattern is written into the bits rather than produced by an invalid
    // operation, because the NaN an invalid operation yields is the target's
    // choice: x86 hands back -nan and CUDA hands back +nan, and the parity
    // claim would break on a bit that the library never promised.
    expect_bits(ulpgrad::rsqrt(from_bits(kOne | 0x80000000u)), kQuietNan);
    expect_bits(ulpgrad::rsqrt(-2.0f), kQuietNan);
    expect_bits(ulpgrad::rsqrt(from_bits(kMinSubnormal | 0x80000000u)), kQuietNan);
    expect_bits(ulpgrad::rsqrt(from_bits(kLargestFinite | 0x80000000u)), kQuietNan);
}

TEST(Rsqrt, PowersOfFourAreExact) {
    // An even exponent means the reduction hands Newton M = 1, and the iterate
    // converges to the exact answer rather than to within an ULP of it. These
    // are the points where a 1 ULP bound is not good enough to hide behind.
    expect_bits(ulpgrad::rsqrt(1.0f), kOne);
    expect_bits(ulpgrad::rsqrt(4.0f), 0x3F000000u);    // 0.5
    expect_bits(ulpgrad::rsqrt(0.25f), 0x40000000u);   // 2.0
    expect_bits(ulpgrad::rsqrt(0.0625f), 0x40800000u);  // 4.0
    expect_bits(ulpgrad::rsqrt(1024.0f), 0x3D000000u);  // 0.03125
}

TEST(Rsqrt, SubnormalArgumentIsRenormalisedNotFlushed) {
    // The smallest normal argument is a power of two, so the reduction hands
    // Newton M = 1 and the answer is 2^63 exactly. A subnormal input is the case
    // that separates this implementation from one that scales the float:
    // 2^-149 * 2^24 is exact on the host and flushed to zero under -ftz=true on
    // the device, so the renormalisation is done with integer shifts on both.
    expect_bits(ulpgrad::rsqrt(from_bits(kMinNormal)), 0x5F000000u);

    // The irrational answers are checked against binary64 with a single
    // rounding. That cannot land on a tie: 1/sqrt(m) is rational only when m is
    // the square of a dyadic rational, and then it is a power of two and lands
    // in the exact case above.
    const std::uint32_t patterns[] = {kMinSubnormal, 0x00000002u, 0x00000003u, 0x00400000u,
                                      0x007FFFFFu, kLargestFinite};
    for (const std::uint32_t bits : patterns) {
        const float x = from_bits(bits);
        const float expected = static_cast<float>(1.0 / std::sqrt(static_cast<double>(x)));
        EXPECT_LE(ulpgrad::detail::ulp_distance(ulpgrad::rsqrt(x), expected),
                  static_cast<unsigned>(ulpgrad::rsqrt_max_ulp))
            << "input 0x" << std::hex << bits << std::dec;
    }
    // The largest finite argument is the smallest result, and it is still
    // normal: no output of rsqrt can be subnormal, which is why the scaling step
    // never rounds.
    expect_bits(ulpgrad::rsqrt(from_bits(kLargestFinite)), 0x1F800000u);
}

// ----------------------------------------------------------------------------- //
// Log, Log2, Log1p
// ----------------------------------------------------------------------------- //

TEST(Log, ZerosLandOnMinusInfinityAndOneIsExact) {
    // log(+-0) is -inf: the sign of the zero does not survive a logarithm, and
    // log(1.0f +- 0) is the one input whose answer the series can give exactly,
    // which is why the reduction's `m - 1` has to be exact for it.
    expect_bits(ulpgrad::log(0.0f), kNegInf);
    expect_bits(ulpgrad::log(from_bits(kNegZero)), kNegInf);
    expect_bits(ulpgrad::log(1.0f), kPosZero);
    expect_bits(ulpgrad::log(from_bits(kOne)), kPosZero);
    expect_bits(ulpgrad::log(from_bits(kPosInf)), kPosInf);
    expect_bits(ulpgrad::log(from_bits(kNegInf)), kQuietNan);
}

TEST(Log, NegativeArgumentReturnsTheCanonicalQuietNan) {
    expect_bits(ulpgrad::log(-1.0f), kQuietNan);
    expect_bits(ulpgrad::log(from_bits(kMinSubnormal | 0x80000000u)), kQuietNan);
    expect_bits(ulpgrad::log(from_bits(kLargestFinite | 0x80000000u)), kQuietNan);
    expect_bits(ulpgrad::log2(-1.0f), kQuietNan);
    expect_bits(ulpgrad::log2(from_bits(kNegInf)), kQuietNan);
}

TEST(Log, SubnormalArgumentsAreRenormalised) {
    // The smallest subnormal is 2^-149, so its logarithm is exactly
    // -149*ln2 = -103.27893; the reduction finds that exponent with integer
    // shifts, since scaling the float would be flushed by -ftz=true.
    expect_bits(ulpgrad::log(from_bits(kMinSubnormal)), 0xC2CE8ED0u);
    expect_bits(ulpgrad::log(from_bits(kMinNormal)), 0xC2AEAC50u);  // -126*ln2
    expect_bits(ulpgrad::log(from_bits(kLargestFinite)), 0x42B17218u);
    // One ULP either side of 1: the answer is tiny, and a reduction that lost
    // the exactness of `m - 1` would return zero for both. These two are not
    // exact answers -- with e = 0 the whole result comes from the series -- so
    // they are checked against the 40-digit reference within the declared bound.
    const float below = ulpgrad::log(from_bits(0x3F7FFFFFu));
    const float above = ulpgrad::log(from_bits(0x3F800001u));
    EXPECT_NE(below, 0.0f);
    EXPECT_NE(above, 0.0f);
    EXPECT_LE(ulpgrad::detail::ulp_distance(below, from_bits(0xB3800000u)),
              static_cast<unsigned>(ulpgrad::log_max_ulp));
    EXPECT_LE(ulpgrad::detail::ulp_distance(above, from_bits(0x33FFFFFFu)),
              static_cast<unsigned>(ulpgrad::log_max_ulp));
}

TEST(Log2, PowersOfTwoAreTheExactInteger) {
    // What the separate fit buys, spot-checked on the three ends of the range.
    expect_bits(ulpgrad::log2(1.0f), kPosZero);
    expect_bits(ulpgrad::log2(1024.0f), 0x41200000u);                 // 10
    expect_bits(ulpgrad::log2(0.5f), 0xBF800000u);                    // -1
    expect_bits(ulpgrad::log2(from_bits(kMinNormal)), 0xC2FC0000u);   // -126
    expect_bits(ulpgrad::log2(from_bits(kMinSubnormal)), 0xC3150000u);  // -149
    expect_bits(ulpgrad::log2(from_bits(kLargestFinite)), 0x43000000u);  // rounds to 128
    expect_bits(ulpgrad::log2(0.0f), kNegInf);
    expect_bits(ulpgrad::log2(from_bits(kNegZero)), kNegInf);
}

TEST(Log1p, DomainEdgesAreHandledWithoutArithmetic) {
    // -1 has no logarithm: it is the pole, and the guard is a comparison rather
    // than a division that would hand the answer to the target's infinity.
    expect_bits(ulpgrad::log1p(-1.0f), kNegInf);
    expect_bits(ulpgrad::log1p(-1.0000001f), kQuietNan);
    expect_bits(ulpgrad::log1p(from_bits(kNegInf)), kQuietNan);
    expect_bits(ulpgrad::log1p(from_bits(kPosInf)), kPosInf);
    // Both zeros, with their signs: log1p is one of the functions where -0 in
    // and -0 out is the whole contract.
    expect_bits(ulpgrad::log1p(0.0f), kPosZero);
    expect_bits(ulpgrad::log1p(from_bits(kNegZero)), kNegZero);
}

TEST(Log1p, ValuesBelowTheGuardAreReturnedUnchanged) {
    // 1e-30 and 1e-38 are the reason the function exists: log(1.0f + x) rounds
    // both of them to zero, while the correct answer is x to within an ULP. The
    // subnormals are covered by the same guard, so no arithmetic ever sees them
    // and -ftz=true cannot change the answer on the device.
    expect_bits(ulpgrad::log1p(1.0e-30f), bits_of(1.0e-30f));
    expect_bits(ulpgrad::log1p(1.0e-38f), bits_of(1.0e-38f));
    expect_bits(ulpgrad::log1p(from_bits(kMinSubnormal)), kMinSubnormal);
    expect_bits(ulpgrad::log1p(from_bits(kMinSubnormal | 0x80000000u)),
                kMinSubnormal | 0x80000000u);
    // Just above the guard the series takes over and must still agree with `x`
    // to within the declared bound.
    const float tiny = std::ldexp(1.0f, -24);
    EXPECT_LE(ulpgrad::detail::ulp_distance(ulpgrad::log1p(tiny), tiny),
              static_cast<unsigned>(ulpgrad::log1p_max_ulp));
}

// ----------------------------------------------------------------------------- //
// Tanh
// ----------------------------------------------------------------------------- //

TEST(Tanh, ZerosKeepTheirSignAndInfinitiesSaturate) {
    // tanh is one of the few functions where -0 in and -0 out is a contract: the
    // polynomial is x plus a correction, and -0 + -0 is -0.
    expect_bits(ulpgrad::tanh(0.0f), kPosZero);
    expect_bits(ulpgrad::tanh(from_bits(kNegZero)), kNegZero);
    expect_bits(ulpgrad::tanh(from_bits(kPosInf)), kOne);
    expect_bits(ulpgrad::tanh(from_bits(kNegInf)), 0xBF800000u);
}

TEST(Tanh, SaturationThresholdIsExactOnBothSides) {
    // 0x41102CB4 is the smallest float32 whose correctly rounded tanh is 1.0,
    // found by bisection against the 40-digit reference. One pattern below it the
    // exp2 branch runs and still returns 1.0 -- that is the point: the guard is a
    // shortcut, not the thing that produces the value.
    expect_bits(ulpgrad::tanh(from_bits(0x41102CB4u)), kOne);
    expect_bits(ulpgrad::tanh(from_bits(0x41102CB5u)), kOne);
    expect_bits(ulpgrad::tanh(from_bits(0x41102CB3u)), kOne);
    expect_bits(ulpgrad::tanh(from_bits(0x41102CB4u | 0x80000000u)), 0xBF800000u);
    // The largest finite argument is saturated in both directions.
    expect_bits(ulpgrad::tanh(from_bits(kLargestFinite)), kOne);
    expect_bits(ulpgrad::tanh(from_bits(kLargestFinite | 0x80000000u)), 0xBF800000u);
}

TEST(Tanh, TinyArgumentsComeBackUnchanged) {
    // |x| < 2^-25 puts x^3*R(x^2) below half an ULP of x, so the correction cannot
    // move the result -- including on the device, where -ftz=true flushes x*x to
    // zero for these arguments and the correction becomes exactly zero instead of
    // a denormal.
    expect_bits(ulpgrad::tanh(from_bits(kMinSubnormal)), kMinSubnormal);
    expect_bits(ulpgrad::tanh(from_bits(kMinSubnormal | 0x80000000u)),
                kMinSubnormal | 0x80000000u);
    const float tiny = std::ldexp(1.0f, -30);
    expect_bits(ulpgrad::tanh(tiny), bits_of(tiny));
    expect_bits(ulpgrad::tanh(-tiny), bits_of(-tiny));
    // Just above that the polynomial takes over and the answer is still within
    // the declared bound of x itself.
    const float borderline = std::ldexp(1.0f, -12);
    EXPECT_LE(ulpgrad::detail::ulp_distance(ulpgrad::tanh(borderline), borderline),
              static_cast<unsigned>(ulpgrad::tanh_max_ulp));
}

TEST(Tanh, IsBoundedByOne) {
    // tanh never leaves [-1, 1], which the whole branch structure has to respect:
    // a wrong sign in the exp2 branch would show up as a value above the bound.
    for (std::uint32_t bits = 0x00000001u; bits < 0x7F7FFFFFu; bits = bits * 3u + 7u) {
        const float x = from_bits(bits);
        if (!(x < 1.0e30f)) {
            continue;
        }
        const float value = ulpgrad::tanh(x);
        EXPECT_LE(value, 1.0f) << "x = " << x;
        EXPECT_GE(value, -1.0f) << "x = " << x;
    }
}

// ----------------------------------------------------------------------------- //
// Sigmoid and Silu
// ----------------------------------------------------------------------------- //

TEST(Sigmoid, ZerosAreOneHalfAndInfinitiesSaturate) {
    expect_bits(ulpgrad::sigmoid(0.0f), 0x3F000000u);                 // 0.5
    expect_bits(ulpgrad::sigmoid(from_bits(kNegZero)), 0x3F000000u);  // 0.5, no sign leak
    expect_bits(ulpgrad::sigmoid(from_bits(kPosInf)), kOne);
    expect_bits(ulpgrad::sigmoid(from_bits(kNegInf)), kPosZero);
}

TEST(Sigmoid, StaysInsideTheUnitInterval) {
    // The branch structure has to keep the result in [0, 1]; a wrong sign in the
    // negative branch would show up as a value outside it.
    for (std::uint32_t bits = 0x00000001u; bits < 0x7F7FFFFFu; bits = bits * 3u + 7u) {
        const float value = ulpgrad::sigmoid(from_bits(bits));
        EXPECT_LE(value, 1.0f) << "x = " << from_bits(bits);
        EXPECT_GE(value, 0.0f) << "x = " << from_bits(bits);
    }
}

TEST(Silu, ZerosKeepTheirSignAndInfinitiesFollowTheLimit) {
    expect_bits(ulpgrad::silu(0.0f), kPosZero);
    expect_bits(ulpgrad::silu(from_bits(kNegZero)), kNegZero);
    expect_bits(ulpgrad::silu(from_bits(kPosInf)), kPosInf);
    // silu(-inf) is the limit of x*sigmoid(x): a negative zero, not -inf*0 = NaN.
    expect_bits(ulpgrad::silu(from_bits(kNegInf)), kNegZero);
}

TEST(Silu, DeepNegativeTailRoundsToNegativeZero) {
    // Below about -104 the magnitude falls under half the smallest subnormal, and
    // the sign still has to be the argument's.
    expect_bits(ulpgrad::silu(-200.0f), kNegZero);
    expect_bits(ulpgrad::silu(from_bits(kLargestFinite | 0x80000000u)), kNegZero);
}
