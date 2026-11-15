#include <ulpgrad/config.h>

#include <gtest/gtest.h>

#include <cmath>
#include <type_traits>

namespace {

// Token pasting is the emptiness check that works on every preprocessor: the
// argument is expanded before ## glues it on, so an empty ULPGRAD_HD leaves the
// identifier exactly as written. If the macro ever expanded to a token, the
// declaration below would name an identifier that does not exist and the build
// would fail instead of silently accepting a placeholder.
#define ULPGRAD_TEST_CAT_IMPL(a, b) a##b
#define ULPGRAD_TEST_CAT(a, b) ULPGRAD_TEST_CAT_IMPL(a, b)

constexpr int ULPGRAD_TEST_CAT(hd_expands_to_nothing_, ULPGRAD_HD) = 1;
static_assert(ULPGRAD_TEST_CAT(hd_expands_to_nothing_, ULPGRAD_HD) == 1,
              "ULPGRAD_HD must expand to nothing outside a CUDA compilation");

// Outside CUDA the qualifiers do not exist, so the macro has to be absent from
// the declaration; inside a CUDA compilation the same line compiles as
// __host__ __device__, and this suite is never built by nvcc.
ULPGRAD_HD float host_echo(float x) { return x; }

// ULPGRAD_FMA has to be usable in an expression and return a float: the device
// branch (__fmaf_rn) and the host branch (std::fmaf) must agree on the type, or
// every header that uses it would compile differently on the two targets.
static_assert(std::is_same<decltype(ULPGRAD_FMA(1.0f, 2.0f, 3.0f)), float>::value,
              "ULPGRAD_FMA must yield float");

}  // namespace

TEST(Config, HostOnlyHarnessRuns) {
    // The suite links, starts and reports: this is the sanity check that the
    // harness itself is not the reason a later suite goes green.
    EXPECT_TRUE(host_echo(1.0f) == 1.0f);
}

TEST(Config, FmaRoundsOnce) {
    // a = 1 + 2^-23 (0x3F800001), b = 1 - 2^-24 (0x3F7FFFFF).
    // The exact product is 1 + 2^-24 - 2^-47: 2^-24 above 1 is below half the
    // 2^-23 spacing, so a separate multiply rounds it away and the residual
    // disappears. Fused, the residual survives: 2^-24 - 2^-47, exactly
    // representable as 0x337FFFFF.
    const float a = 1.0f + std::ldexp(1.0f, -23);
    const float b = 1.0f - std::ldexp(1.0f, -24);
    const float expected = std::ldexp(1.0f, -24) - std::ldexp(1.0f, -47);

    EXPECT_EQ(ULPGRAD_FMA(a, b, -1.0f), expected);

    // Same operands, unfused: the volatile store forces the rounded product so
    // the compiler cannot contract it either way.
    volatile float product = a * b;
    EXPECT_EQ(product - 1.0f, 0.0f);
}
