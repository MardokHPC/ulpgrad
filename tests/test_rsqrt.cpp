#include <ulpgrad/rsqrt.h>
#include <ulpgrad/tables.h>
#include <ulpgrad/detail/bits.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

// Same |x| split as tools/measure_bounds.cpp: bucket i covers [2^(i-1), 2^i),
// bucket 0 also covers every subnormal magnitude, and the last bucket is
// open-ended. The counts below are compared with the declared table, so a
// classifier that disagreed with the tool's would fail on arithmetic, not on a
// comment nobody reads.
constexpr int kBucketCount = ulpgrad::rsqrt_error_bounds_count;

int bucket_of(float x) {
    const std::uint32_t magnitude = ulpgrad::detail::float_to_bit_pattern(x) & 0x7FFFFFFFu;
    const std::uint32_t field = magnitude >> 23;
    if (field == 0xFFu) {
        return kBucketCount - 1;
    }
    const int exponent = static_cast<int>(field) - 127;
    if (exponent < 0) {
        return 0;
    }
    const int index = exponent + 1;
    return index < kBucketCount ? index : kBucketCount - 1;
}

bool is_nan_bits(std::uint32_t bits) {
    return (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u;
}

struct Point {
    std::uint32_t input;
    std::uint32_t expected;
};

// Hex field parser that cannot throw and rejects trailing garbage: the golden
// file is committed data, and a typo in it has to be reported, not crash.
bool parse_hex(const std::string& text, std::uint32_t* value) {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(text.c_str(), &end, 16);
    if (end == text.c_str() || *end != '\0' || parsed > 0xFFFFFFFFul) {
        return false;
    }
    *value = static_cast<std::uint32_t>(parsed);
    return true;
}

struct Stats {
    unsigned max_ulp = 0;
    std::uint32_t worst_input = 0;
    long long measured = 0;
    long long nan_inputs = 0;
    long long nan_outputs = 0;
    long long nan_mismatches = 0;
    unsigned bucket_max[kBucketCount] = {};
    long long bucket_points[kBucketCount] = {};
};

// Reads a golden file. A missing or malformed file sets `error` and returns an
// empty vector: the caller turns that into a failure, never into a skip.
std::vector<Point> read_golden(const std::string& path, std::string* error) {
    std::vector<Point> points;
    std::ifstream stream(path);
    if (!stream) {
        *error = "cannot open the golden reference " + path;
        return points;
    }
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const std::size_t comma = line.find(',');
        std::uint32_t input = 0;
        std::uint32_t expected = 0;
        if (comma == std::string::npos || !parse_hex(line.substr(0, comma), &input) ||
            !parse_hex(line.substr(comma + 1), &expected)) {
            *error = "malformed record '" + line + "' in " + path;
            return std::vector<Point>();
        }
        points.push_back(Point{input, expected});
    }
    return points;
}

// Mirrors tools/measure_bounds.cpp: a NaN at either end of the comparison is
// checked on the payload, because a NaN pattern has no position on the real
// line and a ULP distance between two of them means nothing.
Stats measure(const std::vector<Point>& points, float (*function)(float)) {
    Stats stats;
    for (const Point& point : points) {
        const float x = ulpgrad::detail::bit_pattern_to_float(point.input);
        const float got = function(x);
        const std::uint32_t got_bits = ulpgrad::detail::float_to_bit_pattern(got);
        if (is_nan_bits(point.input) || is_nan_bits(point.expected)) {
            if (is_nan_bits(point.input)) {
                ++stats.nan_inputs;
            } else {
                ++stats.nan_outputs;
            }
            if (got_bits != point.expected) {
                ++stats.nan_mismatches;
            }
            continue;
        }
        const unsigned distance = ulpgrad::detail::ulp_distance(
            got, ulpgrad::detail::bit_pattern_to_float(point.expected));
        ++stats.measured;
        if (distance > stats.max_ulp) {
            stats.max_ulp = distance;
            stats.worst_input = point.input;
        }
        const int bucket = bucket_of(x);
        ++stats.bucket_points[bucket];
        if (distance > stats.bucket_max[bucket]) {
            stats.bucket_max[bucket] = distance;
        }
    }
    return stats;
}

void expect_bucket_counts_match(const Stats& stats, const ulpgrad::ErrorBound* bounds) {
    for (int bucket = 0; bucket < kBucketCount; ++bucket) {
        EXPECT_EQ(stats.bucket_points[bucket], bounds[bucket].points)
            << "bucket " << bounds[bucket].range
            << ": the test and tools/measure_bounds.cpp disagree on which points land here";
    }
}

// Deterministic 32-bit LCG, so a failure reproduces without a random engine
// whose sequence is a library implementation detail.
std::uint32_t next_random(std::uint32_t* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

}  // namespace

// ----------------------------------------------------------------------------- //
// Rsqrt
// ----------------------------------------------------------------------------- //

TEST(Rsqrt, GoldenReferenceUlpIsWithinDeclaredBound) {
    std::string error;
    const std::vector<Point> points =
        read_golden(std::string(ULPGRAD_TEST_DATA_DIR) + "/reference/rsqrt.csv", &error);
    ASSERT_TRUE(error.empty()) << error << " (the golden reference is committed: regenerate it "
                                          "with experiments/m3_fit/gen_golden.py)";
    ASSERT_FALSE(points.empty());

    const Stats stats = measure(points, &ulpgrad::rsqrt);
    SCOPED_TRACE("golden points: " + std::to_string(points.size()));
    EXPECT_EQ(stats.nan_inputs, 3);
    EXPECT_GT(stats.nan_outputs, 0) << "no negative argument was measured: the sample lost them";
    EXPECT_EQ(stats.nan_mismatches, 0) << "NaN results must be bit-identical to the reference";
    EXPECT_LE(stats.max_ulp, static_cast<unsigned>(ulpgrad::rsqrt_max_ulp))
        << "worst input 0x" << std::hex << stats.worst_input << std::dec << " ("
        << ulpgrad::detail::bit_pattern_to_float(stats.worst_input) << "), measured "
        << stats.max_ulp << " ULP against a declared " << ulpgrad::rsqrt_max_ulp;
    // A test that measures nothing is a bug, not a pass.
    EXPECT_GT(stats.measured, 5000);
}

TEST(Rsqrt, GoldenReferenceUlpIsWithinTheDeclaredBoundOfEveryBucket) {
    std::string error;
    const std::vector<Point> points =
        read_golden(std::string(ULPGRAD_TEST_DATA_DIR) + "/reference/rsqrt.csv", &error);
    ASSERT_TRUE(error.empty()) << error;
    ASSERT_FALSE(points.empty());

    const Stats stats = measure(points, &ulpgrad::rsqrt);
    expect_bucket_counts_match(stats, ulpgrad::rsqrt_error_bounds);
    for (int bucket = 0; bucket < kBucketCount; ++bucket) {
        const ulpgrad::ErrorBound& bound = ulpgrad::rsqrt_error_bounds[bucket];
        EXPECT_GT(bound.points, 0) << "bucket " << bound.range << " measured nothing";
        EXPECT_LE(stats.bucket_max[bucket], static_cast<unsigned>(bound.max_ulp))
            << "bucket " << bound.range << " measured " << stats.bucket_max[bucket]
            << " ULP against a declared " << bound.max_ulp;
    }
}

TEST(Rsqrt, EveryExponentFieldStaysWithinTheBound) {
    // The reduction maps every positive input onto a mantissa in [1, 4) and an
    // exact power-of-two scale, so walking all 254 exponent fields with a
    // mantissa stride is the cheapest way to prove the bound does not depend on
    // the magnitude. The complete enumeration of all 2^24 mantissas lives in
    // experiments/m4_transcendental/generate_coefficients.py and is what the
    // documented bound rests on; this case is the cheap regression net.
    constexpr std::uint32_t kStride = 2048;  // 4096 mantissas per binade
    unsigned worst = 0;
    std::uint32_t worst_bits = 0;
    long long seen = 0;
    for (std::uint32_t field = 1u; field < 0xFFu; ++field) {
        for (std::uint32_t fraction = 0u; fraction < 0x00800000u; fraction += kStride) {
            const std::uint32_t bits = (field << 23) | fraction;
            const float x = ulpgrad::detail::bit_pattern_to_float(bits);
            // binary64 is exact for the argument and its reciprocal square root
            // is irrational for every non-square mantissa, so the single
            // rounding to float32 below cannot land on a tie.
            const float expected = static_cast<float>(1.0 / std::sqrt(static_cast<double>(x)));
            const unsigned distance = ulpgrad::detail::ulp_distance(ulpgrad::rsqrt(x), expected);
            ++seen;
            if (distance > worst) {
                worst = distance;
                worst_bits = bits;
            }
        }
    }
    EXPECT_GT(seen, 1000000);
    EXPECT_LE(worst, static_cast<unsigned>(ulpgrad::rsqrt_max_ulp))
        << "worst input 0x" << std::hex << worst_bits << std::dec << ", measured " << worst
        << " ULP against a declared " << ulpgrad::rsqrt_max_ulp;
}

TEST(Rsqrt, SubnormalArgumentsStayWithinTheBound) {
    // A subnormal magnitude has no exponent field to split, so it is
    // renormalised with integer shifts; this case is the one that would catch a
    // shift that is off by one. 200,000 deterministic patterns, both ends of the
    // range included.
    std::uint32_t state = 0x5EED1234u;
    unsigned worst = 0;
    std::uint32_t worst_bits = 0;
    for (int i = 0; i < 200000; ++i) {
        const std::uint32_t magnitude = (next_random(&state) & 0x007FFFFFu) | 1u;
        const float x = ulpgrad::detail::bit_pattern_to_float(magnitude);
        const float expected = static_cast<float>(1.0 / std::sqrt(static_cast<double>(x)));
        const unsigned distance = ulpgrad::detail::ulp_distance(ulpgrad::rsqrt(x), expected);
        if (distance > worst) {
            worst = distance;
            worst_bits = magnitude;
        }
    }
    EXPECT_LE(worst, static_cast<unsigned>(ulpgrad::rsqrt_max_ulp))
        << "worst input 0x" << std::hex << worst_bits << std::dec << ", measured " << worst
        << " ULP against a declared " << ulpgrad::rsqrt_max_ulp;
}
