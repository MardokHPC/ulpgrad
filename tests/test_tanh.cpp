#include <ulpgrad/tanh.h>
#include <ulpgrad/tables.h>
#include <ulpgrad/detail/bits.h>

#include <gtest/gtest.h>

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
constexpr int kBucketCount = ulpgrad::tanh_error_bounds_count;

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

Stats measure(const std::vector<Point>& points, float (*function)(float)) {
    Stats stats;
    for (const Point& point : points) {
        const float x = ulpgrad::detail::bit_pattern_to_float(point.input);
        const float got = function(x);
        if (is_nan_bits(point.input)) {
            ++stats.nan_inputs;
            if (ulpgrad::detail::float_to_bit_pattern(got) != point.expected) {
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

}  // namespace

// ----------------------------------------------------------------------------- //
// Tanh
// ----------------------------------------------------------------------------- //

TEST(Tanh, GoldenReferenceUlpIsWithinDeclaredBound) {
    std::string error;
    const std::vector<Point> points =
        read_golden(std::string(ULPGRAD_TEST_DATA_DIR) + "/reference/tanh.csv", &error);
    ASSERT_TRUE(error.empty()) << error << " (the golden reference is committed: regenerate it "
                                          "with experiments/m3_fit/gen_golden.py)";
    ASSERT_FALSE(points.empty());

    const Stats stats = measure(points, &ulpgrad::tanh);
    SCOPED_TRACE("golden points: " + std::to_string(points.size()));
    EXPECT_EQ(stats.nan_inputs, 3);
    EXPECT_EQ(stats.nan_mismatches, 0) << "a NaN input must come back with its own payload";
    EXPECT_LE(stats.max_ulp, static_cast<unsigned>(ulpgrad::tanh_max_ulp))
        << "worst input 0x" << std::hex << stats.worst_input << std::dec << " ("
        << ulpgrad::detail::bit_pattern_to_float(stats.worst_input) << "), measured "
        << stats.max_ulp << " ULP against a declared " << ulpgrad::tanh_max_ulp;
    // A test that measures nothing is a bug, not a pass.
    EXPECT_GT(stats.measured, 5000);
}

TEST(Tanh, GoldenReferenceUlpIsWithinTheDeclaredBoundOfEveryBucket) {
    std::string error;
    const std::vector<Point> points =
        read_golden(std::string(ULPGRAD_TEST_DATA_DIR) + "/reference/tanh.csv", &error);
    ASSERT_TRUE(error.empty()) << error;
    ASSERT_FALSE(points.empty());

    const Stats stats = measure(points, &ulpgrad::tanh);
    expect_bucket_counts_match(stats, ulpgrad::tanh_error_bounds);
    for (int bucket = 0; bucket < kBucketCount; ++bucket) {
        const ulpgrad::ErrorBound& bound = ulpgrad::tanh_error_bounds[bucket];
        EXPECT_GT(bound.points, 0) << "bucket " << bound.range << " measured nothing";
        EXPECT_LE(stats.bucket_max[bucket], static_cast<unsigned>(bound.max_ulp))
            << "bucket " << bound.range << " measured " << stats.bucket_max[bucket]
            << " ULP against a declared " << bound.max_ulp;
    }
}

TEST(Tanh, BothSignsAgreeToWithinTheBound) {
    // tanh is odd as a real function, and this implementation is not odd to the
    // bit: the upper branch computes exp2(-w) in one direction and exp2(w) in the
    // other, and a reciprocal is not the same float as the negative of its
    // partner's result. That is fine for a correctly rounded function -- the
    // correctly rounded tanh is not odd to the bit either, since tanh(-x) and
    // -tanh(x) round independently -- but the two sides must not drift apart
    // beyond the declared bound.
    for (std::uint32_t bits = 0x00000001u; bits < 0x41100000u; bits = bits * 2u + 1u) {
        const float x = ulpgrad::detail::bit_pattern_to_float(bits);
        const float positive = ulpgrad::tanh(x);
        const float negative = ulpgrad::tanh(-x);
        const unsigned distance = ulpgrad::detail::ulp_distance(negative, -positive);
        EXPECT_LE(distance, 2u * static_cast<unsigned>(ulpgrad::tanh_max_ulp))
            << "x = " << x << ", tanh(x) = " << positive << ", tanh(-x) = " << negative;
    }
}
