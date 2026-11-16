#include <ulpgrad/sigmoid.h>
#include <ulpgrad/silu.h>
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
// open-ended.
constexpr int kBucketCount = ulpgrad::sigmoid_error_bounds_count;

static_assert(ulpgrad::silu_error_bounds_count == kBucketCount,
              "the two functions share one bucket table");

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

bool parse_hex(const std::string& text, std::uint32_t* value) {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(text.c_str(), &end, 16);
    if (end == text.c_str() || *end != '\0' || parsed > 0xFFFFFFFFul) {
        return false;
    }
    *value = static_cast<std::uint32_t>(parsed);
    return true;
}

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

struct Stats {
    unsigned max_ulp = 0;
    std::uint32_t worst_input = 0;
    long long measured = 0;
    long long nan_inputs = 0;
    long long nan_mismatches = 0;
    unsigned bucket_max[kBucketCount] = {};
    long long bucket_points[kBucketCount] = {};
};

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

void expect_within_bounds(const char* name, const std::string& csv,
                          float (*function)(float), const ulpgrad::ErrorBound* bounds,
                          int declared_max_ulp) {
    std::string error;
    const std::vector<Point> points =
        read_golden(std::string(ULPGRAD_TEST_DATA_DIR) + "/reference/" + csv, &error);
    ASSERT_TRUE(error.empty()) << error << " (regenerate with experiments/m3_fit/gen_golden.py)";
    ASSERT_FALSE(points.empty());

    const Stats stats = measure(points, function);
    SCOPED_TRACE(name);
    EXPECT_EQ(stats.nan_inputs, 3);
    EXPECT_EQ(stats.nan_mismatches, 0) << "a NaN input must come back with its own payload";
    EXPECT_LE(stats.max_ulp, static_cast<unsigned>(declared_max_ulp))
        << "worst input 0x" << std::hex << stats.worst_input << std::dec << " ("
        << ulpgrad::detail::bit_pattern_to_float(stats.worst_input) << "), measured "
        << stats.max_ulp << " ULP against a declared " << declared_max_ulp;
    EXPECT_GT(stats.measured, 5000);

    expect_bucket_counts_match(stats, bounds);
    for (int bucket = 0; bucket < kBucketCount; ++bucket) {
        EXPECT_GT(bounds[bucket].points, 0) << "bucket " << bounds[bucket].range << " is empty";
        EXPECT_LE(stats.bucket_max[bucket], static_cast<unsigned>(bounds[bucket].max_ulp))
            << "bucket " << bounds[bucket].range << " measured " << stats.bucket_max[bucket]
            << " ULP against a declared " << bounds[bucket].max_ulp;
    }
}

}  // namespace

// ----------------------------------------------------------------------------- //
// Sigmoid
// ----------------------------------------------------------------------------- //

TEST(Sigmoid, GoldenReferenceUlpIsWithinTheDeclaredBound) {
    expect_within_bounds("sigmoid", "sigmoid.csv", &ulpgrad::sigmoid,
                         ulpgrad::sigmoid_error_bounds, ulpgrad::sigmoid_max_ulp);
}

TEST(Sigmoid, IsNonDecreasing) {
    // A monotone check needs no complement: `1 - sigmoid(x)` cancels catastrophically
    // for x above ~1, so it is not a reference for sigmoid(-x). Monotonicity of the
    // computed function is the property that actually has to hold.
    float previous = ulpgrad::sigmoid(-40.0f);
    for (float x = -40.0f; x <= 40.0f; x += 0.25f) {
        const float current = ulpgrad::sigmoid(x);
        EXPECT_GE(current, previous) << "x = " << x;
        previous = current;
    }
}

// ----------------------------------------------------------------------------- //
// Silu
// ----------------------------------------------------------------------------- //

TEST(Silu, GoldenReferenceUlpIsWithinTheDeclaredBound) {
    expect_within_bounds("silu", "silu.csv", &ulpgrad::silu, ulpgrad::silu_error_bounds,
                         ulpgrad::silu_max_ulp);
}

TEST(Silu, IsTheProductOfXAndSigmoid) {
    // Within the declared budget, silu(x) is x times sigmoid(x): the two share a
    // branch structure, and a wrong sign on one side would drift apart here.
    for (float x = -20.0f; x < 20.0f; x += 0.5f) {
        const float product = x * ulpgrad::sigmoid(x);
        const unsigned distance = ulpgrad::detail::ulp_distance(ulpgrad::silu(x), product);
        EXPECT_LE(distance, 2u * static_cast<unsigned>(ulpgrad::silu_max_ulp))
            << "x = " << x;
    }
}
