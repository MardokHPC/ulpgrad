#include <ulpgrad/log.h>
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
constexpr int kBucketCount = ulpgrad::log_error_bounds_count;

static_assert(ulpgrad::log2_error_bounds_count == kBucketCount,
              "both logarithms must be bucketed the same way for one classifier to be valid");
static_assert(ulpgrad::log1p_error_bounds_count == kBucketCount,
              "log1p must be bucketed the same way too");

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
// checked on the payload, because `log(-1)` has no ULP distance to anything.
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

// One golden file, two assertions: the overall bound and the per-bucket bound.
void expect_golden_bounds(const char* name, float (*function)(float), int declared,
                          const ulpgrad::ErrorBound* bounds) {
    std::string error;
    const std::vector<Point> points =
        read_golden(std::string(ULPGRAD_TEST_DATA_DIR) + "/reference/" + name + ".csv", &error);
    ASSERT_TRUE(error.empty()) << error << " (the golden reference is committed: regenerate it "
                                          "with experiments/m3_fit/gen_golden.py)";
    ASSERT_FALSE(points.empty());

    const Stats stats = measure(points, function);
    SCOPED_TRACE(std::string("golden points: ") + std::to_string(points.size()));
    EXPECT_EQ(stats.nan_inputs, 3);
    EXPECT_EQ(stats.nan_mismatches, 0) << "a NaN result must match the reference pattern exactly";
    EXPECT_LE(stats.max_ulp, static_cast<unsigned>(declared))
        << "worst input 0x" << std::hex << stats.worst_input << std::dec << " ("
        << ulpgrad::detail::bit_pattern_to_float(stats.worst_input) << "), measured "
        << stats.max_ulp << " ULP against a declared " << declared;
    // A test that measures nothing is a bug, not a pass.
    EXPECT_GT(stats.measured, 5000);

    expect_bucket_counts_match(stats, bounds);
    for (int bucket = 0; bucket < kBucketCount; ++bucket) {
        EXPECT_LE(stats.bucket_max[bucket], static_cast<unsigned>(bounds[bucket].max_ulp))
            << "bucket " << bounds[bucket].range << " measured " << stats.bucket_max[bucket]
            << " ULP against a declared " << bounds[bucket].max_ulp;
    }
}

}  // namespace

// ----------------------------------------------------------------------------- //
// Log
// ----------------------------------------------------------------------------- //

TEST(Log, GoldenReferenceUlpIsWithinDeclaredBound) {
    expect_golden_bounds("log", &ulpgrad::log, ulpgrad::log_max_ulp, ulpgrad::log_error_bounds);
}

TEST(Log, NegativeArgumentsProduceTheCanonicalNan) {
    // The golden sample carries them as NaN outputs; this pins the payload the
    // header promises, which the ULP metric alone cannot check.
    std::string error;
    const std::vector<Point> points =
        read_golden(std::string(ULPGRAD_TEST_DATA_DIR) + "/reference/log.csv", &error);
    ASSERT_TRUE(error.empty()) << error;
    const Stats stats = measure(points, &ulpgrad::log);
    EXPECT_GT(stats.nan_outputs, 200) << "the sample lost its negative arguments";
}

// ----------------------------------------------------------------------------- //
// Log2
// ----------------------------------------------------------------------------- //

TEST(Log2, GoldenReferenceUlpIsWithinDeclaredBound) {
    expect_golden_bounds("log2", &ulpgrad::log2, ulpgrad::log2_max_ulp,
                         ulpgrad::log2_error_bounds);
}

TEST(Log2, PowersOfTwoAreExactlyRepresentable) {
    // The separate fit exists for this: `log2(2^k) == k` exactly, and `log(x) *
    // log2(e)` loses that on most of them.
    for (int exponent = -149; exponent <= 127; ++exponent) {
        const float x = std::ldexp(1.0f, exponent);
        const float expected = static_cast<float>(exponent);
        EXPECT_EQ(ulpgrad::detail::ulp_distance(ulpgrad::log2(x), expected), 0u)
            << "log2(2^" << exponent << ") = " << ulpgrad::log2(x);
    }
}

// ----------------------------------------------------------------------------- //
// Log1p
// ----------------------------------------------------------------------------- //

TEST(Log1p, GoldenReferenceUlpIsWithinDeclaredBound) {
    expect_golden_bounds("log1p", &ulpgrad::log1p, ulpgrad::log1p_max_ulp,
                         ulpgrad::log1p_error_bounds);
}

TEST(Log1p, SmallArgumentsAreReturnedExactly) {
    // The whole point of the function: log(1 + x) loses every bit of x below
    // 2^-24, and log1p must not. The guard is at 2^-25, so this walks every
    // binade from there down, plus the subnormals, where the answer is x itself.
    for (int exponent = -25; exponent >= -149; --exponent) {
        const float x = std::ldexp(1.0f, exponent);
        EXPECT_EQ(ulpgrad::detail::ulp_distance(ulpgrad::log1p(x), x), 0u)
            << "log1p(2^" << exponent << ") moved";
        EXPECT_EQ(ulpgrad::detail::ulp_distance(ulpgrad::log1p(-x), -x), 0u)
            << "log1p(-2^" << exponent << ") moved";
    }
}
