// Measures the ULP error of the library functions against the committed golden
// reference and writes data/error-bounds.json.
//
// This is the only producer of that file. The declared bounds in
// include/ulpgrad/tables.h and the prose table in docs/error-tables.md are both
// generated from its output, so a number in the docs cannot disagree with a
// number the code can read: either they come from the same measurement, or
// tests/test_table_freshness.cpp fails.
//
// The buckets are powers of two of |x|, the same split experiments/m0_gpu used,
// because that is the split that shows whether the error grows with the
// argument -- the property the library claims. A bucket with no points is
// reported with max_ulp 0 and points 0 rather than omitted: a hole in the
// sample is a number too.
//
// Usage:
//   ulpgrad_measure_bounds [--data-dir data] [--out data/error-bounds.json]
//
// Exit code is 0 only when every reference file was read, every NaN input came
// back bit-identical, and every NaN *result* matched the pattern the reference
// generator promised.

#include "ulpgrad/exp.h"
#include "ulpgrad/log.h"
#include "ulpgrad/rsqrt.h"
#include "ulpgrad/sigmoid.h"
#include "ulpgrad/silu.h"
#include "ulpgrad/tanh.h"
#include "ulpgrad/detail/bits.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr int kBucketCount = 8;

// The functions under measurement, in the order they appear in the JSON.
//
// One entry per shipped function: the name is the CSV stem and the key in the
// generated table, so adding a function to the library means adding a line here
// and nothing else. Capture-less lambdas convert to the function pointer.
constexpr struct {
    const char* name;
    float (*evaluate)(float);
} kFunctions[] = {
    {"exp", [](float x) { return ulpgrad::exp(x); }},
    {"exp2", [](float x) { return ulpgrad::exp2(x); }},
    {"log", [](float x) { return ulpgrad::log(x); }},
    {"log1p", [](float x) { return ulpgrad::log1p(x); }},
    {"log2", [](float x) { return ulpgrad::log2(x); }},
    {"rsqrt", [](float x) { return ulpgrad::rsqrt(x); }},
    {"sigmoid", [](float x) { return ulpgrad::sigmoid(x); }},
    {"silu", [](float x) { return ulpgrad::silu(x); }},
    {"tanh", [](float x) { return ulpgrad::tanh(x); }},
};

constexpr int kFunctionCount = static_cast<int>(sizeof kFunctions / sizeof kFunctions[0]);

// Upper bounds of the |x| buckets; the last bucket is open. Matches the split
// in docs/error-tables.md and in the generated tables.h.
constexpr struct {
    const char* range;
    float upper;  // exclusive; the last entry is unused
} kBucketTable[kBucketCount] = {
    {"[0, 1)", 1.0f},
    {"[1, 2)", 2.0f},
    {"[2, 4)", 4.0f},
    {"[4, 8)", 8.0f},
    {"[8, 16)", 16.0f},
    {"[16, 32)", 32.0f},
    {"[32, 64)", 64.0f},
    {"[64, inf)", 0.0f},  // open: everything from 64 up
};

struct Bucket {
    unsigned max_ulp = 0;
    long long points = 0;
    long long exact = 0;
    std::uint32_t worst_input = 0;
};

struct Measurement {
    std::string name;
    std::string reference;
    std::string seed;
    long long points = 0;
    long long measured = 0;      // points that carry a ULP distance (NaN excluded)
    long long nan_inputs = 0;    // NaN arguments, checked on the payload
    long long nan_outputs = 0;   // NaN results for a real argument, e.g. log(-1)
    long long nan_mismatches = 0;
    unsigned max_ulp = 0;
    long long exact = 0;
    std::uint32_t worst_input = 0;
    Bucket buckets[kBucketCount];
};

int bucket_of(float x) {
    const std::uint32_t magnitude = ulpgrad::detail::float_to_bit_pattern(x) & 0x7FFFFFFFu;
    const std::uint32_t field = magnitude >> 23;
    if (field == 0xFFu) {
        return kBucketCount - 1;  // infinity: the last bucket is open-ended
    }
    const int exponent = static_cast<int>(field) - 127;
    if (exponent < 0) {
        return 0;  // subnormal magnitudes and [2^-126, 1) share the first bucket
    }
    const int index = exponent + 1;
    return index < kBucketCount ? index : kBucketCount - 1;
}

bool is_nan_bits(std::uint32_t bits) {
    return (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u;
}

// One parsed reference file: the header fields plus (input, expected) pairs.
struct ReferenceFile {
    Measurement meta;
    std::vector<std::uint32_t> inputs;
    std::vector<std::uint32_t> expected;
};

bool parse_hex(const char* begin, const char* end, std::uint32_t* out) {
    if (begin == end) {
        return false;
    }
    std::uint32_t value = 0;
    for (const char* p = begin; p != end; ++p) {
        const char c = *p;
        unsigned digit = 0;
        if (c >= '0' && c <= '9') {
            digit = static_cast<unsigned>(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            digit = static_cast<unsigned>(c - 'a') + 10u;
        } else if (c >= 'A' && c <= 'F') {
            digit = static_cast<unsigned>(c - 'A') + 10u;
        } else {
            return false;
        }
        value = (value << 4) | digit;
    }
    *out = value;
    return true;
}

bool read_reference(const std::string& path, ReferenceFile* file) {
    std::ifstream stream(path);
    if (!stream) {
        std::fprintf(stderr, "error: cannot open %s\n", path.c_str());
        return false;
    }

    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            const std::string body = line.substr(1);
            const std::size_t colon = body.find(':');
            if (colon == std::string::npos) {
                continue;
            }
            std::string key = body.substr(0, colon);
            std::string value = body.substr(colon + 1);
            while (!key.empty() && key.front() == ' ') {
                key.erase(key.begin());
            }
            while (!key.empty() && key.back() == ' ') {
                key.pop_back();
            }
            while (!value.empty() && value.front() == ' ') {
                value.erase(value.begin());
            }
            if (key == "function") {
                file->meta.name = value;
            } else if (key == "reference") {
                file->meta.reference = value;
            } else if (key == "seed") {
                file->meta.seed = value;
            }
            continue;
        }
        const std::size_t comma = line.find(',');
        if (comma == std::string::npos) {
            std::fprintf(stderr, "error: %s: malformed record '%s'\n", path.c_str(), line.c_str());
            return false;
        }
        std::uint32_t in = 0;
        std::uint32_t out = 0;
        if (!parse_hex(line.data(), line.data() + comma, &in) ||
            !parse_hex(line.data() + comma + 1, line.data() + line.size(), &out)) {
            std::fprintf(stderr, "error: %s: non-hex record '%s'\n", path.c_str(), line.c_str());
            return false;
        }
        file->inputs.push_back(in);
        file->expected.push_back(out);
    }

    if (file->meta.name.empty() || file->inputs.empty()) {
        std::fprintf(stderr, "error: %s: no function name or no records\n", path.c_str());
        return false;
    }
    return true;
}

// ULP statistics of one function, computed from its golden file.
template <typename Function>
void measure(const ReferenceFile& file, Function function, Measurement* out) {
    *out = file.meta;
    out->points = static_cast<long long>(file.inputs.size());

    for (std::size_t i = 0; i < file.inputs.size(); ++i) {
        const std::uint32_t in_bits = file.inputs[i];
        const std::uint32_t expected_bits = file.expected[i];
        const float x = ulpgrad::detail::bit_pattern_to_float(in_bits);
        const float got = function(x);
        const std::uint32_t got_bits = ulpgrad::detail::float_to_bit_pattern(got);

        // A NaN pattern has no position on the real line, so the ULP metric
        // cannot say anything about it: both ends of such a comparison are
        // checked bit for bit instead. That covers a NaN argument (whose payload
        // the headers promise to return unchanged) and a NaN *result* for a real
        // argument, which is a signed-zero or a domain contract in disguise.
        if (is_nan_bits(in_bits) || is_nan_bits(expected_bits)) {
            if (is_nan_bits(in_bits)) {
                ++out->nan_inputs;
            } else {
                ++out->nan_outputs;
            }
            if (got_bits != expected_bits) {
                ++out->nan_mismatches;
            }
            continue;
        }

        const float expected = ulpgrad::detail::bit_pattern_to_float(expected_bits);
        const unsigned distance = ulpgrad::detail::ulp_distance(got, expected);
        ++out->measured;
        if (distance == 0u) {
            ++out->exact;
        }
        if (distance > out->max_ulp) {
            out->max_ulp = distance;
            out->worst_input = in_bits;
        }

        Bucket& bucket = out->buckets[bucket_of(x)];
        ++bucket.points;
        if (distance == 0u) {
            ++bucket.exact;
        }
        if (distance > bucket.max_ulp) {
            bucket.max_ulp = distance;
            bucket.worst_input = in_bits;
        }
    }
}

double percentage(long long part, long long whole) {
    if (whole == 0) {
        return 0.0;
    }
    return 100.0 * static_cast<double>(part) / static_cast<double>(whole);
}

const char* compiler_name() {
#if defined(_MSC_VER)
    return "MSVC";
#elif defined(__clang__)
    return "Clang";
#elif defined(__GNUC__)
    return "GCC";
#else
    return "unknown";
#endif
}

int compiler_version() {
#if defined(_MSC_VER)
    return static_cast<int>(_MSC_FULL_VER % 100000);
#elif defined(__clang__)
    return __clang_major__ * 100 + __clang_minor__;
#elif defined(__GNUC__)
    return __GNUC__ * 100 + __GNUC_MINOR__;
#else
    return 0;
#endif
}

const char* os_name() {
#if defined(_WIN32)
    return "Windows";
#elif defined(__linux__)
    return "Linux";
#elif defined(__APPLE__)
    return "macOS";
#else
    return "unknown";
#endif
}

const char* architecture() {
#if defined(_M_X64) || defined(__x86_64__)
    return "x64";
#elif defined(_M_ARM64) || defined(__aarch64__)
    return "arm64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#else
    return "unknown";
#endif
}

void emit_json(FILE* out, const std::vector<Measurement>& measurements) {
    const char* cpu = std::getenv("PROCESSOR_IDENTIFIER");
    std::fprintf(out, "{\n");
    std::fprintf(out, "  \"schema\": 1,\n");
    std::fprintf(out, "  \"generated_by\": \"tools/measure_bounds.cpp\",\n");
    std::fprintf(out, "  \"date\": \"%s\",\n", __DATE__);
    std::fprintf(out, "  \"machine\": {\n");
    std::fprintf(out, "    \"os\": \"%s\",\n", os_name());
    std::fprintf(out, "    \"arch\": \"%s\",\n", architecture());
    std::fprintf(out, "    \"cpu\": \"%s\",\n", (cpu != nullptr) ? cpu : "unknown");
    std::fprintf(out, "    \"compiler\": \"%s %d\"\n", compiler_name(), compiler_version());
    std::fprintf(out, "  },\n");
    std::fprintf(out, "  \"functions\": [\n");
    for (std::size_t f = 0; f < measurements.size(); ++f) {
        const Measurement& m = measurements[f];
        std::fprintf(out, "    {\n");
        std::fprintf(out, "      \"name\": \"%s\",\n", m.name.c_str());
        std::fprintf(out, "      \"reference\": \"%s\",\n", m.reference.c_str());
        std::fprintf(out, "      \"seed\": \"%s\",\n", m.seed.c_str());
        std::fprintf(out, "      \"points\": %lld,\n", m.points);
        std::fprintf(out, "      \"measured_points\": %lld,\n", m.measured);
        std::fprintf(out, "      \"nan_inputs\": %lld,\n", m.nan_inputs);
        std::fprintf(out, "      \"nan_outputs\": %lld,\n", m.nan_outputs);
        std::fprintf(out, "      \"nan_mismatches\": %lld,\n", m.nan_mismatches);
        std::fprintf(out, "      \"max_ulp\": %u,\n", m.max_ulp);
        std::fprintf(out, "      \"worst_input\": \"0x%08X\",\n", m.worst_input);
        std::fprintf(out, "      \"exactly_rounded_pct\": %.2f,\n",
                     percentage(m.exact, m.measured));
        std::fprintf(out, "      \"buckets\": [\n");
        for (int b = 0; b < kBucketCount; ++b) {
            const Bucket& bucket = m.buckets[b];
            std::fprintf(out,
                         "        {\"range\": \"%s\", \"max_ulp\": %u, \"points\": %lld, "
                         "\"exactly_rounded_pct\": %.2f, \"worst_input\": \"0x%08X\"}%s\n",
                         kBucketTable[b].range, bucket.max_ulp, bucket.points,
                         percentage(bucket.exact, bucket.points), bucket.worst_input,
                         (b + 1 < kBucketCount) ? "," : "");
        }
        std::fprintf(out, "      ]\n");
        std::fprintf(out, "    }%s\n", (f + 1 < measurements.size()) ? "," : "");
    }
    std::fprintf(out, "  ]\n");
    std::fprintf(out, "}\n");
}

void print_summary(const Measurement& m) {
    std::printf("%-6s %6lld points (%lld measured, %lld NaN in, %lld NaN out)  max %u ulp  "
                "exactly rounded %.2f%%  worst 0x%08X\n",
                m.name.c_str(), m.points, m.measured, m.nan_inputs, m.nan_outputs, m.max_ulp,
                percentage(m.exact, m.measured), m.worst_input);
    for (int b = 0; b < kBucketCount; ++b) {
        const Bucket& bucket = m.buckets[b];
        std::printf("        |x| %-9s %6lld points  max %u ulp  exactly rounded %.2f%%\n",
                    kBucketTable[b].range, bucket.points, bucket.max_ulp,
                    percentage(bucket.exact, bucket.points));
    }
    if (m.nan_mismatches != 0) {
        std::printf("        %lld NaN results did not match the reference pattern\n",
                    m.nan_mismatches);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string data_dir = "data";
    std::string out_path = "data/error-bounds.json";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else {
            std::fprintf(stderr, "usage: %s [--data-dir DIR] [--out FILE]\n", argv[0]);
            return 2;
        }
    }

    std::vector<ReferenceFile> files(kFunctionCount);
    for (int i = 0; i < kFunctionCount; ++i) {
        const std::string path = data_dir + "/reference/" + kFunctions[i].name + ".csv";
        if (!read_reference(path, &files[i])) {
            return 1;
        }
        if (files[i].meta.name != kFunctions[i].name) {
            std::fprintf(stderr, "error: %s names %s, expected %s\n", path.c_str(),
                         files[i].meta.name.c_str(), kFunctions[i].name);
            return 1;
        }
    }

    std::vector<Measurement> measurements(kFunctionCount);
    for (int i = 0; i < kFunctionCount; ++i) {
        measure(files[i], kFunctions[i].evaluate, &measurements[i]);
    }

    for (const Measurement& m : measurements) {
        print_summary(m);
    }

    FILE* out = std::fopen(out_path.c_str(), "wb");
    if (out == nullptr) {
        std::fprintf(stderr, "error: cannot write %s\n", out_path.c_str());
        return 1;
    }
    emit_json(out, measurements);
    std::fclose(out);
    std::printf("wrote %s\n", out_path.c_str());

    long long mismatches = 0;
    for (const Measurement& m : measurements) {
        mismatches += m.nan_mismatches;
    }
    if (mismatches != 0) {
        std::fprintf(stderr, "error: %lld NaN results did not round-trip\n", mismatches);
        return 1;
    }
    return 0;
}
