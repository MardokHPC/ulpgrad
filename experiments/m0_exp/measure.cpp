// M0 spike: measured ULP error of a float-only exp(x) against a correctly
// rounded reference.
//
// This file is evidence, not library code: it exists to put one real number on
// the table before a single line of ulpgrad is designed. Everything the plan
// claims about "Accurate <= 1.0 ulp" is settled here or nowhere.
//
// Layout: reduction -> degree-6 minimax polynomial -> exact 2^k reconstruction,
// with an explicit fmaf chain (never a*b+c).

#include "coeffs.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

// --------------------------------------------------------------------------- //
// helpers
// --------------------------------------------------------------------------- //

constexpr float k_inf = std::numeric_limits<float>::infinity();

// Polynomial degree of the generated coefficients. Only the degree-6 block is
// the canonical implementation; 5 and 7 exist to answer "does the degree
// matter?" with a measurement instead of an opinion.
#ifndef M0_DEGREE
#define M0_DEGREE 6
#endif
#if (M0_DEGREE != 5) && (M0_DEGREE != 6) && (M0_DEGREE != 7)
#error "M0_DEGREE must be 5, 6 or 7"
#endif

uint32_t bits_of(float x) {
    uint32_t b = 0;
    std::memcpy(&b, &x, sizeof b);
    return b;
}

float float_of(uint32_t b) {
    float x = 0.0f;
    std::memcpy(&x, &b, sizeof x);
    return x;
}

bool is_nan_bits(uint32_t b) {
    return (b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu) != 0u;
}

bool is_inf_bits(uint32_t b) {
    return (b & 0x7FFFFFFFu) == 0x7F800000u;
}

// 2^e for e in [-126, 127].
float power_of_two(int e) {
    return float_of(static_cast<uint32_t>(e + 127) << 23);
}

// --------------------------------------------------------------------------- //
// the function under measurement
// --------------------------------------------------------------------------- //

float m0_exp(float x) noexcept {
    if (std::isnan(x)) {
        return x;                       // payload and sign preserved, bit for bit
    }
    if (x >= overflow_threshold) {
        return k_inf;
    }
    if (x <= underflow_threshold) {
        return 0.0f;                    // exp never returns -0
    }

    // Cody-Waite reduction in two parts; fma keeps the product exact.
    const float kf = std::rint(x * inv_ln2);
    const int k = static_cast<int>(kf);
    float r = std::fma(-kf, ln2_hi, x);
    r = std::fma(-kf, ln2_lo, r);

    // Q(r) = (exp(r) - 1) / r, Horner as an explicit fma chain.
#if M0_DEGREE == 7
    float p = c7;
    p = std::fma(p, r, c6);
    p = std::fma(p, r, c5);
#elif M0_DEGREE == 6
    float p = c6;
    p = std::fma(p, r, c5);
#else
    float p = c5;
#endif
    p = std::fma(p, r, c4);
    p = std::fma(p, r, c3);
    p = std::fma(p, r, c2);
    p = std::fma(p, r, c1);
    p = std::fma(p, r, c0);
    const float y = std::fma(p, r, 1.0f);

    // Scale by 2^k through the exponent field. Exact while the result is
    // normal; the subnormal tail gets one exact scale and one rounded scale.
    // e == 255 is unreachable (k = 128 would need x >= 128*ln2, and the smallest
    // float32 above that already rounds to +inf), but a requirement that is only
    // unreachable by accident should be written down.
    const uint32_t bits = bits_of(y);
    const int32_t e = static_cast<int32_t>((bits >> 23) & 0xFFu) + k;
    if (e >= 0xFF) {
        return k_inf;
    }
    if (e > 0) {
        return float_of((bits & 0x807FFFFFu) | (static_cast<uint32_t>(e) << 23));
    }
    return (y * power_of_two(k + 64)) * power_of_two(-64);
}

// --------------------------------------------------------------------------- //
// ULP distance on the monotone integer image of the float pattern
// --------------------------------------------------------------------------- //

int64_t monotone_key(uint32_t bits) {
    const int32_t s = static_cast<int32_t>(bits);
    if (s >= 0) {
        return static_cast<int64_t>(s);
    }
    return -static_cast<int64_t>(s & 0x7FFFFFFF);
}

uint32_t ulp_distance(uint32_t a, uint32_t b) {
    const int64_t d = monotone_key(a) - monotone_key(b);
    const int64_t m = (d < 0) ? -d : d;
    return static_cast<uint32_t>(m);
}

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL  %s\n", what);
        ++failures;
    }
}

// The mapping is the foundation of every number in this file: if it is wrong,
// the measurement is wrong. Pinned on cases with hand-computed answers.
void verify_ulp_mapping() {
    const float one = 1.0f;
    const float neg_one = -1.0f;
    const float two = 2.0f;

    check(ulp_distance(bits_of(one), bits_of(std::nextafter(one, two))) == 1u,
          "+1.0 and its successor are 1 ulp apart");
    check(ulp_distance(bits_of(neg_one), bits_of(std::nextafter(neg_one, -two))) == 1u,
          "-1.0 and its successor toward -inf are 1 ulp apart");
    check(ulp_distance(bits_of(2.0f), bits_of(1.0f)) == 8388608u,
          "1.0 and 2.0 are 2^23 ulp apart (one binade boundary)");
    check(ulp_distance(bits_of(0.0f), bits_of(-0.0f)) == 0u, "+0 and -0 are 0 ulp apart");
    check(ulp_distance(bits_of(one), bits_of(neg_one)) == 2130706432u,
          "+1.0 and -1.0 are 2*(2^30 - 2^23) ulp apart");
    check(ulp_distance(bits_of(k_inf), bits_of(-k_inf)) == 2u * 2139095040u,
          "+inf and -inf are 2*(2^31 - 2^23 - 1) ulp apart");
    check(ulp_distance(bits_of(1.0f), bits_of(-1.0f)) == 2u * ulp_distance(bits_of(1.0f), 0u),
          "the negative half mirrors the positive half");
    float tiny = float_of(1u);
    check(ulp_distance(bits_of(tiny), 0u) == 1u, "smallest subnormal is 1 ulp from zero");
    check(ulp_distance(bits_of(1.0f), bits_of(1.0f + 2.0f * 1.19209290e-7f)) == 2u,
          "two steps up from 1.0 is 2 ulp");
}

// --------------------------------------------------------------------------- //
// sample plumbing
// --------------------------------------------------------------------------- //

struct Record {
    uint32_t x_bits;
    uint32_t ref_bits;
};
static_assert(sizeof(Record) == 8, "ref.bin records are 8 bytes: uint32 x_bits, uint32 ref_bits");

struct Category {
    std::string name;
    uint32_t first = 0;
    uint32_t count = 0;
    uint32_t max_ulp = 0;
    uint32_t arg = 0;
    uint32_t p99 = 0;
};

int exponent_bucket(uint32_t x_bits) {
    const uint32_t magnitude = x_bits & 0x7FFFFFFFu;
    if (magnitude == 0u) {
        return 0;                                  // +-0 shares the subnormal bucket
    }
    return static_cast<int>((magnitude >> 23) & 0xFFu);
}

uint32_t percentile(const std::vector<uint32_t>& sorted, double p) {
    if (sorted.empty()) {
        return 0;
    }
    const double n = static_cast<double>(sorted.size());
    double idx = std::ceil(p * n) - 1.0;
    if (idx < 0.0) {
        idx = 0.0;
    }
    if (idx > n - 1.0) {
        idx = n - 1.0;
    }
    return sorted[static_cast<size_t>(idx)];
}

// --------------------------------------------------------------------------- //
// timing
// --------------------------------------------------------------------------- //

template <typename Fn>
struct Timing {
    double ns_per_call = 0.0;
    double checksum = 0.0;      // proves the loop body was not optimised away
};

template <typename Fn>
Timing<Fn> measure_ns_per_call(Fn fn, const std::vector<float>& xs, int repeats) {
    using clock = std::chrono::steady_clock;
    Timing<Fn> result;
    result.ns_per_call = 1e30;
    double checksum = 0.0;
    for (int rep = 0; rep < repeats; ++rep) {
        const auto t0 = clock::now();
        float acc = 0.0f;
        for (const float x : xs) {
            acc += fn(x);
        }
        const auto t1 = clock::now();
        const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        const double per_call = ns / static_cast<double>(xs.size());
        if (per_call < result.ns_per_call) {
            result.ns_per_call = per_call;
        }
        checksum += static_cast<double>(acc);
    }
    result.checksum = checksum;
    return result;
}

}  // namespace

// --------------------------------------------------------------------------- //
// main
// --------------------------------------------------------------------------- //

int main(int argc, char** argv) {
    const char* ref_path = (argc > 1) ? argv[1] : "ref.bin";
    const char* manifest_path = (argc > 2) ? argv[2] : "manifest.txt";
    const char* report_path = (argc > 3) ? argv[3] : "report.json";

    std::printf("ulp mapping self-check\n");
    verify_ulp_mapping();
    if (failures != 0) {
        std::printf("%d assertion(s) failed, refusing to report numbers\n", failures);
        return 1;
    }
    std::printf("  ok  monotone mapping and ulp_distance\n");

    std::ifstream ref_in(ref_path, std::ios::binary);
    if (!ref_in) {
        std::printf("cannot open %s\n", ref_path);
        return 1;
    }
    ref_in.seekg(0, std::ios::end);
    const std::streamoff byte_count = ref_in.tellg();
    ref_in.seekg(0, std::ios::beg);
    if (byte_count <= 0 || (byte_count % 8) != 0) {
        std::printf("%s is not a multiple of 8 bytes\n", ref_path);
        return 1;
    }
    std::vector<Record> records(static_cast<size_t>(byte_count) / sizeof(Record));
    ref_in.read(reinterpret_cast<char*>(records.data()), static_cast<std::streamsize>(byte_count));
    if (!ref_in) {
        std::printf("short read on %s\n", ref_path);
        return 1;
    }
    const uint32_t n = static_cast<uint32_t>(records.size());
    std::printf("sample: %u points from %s\n", n, ref_path);

    // ---- categories -------------------------------------------------------
    std::vector<Category> categories;
    {
        std::ifstream man_in(manifest_path);
        if (!man_in) {
            std::printf("cannot open %s\n", manifest_path);
            return 1;
        }
        std::string name;
        unsigned long long first = 0;
        unsigned long long count = 0;
        while (man_in >> name >> first >> count) {
            Category c;
            c.name = name;
            c.first = static_cast<uint32_t>(first);
            c.count = static_cast<uint32_t>(count);
            categories.push_back(c);
        }
    }

    // ---- measure ----------------------------------------------------------
    struct Bucket {
        uint64_t count = 0;
        uint64_t nonzero = 0;
        uint32_t max_ulp = 0;
        uint32_t arg = 0;
    };
    std::vector<Bucket> buckets(256);

    std::vector<uint32_t> distances;
    distances.reserve(n);

    std::vector<float> xs;
    xs.reserve(n);

    // FNV-1a over every result bit pattern: two builds that print the same hash
    // produced bit-identical output, whatever the FP flags did.
    uint64_t out_hash = 1469598103934665603ull;
    const uint64_t fnv_prime = 1099511628211ull;

    uint32_t max_ulp = 0;
    uint32_t arg = 0;
    uint64_t count_over_one = 0;
    uint64_t count_over_zero = 0;
    uint64_t count_zero = 0;
    uint64_t nan_inputs = 0;
    uint64_t nan_mismatch = 0;
    uint64_t special_mismatch = 0;
    uint64_t ref_inf = 0;
    uint64_t ref_inf_mismatch = 0;

    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t xb = records[i].x_bits;
        const uint32_t rb = records[i].ref_bits;
        const float x = float_of(xb);
        xs.push_back(x);

        const uint32_t got = bits_of(m0_exp(x));
        out_hash = (out_hash ^ static_cast<uint64_t>(got)) * fnv_prime;

        if (is_nan_bits(xb)) {
            ++nan_inputs;
            if (got != rb) {
                ++nan_mismatch;
            }
            continue;
        }
        if (is_nan_bits(rb)) {
            ++special_mismatch;                   // reference NaN for a non-NaN input
            continue;
        }

        if (is_inf_bits(rb)) {
            ++ref_inf;
            if (got != rb) {
                ++ref_inf_mismatch;
            }
        }
        if (is_nan_bits(got)) {
            ++special_mismatch;
            continue;
        }

        const uint32_t d = ulp_distance(got, rb);
        distances.push_back(d);
        if (d == 0u) {
            ++count_zero;
        } else {
            ++count_over_zero;
        }
        if (d > 1u) {
            ++count_over_one;
        }
        if (d > max_ulp) {
            max_ulp = d;
            arg = i;
        }

        Bucket& b = buckets[static_cast<size_t>(exponent_bucket(xb))];
        ++b.count;
        if (d > 0u) {
            ++b.nonzero;
        }
        if (d > b.max_ulp) {
            b.max_ulp = d;
            b.arg = i;
        }
    }

    // ---- per-category aggregates -----------------------------------------
    for (Category& c : categories) {
        std::vector<uint32_t> local;
        local.reserve(c.count);
        for (uint32_t i = c.first; i < c.first + c.count && i < n; ++i) {
            const uint32_t xb = records[i].x_bits;
            if (is_nan_bits(xb)) {
                continue;
            }
            const uint32_t d = ulp_distance(bits_of(m0_exp(float_of(xb))), records[i].ref_bits);
            local.push_back(d);
            if (d > c.max_ulp) {
                c.max_ulp = d;
                c.arg = i;
            }
        }
        std::sort(local.begin(), local.end());
        c.p99 = percentile(local, 0.99);
    }

    std::vector<uint32_t> sorted = distances;
    std::sort(sorted.begin(), sorted.end());
    const uint32_t p50 = percentile(sorted, 0.50);
    const uint32_t p99 = percentile(sorted, 0.99);
    const uint32_t p999 = percentile(sorted, 0.999);

    // ---- the control: MSVC expf on the same sample ------------------------
    uint32_t expf_max = 0;
    uint32_t expf_arg = 0;
    uint64_t expf_over_one = 0;
    std::vector<uint32_t> expf_distances;
    expf_distances.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t xb = records[i].x_bits;
        const uint32_t rb = records[i].ref_bits;
        if (is_nan_bits(xb) || is_nan_bits(rb)) {
            continue;
        }
        const uint32_t got = bits_of(std::expf(float_of(xb)));
        if (is_nan_bits(got)) {
            continue;
        }
        const uint32_t d = ulp_distance(got, rb);
        expf_distances.push_back(d);
        if (d > expf_max) {
            expf_max = d;
            expf_arg = i;
        }
        if (d > 1u) {
            ++expf_over_one;
        }
    }
    std::sort(expf_distances.begin(), expf_distances.end());
    const uint32_t expf_p99 = percentile(expf_distances, 0.99);
    const uint32_t expf_p999 = percentile(expf_distances, 0.999);

    // ---- where the maximum is --------------------------------------------
    const uint32_t arg_x_bits = records[arg].x_bits;
    const float arg_x = float_of(arg_x_bits);
    const float arg_kf = std::rint(arg_x * inv_ln2);
    const int arg_k = static_cast<int>(arg_kf);
    float arg_r = std::fma(-arg_kf, ln2_hi, arg_x);
    arg_r = std::fma(-arg_kf, ln2_lo, arg_r);

    // ---- timing -----------------------------------------------------------
    const auto time_ours = measure_ns_per_call([](float x) { return m0_exp(x); }, xs, 5);
    const auto time_expf = measure_ns_per_call([](float x) { return std::expf(x); }, xs, 5);
    const double ns_ours = time_ours.ns_per_call;
    const double ns_expf = time_expf.ns_per_call;

    // ---- stdout -----------------------------------------------------------
    const double measured = static_cast<double>(distances.size());
    std::printf("\n");
    std::printf("points measured (NaN excluded) : %u of %u\n",
                static_cast<unsigned>(distances.size()), n);
    std::printf("max ulp                        : %u\n", max_ulp);
    std::printf("  at x                         : %.9g (0x%08X)\n", static_cast<double>(arg_x),
                arg_x_bits);
    std::printf("  k = %d, r = %.9g\n", arg_k, static_cast<double>(arg_r));
    std::printf("p50 / p99 / p999 ulp           : %u / %u / %u\n", p50, p99, p999);
    std::printf("points with ulp > 1            : %llu\n",
                static_cast<unsigned long long>(count_over_one));
    std::printf("points with ulp > 0            : %llu (%.2f%%)\n",
                static_cast<unsigned long long>(count_over_zero),
                100.0 * static_cast<double>(count_over_zero) / measured);
    std::printf("points exactly rounded         : %llu (%.2f%%)\n",
                static_cast<unsigned long long>(count_zero),
                100.0 * static_cast<double>(count_zero) / measured);
    std::printf("output bit hash (FNV-1a)       : 0x%016llX\n",
                static_cast<unsigned long long>(out_hash));

    std::printf("\nspecials\n");
    std::printf("  NaN inputs                   : %llu, bit-exact failures %llu\n",
                static_cast<unsigned long long>(nan_inputs),
                static_cast<unsigned long long>(nan_mismatch));
    std::printf("  reference +-inf              : %llu, mismatches %llu\n",
                static_cast<unsigned long long>(ref_inf),
                static_cast<unsigned long long>(ref_inf_mismatch));
    std::printf("  other mismatches             : %llu\n",
                static_cast<unsigned long long>(special_mismatch));

    std::printf("\ncontrol: MSVC expf on the same sample\n");
    std::printf("  max ulp %u at x = %.9g (0x%08X), p99 %u, p999 %u, > 1 ulp: %llu\n",
                expf_max, static_cast<double>(float_of(records[expf_arg].x_bits)),
                records[expf_arg].x_bits, expf_p99, expf_p999,
                static_cast<unsigned long long>(expf_over_one));

    std::printf("\ntiming (best of 5 passes over the whole sample)\n");
    std::printf("  m0_exp : %.2f ns/call\n", ns_ours);
    std::printf("  expf   : %.2f ns/call\n", ns_expf);
    std::printf("  accumulator checksum %.6g and %.6g (kept so the loops cannot be"
                " optimised away)\n",
                static_cast<double>(time_ours.checksum), static_cast<double>(time_expf.checksum));

    std::printf("\nper category\n");
    std::printf("  %-12s %10s %8s %8s %s\n", "category", "count", "max ulp", "p99", "worst x");
    for (const Category& c : categories) {
        char worst[32];
        if (c.max_ulp == 0u || c.arg >= n) {
            std::snprintf(worst, sizeof worst, "-");
        } else {
            std::snprintf(worst, sizeof worst, "%.9g", static_cast<double>(
                                                          float_of(records[c.arg].x_bits)));
        }
        std::printf("  %-12s %10u %8u %8u   %s\n", c.name.c_str(), c.count, c.max_ulp, c.p99,
                    worst);
    }

    std::printf("\nper exponent bucket: buckets that are not exactly rounded\n");
    std::printf("  %6s %10s %10s %7s %20s %13s\n", "bucket", "count", "max ulp", "% at 1",
                "range", "worst x");
    size_t active = 0;
    size_t covered = 0;
    for (size_t b = 0; b < buckets.size(); ++b) {
        if (buckets[b].count == 0) {
            continue;
        }
        ++covered;
        if (buckets[b].max_ulp == 0) {
            continue;
        }
        ++active;
        char range[32];
        if (b == 0) {
            std::snprintf(range, sizeof range, "subnormal/zero");
        } else if (b == 255) {
            std::snprintf(range, sizeof range, "inf/nan");
        } else {
            std::snprintf(range, sizeof range, "[2^%d, 2^%d)", static_cast<int>(b) - 127,
                          static_cast<int>(b) - 126);
        }
        const uint32_t worst = (buckets[b].arg < n) ? records[buckets[b].arg].x_bits : 0u;
        const double share = 100.0 * static_cast<double>(buckets[b].nonzero) /
                             static_cast<double>(buckets[b].count);
        std::printf("  %6zu %10llu %10u %6.1f%% %20s %13.9g\n", b,
                    static_cast<unsigned long long>(buckets[b].count), buckets[b].max_ulp, share,
                    range, static_cast<double>(float_of(worst)));
    }
    std::printf("  %zu of %zu non-empty buckets reached 1 ulp; the rest stayed exactly rounded\n",
                active, covered);

    // ---- report.json ------------------------------------------------------
    std::ofstream out(report_path);
    if (!out) {
        std::printf("cannot write %s\n", report_path);
        return 1;
    }
    out.precision(17);
    out << "{\n";
#if defined(_MSC_VER)
    out << "  \"compiler\": \"MSVC " << _MSC_VER << "\",\n";
#else
    out << "  \"compiler\": \"unknown\",\n";
#endif
    out << "  \"sample_points\": " << n << ",\n";
    out << "  \"points_measured\": " << distances.size() << ",\n";
    out << "  \"nan_inputs\": " << nan_inputs << ",\n";
    out << "  \"max_ulp\": " << max_ulp << ",\n";
    out << "  \"max_ulp_index\": " << arg << ",\n";
    out << "  \"max_ulp_x\": " << static_cast<double>(arg_x) << ",\n";
    out << "  \"max_ulp_x_bits\": \"0x" << std::hex << arg_x_bits << std::dec << "\",\n";
    out << "  \"max_ulp_k\": " << arg_k << ",\n";
    out << "  \"max_ulp_r\": " << static_cast<double>(arg_r) << ",\n";
    out << "  \"p50_ulp\": " << p50 << ",\n";
    out << "  \"p99_ulp\": " << p99 << ",\n";
    out << "  \"p999_ulp\": " << p999 << ",\n";
    out << "  \"count_ulp_gt_1\": " << count_over_one << ",\n";
    out << "  \"count_ulp_gt_0\": " << count_over_zero << ",\n";
    out << "  \"count_ulp_eq_0\": " << count_zero << ",\n";
    out << "  \"output_hash\": \"0x" << std::hex << out_hash << std::dec << "\",\n";
    out << "  \"nan_bit_exact_failures\": " << nan_mismatch << ",\n";
    out << "  \"ref_inf_cases\": " << ref_inf << ",\n";
    out << "  \"ref_inf_mismatches\": " << ref_inf_mismatch << ",\n";
    out << "  \"other_special_mismatches\": " << special_mismatch << ",\n";
    out << "  \"mean_ns_per_call\": " << ns_ours << ",\n";
    out << "  \"expf_max_ulp\": " << expf_max << ",\n";
    out << "  \"expf_p99_ulp\": " << expf_p99 << ",\n";
    out << "  \"expf_p999_ulp\": " << expf_p999 << ",\n";
    out << "  \"expf_count_ulp_gt_1\": " << expf_over_one << ",\n";
    out << "  \"expf_ns_per_call\": " << ns_expf << ",\n";
    out << "  \"categories\": [\n";
    for (size_t i = 0; i < categories.size(); ++i) {
        const Category& c = categories[i];
        out << "    {\"name\": \"" << c.name << "\", \"count\": " << c.count
            << ", \"max_ulp\": " << c.max_ulp << ", \"p99_ulp\": " << c.p99 << "}";
        out << ((i + 1 < categories.size()) ? ",\n" : "\n");
    }
    out << "  ],\n";
    out << "  \"exponent_buckets\": [\n";
    bool first_bucket = true;
    for (size_t b = 0; b < buckets.size(); ++b) {
        if (buckets[b].count == 0) {
            continue;
        }
        if (!first_bucket) {
            out << ",\n";
        }
        first_bucket = false;
        const uint32_t worst = (buckets[b].arg < n) ? records[buckets[b].arg].x_bits : 0u;
        out << "    {\"bucket\": " << b << ", \"count\": " << buckets[b].count
            << ", \"count_ulp_gt_0\": " << buckets[b].nonzero
            << ", \"max_ulp\": " << buckets[b].max_ulp << ", \"worst_x_bits\": \"0x" << std::hex
            << worst << std::dec << "\", \"worst_x\": " << static_cast<double>(float_of(worst))
            << "}";
    }
    out << "\n  ]\n";
    out << "}\n";
    out.close();

    std::printf("\nwrote %s\n", report_path);

    const bool ok = (max_ulp <= 1u) && (nan_mismatch == 0) && (ref_inf_mismatch == 0) &&
                    (special_mismatch == 0);
    std::printf("verdict: max ulp %u -> %s\n", max_ulp,
                ok ? "within the 1.0 ulp target" : "ABOVE the 1.0 ulp target");
    return 0;
}
