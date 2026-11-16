// Device-side check for the public headers, measured against the golden
// reference rather than against a hand-written list of inputs.
//
// Two questions, both part of the declared contract, and the second one is the
// reason this file exists:
//
//   1. Do the headers compile for the device at all? `ULPGRAD_HD` selects
//      `__host__ __device__`, and the intrinsics inside are chosen on
//      `__CUDA_ARCH__`. Getting that switch wrong compiles here and fails there,
//      which is the kind of thing a host-only suite cannot see.
//   2. Do the device results agree with the host, and with the reference the
//      bound is declared against? Both questions are answered on every point of
//      every committed golden file in `data/reference/`, not on a sample:
//
//        * parity: host against device, bit pattern against bit pattern, for
//          every function on every point. A single differing bit means the
//          explicit-`fma` contract broke, which is what the host and the device
//          must never do.
//        * bound: device against the mpmath reference stored in the CSV,
//          compared with the declared `ulpgrad::<name>_max_ulp`. The two are
//          reported separately because they fail for different reasons and one
//          of them can pass while the other does not.
//
// NaN patterns are compared bit for bit and never on the ULP metric: the metric
// is defined on the monotone image of the bit pattern, and a NaN has no position
// on that order. `log(-1)` is the case the reference contains, and it must come
// back as the canonical quiet NaN on both targets.
//
// Exit codes follow the convention used by the rest of these suites: 0 pass,
// 1 failure, 77 skipped because there is no device. The reference directory is
// the first argument and it is required: a hardcoded path is a path that lies on
// the machine of whoever runs the command second.
//
// Build and run:
//   nvcc -arch=sm_75 -std=c++17 -O2 -I include -o exp_smoke exp_smoke.cu
//   ./exp_smoke ../../data/reference

#include <ulpgrad/exp.h>
#include <ulpgrad/log.h>
#include <ulpgrad/rsqrt.h>
#include <ulpgrad/sigmoid.h>
#include <ulpgrad/silu.h>
#include <ulpgrad/tables.h>
#include <ulpgrad/tanh.h>
#include <ulpgrad/detail/bits.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// One output slot per function. The order of `kFunctions` is the order of the
// kernel's assignments, and `kFunctionCount` is what keeps the two in step: a
// slot without a name would make the report blame the wrong function.
constexpr int kFunctionCount = 9;

using Evaluate = float (*)(float);

// One entry per shipped function: the CSV stem, the host evaluator and the bound
// the header declares. The bound is read from the generated table, never copied
// here, so a stale number is impossible.
struct Function {
    const char* name;
    Evaluate evaluate;
    int bound;
};

constexpr Function kFunctions[kFunctionCount] = {
    {"exp", [](float x) { return ulpgrad::exp(x); }, ulpgrad::exp_max_ulp},
    {"exp2", [](float x) { return ulpgrad::exp2(x); }, ulpgrad::exp2_max_ulp},
    {"log", [](float x) { return ulpgrad::log(x); }, ulpgrad::log_max_ulp},
    {"log2", [](float x) { return ulpgrad::log2(x); }, ulpgrad::log2_max_ulp},
    {"log1p", [](float x) { return ulpgrad::log1p(x); }, ulpgrad::log1p_max_ulp},
    {"rsqrt", [](float x) { return ulpgrad::rsqrt(x); }, ulpgrad::rsqrt_max_ulp},
    {"sigmoid", [](float x) { return ulpgrad::sigmoid(x); }, ulpgrad::sigmoid_max_ulp},
    {"silu", [](float x) { return ulpgrad::silu(x); }, ulpgrad::silu_max_ulp},
    {"tanh", [](float x) { return ulpgrad::tanh(x); }, ulpgrad::tanh_max_ulp},
};

__global__ void apply(const float* x, float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // One assignment per slot, never a combination: `exp(x) + 3.0f * exp2(x)`
        // is a product-plus-add, and nvcc contracts it into an fma while the host
        // compiler does not. That single line of test code once reported a parity
        // failure that belonged to the harness, not to the library.
        y[0 * n + i] = ulpgrad::exp(x[i]);
        y[1 * n + i] = ulpgrad::exp2(x[i]);
        y[2 * n + i] = ulpgrad::log(x[i]);
        y[3 * n + i] = ulpgrad::log2(x[i]);
        y[4 * n + i] = ulpgrad::log1p(x[i]);
        y[5 * n + i] = ulpgrad::rsqrt(x[i]);
        y[6 * n + i] = ulpgrad::sigmoid(x[i]);
        y[7 * n + i] = ulpgrad::silu(x[i]);
        y[8 * n + i] = ulpgrad::tanh(x[i]);
    }
}

// One parsed golden file: the input patterns and the reference patterns, in the
// order they appear in the CSV.
struct Reference {
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

// Reads `input_bits_hex,output_bits_hex` records and skips the `#` header block.
bool read_reference(const std::string& path, Reference* reference) {
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        std::fprintf(stderr, "error: cannot open %s\n", path.c_str());
        return false;
    }
    char buffer[256];
    while (std::fgets(buffer, sizeof buffer, file) != nullptr) {
        std::string line(buffer);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const std::size_t comma = line.find(',');
        if (comma == std::string::npos) {
            std::fprintf(stderr, "error: %s: malformed record '%s'\n", path.c_str(),
                         line.c_str());
            std::fclose(file);
            return false;
        }
        std::uint32_t in = 0;
        std::uint32_t out = 0;
        if (!parse_hex(line.data(), line.data() + comma, &in) ||
            !parse_hex(line.data() + comma + 1, line.data() + line.size(), &out)) {
            std::fprintf(stderr, "error: %s: non-hex record '%s'\n", path.c_str(), line.c_str());
            std::fclose(file);
            return false;
        }
        reference->inputs.push_back(in);
        reference->expected.push_back(out);
    }
    std::fclose(file);
    if (reference->inputs.empty()) {
        std::fprintf(stderr, "error: %s: no records\n", path.c_str());
        return false;
    }
    return true;
}

bool is_nan_bits(std::uint32_t bits) {
    return (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u;
}

// What one function accumulated, split by the question it answers.
struct Tally {
    long long parity_points = 0;
    long long parity_mismatches = 0;
    long long bound_points = 0;     // points that carry a ULP distance
    long long bit_points = 0;       // points compared bit for bit, i.e. NaN at an end
    long long bit_mismatches = 0;
    unsigned max_ulp = 0;
    std::uint32_t worst_input = 0;
    std::uint32_t worst_got = 0;
    std::uint32_t worst_expected = 0;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: %s <reference-dir>\n"
                     "  <reference-dir> is the directory holding the golden CSVs,\n"
                     "  for example data/reference or ..\\..\\data\\reference\n",
                     argv[0]);
        return 1;
    }
    const std::string directory = argv[1];

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("[ SKIPPED ] no CUDA device available\n");
        return 77;
    }

    Tally tally[kFunctionCount];
    long long golden_points = 0;

    // One golden file at a time: every function is evaluated on that file's
    // inputs, so parity is checked on the union of the samples (each
    // sample has its own domain, and a function is as worth checking on the
    // points of its neighbours as on its own), while the bound is checked only
    // against the file that belongs to the function.
    for (int file = 0; file < kFunctionCount; ++file) {
        Reference reference;
        const std::string path = directory + "/" + kFunctions[file].name + ".csv";
        if (!read_reference(path, &reference)) {
            return 1;
        }
        const int n = static_cast<int>(reference.inputs.size());
        golden_points += n;

        std::vector<float> inputs(reference.inputs.size());
        for (std::size_t i = 0; i < reference.inputs.size(); ++i) {
            inputs[i] = ulpgrad::detail::bit_pattern_to_float(reference.inputs[i]);
        }

        const std::size_t byte_count = inputs.size() * sizeof(float);
        float* device_x = nullptr;
        float* device_y = nullptr;
        if (cudaMalloc(&device_x, byte_count) != cudaSuccess ||
            cudaMalloc(&device_y, kFunctionCount * byte_count) != cudaSuccess) {
            std::printf("cudaMalloc failed\n");
            return 1;
        }
        if (cudaMemcpy(device_x, inputs.data(), byte_count, cudaMemcpyHostToDevice) !=
            cudaSuccess) {
            std::printf("cudaMemcpy to device failed\n");
            return 1;
        }
        apply<<<(n + 255) / 256, 256>>>(device_x, device_y, n);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::printf("kernel failed: %s\n", cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
        std::vector<float> results(kFunctionCount * inputs.size());
        if (cudaMemcpy(results.data(), device_y, kFunctionCount * byte_count,
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::printf("cudaMemcpy to host failed\n");
            return 1;
        }
        cudaFree(device_x);
        cudaFree(device_y);

        // Parity: the same bits, on every point, for every function. Raw patterns
        // rather than `ulp_distance`, because a NaN payload and the sign of a
        // zero are part of the answer.
        for (int slot = 0; slot < kFunctionCount; ++slot) {
            for (int i = 0; i < n; ++i) {
                const std::uint32_t host_bits = ulpgrad::detail::float_to_bit_pattern(
                    kFunctions[slot].evaluate(inputs[i]));
                const std::uint32_t device_bits = ulpgrad::detail::float_to_bit_pattern(
                    results[static_cast<std::size_t>(slot) * n + i]);
                ++tally[slot].parity_points;
                if (host_bits != device_bits) {
                    ++tally[slot].parity_mismatches;
                    std::printf("PARITY  %-5s x=0x%08X host=0x%08X device=0x%08X\n",
                                kFunctions[slot].name, reference.inputs[i], host_bits,
                                device_bits);
                }
            }
        }

        // Bound: the device against the reference, for this file's own function.
        for (int i = 0; i < n; ++i) {
            const std::uint32_t in_bits = reference.inputs[i];
            const std::uint32_t expected_bits = reference.expected[i];
            const std::uint32_t got_bits = ulpgrad::detail::float_to_bit_pattern(
                results[static_cast<std::size_t>(file) * n + i]);
            if (is_nan_bits(in_bits) || is_nan_bits(expected_bits)) {
                ++tally[file].bit_points;
                if (got_bits != expected_bits) {
                    ++tally[file].bit_mismatches;
                    std::printf("BOUND   %-5s x=0x%08X got=0x%08X reference=0x%08X (NaN)\n",
                                kFunctions[file].name, in_bits, got_bits, expected_bits);
                }
                continue;
            }
            const unsigned distance = ulpgrad::detail::ulp_distance(
                ulpgrad::detail::bit_pattern_to_float(got_bits),
                ulpgrad::detail::bit_pattern_to_float(expected_bits));
            ++tally[file].bound_points;
            if (distance > tally[file].max_ulp) {
                tally[file].max_ulp = distance;
                tally[file].worst_input = in_bits;
                tally[file].worst_got = got_bits;
                tally[file].worst_expected = expected_bits;
            }
        }
    }

    std::printf("reference: %s, %lld points in %d golden files\n", directory.c_str(),
                golden_points, kFunctionCount);
    std::printf("parity: host against device, bit for bit, every function on every point\n");
    std::printf("bound: device against the mpmath reference, declared bound from tables.h\n");

    int failures = 0;
    for (int slot = 0; slot < kFunctionCount; ++slot) {
        const Tally& t = tally[slot];
        const bool parity_ok = t.parity_mismatches == 0;
        const bool bound_ok =
            t.max_ulp <= static_cast<unsigned>(kFunctions[slot].bound) && t.bit_mismatches == 0;
        if (!parity_ok || !bound_ok) {
            ++failures;
        }
        std::printf(
            "%-5s  parity %6lld points, %lld mismatch  |  bound %6lld points"
            " (+%lld NaN), max %u ulp (bound %d)  %s\n",
            kFunctions[slot].name, t.parity_points, t.parity_mismatches, t.bound_points,
            t.bit_points, t.max_ulp, kFunctions[slot].bound,
            (parity_ok && bound_ok) ? "ok" : "FAIL");
        if (!bound_ok) {
            std::printf("       worst x=0x%08X got=0x%08X reference=0x%08X\n", t.worst_input,
                        t.worst_got, t.worst_expected);
        }
    }

    std::printf("%s: %d of %d functions within parity and bound\n",
                failures == 0 ? "PASS" : "FAIL", kFunctionCount - failures, kFunctionCount);
    return failures == 0 ? 0 : 1;
}
