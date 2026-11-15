#include <gtest/gtest.h>

// Host-only main: every suite in this repository is host code, so the binary
// runs on GPU-less CI runners. Device suites arrive in M7 and link a separate
// main that exits 77 when no device is present, which ctest reports as SKIPPED
// rather than as a silent pass.
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
