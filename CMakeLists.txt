cmake_minimum_required(VERSION 3.18)

project(ulpgrad
    LANGUAGES CXX
    VERSION 0.1.0
    DESCRIPTION "Header-only float32 math functions with a measured, CI-verified ULP bound"
    HOMEPAGE_URL "https://github.com/MardokHPC/ulpgrad"
)

option(ULPGRAD_BUILD_TESTS "Build the host-only GoogleTest suites" OFF)
option(ULPGRAD_BUILD_BENCH "Build the GPU benchmark (reserved: requires CUDA)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# One generator expression per flag. A multi-word genex such as
# `$<...:-Wall -Wextra>` is not split by CMake: it reaches the compiler as a single
# argument, which both cl.exe and g++ reject. Keep one flag per line.
function(ulpgrad_apply_warnings target)
    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
        $<$<CXX_COMPILER_ID:MSVC>:/WX>
        $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall>
        $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wextra>
        $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wpedantic>
        $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Werror>
    )
endfunction()

# Header-only INTERFACE target: the same name whether consumed through
# add_subdirectory/FetchContent or through find_package().
add_library(ulpgrad INTERFACE)
add_library(ulpgrad::ulpgrad ALIAS ulpgrad)

target_include_directories(ulpgrad INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(ulpgrad INTERFACE cxx_std_17)

# Measurement tool: reads data/reference/*.csv, exercises the library, and
# rewrites data/error-bounds.json. Not installed and not a test: it is the input
# of the generated table, so CI has to keep it compiling, but the artifacts it
# produces are committed and the tests verify them.
add_executable(ulpgrad_measure_bounds tools/measure_bounds.cpp)
target_link_libraries(ulpgrad_measure_bounds PRIVATE ulpgrad::ulpgrad)
set_target_properties(ulpgrad_measure_bounds PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
ulpgrad_apply_warnings(ulpgrad_measure_bounds)
# The tool writes one file and reads one environment variable. The MSVC "safe"
# replacements for fopen/getenv are not portable and exist in no other toolchain,
# so the deprecation warning is silenced here rather than in the source, which
# keeps the single source compilable everywhere with the same code path.
target_compile_definitions(ulpgrad_measure_bounds PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
)

# Warnings are not attached to `ulpgrad`: it is INTERFACE, nothing is compiled
# with it, and an INTERFACE target cannot take PRIVATE options. Consumers keep
# their own policy; every target this project compiles goes through the helper.

# Install + export, three blocks. The headers land in include/ulpgrad/, the
# target file carries the include dir and the C++17 requirement.
install(DIRECTORY include/ulpgrad
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

install(TARGETS ulpgrad
    EXPORT ulpgrad-targets
)

install(EXPORT ulpgrad-targets
    FILE ulpgrad-targets.cmake
    NAMESPACE ulpgrad::
    DESTINATION lib/cmake/ulpgrad
)

if(ULPGRAD_BUILD_TESTS)
    enable_testing()

    # GoogleTest 1.15.2 by tag, so a build in 2027 fetches the same code. The
    # timestamp policy keeps the archive's own timestamps: without CMP0135 NEW
    # the extracted files are stamped "now" and every configure looks dirty.
    if(POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)
    endif()
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
    )
    # Keep GoogleTest on the parent project's CRT and out of our install tree.
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # GoogleTest is not ours and is not warning-clean at /W4 -Werror. An include
    # directory marked SYSTEM makes the compiler stop reporting warnings from it,
    # which is the only way to keep -Werror on. FetchContent creates `gtest` as a
    # plain include dir, so the wrapper moves that directory onto the SYSTEM list
    # and drops it from the ordinary one: two entries for the same path would let
    # the non-system `-I` win the search order and the warnings would come back.
    add_library(ulpgrad_gtest INTERFACE)
    target_link_libraries(ulpgrad_gtest INTERFACE gtest)
    target_include_directories(ulpgrad_gtest SYSTEM INTERFACE
        ${googletest_SOURCE_DIR}/googletest/include
    )
    set_target_properties(gtest PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")

    function(ulpgrad_add_test name source)
        add_executable(${name} ${source} ${PROJECT_SOURCE_DIR}/tests/test_main.cpp)
        target_link_libraries(${name} PRIVATE ulpgrad::ulpgrad ulpgrad_gtest)
        # The golden reference files live next to the source, not next to the
        # build tree: a test that reads them must fail when they are missing,
        # and a path baked in at configure time keeps the binary independent of
        # the working directory ctest happens to use.
        target_compile_definitions(${name} PRIVATE
            ULPGRAD_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/data"
            ULPGRAD_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        )
        set_target_properties(${name} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
        )
        ulpgrad_apply_warnings(${name})
        add_test(NAME ${name} COMMAND ${name})
        # Exit code 77 means "the harness measured nothing": ctest reports it as
        # SKIPPED, never as a silent pass. No suite in M1 can reach 77; the hook
        # is in place so the device suites of M7 do not have to change this file.
        set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
    endfunction()

    ulpgrad_add_test(test_config tests/test_config.cpp)
    ulpgrad_add_test(test_bits tests/test_bits.cpp)
    ulpgrad_add_test(test_exp tests/test_exp.cpp)
    ulpgrad_add_test(test_log tests/test_log.cpp)
    ulpgrad_add_test(test_rsqrt tests/test_rsqrt.cpp)
    ulpgrad_add_test(test_tanh tests/test_tanh.cpp)
    ulpgrad_add_test(test_special_cases tests/test_special_cases.cpp)
    ulpgrad_add_test(test_table_freshness tests/test_table_freshness.cpp)
endif()

# Package config and version file. VERSION comes from project(): one source, so
# the installed package can never disagree with the build tree.
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ulpgrad-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ulpgrad-config.cmake
    INSTALL_DESTINATION lib/cmake/ulpgrad
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ulpgrad-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ulpgrad-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ulpgrad-config-version.cmake
    DESTINATION lib/cmake/ulpgrad
)
