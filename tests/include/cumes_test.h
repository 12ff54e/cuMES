#ifndef CUMES_TESTS_INCLUDE_CUMES_TEST_H_
#define CUMES_TESTS_INCLUDE_CUMES_TEST_H_
// cumes_test.h — CUDA-free test harness (assertions, comparison, summary).
//
// Included by every test: directly by the host-only .cpp tests, transitively
// by the .cu tests via cumes_test_cuda_helper.cuh. CUDA-free (no cuda_runtime);
// the project builds as C++20 throughout (root CMakeLists.txt).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <span>
#include <string_view>

// cumes::test::format(...): std::format where the toolchain has it (gcc 13+),
// an ostream-based fallback with the same {} syntax for the g++-12 CUDA host
// pass. Every test uses it instead of printf.
#include "cumes_test_format.h"

namespace cumes::test {

// The single shared failure counter (function-local static, so header-only and
// shared across TUs). Capture `int before = failures();` to count only the
// failures a sub-check adds.
inline int& failures() {
    static int n = 0;
    return n;
}

// Assert `ok`; on failure print "FAIL <msg>" and increment the counter.
inline void check(bool ok, std::string_view msg) {
    if (ok) {
        std::cout << "PASS " << msg << '\n';
    } else {
        std::cout << "FAIL " << msg << '\n';
        ++failures();
    }
}

// Max element-wise |a[i] - b[i]| over the shorter of the two ranges. Accepts
// any contiguous range (std::vector, std::array, std::span, raw C arrays) via
// the generic RangeA/RangeB deduction — a std::span parameter would not deduce
// from std::vector (implicit conversions don't participate in deduction).
template <typename RangeA, typename RangeB>
double max_diff(const RangeA& a, const RangeB& b) {
    const std::size_t n = std::min(a.size(), b.size());
    double m = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        m = std::max(m, std::fabs(double(a[i]) - double(b[i])));
    return m;
}

// Span convenience overload (raw device→host buffers): element-wise max
// over the shorter span.
template <typename T>
double max_diff(std::span<const T> a, std::span<const T> b) {
    const std::size_t n = std::min(a.size(), b.size());
    double m = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        m = std::max(m, std::fabs(double(a[i]) - double(b[i])));
    return m;
}

// Assert |a - b| <= tol; on failure print the values and the message.
inline void expect_near(double a, double b, double tol, std::string_view msg) {
    if (std::fabs(a - b) <= tol) {
        std::cout << "PASS " << msg << '\n';
    } else {
        std::cout << "FAIL " << msg << " (a=" << a << " b=" << b
                  << " diff=" << std::fabs(a - b) << ")\n";
        ++failures();
    }
}

// Print the final summary and return the process exit code (0 = all pass).
inline int summary() {
    if (failures()) {
        std::cout << failures() << " FAILURES\n";
        return 1;
    }
    std::cout << "ALL PASS\n";
    return 0;
}

}  // namespace cumes::test

#endif  // CUMES_TESTS_INCLUDE_CUMES_TEST_H_
