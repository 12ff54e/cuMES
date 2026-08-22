// test_accumulation.cu — differential test for the mixed-float
// double-accumulation norm policy (blueprint §8.8, §8.12).
//
// The solver's control-feeding reductions (computeResidualsKernel,
// rzNormKernel, forceNormReduceKernel) accumulate in NormAccum<T>::type —
// double for float state, double for double state. This test pins that trait
// contract and demonstrates the numerical benefit: summing a large array of
// squares whose terms span a wide dynamic range in float accumulation silently
// drops the small terms, while double accumulation tracks the CPU double
// reference. The crafted input has one 1e4 term (square 1e8) plus a million 0.1
// terms (square 1e-2): at 1e8 the float ulp is 8, so every small add vanishes
// in float but survives in double.
#include "cumes/numerics/accumulation.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes_test.h"

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>
using namespace cumes::test;

// acc += A(x[i]) * A(x[i]) over a grid-stride loop + a fixed block tree, the
// same reduction shape the solver kernels use. `A` is the accumulator type.
template <typename T, typename A>
__global__ void sumSquaresKernel(const T* __restrict__ x,
                                 int n,
                                 T* __restrict__ out) {
    A acc = A(0);
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += A(x[i]) * A(x[i]);
    __shared__ A s[256];
    int tid = threadIdx.x;
    s[tid] = acc;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k) s[tid] += s[tid + k];
        __syncthreads();
    }
    if (tid == 0) out[0] = T(s[0]);
}

int main() {
    // Trait contract (the production kernels rely on these).
    static_assert(std::is_same_v<cumes::NormAccum<float>::type, double>);
    static_assert(std::is_same_v<cumes::NormAccum<double>::type, double>);

    const int n = 1 << 20;  // 1M terms
    std::vector<float> h(n);
    long double cpu_ref = 0.0L;  // extended-precision reference (near-exact)
    h[0] = 1.0e4f;
    for (int i = 1; i < n; ++i) h[i] = 0.1f;
    for (int i = 0; i < n; ++i)
        cpu_ref += (long double)h[i] * (long double)h[i];

    float* d_x = nullptr;
    float* d_out = nullptr;
    cumes::check_cuda(cudaMalloc(&d_x, n * sizeof(float)), "alloc x");
    cumes::check_cuda(cudaMalloc(&d_out, sizeof(float)), "alloc out");
    cumes::check_cuda(
        cudaMemcpy(d_x, h.data(), n * sizeof(float), cudaMemcpyHostToDevice),
        "cpy x");

    float f_acc = 0.0f, d_acc = 0.0f;
    sumSquaresKernel<float, float><<<1, 256>>>(d_x, n, d_out);
    cumes::check_cuda(
        cudaMemcpy(&f_acc, d_out, sizeof(float), cudaMemcpyDeviceToHost),
        "cpy float out");
    sumSquaresKernel<float, double><<<1, 256>>>(d_x, n, d_out);
    cumes::check_cuda(
        cudaMemcpy(&d_acc, d_out, sizeof(float), cudaMemcpyDeviceToHost),
        "cpy double out");

    const double ref = (double)cpu_ref;
    const double ferr = std::fabs((double)f_acc - ref);
    const double derr = std::fabs((double)d_acc - ref);

    std::cout << format("sum-of-squares reference  = {:.6f}\n", ref);
    std::cout << format("  float accumulation      = {:.6f} (abs err {:.3e})\n",
                        (double)f_acc, ferr);
    std::cout << format("  double accumulation     = {:.6f} (abs err {:.3e})\n",
                        (double)d_acc, derr);

    // double accumulation must be decisively better than float accumulation on
    // this dynamic-range case, and stay within a generous relative band of the
    // extended-precision reference (the GPU reduction uses a different order,
    // so a ~1e-8 relative deviation from the exact sum is expected).
    check(derr < ferr,
          "double accumulation not better than float accumulation");
    check(derr < 1e-6 * ref, "double accumulation too far from reference");

    cudaFree(d_x);
    cudaFree(d_out);

    return summary();
}
