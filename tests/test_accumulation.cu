// A float-float reduction must retain small summands lost by ordinary float
// accumulation. The input spans 1e8 down to 1e-2 in its squared terms; compare
// both GPU reductions with an extended-precision host reference.
#include "cumes/numerics/accumulation.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/runtime/device_buffer.cuh"
#include "cumes_test.h"

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>
using namespace cumes::test;

// acc += A(x[i]) * A(x[i]) over a grid-stride loop + a fixed block tree, the
// same reduction shape the solver kernels use. `A` is the accumulator type.
template <typename T, typename A>
__global__ void sum_squares_kernel(const T* __restrict__ x,
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
    static_assert(
        std::is_same_v<cumes::NormAccum<float>::type, cumes::FloatFloat>);
    static_assert(std::is_same_v<cumes::NormAccum<double>::type, double>);

    const int n = 1 << 20;  // 1M terms
    std::vector<float> h(n);
    long double cpu_ref = 0.0L;  // extended-precision reference (near-exact)
    h[0] = 1.0e4f;
    for (int i = 1; i < n; ++i) h[i] = 0.1f;
    for (int i = 0; i < n; ++i)
        cpu_ref += (long double)h[i] * (long double)h[i];

    cumes::DeviceBuffer<float> d_x(n), d_out(1);
    cumes::check_cuda(cudaMemcpy(d_x.data(), h.data(), n * sizeof(float),
                                 cudaMemcpyHostToDevice),
                      "cpy x");

    float f_acc = 0.0f, ff_acc = 0.0f;
    sum_squares_kernel<float, float><<<1, 256>>>(d_x.data(), n, d_out.data());
    cumes::check_cuda(
        cudaMemcpy(&f_acc, d_out.data(), sizeof(float), cudaMemcpyDeviceToHost),
        "cpy float out");
    sum_squares_kernel<float, cumes::FloatFloat>
        <<<1, 256>>>(d_x.data(), n, d_out.data());
    cumes::check_cuda(cudaMemcpy(&ff_acc, d_out.data(), sizeof(float),
                                 cudaMemcpyDeviceToHost),
                      "cpy float-float out");

    const double ref = (double)cpu_ref;
    const double ferr = std::fabs((double)f_acc - ref);
    const double fferr = std::fabs((double)ff_acc - ref);

    std::cout << format("sum-of-squares reference  = {:.6f}\n", ref);
    std::cout << format("  float accumulation      = {:.6f} (abs err {:.3e})\n",
                        (double)f_acc, ferr);
    std::cout << format(
        "  float-float accumulation     = {:.6f} (abs err {:.3e})\n",
        (double)ff_acc, fferr);

    // float-float accumulation must be decisively better than float
    // accumulation on this dynamic-range case, and stay within a generous
    // relative band of the extended-precision reference (the GPU reduction uses
    // a different order, so a ~1e-8 relative deviation from the exact sum is
    // expected).
    check(fferr < ferr,
          "float-float accumulation not better than float accumulation");
    check(fferr < 1e-6 * ref,
          "float-float accumulation too far from reference");

    return summary();
}
