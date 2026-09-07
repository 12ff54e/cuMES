#include "cumes/numerics/forward_dual.cuh"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/runtime/device_buffer.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

using Dual = cumes::ForwardDual<double>;

__global__ void evaluate_kernel(const Dual* x, Dual* y) {
    const Dual a = x[0];
    const Dual b = x[1];
    y[0] = sqrt(a * a + b) / (a - Dual(0.5));
    y[1] = pow(a, 3) + fabs(b);
}

void require_close(double actual, double expected, const char* message) {
    if (std::abs(actual - expected) > 2.0e-13) {
        std::cerr << "FAIL: " << message << " actual=" << actual
                  << " expected=" << expected << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    const Dual input[2] = {{2.0, 0.3}, {1.5, -0.2}};
    cumes::DeviceBuffer<Dual> d_input(2);
    cumes::DeviceBuffer<Dual> d_output(2);
    cumes::check_cuda(cudaMemcpy(d_input.data(), input, sizeof(input),
                                 cudaMemcpyHostToDevice),
                      "forward dual input");
    evaluate_kernel<<<1, 1>>>(d_input.data(), d_output.data());
    cumes::check_cuda(cudaGetLastError(), "forward dual kernel");
    Dual output[2];
    cumes::check_cuda(cudaMemcpy(output, d_output.data(), sizeof(output),
                                 cudaMemcpyDeviceToHost),
                      "forward dual output");

    const double root = std::sqrt(5.5);
    const double numerator_tangent = (2.0 * 2.0 * 0.3 - 0.2) / (2.0 * root);
    const double quotient_tangent =
        (numerator_tangent * 1.5 - root * 0.3) / (1.5 * 1.5);
    require_close(output[0].value, root / 1.5, "quotient primal");
    require_close(output[0].tangent, quotient_tangent, "quotient tangent");
    require_close(output[1].value, 9.5, "power/absolute primal");
    require_close(output[1].tangent, 3.0 * 4.0 * 0.3 - 0.2,
                  "power/absolute tangent");
    std::cout << "forward dual CUDA tests passed\n";
    return 0;
}
