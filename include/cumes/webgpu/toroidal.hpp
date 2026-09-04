#ifndef CUMES_INCLUDE_CUMES_WEBGPU_TOROIDAL_HPP_
#define CUMES_INCLUDE_CUMES_WEBGPU_TOROIDAL_HPP_

#include "cumes/webgpu/axisymmetric.hpp"

#include <functional>
#include <string>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace cumes::webgpu {

struct ToroidalInverseCase {
    int ns = 0;
    int mpol = 0;
    int ntor = 0;
    int ntheta = 0;
    int nzeta = 0;
    int nfp = 0;
    // Component-major [component][mode][surface], with
    // mode=m*(ntor+1)+n.
    std::vector<float> state;
};

using ToroidalInverseResult = AxisymmetricInverseResult;
using ToroidalInverseCallback =
    std::function<void(std::string, ToroidalInverseResult)>;

void enqueue_toroidal_inverse(const wgpu::Device& device,
                              const ToroidalInverseCase& input,
                              ToroidalInverseCallback callback);

ToroidalInverseResult toroidal_inverse_reference(
    const ToroidalInverseCase& input);

inline constexpr std::size_t TOROIDAL_FORWARD_FIELD_COUNT = 20;

struct ToroidalForwardCase {
    int ns = 0;
    int mpol = 0;
    int ntor = 0;
    int ntheta = 0;
    int nzeta = 0;
    int nfp = 0;
    bool include_lcfs = false;
    // Field-major [field][surface][zeta][theta]: armn e/o, azmn e/o,
    // brmn e/o, bzmn e/o, blmn e/o, crmn e/o, czmn e/o, clmn e/o,
    // frcon e/o, fzcon e/o.
    std::vector<float> fields;
};

struct ToroidalForwardResult {
    std::vector<float> residual;
};

using ToroidalForwardCallback =
    std::function<void(std::string, ToroidalForwardResult)>;

void enqueue_toroidal_forward(const wgpu::Device& device,
                              const ToroidalForwardCase& input,
                              ToroidalForwardCallback callback);

ToroidalForwardResult toroidal_forward_reference(
    const ToroidalForwardCase& input);

}  // namespace cumes::webgpu

#endif  // CUMES_INCLUDE_CUMES_WEBGPU_TOROIDAL_HPP_
