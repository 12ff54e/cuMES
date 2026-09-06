#ifndef CUMES_INCLUDE_CUMES_WEBGPU_TOROIDAL_HPP_
#define CUMES_INCLUDE_CUMES_WEBGPU_TOROIDAL_HPP_

#include "cumes/webgpu/axisymmetric.hpp"

#include <functional>
#include <string>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace cumes::webgpu {

using ToroidalInverseResult = AxisymmetricInverseResult;

struct ToroidalInverseCase {
    // Batched resident solves can replace host vectors with a GPU finite scan.
    bool readback_values = true;
    // A pending descent state is consumed without mapping. Axis extrapolation
    // is applied by index selection; the host mirror is updated at the fence.
    DeviceFields device_state;
    BatchedReadback<ToroidalInverseResult> readback;
    int ns = 0;
    int mpol = 0;
    int ntor = 0;
    int ntheta = 0;
    int nzeta = 0;
    int nfp = 0;
    bool double_single = false;
    // Component-major [component][mode][surface], with
    // mode=m*(ntor+1)+n.
    std::vector<float> state;
    std::vector<float> state_lo;
};

using ToroidalInverseCallback =
    std::function<void(std::string, ToroidalInverseResult)>;

void enqueue_toroidal_inverse(const wgpu::Device& device,
                              const ToroidalInverseCase& input,
                              ToroidalInverseCallback callback);

ToroidalInverseResult toroidal_inverse_reference(
    const ToroidalInverseCase& input);

inline constexpr std::size_t TOROIDAL_FORWARD_FIELD_COUNT = 20;

struct ToroidalForwardCase {
    DeviceFields device_fields;
    int ns = 0;
    int mpol = 0;
    int ntor = 0;
    int ntheta = 0;
    int nzeta = 0;
    int nfp = 0;
    bool include_lcfs = false;
    bool double_single = false;
    bool use_fft = true;
    bool optimized_fft = true;
    bool readback = true;
    // Diagnostic/reference option: split double-precision zeta roots directly.
    // The poloidal basis and inverse transform remain unchanged.
    bool canonical_zeta = false;
    // Field-major [field][surface][zeta][theta]: armn e/o, azmn e/o,
    // brmn e/o, bzmn e/o, blmn e/o, crmn e/o, czmn e/o, clmn e/o,
    // frcon e/o, fzcon e/o.
    std::vector<float> fields;
    std::vector<float> fields_lo;
};

struct ToroidalForwardResult {
    DeviceFields device_residual;
    std::vector<float> residual;
    std::vector<float> residual_lo;
};

using ToroidalForwardCallback =
    std::function<void(std::string, ToroidalForwardResult)>;

void enqueue_toroidal_forward(const wgpu::Device& device,
                              const ToroidalForwardCase& input,
                              ToroidalForwardCallback callback);

ToroidalForwardResult toroidal_forward_reference(
    const ToroidalForwardCase& input);

struct ToroidalDealiasResult;

struct ToroidalDealiasCase {
    BatchedReadback<ToroidalDealiasResult> readback;
    bool readback_values = true;
    DeviceFields device_g_con_eff;
    DeviceFields device_tcon;
    int ns = 0;
    int mpol = 0;
    int ntor = 0;
    int ntheta = 0;
    int nzeta = 0;
    std::vector<float> g_con_eff;
    std::vector<float> tcon;
    std::vector<float> faccon;
};

struct ToroidalDealiasResult {
    DeviceFields device_g_con;
    bool finite = true;
    std::vector<float> g_con;
};

using ToroidalDealiasCallback =
    std::function<void(std::string, ToroidalDealiasResult)>;

void enqueue_toroidal_dealias(const wgpu::Device& device,
                              const ToroidalDealiasCase& input,
                              ToroidalDealiasCallback callback);

ToroidalDealiasResult toroidal_dealias_reference(
    const ToroidalDealiasCase& input);

}  // namespace cumes::webgpu

#endif  // CUMES_INCLUDE_CUMES_WEBGPU_TOROIDAL_HPP_
