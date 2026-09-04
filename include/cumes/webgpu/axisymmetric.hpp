#ifndef CUMES_INCLUDE_CUMES_WEBGPU_AXISYMMETRIC_HPP_
#define CUMES_INCLUDE_CUMES_WEBGPU_AXISYMMETRIC_HPP_

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace cumes::webgpu {

inline constexpr std::size_t SPECTRAL_COMPONENT_COUNT = 6;
inline constexpr std::size_t GEOMETRY_PARITY_FIELD_COUNT = 18;
inline constexpr std::size_t FORWARD_INPUT_FIELD_COUNT = 14;

struct AxisymmetricInverseCase {
    int ns = 0;
    int mpol = 0;
    int ntheta = 0;
    // Component-major spectral layout: [component][mode][surface].
    std::vector<float> state;
};

struct AxisymmetricInverseResult {
    // Field-major real-space layout: [field][surface][theta]. Field order is
    // r/z/l, ru/zu/lu even; r/z/l, ru/zu/lu odd; then the six zero toroidal
    // derivatives, matching GeometryParityViews.
    std::vector<float> geometry;
    std::vector<float> r_con;
    std::vector<float> z_con;
    std::vector<float> geometry_lo;
    std::vector<float> r_con_lo;
    std::vector<float> z_con_lo;
};

using AxisymmetricInverseCallback =
    std::function<void(std::string, AxisymmetricInverseResult)>;

void enqueue_axisymmetric_inverse(const wgpu::Device& device,
                                  const AxisymmetricInverseCase& input,
                                  AxisymmetricInverseCallback callback);

AxisymmetricInverseResult axisymmetric_inverse_reference(
    const AxisymmetricInverseCase& input);

struct AxisymmetricForwardCase {
    int ns = 0;
    int mpol = 0;
    int ntheta = 0;
    bool include_lcfs = false;
    // Field-major [field][surface][theta]: armn e/o, azmn e/o, brmn e/o,
    // bzmn e/o, blmn e/o, frcon e/o, fzcon e/o.
    std::vector<float> fields;
};

struct AxisymmetricForwardResult {
    // Component-major [component][mode][surface].
    std::vector<float> residual;
};

using AxisymmetricForwardCallback =
    std::function<void(std::string, AxisymmetricForwardResult)>;

void enqueue_axisymmetric_forward(const wgpu::Device& device,
                                  const AxisymmetricForwardCase& input,
                                  AxisymmetricForwardCallback callback);

AxisymmetricForwardResult axisymmetric_forward_reference(
    const AxisymmetricForwardCase& input);

struct AxisymmetricDealiasCase {
    int ns = 0;
    int mpol = 0;
    int ntheta = 0;
    std::vector<float> g_con_eff;
    std::vector<float> tcon;
    std::vector<float> faccon;
};

struct AxisymmetricDealiasResult {
    std::vector<float> g_con;
};

using AxisymmetricDealiasCallback =
    std::function<void(std::string, AxisymmetricDealiasResult)>;

void enqueue_axisymmetric_dealias(const wgpu::Device& device,
                                  const AxisymmetricDealiasCase& input,
                                  AxisymmetricDealiasCallback callback);

AxisymmetricDealiasResult axisymmetric_dealias_reference(
    const AxisymmetricDealiasCase& input);

}  // namespace cumes::webgpu

#endif  // CUMES_INCLUDE_CUMES_WEBGPU_AXISYMMETRIC_HPP_
