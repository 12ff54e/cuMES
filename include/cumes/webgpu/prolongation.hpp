#ifndef CUMES_INCLUDE_CUMES_WEBGPU_PROLONGATION_HPP_
#define CUMES_INCLUDE_CUMES_WEBGPU_PROLONGATION_HPP_

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace cumes::webgpu {

enum class RadialInterpolation : unsigned {
    LINEAR = 0,
    CATMULL_ROM = 1,
};

struct ProlongationCase {
    int ns_old = 0;
    int ns_new = 0;
    int mnmax = 0;
    int ntor = 0;
    RadialInterpolation interpolation = RadialInterpolation::LINEAR;
    std::vector<float> state;
};

struct ProlongationResult {
    std::vector<float> state;
    std::vector<float> velocity;
};

using ProlongationCallback =
    std::function<void(std::string, ProlongationResult)>;

// Dispatches the mixed-float radial-transfer shader and asynchronously maps
// its result. The callback receives an empty error on success. All resources
// needed by an in-flight dispatch are retained until the map callback fires.
void enqueue_prolongation(const wgpu::Device& device,
                          const ProlongationCase& input,
                          ProlongationCallback callback);

// CPU mirror of the WGSL contract, used by the browser self-test and by later
// backend-conformance tests.
ProlongationResult prolongation_reference(const ProlongationCase& input);

}  // namespace cumes::webgpu

#endif  // CUMES_INCLUDE_CUMES_WEBGPU_PROLONGATION_HPP_
