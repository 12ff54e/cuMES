#ifndef CUMES_INCLUDE_CUMES_WEBGPU_INITIALIZATION_HPP_
#define CUMES_INCLUDE_CUMES_WEBGPU_INITIALIZATION_HPP_

#include <cstddef>
#include <vector>

namespace cumes {
class ValidatedProblem;
}

namespace cumes::webgpu {

struct RadialProfiles {
    using val_type = float;

    std::vector<float> iota_f;
    std::vector<float> phip_f;
    std::vector<float> chi_f;
    std::vector<float> sqrt_s_f;
    std::vector<float> iota_h;
    std::vector<float> pres_h;
    std::vector<float> phip_h;
    std::vector<float> dvds_h;
    std::vector<float> sqrt_s_h;
    std::vector<float> curr_h;
    std::vector<float> chip_h;
    float delta_s = 0.0F;
    float lamscale = 0.0F;
};

struct AxisymmetricStageData {
    using val_type = float;

    int ns = 0;
    int mpol = 0;
    int ntor = 0;
    int ntheta = 0;
    int nzeta = 1;
    int nfp = 1;
    int max_iterations = 0;
    double tolerance = 0.0;
    float delta_t = 0.9F;
    float tcon0 = 1.0F;
    bool free_boundary = false;
    std::vector<float> state;
    RadialProfiles profiles;
    float envelope_correction = 0.0F;
    float lambda_seed_scale = 0.0F;
};

// Constructs the same f32 cold-start state and immutable radial profiles as
// the CUDA stage setup for either axisymmetric or folded 3-D modes.
AxisymmetricStageData initialize_stage(const ValidatedProblem& problem,
                                       std::size_t stage_index);

AxisymmetricStageData initialize_axisymmetric_stage(
    const ValidatedProblem& problem,
    std::size_t stage_index);

}  // namespace cumes::webgpu

#endif  // CUMES_INCLUDE_CUMES_WEBGPU_INITIALIZATION_HPP_
