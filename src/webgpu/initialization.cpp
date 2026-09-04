#include "cumes/webgpu/initialization.hpp"

#include "cumes/config/device_params.hpp"
#include "cumes/config/profile_functions.hpp"
#include "cumes/config/validated_problem.hpp"
#include "cumes/state/axisymmetric_lambda_seed.hpp"
#include "cumes/state/seed_envelope.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>

namespace cumes::webgpu {
namespace {

RadialProfiles initialize_profiles(const ProblemSpec& spec, int ns) {
    RadialProfiles profiles;
    profiles.delta_s = 1.0F / static_cast<float>(ns - 1);

    float max_toroidal_flux =
        static_cast<float>(DeviceParams<float>::SIGN_JACOBIAN) *
        static_cast<float>(spec.physical.phiedge) /
        static_cast<float>(2.0 * M_PI);
    const float torflux_edge = torflux<float>(spec, 1.0F);
    if (torflux_edge != 0.0F) max_toroidal_flux /= torflux_edge;

    float current_scale = 0.0F;
    if (spec.current_model == CurrentModel::PRESCRIBED_CURRENT) {
        const float edge_current = eval_curr_profile<float>(spec, 1.0F);
        if (edge_current == 0.0F) {
            throw std::runtime_error(
                "WebGPU profiles: prescribed current has zero edge integral");
        }
        current_scale = static_cast<float>(DeviceParams<float>::SIGN_JACOBIAN) *
                        DeviceParams<float>::MU_0 *
                        static_cast<float>(spec.physical.curtor) /
                        (static_cast<float>(2.0 * M_PI) * edge_current);
    }

    profiles.iota_f.resize(ns);
    profiles.phip_f.resize(ns);
    profiles.chi_f.resize(ns);
    profiles.sqrt_s_f.resize(ns);
    for (int surface = 0; surface < ns; ++surface) {
        const float s = profiles.delta_s * static_cast<float>(surface);
        const float flux = std::min(torflux<float>(spec, s), 1.0F);
        const float derivative = torflux_deriv<float>(spec, s);
        const float iota = eval_iota_profile<float>(spec, flux);
        profiles.iota_f[surface] = iota;
        profiles.phip_f[surface] = max_toroidal_flux * derivative;
        profiles.chi_f[surface] = max_toroidal_flux * iota * derivative;
        profiles.sqrt_s_f[surface] = std::sqrt(s + 1.0e-12F);
    }

    const int half_surfaces = ns - 1;
    profiles.iota_h.resize(half_surfaces);
    profiles.pres_h.resize(half_surfaces);
    profiles.phip_h.resize(half_surfaces);
    profiles.dvds_h.assign(half_surfaces, 1.0F);
    profiles.sqrt_s_h.resize(half_surfaces);
    profiles.curr_h.resize(half_surfaces);
    profiles.chip_h.resize(half_surfaces);
    float phip_square_sum = 0.0F;
    for (int surface = 0; surface < half_surfaces; ++surface) {
        const float s = profiles.delta_s * (static_cast<float>(surface) + 0.5F);
        const float flux = std::min(torflux<float>(spec, s), 1.0F);
        const float pressure_flux = std::min(
            torflux<float>(
                spec, std::min(s, static_cast<float>(spec.physical.spres_ped))),
            1.0F);
        const float derivative = torflux_deriv<float>(spec, s);
        const float iota = eval_iota_profile<float>(spec, flux);
        profiles.iota_h[surface] = iota;
        profiles.pres_h[surface] =
            eval_mass_profile<float>(spec, pressure_flux);
        profiles.phip_h[surface] = max_toroidal_flux * derivative;
        profiles.sqrt_s_h[surface] = std::sqrt(s);
        profiles.curr_h[surface] =
            current_scale * eval_curr_profile<float>(spec, flux);
        profiles.chip_h[surface] = max_toroidal_flux * iota * derivative;
        phip_square_sum += profiles.phip_h[surface] * profiles.phip_h[surface];
    }
    profiles.lamscale = std::sqrt(phip_square_sum * profiles.delta_s);
    return profiles;
}

}  // namespace

AxisymmetricStageData initialize_stage(const ValidatedProblem& problem,
                                       std::size_t stage_index) {
    if (stage_index >= problem.stage_shapes().size()) {
        throw std::runtime_error("WebGPU stage index is out of range");
    }
    const GridShape& shape = problem.stage_shapes()[stage_index];
    const ProblemSpec& spec = problem.spec();
    const FoldedBoundary& boundary = problem.boundary();
    AxisymmetricStageData stage;
    stage.ns = shape.ns;
    stage.mpol = shape.mpol;
    stage.ntor = shape.ntor;
    stage.ntheta = shape.ntheta;
    stage.nzeta = shape.nzeta;
    stage.nfp = shape.nfp;
    stage.max_iterations =
        static_cast<int>(spec.stages[stage_index].max_iterations);
    stage.tolerance = spec.stages[stage_index].tolerance;
    stage.delta_t = static_cast<float>(spec.delt);
    stage.tcon0 = static_cast<float>(spec.physical.tcon0);
    stage.free_boundary = spec.free_boundary.lfreeb;
    const std::size_t family_values =
        static_cast<std::size_t>(shape.ns) * shape.modes();
    stage.state.assign(6 * family_values, 0.0F);
    stage.envelope_correction = static_cast<float>(
        default_seed_envelope(shape.ntor, spec.free_boundary.lfreeb, shape.ns,
                              static_cast<int>(spec.stages.size())));
    stage.lambda_seed_scale =
        static_cast<float>(default_axisymmetric_lambda_seed(
            shape.ntor, spec.free_boundary.lfreeb));

    auto family = [&](std::size_t component) {
        return std::span<float>(stage.state.data() + component * family_values,
                                family_values);
    };
    auto rcc = family(0);
    auto zsc = family(1);
    auto lsc = family(2);
    auto rss = family(3);
    auto zcs = family(4);
    for (int surface = 0; surface < shape.ns; ++surface) {
        const float s =
            static_cast<float>(surface) / static_cast<float>(shape.ns - 1);
        for (int m = 0; m < shape.mpol; ++m) {
            for (int n = 0; n <= shape.ntor; ++n) {
                const int mode = m * (shape.ntor + 1) + n;
                const std::size_t index =
                    static_cast<std::size_t>(mode) * shape.ns + surface;
                if (m == 0) {
                    rcc[index] =
                        s * static_cast<float>(boundary.rbcc[mode]) +
                        (1.0F - s) * static_cast<float>(spec.raxis_c[n]);
                    zcs[index] =
                        s * static_cast<float>(boundary.zbcs[mode]) -
                        (1.0F - s) * static_cast<float>(spec.zaxis_s[n]);
                    continue;
                }
                const float weight =
                    seed_radial_weight(m, s, stage.envelope_correction);
                rcc[index] = weight * static_cast<float>(boundary.rbcc[mode]);
                rss[index] = weight * static_cast<float>(boundary.rbss[mode]);
                zsc[index] = weight * static_cast<float>(boundary.zbsc[mode]);
                zcs[index] = weight * static_cast<float>(boundary.zbcs[mode]);
            }
        }
    }

    if (stage.lambda_seed_scale != 0.0F &&
        !seed_axisymmetric_lambda<float>(
            shape.ns, shape.mpol, rcc, zsc, lsc, boundary.rbcc, boundary.zbsc,
            spec.raxis_c[0], stage.envelope_correction,
            stage.lambda_seed_scale)) {
        stage.lambda_seed_scale = 0.0F;
        std::fill(lsc.begin(), lsc.end(), 0.0F);
    }
    stage.profiles = initialize_profiles(spec, shape.ns);
    return stage;
}

AxisymmetricStageData initialize_axisymmetric_stage(
    const ValidatedProblem& problem,
    std::size_t stage_index) {
    const auto& shape = problem.stage_shapes().at(stage_index);
    if (shape.ntor != 0 || shape.nzeta != 1) {
        throw std::runtime_error(
            "axisymmetric WebGPU initialization requires ntor=0 and nzeta=1");
    }
    return initialize_stage(problem, stage_index);
}

}  // namespace cumes::webgpu
