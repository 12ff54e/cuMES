#include "cumes/webgpu/initialization.hpp"

#include "cumes/config/validated_problem.hpp"
#include "cumes/state/axisymmetric_lambda_seed.hpp"
#include "cumes/state/seed_envelope.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>
#include <utility>

namespace cumes::webgpu {
namespace {

RadialProfiles initialize_profiles(const ProblemSpec& spec, int ns) {
    const auto minimum = [](auto a, auto b) { return std::min(a, b); };
    RadialProfiles profiles;
    static_cast<HostRadialProfiles<float>&>(profiles) =
        evaluate_radial_profiles<float>(spec, ns, minimum,
                                        /*skip_zero_current=*/false);
    const auto exact = evaluate_radial_profiles<double>(
        spec, ns, minimum, /*skip_zero_current=*/false);
    // Preserve the original f32 high words; the independently evaluated
    // double profiles supply only their low-word corrections.
    const auto low_words = [](std::vector<float>& low,
                              const std::vector<float>& high,
                              const std::vector<double>& exact) {
        low.resize(high.size());
        for (std::size_t i = 0; i < high.size(); ++i)
            low[i] = static_cast<float>(exact[i] - high[i]);
    };
    low_words(profiles.iota_f_lo, profiles.iota_f, exact.iota_f);
    low_words(profiles.phip_f_lo, profiles.phip_f, exact.phip_f);
    low_words(profiles.chi_f_lo, profiles.chi_f, exact.chi_f);
    low_words(profiles.sqrt_s_f_lo, profiles.sqrt_s_f, exact.sqrt_s_f);
    low_words(profiles.iota_h_lo, profiles.iota_h, exact.iota_h);
    low_words(profiles.pres_h_lo, profiles.pres_h, exact.pres_h);
    low_words(profiles.phip_h_lo, profiles.phip_h, exact.phip_h);
    low_words(profiles.sqrt_s_h_lo, profiles.sqrt_s_h, exact.sqrt_s_h);
    low_words(profiles.curr_h_lo, profiles.curr_h, exact.curr_h);
    low_words(profiles.chip_h_lo, profiles.chip_h, exact.chip_h);
    profiles.delta_s_lo = static_cast<float>(exact.delta_s - profiles.delta_s);
    profiles.lamscale_lo =
        static_cast<float>(exact.lamscale - profiles.lamscale);
    return profiles;
}

}  // namespace

AxisymmetricStageData initialize_stage(const ValidatedProblem& problem,
                                       std::size_t stage_index,
                                       bool float_radius_reference,
                                       bool compensated_geometry,
                                       bool compensated_toroidal_geometry) {
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
    stage.prescribed_current =
        spec.current_model == CurrentModel::PRESCRIBED_CURRENT;
    const std::size_t family_values =
        static_cast<std::size_t>(shape.ns) * shape.modes();
    stage.state.assign(6 * family_values, 0.0F);
    stage.state_lo.assign(6 * family_values, 0.0F);
    const double envelope_correction =
        default_seed_envelope(shape.ntor, spec.free_boundary.lfreeb, shape.ns,
                              static_cast<int>(spec.stages.size()));
    stage.envelope_correction = static_cast<float>(envelope_correction);
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
                    const double exact_s = static_cast<double>(surface) /
                                           static_cast<double>(shape.ns - 1);
                    stage.state_lo[index] =
                        static_cast<float>(exact_s * boundary.rbcc[mode] +
                                           (1.0 - exact_s) * spec.raxis_c[n] -
                                           static_cast<double>(rcc[index]));
                    stage.state_lo[4 * family_values + index] =
                        static_cast<float>(exact_s * boundary.zbcs[mode] -
                                           (1.0 - exact_s) * spec.zaxis_s[n] -
                                           static_cast<double>(zcs[index]));
                    continue;
                }
                const float weight =
                    seed_radial_weight(m, s, stage.envelope_correction);
                rcc[index] = weight * static_cast<float>(boundary.rbcc[mode]);
                rss[index] = weight * static_cast<float>(boundary.rbss[mode]);
                zsc[index] = weight * static_cast<float>(boundary.zbsc[mode]);
                zcs[index] = weight * static_cast<float>(boundary.zbcs[mode]);
                const double exact_s = static_cast<double>(surface) /
                                       static_cast<double>(shape.ns - 1);
                const double exact_weight =
                    seed_radial_weight(m, exact_s, envelope_correction);
                stage.state_lo[index] =
                    static_cast<float>(exact_weight * boundary.rbcc[mode] -
                                       static_cast<double>(rcc[index]));
                stage.state_lo[3 * family_values + index] =
                    static_cast<float>(exact_weight * boundary.rbss[mode] -
                                       static_cast<double>(rss[index]));
                stage.state_lo[family_values + index] =
                    static_cast<float>(exact_weight * boundary.zbsc[mode] -
                                       static_cast<double>(zsc[index]));
                stage.state_lo[4 * family_values + index] =
                    static_cast<float>(exact_weight * boundary.zbcs[mode] -
                                       static_cast<double>(zcs[index]));
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
    if ((float_radius_reference || compensated_geometry ||
         compensated_toroidal_geometry) &&
        (shape.ntor == 0 || spec.free_boundary.lfreeb)) {
        throw std::runtime_error(
            "float geometry fixes require fixed-boundary 3-D geometry");
    }
    stage.compensated_geometry =
        compensated_geometry || compensated_toroidal_geometry;
    stage.compensated_toroidal_geometry = compensated_toroidal_geometry;
    if (float_radius_reference) {
        auto reference = std::make_shared<FloatRadiusReference>();
        reference->coefficients.assign(boundary.rbcc.begin(),
                                       boundary.rbcc.begin() + shape.ntor + 1);
        reference->angular.resize(shape.ntheta * shape.nzeta);
        for (int zeta = 0; zeta < shape.nzeta; ++zeta) {
            const float angle = 2.0F * std::numbers::pi_v<float> *
                                static_cast<float>(zeta) /
                                static_cast<float>(shape.nzeta);
            // Setup-only wide accumulation of the same f32 products as the
            // inverse basis. No double arithmetic is sent to the GPU.
            double angular = 0.0;
            for (int n = 1; n <= shape.ntor; ++n) {
                angular += static_cast<double>(
                               static_cast<float>(reference->coefficients[n])) *
                           std::cos(static_cast<float>(n) * angle);
            }
            for (int theta = 0; theta < shape.ntheta; ++theta)
                reference->angular[zeta * shape.ntheta + theta] =
                    static_cast<float>(angular);
        }
        for (int n = 0; n <= shape.ntor; ++n) {
            for (int surface = 0; surface < shape.ns; ++surface) {
                const double s = static_cast<double>(surface) / (shape.ns - 1);
                const auto index = n * shape.ns + surface;
                // Subtract before conversion; subtracting two rounded radii
                // would already have lost the small displacement.
                stage.state[index] = static_cast<float>(
                    (1.0 - s) * (spec.raxis_c[n] - boundary.rbcc[n]));
                stage.state_lo[index] = 0.0F;
            }
        }
        stage.radius_reference = std::move(reference);
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
