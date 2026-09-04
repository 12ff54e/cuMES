#include "cumes/config/json_reader.hpp"
#include "cumes/webgpu/axisymmetric.hpp"
#include "cumes/webgpu/constraint.hpp"
#include "cumes/webgpu/force.hpp"
#include "cumes/webgpu/geometry.hpp"
#include "cumes/webgpu/initialization.hpp"
#include "cumes/webgpu/numerics.hpp"
#include "cumes/webgpu/prolongation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <emscripten.h>
#include <emscripten/em_js.h>
#include <webgpu/webgpu_cpp.h>

namespace {

EM_JS(void, publish_browser_result, (int success, const char* detail), {
    document.body.dataset.cumesWebgpu = success ? 'pass' : 'fail';
    document.body.dataset.cumesDetail = UTF8ToString(detail);
    clearTimeout(window.cumesDeadline);
    clearInterval(window.cumesKeepAlive);
});

std::string message_text(wgpu::StringView message) {
    return message.length == 0 ? std::string{}
                               : std::string(message.data, message.length);
}

class BrowserSelfTest : public std::enable_shared_from_this<BrowserSelfTest> {
   public:
    void start() {
        instance_ = wgpu::Instance(wgpuCreateInstance(nullptr));
        if (!instance_) {
            finish(false, "wgpuCreateInstance returned null");
            return;
        }
        std::printf("cuMES WebGPU milestone: requesting adapter\n");
        const auto self = shared_from_this();
        instance_.RequestAdapter(
            nullptr, wgpu::CallbackMode::AllowSpontaneous,
            [self](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter,
                   wgpu::StringView message) {
                if (status != wgpu::RequestAdapterStatus::Success) {
                    self->finish(false, "WebGPU adapter request failed: " +
                                            message_text(message));
                    return;
                }
                self->adapter_ = std::move(adapter);
                self->request_device();
            });
    }

   private:
    void request_device() {
        const auto self = shared_from_this();
        wgpu::DeviceDescriptor descriptor{};
        descriptor.label = "cuMES WebGPU device";
        descriptor.SetUncapturedErrorCallback(
            [](const wgpu::Device&, wgpu::ErrorType type,
               wgpu::StringView message, BrowserSelfTest* test) {
                test->gpu_error_ = "uncaptured WebGPU error (" +
                                   std::to_string(static_cast<unsigned>(type)) +
                                   "): " + message_text(message);
                std::fprintf(stderr, "%s\n", test->gpu_error_.c_str());
            },
            this);
        adapter_.RequestDevice(
            &descriptor, wgpu::CallbackMode::AllowSpontaneous,
            [self](wgpu::RequestDeviceStatus status, wgpu::Device device,
                   wgpu::StringView message) {
                if (status != wgpu::RequestDeviceStatus::Success) {
                    self->finish(false, "WebGPU device request failed: " +
                                            message_text(message));
                    return;
                }
                self->device_ = std::move(device);
                self->cases_ = make_cases();
                std::printf(
                    "adapter/device ready; running %zu radial-transfer "
                    "cases\n",
                    self->cases_.size());
                self->run_next();
            });
    }

    static std::vector<cumes::webgpu::ProlongationCase> make_cases() {
        std::vector<cumes::webgpu::ProlongationCase> cases;
        for (const auto interpolation :
             {cumes::webgpu::RadialInterpolation::LINEAR,
              cumes::webgpu::RadialInterpolation::CATMULL_ROM}) {
            cumes::webgpu::ProlongationCase test;
            test.ns_old = 5;
            test.ns_new = 8;
            test.ntor = 1;
            test.mnmax = 6;  // m=0..2, n=0..1 exercises true m parity.
            test.interpolation = interpolation;
            test.state.resize(6 * test.mnmax * test.ns_old);
            for (int family = 0; family < 6; ++family) {
                for (int mode = 0; mode < test.mnmax; ++mode) {
                    const bool odd = ((mode / (test.ntor + 1)) % 2) == 1;
                    for (int j = 0; j < test.ns_old; ++j) {
                        const float base =
                            0.4F * static_cast<float>(family + 1) +
                            0.2F * static_cast<float>(mode);
                        float value = base + 0.08F * static_cast<float>(j) +
                                      0.013F * static_cast<float>(j * j);
                        if (odd) {
                            const float s = static_cast<float>(j) /
                                            static_cast<float>(test.ns_old - 1);
                            const float radial = std::max(
                                std::sqrt(s),
                                std::sqrt(1.0F /
                                          static_cast<float>(test.ns_old - 1)));
                            value *= radial;
                            if (j == 0) value = 0.0F;
                        }
                        const auto index =
                            (static_cast<std::size_t>(family) * test.mnmax +
                             mode) *
                                test.ns_old +
                            j;
                        test.state[index] = value;
                    }
                }
            }
            cases.push_back(std::move(test));
        }
        return cases;
    }

    void run_next() {
        if (case_index_ == cases_.size()) {
            if (!gpu_error_.empty()) {
                finish(false, gpu_error_);
            } else {
                run_axisymmetric_inverse();
            }
            return;
        }

        const auto self = shared_from_this();
        const auto interpolation = cases_[case_index_].interpolation;
        cumes::webgpu::enqueue_prolongation(
            device_, cases_[case_index_],
            [self, interpolation](std::string error,
                                  cumes::webgpu::ProlongationResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected = cumes::webgpu::prolongation_reference(
                    self->cases_[self->case_index_]);
                if (actual.state.size() != expected.state.size() ||
                    actual.velocity.size() != expected.velocity.size()) {
                    self->finish(false, "WebGPU result shape mismatch");
                    return;
                }
                float max_error = 0.0F;
                for (std::size_t i = 0; i < actual.state.size(); ++i) {
                    max_error =
                        std::max(max_error,
                                 std::abs(actual.state[i] - expected.state[i]));
                }
                const bool velocity_zero =
                    std::all_of(actual.velocity.begin(), actual.velocity.end(),
                                [](float value) { return value == 0.0F; });
                if (max_error > 4.0e-6F || !velocity_zero) {
                    self->finish(
                        false,
                        "radial-transfer mismatch: max_error=" +
                            std::to_string(max_error) + " velocity_zero=" +
                            std::string(velocity_zero ? "true" : "false"));
                    return;
                }
                std::printf(
                    "  %s: PASS (max |GPU-CPU| = %.3e)\n",
                    interpolation == cumes::webgpu::RadialInterpolation::LINEAR
                        ? "linear"
                        : "Catmull-Rom",
                    static_cast<double>(max_error));
                ++self->case_index_;
                self->run_next();
            });
    }

    static cumes::webgpu::AxisymmetricInverseCase make_axisymmetric_case() {
        cumes::webgpu::AxisymmetricInverseCase test;
        test.ns = 5;
        test.mpol = 6;
        test.ntheta = 18;
        test.state.resize(cumes::webgpu::SPECTRAL_COMPONENT_COUNT * test.mpol *
                          test.ns);
        for (std::size_t component = 0;
             component < cumes::webgpu::SPECTRAL_COMPONENT_COUNT; ++component) {
            for (int mode = 0; mode < test.mpol; ++mode) {
                for (int surface = 0; surface < test.ns; ++surface) {
                    const auto index =
                        (component * test.mpol + mode) * test.ns + surface;
                    test.state[index] =
                        0.17F * static_cast<float>(component + 1) +
                        0.031F * static_cast<float>(mode * mode + 1) +
                        0.043F * static_cast<float>(surface) -
                        0.007F * static_cast<float>(mode * surface);
                }
            }
        }
        return test;
    }

    void run_axisymmetric_inverse() {
        axisymmetric_case_ = make_axisymmetric_case();
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_axisymmetric_inverse(
            device_, axisymmetric_case_,
            [self](std::string error,
                   cumes::webgpu::AxisymmetricInverseResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected =
                    cumes::webgpu::axisymmetric_inverse_reference(
                        self->axisymmetric_case_);
                if (actual.geometry.size() != expected.geometry.size() ||
                    actual.r_con.size() != expected.r_con.size() ||
                    actual.z_con.size() != expected.z_con.size()) {
                    self->finish(false,
                                 "axisymmetric inverse result shape mismatch");
                    return;
                }
                float max_error = 0.0F;
                const auto compare = [&max_error](const auto& gpu,
                                                  const auto& cpu) {
                    for (std::size_t i = 0; i < gpu.size(); ++i) {
                        max_error =
                            std::max(max_error, std::abs(gpu[i] - cpu[i]));
                    }
                };
                compare(actual.geometry, expected.geometry);
                compare(actual.r_con, expected.r_con);
                compare(actual.z_con, expected.z_con);

                const std::size_t points =
                    static_cast<std::size_t>(self->axisymmetric_case_.ns) *
                    self->axisymmetric_case_.ntheta;
                bool toroidal_derivatives_zero = true;
                for (std::size_t field = 12;
                     field < cumes::webgpu::GEOMETRY_PARITY_FIELD_COUNT;
                     ++field) {
                    const auto first = actual.geometry.begin() + field * points;
                    toroidal_derivatives_zero &=
                        std::all_of(first, first + points,
                                    [](float value) { return value == 0.0F; });
                }
                if (max_error > 1.0e-4F || !toroidal_derivatives_zero) {
                    self->finish(
                        false,
                        "axisymmetric inverse mismatch: max_error=" +
                            std::to_string(max_error) +
                            " toroidal_derivatives_zero=" +
                            (toroidal_derivatives_zero ? "true" : "false"));
                    return;
                }
                std::printf(
                    "  axisymmetric inverse+rCon/zCon: PASS "
                    "(max |GPU-CPU| = %.3e)\n",
                    static_cast<double>(max_error));
                if (!self->gpu_error_.empty()) {
                    self->finish(false, self->gpu_error_);
                    return;
                }
                self->forward_cases_ = make_forward_cases();
                self->run_axisymmetric_forward();
            });
    }

    static std::vector<cumes::webgpu::AxisymmetricForwardCase>
    make_forward_cases() {
        cumes::webgpu::AxisymmetricForwardCase base;
        base.ns = 5;
        base.mpol = 6;
        base.ntheta = 18;
        const std::size_t points =
            static_cast<std::size_t>(base.ns) * base.ntheta;
        base.fields.resize(cumes::webgpu::FORWARD_INPUT_FIELD_COUNT * points);
        for (std::size_t field = 0;
             field < cumes::webgpu::FORWARD_INPUT_FIELD_COUNT; ++field) {
            for (int surface = 0; surface < base.ns; ++surface) {
                for (int theta = 0; theta < base.ntheta; ++theta) {
                    const auto index =
                        field * points +
                        static_cast<std::size_t>(surface) * base.ntheta + theta;
                    base.fields[index] =
                        0.023F * static_cast<float>(field + 1) +
                        0.017F * static_cast<float>(surface + 1) +
                        0.009F * static_cast<float>((theta * 5 + field) % 11);
                }
            }
        }
        auto include_lcfs = base;
        include_lcfs.include_lcfs = true;
        return {std::move(base), std::move(include_lcfs)};
    }

    void run_axisymmetric_forward() {
        if (forward_case_index_ == forward_cases_.size()) {
            run_axisymmetric_dealias();
            return;
        }
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_axisymmetric_forward(
            device_, forward_cases_[forward_case_index_],
            [self](std::string error,
                   cumes::webgpu::AxisymmetricForwardResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto& test =
                    self->forward_cases_[self->forward_case_index_];
                const auto expected =
                    cumes::webgpu::axisymmetric_forward_reference(test);
                if (actual.residual.size() != expected.residual.size()) {
                    self->finish(false,
                                 "axisymmetric forward result shape mismatch");
                    return;
                }
                float max_error = 0.0F;
                for (std::size_t i = 0; i < actual.residual.size(); ++i) {
                    max_error = std::max(
                        max_error,
                        std::abs(actual.residual[i] - expected.residual[i]));
                }
                const auto value = [&test, &actual](int component, int mode,
                                                    int surface) {
                    return actual
                        .residual[(static_cast<std::size_t>(component) *
                                       test.mpol +
                                   mode) *
                                      test.ns +
                                  surface];
                };
                bool boundary_contract = true;
                for (int component = 3; component < 6; ++component) {
                    for (int mode = 0; mode < test.mpol; ++mode) {
                        for (int surface = 0; surface < test.ns; ++surface) {
                            boundary_contract &=
                                value(component, mode, surface) == 0.0F;
                        }
                    }
                }
                for (int mode = 0; mode < test.mpol; ++mode) {
                    boundary_contract &= value(1, mode, 0) == 0.0F;
                    boundary_contract &= value(2, mode, 0) == 0.0F;
                    if (mode > 0) {
                        boundary_contract &= value(0, mode, 0) == 0.0F;
                    }
                    if (!test.include_lcfs) {
                        boundary_contract &=
                            value(0, mode, test.ns - 1) == 0.0F;
                        boundary_contract &=
                            value(1, mode, test.ns - 1) == 0.0F;
                    }
                }
                if (max_error > 1.0e-4F || !boundary_contract) {
                    self->finish(false,
                                 "axisymmetric forward mismatch: max_error=" +
                                     std::to_string(max_error) +
                                     " boundary_contract=" +
                                     (boundary_contract ? "true" : "false"));
                    return;
                }
                std::printf(
                    "  axisymmetric forward (%s LCFS): PASS "
                    "(max |GPU-CPU| = %.3e)\n",
                    test.include_lcfs ? "included" : "fixed",
                    static_cast<double>(max_error));
                ++self->forward_case_index_;
                self->run_axisymmetric_forward();
            });
    }

    static cumes::webgpu::AxisymmetricDealiasCase make_dealias_case() {
        cumes::webgpu::AxisymmetricDealiasCase test;
        test.ns = 5;
        test.mpol = 6;
        test.ntheta = 18;
        const std::size_t points =
            static_cast<std::size_t>(test.ns) * test.ntheta;
        test.g_con_eff.resize(points);
        for (std::size_t i = 0; i < points; ++i) {
            test.g_con_eff[i] =
                std::sin(0.21F * static_cast<float>(i)) +
                0.13F * std::cos(0.07F * static_cast<float>(i * i));
        }
        test.tcon.resize(test.ns);
        for (int surface = 0; surface < test.ns; ++surface) {
            test.tcon[surface] = 0.5F + 0.1F * static_cast<float>(surface);
        }
        test.faccon.resize(test.mpol, 0.0F);
        for (int mode = 1; mode < test.mpol; ++mode) {
            const float xmpq = static_cast<float>((mode + 1) * mode);
            test.faccon[mode] = 0.25F / (xmpq * xmpq);
        }
        return test;
    }

    void run_axisymmetric_dealias() {
        dealias_case_ = make_dealias_case();
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_axisymmetric_dealias(
            device_, dealias_case_,
            [self](std::string error,
                   cumes::webgpu::AxisymmetricDealiasResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected =
                    cumes::webgpu::axisymmetric_dealias_reference(
                        self->dealias_case_);
                if (actual.g_con.size() != expected.g_con.size()) {
                    self->finish(false,
                                 "axisymmetric dealias result shape mismatch");
                    return;
                }
                float max_error = 0.0F;
                for (std::size_t i = 0; i < actual.g_con.size(); ++i) {
                    max_error =
                        std::max(max_error,
                                 std::abs(actual.g_con[i] - expected.g_con[i]));
                }
                const bool axis_zero = std::all_of(
                    actual.g_con.begin(),
                    actual.g_con.begin() + self->dealias_case_.ntheta,
                    [](float value) { return value == 0.0F; });
                if (max_error > 1.0e-4F || !axis_zero) {
                    self->finish(false,
                                 "axisymmetric dealias mismatch: max_error=" +
                                     std::to_string(max_error) + " axis_zero=" +
                                     (axis_zero ? "true" : "false"));
                    return;
                }
                std::printf(
                    "  axisymmetric constraint dealias: PASS "
                    "(max |GPU-CPU| = %.3e)\n",
                    static_cast<double>(max_error));
                if (!self->gpu_error_.empty()) {
                    self->finish(false, self->gpu_error_);
                    return;
                }
                self->run_solovev_initialization();
            });
    }

    void run_solovev_initialization() {
        try {
            cumes::SolverOptions options;
            options.precision = cumes::PrecisionPolicy::MIXED_FLOAT;
            auto parsed =
                cumes::read_problem_spec("/inputs/solovev.json", options);
            if (parsed.report.has_errors()) {
                const auto errors = parsed.report.errors();
                finish(false, "Solovev JSON mapping failed: " + errors.front());
                return;
            }
            for (auto& stage : parsed.spec.stages) {
                stage.tolerance = std::max(
                    stage.tolerance, cumes::tolerance_floor(options.precision));
            }
            auto validated = cumes::validate(std::move(parsed.spec), options);
            if (!validated.has_value()) {
                const auto errors = validated.error().errors();
                finish(false, "Solovev validation failed: " +
                                  (errors.empty() ? std::string("unknown error")
                                                  : errors.front()));
                return;
            }
            const auto& boundary = validated.value().boundary();
            initialized_stage_ = cumes::webgpu::initialize_axisymmetric_stage(
                validated.value(), 0);
            const auto& stage = initialized_stage_;
            const std::size_t family_values =
                static_cast<std::size_t>(stage.ns) * stage.mpol;
            bool state_contract =
                stage.ns == 5 && stage.mpol == 6 && stage.ntheta == 18 &&
                stage.state.size() ==
                    cumes::webgpu::SPECTRAL_COMPONENT_COUNT * family_values &&
                stage.envelope_correction == -0.07F &&
                stage.lambda_seed_scale == 0.65F;
            for (int mode = 0; mode < stage.mpol; ++mode) {
                const std::size_t lcfs =
                    static_cast<std::size_t>(mode) * stage.ns + stage.ns - 1;
                state_contract &= stage.state[lcfs] ==
                                  static_cast<float>(boundary.rbcc[mode]);
                state_contract &= stage.state[family_values + lcfs] ==
                                  static_cast<float>(boundary.zbsc[mode]);
                if (mode > 0) {
                    const std::size_t axis =
                        static_cast<std::size_t>(mode) * stage.ns;
                    for (std::size_t component = 0;
                         component < cumes::webgpu::SPECTRAL_COMPONENT_COUNT;
                         ++component) {
                        state_contract &=
                            stage.state[component * family_values + axis] ==
                            0.0F;
                    }
                }
            }
            const auto& profiles = stage.profiles;
            const bool profile_contract =
                profiles.iota_f.size() == static_cast<std::size_t>(stage.ns) &&
                profiles.phip_f.size() == static_cast<std::size_t>(stage.ns) &&
                profiles.iota_h.size() ==
                    static_cast<std::size_t>(stage.ns - 1) &&
                profiles.pres_h.size() ==
                    static_cast<std::size_t>(stage.ns - 1) &&
                profiles.delta_s == 0.25F && profiles.lamscale > 0.0F &&
                std::all_of(profiles.iota_f.begin(), profiles.iota_f.end(),
                            [](float value) { return value == 1.0F; });
            if (!state_contract || !profile_contract) {
                finish(false,
                       "Solovev initialization violated state/profile layout "
                       "contract");
                return;
            }

            cumes::webgpu::AxisymmetricInverseCase inverse;
            inverse.ns = stage.ns;
            inverse.mpol = stage.mpol;
            inverse.ntheta = stage.ntheta;
            inverse.state = stage.state;
            const auto self = shared_from_this();
            cumes::webgpu::enqueue_axisymmetric_inverse(
                device_, inverse,
                [self, inverse](
                    std::string error,
                    cumes::webgpu::AxisymmetricInverseResult actual) {
                    if (!error.empty()) {
                        self->finish(false, std::move(error));
                        return;
                    }
                    const auto expected =
                        cumes::webgpu::axisymmetric_inverse_reference(inverse);
                    float max_error = 0.0F;
                    const auto compare = [&max_error](const auto& gpu,
                                                      const auto& cpu) {
                        if (gpu.size() != cpu.size()) {
                            max_error = std::numeric_limits<float>::infinity();
                            return;
                        }
                        for (std::size_t i = 0; i < gpu.size(); ++i) {
                            max_error =
                                std::max(max_error, std::abs(gpu[i] - cpu[i]));
                        }
                    };
                    compare(actual.geometry, expected.geometry);
                    compare(actual.r_con, expected.r_con);
                    compare(actual.z_con, expected.z_con);
                    if (max_error > 1.0e-4F) {
                        self->finish(
                            false,
                            "initialized Solovev inverse mismatch: max_error=" +
                                std::to_string(max_error));
                        return;
                    }
                    std::printf(
                        "  parsed Solovev seed+profiles+inverse: PASS "
                        "(max |GPU-CPU| = %.3e)\n",
                        static_cast<double>(max_error));
                    self->solovev_r_con_ = std::move(actual.r_con);
                    self->solovev_z_con_ = std::move(actual.z_con);
                    self->run_base_geometry(std::move(actual.geometry));
                });
        } catch (const std::exception& error) {
            finish(false, "Solovev initialization failed: " +
                              std::string(error.what()));
        }
    }

    void run_base_geometry(std::vector<float> geometry) {
        base_geometry_case_.ns = initialized_stage_.ns;
        base_geometry_case_.ntheta = initialized_stage_.ntheta;
        base_geometry_case_.delta_s = initialized_stage_.profiles.delta_s;
        base_geometry_case_.geometry = std::move(geometry);
        base_geometry_case_.sqrt_s_f = initialized_stage_.profiles.sqrt_s_f;
        base_geometry_case_.sqrt_s_h = initialized_stage_.profiles.sqrt_s_h;
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_base_geometry(
            device_, base_geometry_case_,
            [self](std::string error,
                   cumes::webgpu::BaseGeometryResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected = cumes::webgpu::base_geometry_reference(
                    self->base_geometry_case_);
                if (actual.fields.size() != expected.fields.size()) {
                    self->finish(false, "base geometry result shape mismatch");
                    return;
                }
                float max_error = 0.0F;
                for (std::size_t i = 0; i < actual.fields.size(); ++i) {
                    max_error = std::max(
                        max_error,
                        std::abs(actual.fields[i] - expected.fields[i]));
                }
                const std::size_t half_points =
                    static_cast<std::size_t>(self->base_geometry_case_.ns - 1) *
                    self->base_geometry_case_.ntheta;
                const auto guv = actual.fields.begin() + 8 * half_points;
                const bool axisymmetric_guv_zero =
                    std::all_of(guv, guv + half_points,
                                [](float value) { return value == 0.0F; });
                const auto gsqrt = actual.fields.begin() + 6 * half_points;
                const bool finite_jacobian =
                    std::all_of(gsqrt, gsqrt + half_points, [](float value) {
                        return std::isfinite(value) && value != 0.0F;
                    });
                if (max_error > 2.0e-4F || !axisymmetric_guv_zero ||
                    !finite_jacobian) {
                    self->finish(
                        false, "base geometry mismatch: max_error=" +
                                   std::to_string(max_error) + " guv_zero=" +
                                   (axisymmetric_guv_zero ? "true" : "false") +
                                   " finite_jacobian=" +
                                   (finite_jacobian ? "true" : "false"));
                    return;
                }
                std::printf(
                    "  Solovev half-grid base geometry: PASS "
                    "(max |GPU-CPU| = %.3e)\n",
                    static_cast<double>(max_error));
                self->run_magnetic_field(std::move(actual.fields));
            });
    }

    void run_magnetic_field(std::vector<float> base_geometry) {
        magnetic_field_case_.ns = initialized_stage_.ns;
        magnetic_field_case_.ntheta = initialized_stage_.ntheta;
        magnetic_field_case_.lamscale = initialized_stage_.profiles.lamscale;
        magnetic_field_case_.geometry = base_geometry_case_.geometry;
        magnetic_field_case_.base_geometry = std::move(base_geometry);
        magnetic_field_case_.sqrt_s_h = initialized_stage_.profiles.sqrt_s_h;
        magnetic_field_case_.phip_f = initialized_stage_.profiles.phip_f;
        magnetic_field_case_.chip_h = initialized_stage_.profiles.chip_h;
        magnetic_field_case_.pres_h = initialized_stage_.profiles.pres_h;
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_magnetic_field(
            device_, magnetic_field_case_,
            [self](std::string error,
                   cumes::webgpu::MagneticFieldResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected = cumes::webgpu::magnetic_field_reference(
                    self->magnetic_field_case_);
                if (actual.fields.size() != expected.fields.size()) {
                    self->finish(false, "magnetic field result shape mismatch");
                    return;
                }
                float max_error = 0.0F;
                bool finite = true;
                for (std::size_t i = 0; i < actual.fields.size(); ++i) {
                    max_error = std::max(
                        max_error,
                        std::abs(actual.fields[i] - expected.fields[i]));
                    finite &= std::isfinite(actual.fields[i]);
                }
                if (max_error > 2.0e-4F || !finite) {
                    self->finish(false,
                                 "magnetic field mismatch: max_error=" +
                                     std::to_string(max_error) +
                                     " finite=" + (finite ? "true" : "false"));
                    return;
                }
                std::printf(
                    "  Solovev magnetic field+pressure: PASS "
                    "(max |GPU-CPU| = %.3e)\n",
                    static_cast<double>(max_error));
                self->run_axisymmetric_force(std::move(actual.fields));
            });
    }

    void run_axisymmetric_force(std::vector<float> magnetic_field) {
        force_case_.ns = initialized_stage_.ns;
        force_case_.ntheta = initialized_stage_.ntheta;
        force_case_.delta_s = initialized_stage_.profiles.delta_s;
        force_case_.lamscale = initialized_stage_.profiles.lamscale;
        force_case_.geometry = magnetic_field_case_.geometry;
        force_case_.base_geometry = magnetic_field_case_.base_geometry;
        force_case_.magnetic_field = std::move(magnetic_field);
        force_case_.sqrt_s_f = initialized_stage_.profiles.sqrt_s_f;
        force_case_.sqrt_s_h = initialized_stage_.profiles.sqrt_s_h;
        force_case_.phip_f = initialized_stage_.profiles.phip_f;
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_axisymmetric_force(
            device_, force_case_,
            [self](std::string error,
                   cumes::webgpu::AxisymmetricForceResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected =
                    cumes::webgpu::axisymmetric_force_reference(
                        self->force_case_);
                if (actual.fields.size() != expected.fields.size()) {
                    self->finish(false, "force result shape mismatch");
                    return;
                }
                float max_error = 0.0F;
                bool finite = true;
                for (std::size_t i = 0; i < actual.fields.size(); ++i) {
                    max_error = std::max(
                        max_error,
                        std::abs(actual.fields[i] - expected.fields[i]));
                    finite &= std::isfinite(actual.fields[i]);
                }
                if (max_error > 5.0e-4F || !finite) {
                    self->finish(false, "axisymmetric force mismatch: " +
                                            std::to_string(max_error));
                    return;
                }
                std::printf(
                    "  Solovev MHD force: PASS "
                    "(max |GPU-CPU| = %.3e)\n",
                    static_cast<double>(max_error));
                self->run_solovev_forward(std::move(actual.fields));
            });
    }

    void run_solovev_forward(std::vector<float> force_fields) {
        solver_forward_case_.ns = initialized_stage_.ns;
        solver_forward_case_.mpol = initialized_stage_.mpol;
        solver_forward_case_.ntheta = initialized_stage_.ntheta;
        solver_forward_case_.include_lcfs = false;
        const std::size_t points =
            static_cast<std::size_t>(solver_forward_case_.ns) *
            solver_forward_case_.ntheta;
        solver_forward_case_.fields.assign(
            cumes::webgpu::FORWARD_INPUT_FIELD_COUNT * points, 0.0F);
        std::copy(force_fields.begin(), force_fields.end(),
                  solver_forward_case_.fields.begin());
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_axisymmetric_forward(
            device_, solver_forward_case_,
            [self](std::string error,
                   cumes::webgpu::AxisymmetricForwardResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected =
                    cumes::webgpu::axisymmetric_forward_reference(
                        self->solver_forward_case_);
                float max_error = 0.0F;
                bool finite =
                    actual.residual.size() == expected.residual.size();
                if (finite) {
                    for (std::size_t i = 0; i < actual.residual.size(); ++i) {
                        max_error =
                            std::max(max_error, std::abs(actual.residual[i] -
                                                         expected.residual[i]));
                        finite &= std::isfinite(actual.residual[i]);
                    }
                }
                const bool nonzero =
                    std::any_of(actual.residual.begin(), actual.residual.end(),
                                [](float value) { return value != 0.0F; });
                if (max_error > 5.0e-4F || !finite || !nonzero) {
                    self->finish(false, "Solovev forward residual mismatch: " +
                                            std::to_string(max_error));
                    return;
                }
                std::printf(
                    "  Solovev spectral residual projection: PASS "
                    "(max |GPU-CPU| = %.3e)\n",
                    static_cast<double>(max_error));
                self->run_residual_decomposition(std::move(actual.residual));
            });
    }

    void run_residual_decomposition(std::vector<float> residual) {
        residual_case_.ns = initialized_stage_.ns;
        residual_case_.mpol = initialized_stage_.mpol;
        residual_case_.include_edge_rz = false;
        residual_case_.zero_m1_z = true;
        residual_case_.residual = std::move(residual);
        residual_case_.sqrt_s_f = initialized_stage_.profiles.sqrt_s_f;
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_residual_decomposition(
            device_, residual_case_,
            [self](std::string error,
                   cumes::webgpu::ResidualDecompositionResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected =
                    cumes::webgpu::residual_decomposition_reference(
                        self->residual_case_);
                float max_error = 0.0F;
                bool valid = actual.residual.size() == expected.residual.size();
                if (valid) {
                    for (std::size_t i = 0; i < actual.residual.size(); ++i) {
                        max_error =
                            std::max(max_error, std::abs(actual.residual[i] -
                                                         expected.residual[i]));
                    }
                }
                for (int group = 0; group < 3; ++group) {
                    valid &= actual.raw_norm[group] == expected.raw_norm[group];
                    valid &= std::isfinite(actual.raw_norm[group]);
                }
                if (!valid || max_error > 5.0e-4F) {
                    self->finish(false, "residual decomposition mismatch: " +
                                            std::to_string(max_error));
                    return;
                }
                std::printf(
                    "  decomposed residual+double norms: PASS "
                    "(max |GPU-CPU| = %.3e)\n",
                    static_cast<double>(max_error));
                self->run_axisymmetric_constraint();
            });
    }

    void run_axisymmetric_constraint() {
        constraint_case_.ns = initialized_stage_.ns;
        constraint_case_.mpol = initialized_stage_.mpol;
        constraint_case_.ntheta = initialized_stage_.ntheta;
        constraint_case_.delta_s = initialized_stage_.profiles.delta_s;
        constraint_case_.tcon0 = 1.0F;
        constraint_case_.reset_reference = true;
        constraint_case_.refresh_preconditioner = true;
        constraint_case_.geometry = base_geometry_case_.geometry;
        constraint_case_.r_con = solovev_r_con_;
        constraint_case_.z_con = solovev_z_con_;
        const std::size_t points =
            static_cast<std::size_t>(constraint_case_.ns) *
            constraint_case_.ntheta;
        constraint_case_.r_con0.assign(points, 0.0F);
        constraint_case_.z_con0.assign(points, 0.0F);
        constraint_case_.tcon.assign(constraint_case_.ns, 0.0F);
        constraint_case_.ard.resize(2 * constraint_case_.ns);
        constraint_case_.azd.resize(2 * constraint_case_.ns);
        for (int surface = 0; surface < constraint_case_.ns; ++surface) {
            constraint_case_.ard[2 * surface] =
                0.25F + 0.03F * static_cast<float>(surface);
            constraint_case_.ard[2 * surface + 1] = 0.0F;
            constraint_case_.azd[2 * surface] =
                0.30F + 0.02F * static_cast<float>(surface);
            constraint_case_.azd[2 * surface + 1] = 0.0F;
        }
        constraint_case_.sqrt_s_f = initialized_stage_.profiles.sqrt_s_f;
        constraint_case_.force_fields.assign(
            solver_forward_case_.fields.begin(),
            solver_forward_case_.fields.begin() + 10 * points);
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_axisymmetric_constraint(
            device_, constraint_case_,
            [self](std::string error,
                   cumes::webgpu::AxisymmetricConstraintResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected =
                    cumes::webgpu::axisymmetric_constraint_reference(
                        self->constraint_case_);
                float max_error = 0.0F;
                bool valid = true;
                const auto compare = [&max_error, &valid](const auto& gpu,
                                                          const auto& cpu) {
                    valid &= gpu.size() == cpu.size();
                    if (gpu.size() != cpu.size()) return;
                    for (std::size_t i = 0; i < gpu.size(); ++i) {
                        max_error =
                            std::max(max_error, std::abs(gpu[i] - cpu[i]));
                        valid &= std::isfinite(gpu[i]);
                    }
                };
                compare(actual.fields, expected.fields);
                compare(actual.r_con0, expected.r_con0);
                compare(actual.z_con0, expected.z_con0);
                compare(actual.tcon, expected.tcon);
                compare(actual.g_con_eff, expected.g_con_eff);
                compare(actual.g_con, expected.g_con);
                const bool active =
                    std::any_of(actual.g_con.begin(), actual.g_con.end(),
                                [](float value) { return value != 0.0F; });
                if (!valid || !active || max_error > 1.0e-3F) {
                    self->finish(false, "axisymmetric constraint mismatch: " +
                                            std::to_string(max_error));
                    return;
                }
                std::printf(
                    "  Solovev constraint refresh+force: PASS "
                    "(max |GPU-CPU| = %.3e)\n",
                    static_cast<double>(max_error));
                self->run_constraint_forward(std::move(actual.fields));
            });
    }

    void run_constraint_forward(std::vector<float> fields) {
        constraint_forward_case_.ns = initialized_stage_.ns;
        constraint_forward_case_.mpol = initialized_stage_.mpol;
        constraint_forward_case_.ntheta = initialized_stage_.ntheta;
        constraint_forward_case_.include_lcfs = false;
        constraint_forward_case_.fields = std::move(fields);
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_axisymmetric_forward(
            device_, constraint_forward_case_,
            [self](std::string error,
                   cumes::webgpu::AxisymmetricForwardResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected =
                    cumes::webgpu::axisymmetric_forward_reference(
                        self->constraint_forward_case_);
                float max_error = 0.0F;
                bool valid = actual.residual.size() == expected.residual.size();
                if (valid) {
                    for (std::size_t i = 0; i < actual.residual.size(); ++i) {
                        max_error =
                            std::max(max_error, std::abs(actual.residual[i] -
                                                         expected.residual[i]));
                        valid &= std::isfinite(actual.residual[i]);
                    }
                }
                if (!valid || max_error > 5.0e-4F) {
                    self->finish(false,
                                 "constraint residual projection mismatch: " +
                                     std::to_string(max_error));
                    return;
                }
                std::printf(
                    "  constrained spectral residual projection: PASS "
                    "(max |GPU-CPU| = %.3e)\n",
                    static_cast<double>(max_error));
                self->finish(
                    true,
                    "parsed Solovev DAG through refreshed constraint force "
                    "and invariant residual verified");
            });
    }

    static void finish(bool success, const std::string& detail) {
        std::printf("cuMES WebGPU self-test: %s — %s\n",
                    success ? "PASS" : "FAIL", detail.c_str());
        std::fflush(stdout);
        std::fflush(stderr);
        publish_browser_result(success ? 1 : 0, detail.c_str());
        emscripten_force_exit(success ? 0 : 1);
    }

    wgpu::Instance instance_;
    wgpu::Adapter adapter_;
    wgpu::Device device_;
    std::vector<cumes::webgpu::ProlongationCase> cases_;
    cumes::webgpu::AxisymmetricInverseCase axisymmetric_case_;
    std::vector<cumes::webgpu::AxisymmetricForwardCase> forward_cases_;
    cumes::webgpu::AxisymmetricDealiasCase dealias_case_;
    cumes::webgpu::AxisymmetricStageData initialized_stage_;
    cumes::webgpu::BaseGeometryCase base_geometry_case_;
    cumes::webgpu::MagneticFieldCase magnetic_field_case_;
    cumes::webgpu::AxisymmetricForceCase force_case_;
    cumes::webgpu::AxisymmetricForwardCase solver_forward_case_;
    cumes::webgpu::ResidualDecompositionCase residual_case_;
    cumes::webgpu::AxisymmetricConstraintCase constraint_case_;
    cumes::webgpu::AxisymmetricForwardCase constraint_forward_case_;
    std::vector<float> solovev_r_con_;
    std::vector<float> solovev_z_con_;
    std::size_t case_index_ = 0;
    std::size_t forward_case_index_ = 0;
    std::string gpu_error_;
};

}  // namespace

int main() {
    std::make_shared<BrowserSelfTest>()->start();
    return 0;
}
