#include "cumes/config/device_params.hpp"
#include "cumes/config/json_reader.hpp"
#include "cumes/io/derived_fields.hpp"
#include "cumes/io/reader.hpp"
#include "cumes/io/writer.hpp"
#include "cumes/solver/iteration_controller.hpp"
#include "cumes/webgpu/axisymmetric.hpp"
#include "cumes/webgpu/constraint.hpp"
#include "cumes/webgpu/descent.hpp"
#include "cumes/webgpu/float_float.hpp"
#include "cumes/webgpu/force.hpp"
#include "cumes/webgpu/geometry.hpp"
#include "cumes/webgpu/initialization.hpp"
#include "cumes/webgpu/numerics.hpp"
#include "cumes/webgpu/preconditioner.hpp"
#include "cumes/webgpu/prolongation.hpp"
#include "cumes/webgpu/toroidal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
#include <iomanip>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <emscripten.h>
#include <webgpu/webgpu_cpp.h>

namespace {

extern "C" {
void publish_browser_result(int success, const char* detail);
int publish_browser_output(const char* path);
int requested_w7x_solve();
int requested_app_mode();
int requested_app_run();
void publish_browser_ready();
void publish_browser_equilibrium(const char* json);
void publish_browser_adapter(const char* device,
                             const char* type,
                             const char* backend);
}

#ifndef CUMES_GIT_REVISION
#define CUMES_GIT_REVISION ""
#endif
#ifndef CUMES_GIT_DIRTY
#define CUMES_GIT_DIRTY 0
#endif
#ifndef CUMES_BUILD_TYPE
#define CUMES_BUILD_TYPE ""
#endif
#ifndef CUMES_PRECISION_POLICY_NAME
#define CUMES_PRECISION_POLICY_NAME "mixed-float"
#endif
#ifndef CUMES_PRECISION_FLAGS
#define CUMES_PRECISION_FLAGS ""
#endif

// The CUDA mixed-float W7-X trajectory has an empirically qualified residual
// floor near 3e-3. Its regression capture therefore uses 1e-2; WebGPU is f32
// and must use the same achievable solver contract.
constexpr double W7X_MIXED_FLOAT_TOLERANCE = 1.0e-2;

std::string message_text(wgpu::StringView message) {
    return message.length == 0 ? std::string{}
                               : std::string(message.data, message.length);
}

const char* adapter_type_name(wgpu::AdapterType type) {
    switch (type) {
        case wgpu::AdapterType::DiscreteGPU:
            return "discrete-gpu";
        case wgpu::AdapterType::IntegratedGPU:
            return "integrated-gpu";
        case wgpu::AdapterType::CPU:
            return "cpu";
        case wgpu::AdapterType::Unknown:
            return "unknown";
    }
    return "unknown";
}

const char* backend_type_name(wgpu::BackendType type) {
    switch (type) {
        case wgpu::BackendType::Undefined:
            return "undefined";
        case wgpu::BackendType::Null:
            return "null";
        case wgpu::BackendType::WebGPU:
            return "webgpu";
        case wgpu::BackendType::D3D11:
            return "d3d11";
        case wgpu::BackendType::D3D12:
            return "d3d12";
        case wgpu::BackendType::Metal:
            return "metal";
        case wgpu::BackendType::Vulkan:
            return "vulkan";
        case wgpu::BackendType::OpenGL:
            return "opengl";
        case wgpu::BackendType::OpenGLES:
            return "opengles";
    }
    return "unknown";
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
        wgpu::RequestAdapterOptions options{};
        options.powerPreference = wgpu::PowerPreference::HighPerformance;
        instance_.RequestAdapter(
            &options, wgpu::CallbackMode::AllowSpontaneous,
            [self](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter,
                   wgpu::StringView message) {
                if (status != wgpu::RequestAdapterStatus::Success) {
                    self->finish(false, "WebGPU adapter request failed: " +
                                            message_text(message));
                    return;
                }
                self->adapter_ = std::move(adapter);
                wgpu::AdapterInfo info{};
                if (self->adapter_.GetInfo(&info)) {
                    const std::string device_name = message_text(info.device);
                    const char* type_name = adapter_type_name(info.adapterType);
                    const char* backend_name =
                        backend_type_name(info.backendType);
                    std::printf("adapter selected: %s (%s, %s)\n",
                                device_name.c_str(), type_name, backend_name);
                    publish_browser_adapter(device_name.c_str(), type_name,
                                            backend_name);
                }
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
                if (requested_w7x_solve() != 0) {
                    self->run_selected_w7x_solver();
                    return;
                }
                if (requested_app_mode() != 0) {
                    self->run_interactive_solver();
                    return;
                }
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
                self->run_toroidal_inverse();
            });
    }

    void run_toroidal_inverse() {
        toroidal_case_.ns = 3;
        toroidal_case_.mpol = 3;
        toroidal_case_.ntor = 2;
        toroidal_case_.ntheta = 16;
        toroidal_case_.nzeta = 8;
        toroidal_case_.nfp = 5;
        const std::size_t values = cumes::webgpu::SPECTRAL_COMPONENT_COUNT *
                                   toroidal_case_.ns * toroidal_case_.mpol *
                                   (toroidal_case_.ntor + 1);
        toroidal_case_.state.resize(values);
        for (std::size_t i = 0; i < values; ++i) {
            toroidal_case_.state[i] =
                0.03F * std::sin(0.17F * static_cast<float>(i + 1)) +
                0.01F * std::cos(0.07F * static_cast<float>(i * i + 3));
        }
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_toroidal_inverse(
            device_, toroidal_case_,
            [self](std::string error,
                   cumes::webgpu::ToroidalInverseResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected = cumes::webgpu::toroidal_inverse_reference(
                    self->toroidal_case_);
                float max_error = 0.0F;
                bool valid = true;
                const auto compare = [&max_error, &valid](const auto& gpu,
                                                          const auto& cpu) {
                    valid &= gpu.size() == cpu.size();
                    if (gpu.size() != cpu.size()) return;
                    for (std::size_t i = 0; i < gpu.size(); ++i) {
                        max_error =
                            std::max(max_error, std::abs(gpu[i] - cpu[i]) /
                                                    (1.0F + std::abs(cpu[i])));
                        valid &= std::isfinite(gpu[i]);
                    }
                };
                compare(actual.geometry, expected.geometry);
                compare(actual.r_con, expected.r_con);
                compare(actual.z_con, expected.z_con);
                const std::size_t points =
                    static_cast<std::size_t>(self->toroidal_case_.ns) *
                    self->toroidal_case_.ntheta * self->toroidal_case_.nzeta;
                const bool toroidal_derivatives =
                    std::any_of(actual.geometry.begin() + 12 * points,
                                actual.geometry.end(),
                                [](float value) { return value != 0.0F; });
                if (!valid || !toroidal_derivatives || max_error > 2.0e-5F) {
                    self->finish(false, "toroidal inverse mismatch: " +
                                            std::to_string(max_error));
                    return;
                }
                std::printf(
                    "  direct 3-D inverse transform: PASS "
                    "(max |GPU-CPU| = %.3e)\n",
                    static_cast<double>(max_error));
                self->run_toroidal_geometry();
            });
    }

    void run_toroidal_geometry() {
        cumes::webgpu::ToroidalInverseCase physical = toroidal_case_;
        std::fill(physical.state.begin(), physical.state.end(), 0.0F);
        const int mnmax = physical.mpol * (physical.ntor + 1);
        const auto coefficient = [&](int component, int m, int n,
                                     int surface) -> float& {
            const int mode = m * (physical.ntor + 1) + n;
            return physical
                .state[(static_cast<std::size_t>(component) * mnmax + mode) *
                           physical.ns +
                       surface];
        };
        for (int surface = 0; surface < physical.ns; ++surface) {
            const float s = static_cast<float>(surface) /
                            static_cast<float>(physical.ns - 1);
            coefficient(0, 0, 0, surface) = 10.0F;
            coefficient(0, 1, 0, surface) = 0.8F * s;
            coefficient(1, 1, 0, surface) = 0.8F * s;
            coefficient(0, 1, 1, surface) = 0.025F * s;
            coefficient(1, 1, 1, surface) = -0.018F * s;
            coefficient(2, 1, 1, surface) = 0.006F * s;
            coefficient(3, 1, 1, surface) = 0.012F * s;
            coefficient(4, 1, 1, surface) = -0.009F * s;
            coefficient(5, 1, 1, surface) = 0.004F * s;
            coefficient(0, 2, 2, surface) = 0.003F * s;
            coefficient(1, 2, 2, surface) = -0.002F * s;
        }
        const auto full = cumes::webgpu::toroidal_inverse_reference(physical);
        toroidal_r_con_ = full.r_con;
        toroidal_z_con_ = full.z_con;
        toroidal_geometry_case_.ns = physical.ns;
        toroidal_geometry_case_.ntheta = physical.ntheta;
        toroidal_geometry_case_.nzeta = physical.nzeta;
        toroidal_geometry_case_.delta_s = 0.5F;
        toroidal_geometry_case_.geometry = full.geometry;
        toroidal_geometry_case_.sqrt_s_f = {0.0F, std::sqrt(0.5F), 1.0F};
        toroidal_geometry_case_.sqrt_s_h = {std::sqrt(0.25F), std::sqrt(0.75F)};
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_base_geometry(
            device_, toroidal_geometry_case_,
            [self](std::string error,
                   cumes::webgpu::BaseGeometryResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected = cumes::webgpu::base_geometry_reference(
                    self->toroidal_geometry_case_);
                float max_error = 0.0F;
                bool valid = actual.fields.size() == expected.fields.size();
                if (valid) {
                    for (std::size_t i = 0; i < actual.fields.size(); ++i) {
                        max_error = std::max(
                            max_error,
                            std::abs(actual.fields[i] - expected.fields[i]));
                        valid &= std::isfinite(actual.fields[i]);
                    }
                }
                const std::size_t half_points =
                    static_cast<std::size_t>(self->toroidal_geometry_case_.ns -
                                             1) *
                    self->toroidal_geometry_case_.ntheta *
                    self->toroidal_geometry_case_.nzeta;
                const bool guv_nonzero =
                    valid &&
                    std::any_of(actual.fields.begin() + 8 * half_points,
                                actual.fields.begin() + 9 * half_points,
                                [](float value) { return value != 0.0F; });
                if (!valid || !guv_nonzero || max_error > 2.0e-4F) {
                    self->finish(false, "3-D base geometry mismatch: " +
                                            std::to_string(max_error));
                    return;
                }
                std::printf(
                    "  3-D half-grid base geometry: PASS "
                    "(max |GPU-CPU| = %.3e)\n",
                    static_cast<double>(max_error));
                self->run_toroidal_magnetic_field(std::move(actual.fields));
            });
    }

    void run_toroidal_magnetic_field(std::vector<float> base_geometry) {
        toroidal_magnetic_case_.ns = toroidal_geometry_case_.ns;
        toroidal_magnetic_case_.ntheta = toroidal_geometry_case_.ntheta;
        toroidal_magnetic_case_.nzeta = toroidal_geometry_case_.nzeta;
        toroidal_magnetic_case_.lamscale = 1.0F;
        toroidal_magnetic_case_.prescribed_current = true;
        toroidal_magnetic_case_.geometry = toroidal_geometry_case_.geometry;
        toroidal_magnetic_case_.base_geometry = std::move(base_geometry);
        toroidal_magnetic_case_.sqrt_s_h = toroidal_geometry_case_.sqrt_s_h;
        toroidal_magnetic_case_.phip_f = {0.9F, 0.85F, 0.8F};
        toroidal_magnetic_case_.chip_h = {0.2F, 0.25F};
        toroidal_magnetic_case_.pres_h = {0.03F, 0.01F};
        toroidal_magnetic_case_.curr_h = {0.18F, 0.22F};
        toroidal_magnetic_case_.phip_h = {0.875F, 0.825F};
        toroidal_magnetic_case_.iota_h = {0.2F, 0.25F};
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_magnetic_field(
            device_, toroidal_magnetic_case_,
            [self](std::string error,
                   cumes::webgpu::MagneticFieldResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected = cumes::webgpu::magnetic_field_reference(
                    self->toroidal_magnetic_case_);
                float max_error = 0.0F;
                bool valid = actual.fields.size() == expected.fields.size() &&
                             actual.chip_h.size() == expected.chip_h.size() &&
                             actual.iota_h.size() == expected.iota_h.size();
                if (valid) {
                    for (std::size_t i = 0; i < actual.fields.size(); ++i) {
                        max_error = std::max(
                            max_error,
                            std::abs(actual.fields[i] - expected.fields[i]));
                        valid &= std::isfinite(actual.fields[i]);
                    }
                }
                if (valid) {
                    for (std::size_t i = 0; i < actual.chip_h.size(); ++i) {
                        max_error = std::max(
                            {max_error,
                             std::abs(actual.chip_h[i] - expected.chip_h[i]),
                             std::abs(actual.iota_h[i] - expected.iota_h[i])});
                        valid &= std::isfinite(actual.chip_h[i]) &&
                                 std::isfinite(actual.iota_h[i]);
                    }
                }
                if (!valid || max_error > 2.0e-4F) {
                    self->finish(false, "3-D magnetic field mismatch: " +
                                            std::to_string(max_error));
                    return;
                }
                std::printf(
                    "  3-D prescribed-current magnetic field+pressure: PASS "
                    "(max |GPU-CPU| = %.3e)\n",
                    static_cast<double>(max_error));
                self->run_toroidal_force(std::move(actual.fields));
            });
    }

    void run_toroidal_force(std::vector<float> magnetic_field) {
        toroidal_force_case_.ns = toroidal_geometry_case_.ns;
        toroidal_force_case_.ntheta = toroidal_geometry_case_.ntheta;
        toroidal_force_case_.nzeta = toroidal_geometry_case_.nzeta;
        toroidal_force_case_.delta_s = toroidal_geometry_case_.delta_s;
        toroidal_force_case_.lamscale = toroidal_magnetic_case_.lamscale;
        toroidal_force_case_.geometry = toroidal_geometry_case_.geometry;
        toroidal_force_case_.base_geometry =
            toroidal_magnetic_case_.base_geometry;
        toroidal_force_case_.magnetic_field = std::move(magnetic_field);
        toroidal_force_case_.sqrt_s_f = toroidal_geometry_case_.sqrt_s_f;
        toroidal_force_case_.sqrt_s_h = toroidal_geometry_case_.sqrt_s_h;
        toroidal_force_case_.phip_f = toroidal_magnetic_case_.phip_f;
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_axisymmetric_force(
            device_, toroidal_force_case_,
            [self](std::string error,
                   cumes::webgpu::AxisymmetricForceResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected =
                    cumes::webgpu::axisymmetric_force_reference(
                        self->toroidal_force_case_);
                float max_error = 0.0F;
                bool valid = actual.fields.size() == expected.fields.size();
                if (valid) {
                    for (std::size_t i = 0; i < actual.fields.size(); ++i) {
                        max_error = std::max(
                            max_error,
                            std::abs(actual.fields[i] - expected.fields[i]));
                        valid &= std::isfinite(actual.fields[i]);
                    }
                }
                const std::size_t points =
                    static_cast<std::size_t>(self->toroidal_force_case_.ns) *
                    self->toroidal_force_case_.ntheta *
                    self->toroidal_force_case_.nzeta;
                const bool toroidal_force_nonzero =
                    valid && std::any_of(actual.fields.begin() + 10 * points,
                                         actual.fields.end(), [](float value) {
                                             return value != 0.0F;
                                         });
                if (!valid || !toroidal_force_nonzero || max_error > 5.0e-4F) {
                    self->finish(false, "3-D MHD force mismatch: " +
                                            std::to_string(max_error));
                    return;
                }
                std::printf("  3-D MHD force: PASS (max |GPU-CPU| = %.3e)\n",
                            static_cast<double>(max_error));
                self->run_toroidal_constraint(std::move(actual.fields));
            });
    }

    void run_toroidal_constraint(std::vector<float> force_fields) {
        toroidal_constraint_case_.ns = toroidal_case_.ns;
        toroidal_constraint_case_.mpol = toroidal_case_.mpol;
        toroidal_constraint_case_.ntor = toroidal_case_.ntor;
        toroidal_constraint_case_.ntheta = toroidal_case_.ntheta;
        toroidal_constraint_case_.nzeta = toroidal_case_.nzeta;
        toroidal_constraint_case_.delta_s = toroidal_geometry_case_.delta_s;
        toroidal_constraint_case_.tcon0 = 1.0F;
        toroidal_constraint_case_.reset_reference = true;
        toroidal_constraint_case_.refresh_preconditioner = true;
        toroidal_constraint_case_.geometry = toroidal_geometry_case_.geometry;
        toroidal_constraint_case_.r_con = toroidal_r_con_;
        toroidal_constraint_case_.z_con = toroidal_z_con_;
        const std::size_t points = toroidal_r_con_.size();
        toroidal_constraint_case_.r_con0.assign(points, 0.0F);
        toroidal_constraint_case_.z_con0.assign(points, 0.0F);
        toroidal_constraint_case_.tcon.assign(toroidal_case_.ns, 0.0F);
        toroidal_constraint_case_.ard.assign(2 * toroidal_case_.ns, 0.8F);
        toroidal_constraint_case_.azd.assign(2 * toroidal_case_.ns, 0.9F);
        toroidal_constraint_case_.sqrt_s_f = toroidal_geometry_case_.sqrt_s_f;
        toroidal_constraint_case_.force_fields = std::move(force_fields);
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_axisymmetric_constraint(
            device_, toroidal_constraint_case_,
            [self](std::string error,
                   cumes::webgpu::AxisymmetricConstraintResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected =
                    cumes::webgpu::axisymmetric_constraint_reference(
                        self->toroidal_constraint_case_);
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
                    self->finish(false, "3-D constraint mismatch: " +
                                            std::to_string(max_error));
                    return;
                }
                std::printf(
                    "  complete 3-D constraint: PASS "
                    "(max |GPU-CPU| = %.3e)\n",
                    static_cast<double>(max_error));
                self->run_toroidal_dealias(std::move(actual.fields));
            });
    }

    void run_toroidal_dealias(std::vector<float> constrained_fields) {
        toroidal_dealias_case_.ns = toroidal_case_.ns;
        toroidal_dealias_case_.mpol = toroidal_case_.mpol;
        toroidal_dealias_case_.ntor = toroidal_case_.ntor;
        toroidal_dealias_case_.ntheta = toroidal_case_.ntheta;
        toroidal_dealias_case_.nzeta = toroidal_case_.nzeta;
        const std::size_t points = static_cast<std::size_t>(toroidal_case_.ns) *
                                   toroidal_case_.ntheta * toroidal_case_.nzeta;
        toroidal_dealias_case_.g_con_eff.resize(points);
        for (std::size_t i = 0; i < points; ++i) {
            toroidal_dealias_case_.g_con_eff[i] =
                0.09F * std::sin(0.071F * static_cast<float>(i + 1)) +
                0.03F * std::cos(0.037F * static_cast<float>(i + 4));
        }
        toroidal_dealias_case_.tcon = {0.0F, 0.45F, 0.7F};
        toroidal_dealias_case_.faccon.assign(toroidal_case_.mpol, 0.0F);
        for (int m = 1; m < toroidal_case_.mpol; ++m) {
            const float xmpq = static_cast<float>((m + 1) * m);
            toroidal_dealias_case_.faccon[m] = 0.25F / (xmpq * xmpq);
        }
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_toroidal_dealias(
            device_, toroidal_dealias_case_,
            [self, constrained_fields = std::move(constrained_fields)](
                std::string error,
                cumes::webgpu::ToroidalDealiasResult actual) mutable {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected = cumes::webgpu::toroidal_dealias_reference(
                    self->toroidal_dealias_case_);
                float max_error = 0.0F;
                bool valid = actual.g_con.size() == expected.g_con.size();
                if (valid) {
                    for (std::size_t i = 0; i < actual.g_con.size(); ++i) {
                        max_error = std::max(
                            max_error,
                            std::abs(actual.g_con[i] - expected.g_con[i]));
                        valid &= std::isfinite(actual.g_con[i]);
                    }
                }
                const bool active =
                    valid &&
                    std::any_of(actual.g_con.begin(), actual.g_con.end(),
                                [](float value) { return value != 0.0F; });
                if (!valid || !active || max_error > 2.0e-5F) {
                    self->finish(false, "3-D constraint dealias mismatch: " +
                                            std::to_string(max_error));
                    return;
                }
                std::printf(
                    "  direct 3-D constraint dealias: PASS "
                    "(max |GPU-CPU| = %.3e)\n",
                    static_cast<double>(max_error));
                self->run_toroidal_forward(std::move(constrained_fields));
            });
    }

    void run_toroidal_forward(std::vector<float> force_fields) {
        toroidal_forward_case_.ns = toroidal_case_.ns;
        toroidal_forward_case_.mpol = toroidal_case_.mpol;
        toroidal_forward_case_.ntor = toroidal_case_.ntor;
        toroidal_forward_case_.ntheta = toroidal_case_.ntheta;
        toroidal_forward_case_.nzeta = toroidal_case_.nzeta;
        toroidal_forward_case_.nfp = toroidal_case_.nfp;
        toroidal_forward_case_.include_lcfs = false;
        const std::size_t points =
            static_cast<std::size_t>(toroidal_forward_case_.ns) *
            toroidal_forward_case_.ntheta * toroidal_forward_case_.nzeta;
        toroidal_forward_case_.fields.assign(
            cumes::webgpu::TOROIDAL_FORWARD_FIELD_COUNT * points, 0.0F);
        std::copy(force_fields.begin(), force_fields.end(),
                  toroidal_forward_case_.fields.begin());
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_toroidal_forward(
            device_, toroidal_forward_case_,
            [self](std::string error,
                   cumes::webgpu::ToroidalForwardResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected = cumes::webgpu::toroidal_forward_reference(
                    self->toroidal_forward_case_);
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
                const int mnmax = self->toroidal_forward_case_.mpol *
                                  (self->toroidal_forward_case_.ntor + 1);
                bool gates_valid = true;
                bool interior_nonzero = false;
                for (int component = 0; component < 6; ++component) {
                    for (int mode = 0; mode < mnmax; ++mode) {
                        const int m =
                            mode / (self->toroidal_forward_case_.ntor + 1);
                        const auto at = [&](int surface) {
                            return actual
                                .residual[(static_cast<std::size_t>(component) *
                                               mnmax +
                                           mode) *
                                              self->toroidal_forward_case_.ns +
                                          surface];
                        };
                        if (m > 0 || (component != 0 && component != 4)) {
                            gates_valid &= at(0) == 0.0F;
                        }
                        if (component != 2 && component != 5) {
                            gates_valid &=
                                at(self->toroidal_forward_case_.ns - 1) == 0.0F;
                        }
                        interior_nonzero |= at(1) != 0.0F;
                    }
                }
                if (!valid || !gates_valid || !interior_nonzero ||
                    max_error > 2.0e-5F) {
                    self->finish(false, "toroidal forward mismatch: " +
                                            std::to_string(max_error));
                    return;
                }
                std::printf(
                    "  direct 3-D forward transform: PASS "
                    "(max |GPU-CPU| = %.3e)\n",
                    static_cast<double>(max_error));
                self->run_toroidal_preconditioner_elements(
                    std::move(actual.residual));
            });
    }

    void run_toroidal_preconditioner_elements(std::vector<float> residual) {
        toroidal_preconditioner_case_.ns = toroidal_geometry_case_.ns;
        toroidal_preconditioner_case_.ntheta = toroidal_geometry_case_.ntheta;
        toroidal_preconditioner_case_.nzeta = toroidal_geometry_case_.nzeta;
        toroidal_preconditioner_case_.delta_s = toroidal_geometry_case_.delta_s;
        toroidal_preconditioner_case_.free_boundary = false;
        toroidal_preconditioner_case_.geometry =
            toroidal_geometry_case_.geometry;
        toroidal_preconditioner_case_.base_geometry =
            toroidal_magnetic_case_.base_geometry;
        toroidal_preconditioner_case_.magnetic_field =
            toroidal_force_case_.magnetic_field;
        toroidal_preconditioner_case_.sqrt_s_f =
            toroidal_geometry_case_.sqrt_s_f;
        toroidal_preconditioner_case_.sqrt_s_h =
            toroidal_geometry_case_.sqrt_s_h;
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_axisymmetric_preconditioner_elements(
            device_, toroidal_preconditioner_case_,
            [self, residual = std::move(residual)](
                std::string error,
                cumes::webgpu::AxisymmetricPreconditionerElements
                    actual) mutable {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected = cumes::webgpu::
                    axisymmetric_preconditioner_element_reference(
                        self->toroidal_preconditioner_case_);
                float max_error = 0.0F;
                bool valid = true;
                const auto compare = [&max_error, &valid](const auto& gpu,
                                                          const auto& cpu) {
                    valid &= gpu.size() == cpu.size();
                    if (gpu.size() != cpu.size()) return;
                    for (std::size_t i = 0; i < gpu.size(); ++i) {
                        max_error =
                            std::max(max_error, std::abs(gpu[i] - cpu[i]) /
                                                    (1.0F + std::abs(cpu[i])));
                        valid &= std::isfinite(gpu[i]);
                    }
                };
                compare(actual.ard, expected.ard);
                compare(actual.brd, expected.brd);
                compare(actual.azd, expected.azd);
                compare(actual.bzd, expected.bzd);
                compare(actual.cxd, expected.cxd);
                compare(actual.arm, expected.arm);
                compare(actual.brm, expected.brm);
                compare(actual.azm, expected.azm);
                compare(actual.bzm, expected.bzm);
                if (!valid || max_error > 5.0e-5F) {
                    self->finish(false,
                                 "3-D preconditioner element mismatch: " +
                                     std::to_string(max_error));
                    return;
                }
                self->toroidal_preconditioner_elements_ = std::move(actual);
                self->run_toroidal_preconditioner_matrix(std::move(residual));
            });
    }

    void run_toroidal_preconditioner_matrix(std::vector<float> residual) {
        toroidal_preconditioner_matrix_case_.ns = toroidal_case_.ns;
        toroidal_preconditioner_matrix_case_.mpol = toroidal_case_.mpol;
        toroidal_preconditioner_matrix_case_.ntor = toroidal_case_.ntor;
        toroidal_preconditioner_matrix_case_.ntheta = toroidal_case_.ntheta;
        toroidal_preconditioner_matrix_case_.nzeta = toroidal_case_.nzeta;
        toroidal_preconditioner_matrix_case_.nfp = toroidal_case_.nfp;
        toroidal_preconditioner_matrix_case_.delta_s =
            toroidal_geometry_case_.delta_s;
        toroidal_preconditioner_matrix_case_.free_boundary = false;
        toroidal_preconditioner_matrix_case_.elements =
            toroidal_preconditioner_elements_;
        toroidal_preconditioner_matrix_case_.base_geometry =
            toroidal_magnetic_case_.base_geometry;
        toroidal_preconditioner_matrix_case_.sqrt_s_f =
            toroidal_geometry_case_.sqrt_s_f;
        toroidal_preconditioner_matrix_case_.phip_h = {0.875F, 0.825F};
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_axisymmetric_preconditioner_matrix(
            device_, toroidal_preconditioner_matrix_case_,
            [self, residual = std::move(residual)](
                std::string error,
                cumes::webgpu::AxisymmetricPreconditionerMatrix
                    actual) mutable {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected =
                    cumes::webgpu::axisymmetric_preconditioner_matrix_reference(
                        self->toroidal_preconditioner_matrix_case_);
                float max_error = 0.0F;
                bool valid = actual.first_surface == expected.first_surface;
                const auto compare = [&max_error, &valid](const auto& gpu,
                                                          const auto& cpu) {
                    valid &= gpu.size() == cpu.size();
                    if (gpu.size() != cpu.size()) return;
                    for (std::size_t i = 0; i < gpu.size(); ++i) {
                        max_error =
                            std::max(max_error, std::abs(gpu[i] - cpu[i]) /
                                                    (1.0F + std::abs(cpu[i])));
                        valid &= std::isfinite(gpu[i]);
                    }
                };
                compare(actual.upper_r, expected.upper_r);
                compare(actual.diagonal_r, expected.diagonal_r);
                compare(actual.lower_r, expected.lower_r);
                compare(actual.upper_z, expected.upper_z);
                compare(actual.diagonal_z, expected.diagonal_z);
                compare(actual.lower_z, expected.lower_z);
                compare(actual.lambda, expected.lambda);
                compare(actual.scale, expected.scale);
                if (!valid || max_error > 2.0e-4F) {
                    self->finish(false, "3-D preconditioner matrix mismatch: " +
                                            std::to_string(max_error));
                    return;
                }
                self->toroidal_preconditioner_matrix_ = std::move(actual);
                self->run_toroidal_preconditioner_apply(std::move(residual));
            });
    }

    void run_toroidal_preconditioner_apply(std::vector<float> residual) {
        toroidal_preconditioner_apply_case_.ns = toroidal_case_.ns;
        toroidal_preconditioner_apply_case_.mpol = toroidal_case_.mpol;
        toroidal_preconditioner_apply_case_.ntor = toroidal_case_.ntor;
        toroidal_preconditioner_apply_case_.include_lcfs = false;
        toroidal_preconditioner_apply_case_.elements =
            toroidal_preconditioner_elements_;
        toroidal_preconditioner_apply_case_.matrix =
            toroidal_preconditioner_matrix_;
        toroidal_preconditioner_apply_case_.residual = std::move(residual);
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_axisymmetric_preconditioner_apply(
            device_, toroidal_preconditioner_apply_case_,
            [self](
                std::string error,
                cumes::webgpu::AxisymmetricPreconditionerApplyResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected =
                    cumes::webgpu::axisymmetric_preconditioner_apply_reference(
                        self->toroidal_preconditioner_apply_case_);
                float max_error = 0.0F;
                bool valid = actual.residual.size() == expected.residual.size();
                if (valid) {
                    for (std::size_t i = 0; i < actual.residual.size(); ++i) {
                        max_error = std::max(
                            max_error,
                            std::abs(actual.residual[i] -
                                     expected.residual[i]) /
                                (1.0F + std::abs(expected.residual[i])));
                        valid &= std::isfinite(actual.residual[i]);
                    }
                }
                valid &= actual.breakdown_count == expected.breakdown_count;
                if (!valid || actual.breakdown_count != 0 ||
                    max_error > 2.0e-4F) {
                    self->finish(false, "3-D preconditioner apply mismatch: " +
                                            std::to_string(max_error));
                    return;
                }
                std::printf(
                    "  complete 3-D preconditioner: PASS "
                    "(max scaled |GPU-CPU| = %.3e)\n",
                    static_cast<double>(max_error));
                self->run_w7x_initialization();
            });
    }

    void run_selected_w7x_solver() {
        try {
            cumes::SolverOptions options;
            options.precision = cumes::PrecisionPolicy::MIXED_FLOAT;
            auto parsed = cumes::read_problem_spec("/inputs/w7x.json", options);
            if (parsed.report.has_errors()) {
                const auto errors = parsed.report.errors();
                finish(false, "W7-X JSON mapping failed: " + errors.front());
                return;
            }
            for (auto& stage : parsed.spec.stages) {
                stage.tolerance =
                    std::max(stage.tolerance, W7X_MIXED_FLOAT_TOLERANCE);
            }
            auto validated = cumes::validate(std::move(parsed.spec), options);
            if (!validated.has_value()) {
                const auto errors = validated.error().errors();
                finish(false, "W7-X validation failed: " +
                                  (errors.empty() ? std::string("unknown error")
                                                  : errors.front()));
                return;
            }
            problem_.emplace(std::move(validated.value()));
            production_solve_ = true;
            double_single_solve_ = true;
            active_case_name_ = "W7-X";
            active_input_path_ = "inputs/w7x.json";
            stage_index_ = 0;
            total_iterations_ = 0;
            stage_iterations_.clear();
            stage_reports_.clear();
            initialized_stage_ =
                cumes::webgpu::initialize_stage(*problem_, stage_index_);
            reset_stage_state();
            std::printf(
                "running complete W7-X fixed-boundary multigrid solve "
                "(%zu stages, mixed-float ftol=%.0e)\n",
                problem_->stage_shapes().size(), W7X_MIXED_FLOAT_TOLERANCE);
            run_stage_inverse();
        } catch (const std::exception& error) {
            finish(false,
                   "W7-X solver startup failed: " + std::string(error.what()));
        }
    }

    void run_interactive_solver() {
        try {
            cumes::SolverOptions options;
            options.precision = cumes::PrecisionPolicy::MIXED_FLOAT;
            auto parsed =
                cumes::read_problem_spec("/inputs/interactive.json", options);
            if (parsed.report.has_errors()) {
                const auto errors = parsed.report.errors();
                finish(false,
                       "Interactive input mapping failed: " + errors.front());
                return;
            }
            for (auto& stage : parsed.spec.stages) {
                stage.tolerance = std::max(
                    stage.tolerance, cumes::tolerance_floor(options.precision));
            }
            auto validated = cumes::validate(std::move(parsed.spec), options);
            if (!validated.has_value()) {
                const auto errors = validated.error().errors();
                finish(false, "Interactive input validation failed: " +
                                  (errors.empty() ? std::string("unknown error")
                                                  : errors.front()));
                return;
            }
            problem_.emplace(std::move(validated.value()));
            production_solve_ = true;
            double_single_solve_ = false;
            active_case_name_ = "Interactive equilibrium";
            active_input_path_ = "browser boundary editor";
            stage_index_ = 0;
            total_iterations_ = 0;
            stage_iterations_.clear();
            stage_reports_.clear();
            initialized_stage_ =
                cumes::webgpu::initialize_stage(*problem_, stage_index_);
            reset_stage_state();
            std::printf(
                "running interactive fixed-boundary multigrid solve "
                "(%zu stages)\n",
                problem_->stage_shapes().size());
            run_stage_inverse();
        } catch (const std::exception& error) {
            finish(false, "Interactive solver startup failed: " +
                              std::string(error.what()));
        }
    }

    void run_w7x_initialization() {
        try {
            cumes::SolverOptions options;
            options.precision = cumes::PrecisionPolicy::MIXED_FLOAT;
            auto parsed = cumes::read_problem_spec("/inputs/w7x.json", options);
            if (parsed.report.has_errors()) {
                const auto errors = parsed.report.errors();
                finish(false, "W7-X JSON mapping failed: " + errors.front());
                return;
            }
            for (auto& stage : parsed.spec.stages) {
                stage.tolerance = std::max(
                    stage.tolerance, cumes::tolerance_floor(options.precision));
            }
            auto validated = cumes::validate(std::move(parsed.spec), options);
            if (!validated.has_value()) {
                const auto errors = validated.error().errors();
                finish(false, "W7-X validation failed: " +
                                  (errors.empty() ? std::string("unknown error")
                                                  : errors.front()));
                return;
            }
            w7x_problem_.emplace(std::move(validated.value()));
            w7x_stage_ = cumes::webgpu::initialize_stage(*w7x_problem_, 0);
            const auto& shape = w7x_problem_->stage_shapes()[0];
            const auto& boundary = w7x_problem_->boundary();
            const std::size_t family_values =
                static_cast<std::size_t>(shape.ns) * shape.modes();
            bool valid = w7x_stage_.ns == 33 && w7x_stage_.mpol == 12 &&
                         w7x_stage_.ntor == 12 && w7x_stage_.ntheta == 30 &&
                         w7x_stage_.nzeta == 36 && w7x_stage_.nfp == 5 &&
                         w7x_stage_.state.size() == 6 * family_values &&
                         w7x_stage_.envelope_correction == 0.12F &&
                         w7x_stage_.lambda_seed_scale == 0.0F;
            for (std::size_t mode = 0; mode < shape.modes(); ++mode) {
                const int m = static_cast<int>(mode) / (shape.ntor + 1);
                const std::size_t axis =
                    mode * static_cast<std::size_t>(shape.ns);
                const std::size_t lcfs = axis + shape.ns - 1;
                valid &= w7x_stage_.state[lcfs] ==
                         static_cast<float>(boundary.rbcc[mode]);
                valid &= w7x_stage_.state[3 * family_values + lcfs] ==
                         static_cast<float>(boundary.rbss[mode]);
                valid &= w7x_stage_.state[family_values + lcfs] ==
                         static_cast<float>(boundary.zbsc[mode]);
                valid &= w7x_stage_.state[4 * family_values + lcfs] ==
                         static_cast<float>(boundary.zbcs[mode]);
                if (m > 0) {
                    for (int component = 0; component < 6; ++component) {
                        valid &= w7x_stage_
                                     .state[component * family_values + axis] ==
                                 0.0F;
                    }
                }
            }
            valid &= w7x_stage_.profiles.curr_h.size() == 32;
            valid &= std::any_of(w7x_stage_.profiles.curr_h.begin(),
                                 w7x_stage_.profiles.curr_h.end(),
                                 [](float value) { return value != 0.0F; });
            if (!valid) {
                finish(false, "W7-X cold-start/profile contract mismatch");
                return;
            }
            std::printf(
                "  parsed W7-X 3-D cold start: PASS "
                "(ns=%d, mnmax=%d, angular=%d)\n",
                shape.ns, static_cast<int>(shape.modes()),
                shape.ntheta * shape.nzeta);
            run_w7x_inverse();
        } catch (const std::exception& error) {
            finish(false,
                   "W7-X initialization failed: " + std::string(error.what()));
        }
    }

    void run_w7x_inverse() {
        const int modes = w7x_stage_.mpol * (w7x_stage_.ntor + 1);
        const std::size_t family_values =
            static_cast<std::size_t>(w7x_stage_.ns) * modes;
        for (int mode = 0; mode < modes; ++mode) {
            const int m = mode / (w7x_stage_.ntor + 1);
            if (m != 0 && m != 1) continue;
            const int first_component = m == 0 ? 5 : 0;
            for (int component = first_component; component < 6; ++component) {
                const std::size_t axis =
                    static_cast<std::size_t>(component) * family_values +
                    static_cast<std::size_t>(mode) * w7x_stage_.ns;
                w7x_stage_.state[axis] = w7x_stage_.state[axis + 1];
                w7x_stage_.state_lo[axis] = w7x_stage_.state_lo[axis + 1];
            }
        }
        w7x_inverse_case_.ns = w7x_stage_.ns;
        w7x_inverse_case_.mpol = w7x_stage_.mpol;
        w7x_inverse_case_.ntor = w7x_stage_.ntor;
        w7x_inverse_case_.ntheta = w7x_stage_.ntheta;
        w7x_inverse_case_.nzeta = w7x_stage_.nzeta;
        w7x_inverse_case_.nfp = w7x_stage_.nfp;
        w7x_inverse_case_.double_single = true;
        w7x_inverse_case_.state = w7x_stage_.state;
        w7x_inverse_case_.state_lo = w7x_stage_.state_lo;
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_toroidal_inverse(
            device_, w7x_inverse_case_,
            [self](std::string error,
                   cumes::webgpu::ToroidalInverseResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected = cumes::webgpu::toroidal_inverse_reference(
                    self->w7x_inverse_case_);
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
                compare(actual.geometry, expected.geometry);
                compare(actual.r_con, expected.r_con);
                compare(actual.z_con, expected.z_con);
                double max_reconstructed_error = 0.0;
                const auto compare_pairs =
                    [&max_reconstructed_error, &valid](
                        const auto& gpu_hi, const auto& gpu_lo,
                        const auto& cpu_hi, const auto& cpu_lo) {
                        valid &= gpu_hi.size() == gpu_lo.size() &&
                                 gpu_hi.size() == cpu_hi.size() &&
                                 gpu_hi.size() == cpu_lo.size();
                        if (!valid) return;
                        for (std::size_t i = 0; i < gpu_hi.size(); ++i) {
                            const double gpu =
                                static_cast<double>(gpu_hi[i]) + gpu_lo[i];
                            const double cpu =
                                static_cast<double>(cpu_hi[i]) + cpu_lo[i];
                            max_reconstructed_error = std::max(
                                max_reconstructed_error, std::abs(gpu - cpu));
                            valid &= std::isfinite(gpu_lo[i]);
                        }
                    };
                compare_pairs(actual.geometry, actual.geometry_lo,
                              expected.geometry, expected.geometry_lo);
                compare_pairs(actual.r_con, actual.r_con_lo, expected.r_con,
                              expected.r_con_lo);
                compare_pairs(actual.z_con, actual.z_con_lo, expected.z_con,
                              expected.z_con_lo);
                if (!valid || max_error > 5.0e-4F ||
                    max_reconstructed_error > 2.0e-10) {
                    self->finish(
                        false,
                        "W7-X inverse mismatch: " + std::to_string(max_error) +
                            " reconstructed=" +
                            std::to_string(max_reconstructed_error));
                    return;
                }
                self->w7x_r_con_ = std::move(actual.r_con);
                self->w7x_z_con_ = std::move(actual.z_con);
                self->w7x_geometry_lo_ = std::move(actual.geometry_lo);
                std::printf(
                    "  W7-X double-single inverse transform: PASS "
                    "(max reconstructed |GPU-CPU| = %.3e)\n",
                    max_reconstructed_error);
                self->run_w7x_geometry(std::move(actual.geometry));
            });
    }

    void run_w7x_geometry(std::vector<float> geometry) {
        w7x_geometry_case_.ns = w7x_stage_.ns;
        w7x_geometry_case_.ntheta = w7x_stage_.ntheta;
        w7x_geometry_case_.nzeta = w7x_stage_.nzeta;
        w7x_geometry_case_.delta_s = w7x_stage_.profiles.delta_s;
        w7x_geometry_case_.double_single = true;
        w7x_geometry_case_.geometry = std::move(geometry);
        w7x_geometry_case_.geometry_lo = w7x_geometry_lo_;
        w7x_geometry_case_.sqrt_s_f = w7x_stage_.profiles.sqrt_s_f;
        w7x_geometry_case_.sqrt_s_h = w7x_stage_.profiles.sqrt_s_h;
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_base_geometry(
            device_, w7x_geometry_case_,
            [self](std::string error,
                   cumes::webgpu::BaseGeometryResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected = cumes::webgpu::base_geometry_reference(
                    self->w7x_geometry_case_);
                float max_error = 0.0F;
                bool valid = actual.fields.size() == expected.fields.size();
                if (valid) {
                    for (std::size_t i = 0; i < actual.fields.size(); ++i) {
                        max_error = std::max(
                            max_error,
                            std::abs(actual.fields[i] - expected.fields[i]));
                        valid &= std::isfinite(actual.fields[i]);
                    }
                }
                double max_reconstructed_error = 0.0;
                valid &= actual.fields_lo.size() == actual.fields.size() &&
                         expected.fields_lo.size() == expected.fields.size();
                if (valid) {
                    for (std::size_t i = 0; i < actual.fields.size(); ++i) {
                        max_reconstructed_error = std::max(
                            max_reconstructed_error,
                            std::abs((static_cast<double>(actual.fields[i]) +
                                      actual.fields_lo[i]) -
                                     (static_cast<double>(expected.fields[i]) +
                                      expected.fields_lo[i])));
                        valid &= std::isfinite(actual.fields_lo[i]);
                    }
                }
                if (!valid || max_error > 5.0e-4F ||
                    max_reconstructed_error > 5.0e-9) {
                    self->finish(
                        false,
                        "W7-X geometry mismatch: " + std::to_string(max_error) +
                            " reconstructed=" +
                            std::to_string(max_reconstructed_error));
                    return;
                }
                std::printf(
                    "  W7-X double-single half-grid geometry: PASS "
                    "(max reconstructed |GPU-CPU| = %.3e)\n",
                    max_reconstructed_error);
                self->w7x_base_geometry_lo_ = std::move(actual.fields_lo);
                self->run_w7x_magnetic_field(std::move(actual.fields));
            });
    }

    void run_w7x_magnetic_field(std::vector<float> base_geometry) {
        w7x_magnetic_case_.ns = w7x_stage_.ns;
        w7x_magnetic_case_.ntheta = w7x_stage_.ntheta;
        w7x_magnetic_case_.nzeta = w7x_stage_.nzeta;
        w7x_magnetic_case_.lamscale = w7x_stage_.profiles.lamscale;
        w7x_magnetic_case_.prescribed_current = true;
        w7x_magnetic_case_.geometry = w7x_geometry_case_.geometry;
        w7x_magnetic_case_.base_geometry = std::move(base_geometry);
        w7x_magnetic_case_.sqrt_s_h = w7x_stage_.profiles.sqrt_s_h;
        w7x_magnetic_case_.phip_f = w7x_stage_.profiles.phip_f;
        w7x_magnetic_case_.chip_h = w7x_stage_.profiles.chip_h;
        w7x_magnetic_case_.pres_h = w7x_stage_.profiles.pres_h;
        w7x_magnetic_case_.curr_h = w7x_stage_.profiles.curr_h;
        w7x_magnetic_case_.phip_h = w7x_stage_.profiles.phip_h;
        w7x_magnetic_case_.iota_h = w7x_stage_.profiles.iota_h;
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_magnetic_field(
            device_, w7x_magnetic_case_,
            [self](std::string error,
                   cumes::webgpu::MagneticFieldResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected = cumes::webgpu::magnetic_field_reference(
                    self->w7x_magnetic_case_);
                float max_error = 0.0F;
                bool valid = actual.fields.size() == expected.fields.size() &&
                             actual.chip_h.size() == expected.chip_h.size() &&
                             actual.iota_h.size() == expected.iota_h.size();
                const auto compare = [&max_error, &valid](const auto& gpu,
                                                          const auto& cpu) {
                    if (gpu.size() != cpu.size()) return;
                    for (std::size_t i = 0; i < gpu.size(); ++i) {
                        max_error =
                            std::max(max_error, std::abs(gpu[i] - cpu[i]) /
                                                    (1.0F + std::abs(cpu[i])));
                        valid &= std::isfinite(gpu[i]);
                    }
                };
                compare(actual.fields, expected.fields);
                compare(actual.chip_h, expected.chip_h);
                compare(actual.iota_h, expected.iota_h);
                if (!valid || max_error > 5.0e-4F) {
                    self->finish(false, "W7-X current solve mismatch: " +
                                            std::to_string(max_error));
                    return;
                }
                self->w7x_stage_.profiles.chip_h = actual.chip_h;
                self->w7x_stage_.profiles.iota_h = actual.iota_h;
                std::printf(
                    "  W7-X prescribed-current magnetic field: PASS "
                    "(max scaled |GPU-CPU| = %.3e)\n",
                    static_cast<double>(max_error));
                self->run_w7x_force(std::move(actual.fields));
            });
    }

    void run_w7x_force(std::vector<float> magnetic_field) {
        w7x_force_case_.ns = w7x_stage_.ns;
        w7x_force_case_.ntheta = w7x_stage_.ntheta;
        w7x_force_case_.nzeta = w7x_stage_.nzeta;
        w7x_force_case_.delta_s = w7x_stage_.profiles.delta_s;
        w7x_force_case_.lamscale = w7x_stage_.profiles.lamscale;
        w7x_force_case_.geometry = w7x_geometry_case_.geometry;
        w7x_force_case_.base_geometry = w7x_magnetic_case_.base_geometry;
        w7x_force_case_.magnetic_field = std::move(magnetic_field);
        w7x_force_case_.sqrt_s_f = w7x_stage_.profiles.sqrt_s_f;
        w7x_force_case_.sqrt_s_h = w7x_stage_.profiles.sqrt_s_h;
        w7x_force_case_.phip_f = w7x_stage_.profiles.phip_f;
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_axisymmetric_force(
            device_, w7x_force_case_,
            [self](std::string error,
                   cumes::webgpu::AxisymmetricForceResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected =
                    cumes::webgpu::axisymmetric_force_reference(
                        self->w7x_force_case_);
                float max_error = 0.0F;
                bool valid = actual.fields.size() == expected.fields.size();
                if (valid) {
                    for (std::size_t i = 0; i < actual.fields.size(); ++i) {
                        max_error = std::max(
                            max_error,
                            std::abs(actual.fields[i] - expected.fields[i]) /
                                (1.0F + std::abs(expected.fields[i])));
                        valid &= std::isfinite(actual.fields[i]);
                    }
                }
                if (!valid || max_error > 5.0e-4F) {
                    self->finish(false, "W7-X force mismatch: " +
                                            std::to_string(max_error));
                    return;
                }
                std::printf(
                    "  W7-X MHD force: PASS "
                    "(max scaled |GPU-CPU| = %.3e)\n",
                    static_cast<double>(max_error));
                self->run_w7x_forward(std::move(actual.fields));
            });
    }

    void run_w7x_forward(std::vector<float> force_fields) {
        w7x_forward_case_.ns = w7x_stage_.ns;
        w7x_forward_case_.mpol = w7x_stage_.mpol;
        w7x_forward_case_.ntor = w7x_stage_.ntor;
        w7x_forward_case_.ntheta = w7x_stage_.ntheta;
        w7x_forward_case_.nzeta = w7x_stage_.nzeta;
        w7x_forward_case_.nfp = w7x_stage_.nfp;
        w7x_forward_case_.include_lcfs = false;
        const std::size_t points = static_cast<std::size_t>(w7x_stage_.ns) *
                                   w7x_stage_.ntheta * w7x_stage_.nzeta;
        w7x_forward_case_.fields.assign(
            cumes::webgpu::TOROIDAL_FORWARD_FIELD_COUNT * points, 0.0F);
        std::copy(force_fields.begin(), force_fields.end(),
                  w7x_forward_case_.fields.begin());
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_toroidal_forward(
            device_, w7x_forward_case_,
            [self](std::string error,
                   cumes::webgpu::ToroidalForwardResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected = cumes::webgpu::toroidal_forward_reference(
                    self->w7x_forward_case_);
                float max_error = 0.0F;
                bool valid = actual.residual.size() == expected.residual.size();
                if (valid) {
                    for (std::size_t i = 0; i < actual.residual.size(); ++i) {
                        max_error = std::max(
                            max_error,
                            std::abs(actual.residual[i] -
                                     expected.residual[i]) /
                                (1.0F + std::abs(expected.residual[i])));
                        valid &= std::isfinite(actual.residual[i]);
                    }
                }
                const bool nonzero =
                    std::any_of(actual.residual.begin(), actual.residual.end(),
                                [](float value) { return value != 0.0F; });
                if (!valid || !nonzero || max_error > 5.0e-4F) {
                    self->finish(false, "W7-X forward residual mismatch: " +
                                            std::to_string(max_error));
                    return;
                }
                std::printf(
                    "  W7-X first physical residual pass: PASS "
                    "(max scaled |GPU-CPU| = %.3e)\n",
                    static_cast<double>(max_error));
                self->run_w7x_decomposition(std::move(actual.residual));
            });
    }

    void run_w7x_decomposition(std::vector<float> residual) {
        w7x_residual_case_.ns = w7x_stage_.ns;
        w7x_residual_case_.mpol = w7x_stage_.mpol;
        w7x_residual_case_.ntor = w7x_stage_.ntor;
        w7x_residual_case_.include_edge_rz = false;
        w7x_residual_case_.zero_m1_z = true;
        w7x_residual_case_.residual = std::move(residual);
        w7x_residual_case_.sqrt_s_f = w7x_stage_.profiles.sqrt_s_f;
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_residual_decomposition(
            device_, w7x_residual_case_,
            [self](std::string error,
                   cumes::webgpu::ResidualDecompositionResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected =
                    cumes::webgpu::residual_decomposition_reference(
                        self->w7x_residual_case_);
                float max_error = 0.0F;
                bool valid = actual.residual.size() == expected.residual.size();
                if (valid) {
                    for (std::size_t i = 0; i < actual.residual.size(); ++i) {
                        max_error = std::max(
                            max_error,
                            std::abs(actual.residual[i] -
                                     expected.residual[i]) /
                                (1.0F + std::abs(expected.residual[i])));
                        valid &= std::isfinite(actual.residual[i]);
                    }
                }
                double max_norm_error = 0.0;
                for (int group = 0; group < 3; ++group) {
                    max_norm_error = std::max(
                        max_norm_error,
                        std::abs(actual.raw_norm[group] -
                                 expected.raw_norm[group]) /
                            (1.0 + std::abs(expected.raw_norm[group])));
                    valid &= std::isfinite(actual.raw_norm[group]);
                }
                if (!valid || max_error > 5.0e-4F || max_norm_error > 5.0e-4) {
                    self->finish(false, "W7-X decomposition mismatch: " +
                                            std::to_string(max_error));
                    return;
                }
                std::printf(
                    "  W7-X residual decomposition: PASS "
                    "(max scaled |GPU-CPU| = %.3e)\n",
                    static_cast<double>(max_error));
                self->run_w7x_descent(std::move(actual.residual));
            });
    }

    void run_w7x_descent(std::vector<float> residual) {
        w7x_descent_case_.ns = w7x_stage_.ns;
        w7x_descent_case_.mpol = w7x_stage_.mpol;
        w7x_descent_case_.ntor = w7x_stage_.ntor;
        w7x_descent_case_.move_lcfs = false;
        w7x_descent_case_.delta_t = 0.01F;
        w7x_descent_case_.damping_b1 = 0.9F;
        w7x_descent_case_.damping_fac = 1.0F;
        w7x_descent_case_.double_single = true;
        w7x_descent_case_.state = w7x_stage_.state;
        w7x_descent_case_.state_lo.assign(w7x_stage_.state.size(), 0.0F);
        // Exercise persistence below the high word's ulp independently of the
        // particular first-pass W7-X force values.
        w7x_descent_case_.state_lo[w7x_stage_.ns - 1] = 1.0e-8F;
        w7x_descent_case_.velocity.assign(w7x_stage_.state.size(), 0.0F);
        w7x_descent_case_.velocity_lo.assign(w7x_stage_.state.size(), 0.0F);
        w7x_descent_case_.residual = std::move(residual);
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_axisymmetric_descent(
            device_, w7x_descent_case_,
            [self](std::string error,
                   cumes::webgpu::AxisymmetricDescentResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected =
                    cumes::webgpu::axisymmetric_descent_reference(
                        self->w7x_descent_case_);
                float max_error = 0.0F;
                bool valid = true;
                const auto compare = [&max_error, &valid](const auto& gpu,
                                                          const auto& cpu) {
                    valid &= gpu.size() == cpu.size();
                    if (gpu.size() != cpu.size()) return;
                    for (std::size_t i = 0; i < gpu.size(); ++i) {
                        max_error =
                            std::max(max_error, std::abs(gpu[i] - cpu[i]) /
                                                    (1.0F + std::abs(cpu[i])));
                        valid &= std::isfinite(gpu[i]);
                    }
                };
                compare(actual.state, expected.state);
                compare(actual.state_lo, expected.state_lo);
                compare(actual.velocity, expected.velocity);
                compare(actual.velocity_lo, expected.velocity_lo);
                double max_reconstructed_error = 0.0;
                if (actual.state_lo.size() == expected.state_lo.size() &&
                    actual.velocity_lo.size() == expected.velocity_lo.size()) {
                    for (std::size_t i = 0; i < actual.state_lo.size(); ++i) {
                        max_reconstructed_error = std::max(
                            {max_reconstructed_error,
                             std::abs((static_cast<double>(actual.state[i]) +
                                       actual.state_lo[i]) -
                                      (static_cast<double>(expected.state[i]) +
                                       expected.state_lo[i])),
                             std::abs(
                                 (static_cast<double>(actual.velocity[i]) +
                                  actual.velocity_lo[i]) -
                                 (static_cast<double>(expected.velocity[i]) +
                                  expected.velocity_lo[i]))});
                    }
                }
                const bool retained_low_bits =
                    std::any_of(actual.state_lo.begin(), actual.state_lo.end(),
                                [](float value) { return value != 0.0F; });
                if (!valid || !retained_low_bits || max_error > 5.0e-4F ||
                    max_reconstructed_error > 2.0e-12) {
                    self->finish(false,
                                 "W7-X descent mismatch: error=" +
                                     std::to_string(max_error) + " state_lo=" +
                                     std::to_string(actual.state_lo.size()) +
                                     " reconstruction_error=" +
                                     std::to_string(max_reconstructed_error) +
                                     " retained=" +
                                     (retained_low_bits ? "true" : "false") +
                                     " valid=" + (valid ? "true" : "false"));
                    return;
                }
                std::printf(
                    "  W7-X accelerated descent: PASS "
                    "(max scaled |GPU-CPU| = %.3e)\n",
                    static_cast<double>(max_error));
                self->initialized_stage_ = self->w7x_stage_;
                self->w7x_stage_slice_ = true;
                self->active_case_name_ = "W7-X";
                self->reset_stage_state();
                self->run_stage_inverse();
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
            problem_.emplace(std::move(validated.value()));
            active_case_name_ = "Solovev";
            active_input_path_ = "inputs/solovev.json";
            const auto& boundary = problem_->boundary();
            initialized_stage_ =
                cumes::webgpu::initialize_axisymmetric_stage(*problem_, 0);
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

            reset_stage_state();
            run_stage_inverse();
        } catch (const std::exception& error) {
            finish(false, "Solovev initialization failed: " +
                              std::string(error.what()));
        }
    }

    void run_stage_inverse() {
        if (attempted_passes_ >= initialized_stage_.max_iterations) {
            finish(false,
                   active_case_name_ +
                       " stage exhausted its iteration limit at iter=" +
                       std::to_string(controller_->effective_iteration()) +
                       " FSQR=" + std::to_string(invariant_normalized_[0]));
            return;
        }
        ++attempted_passes_;
        if (controller_->next_schedule()) {
            restore_checkpoint();
            std::printf(
                "  controller maintenance restore: iter=%d delta=%.3e\n",
                controller_->effective_iteration(), controller_->delta_t());
            run_stage_inverse();
            return;
        }
        // CUDA's extrapolateTowardsAxis mutates the physical state immediately
        // before every inverse pass.
        const int mode_count =
            initialized_stage_.mpol * (initialized_stage_.ntor + 1);
        const std::size_t family_values =
            static_cast<std::size_t>(initialized_stage_.ns) * mode_count;
        for (int mode = 0; mode < mode_count; ++mode) {
            const int m = mode / (initialized_stage_.ntor + 1);
            if (m != 0 && m != 1) continue;
            const int first_component = m == 0 ? 5 : 0;
            for (int component = first_component; component < 6; ++component) {
                const std::size_t axis =
                    static_cast<std::size_t>(component) * family_values +
                    static_cast<std::size_t>(mode) * initialized_stage_.ns;
                initialized_stage_.state[axis] =
                    initialized_stage_.state[axis + 1];
                if (double_single_solve_) {
                    stage_state_lo_[axis] = stage_state_lo_[axis + 1];
                }
            }
        }
        const auto self = shared_from_this();
        if (initialized_stage_.ntor == 0) {
            cumes::webgpu::AxisymmetricInverseCase inverse;
            inverse.ns = initialized_stage_.ns;
            inverse.mpol = initialized_stage_.mpol;
            inverse.ntheta = initialized_stage_.ntheta;
            inverse.state = initialized_stage_.state;
            cumes::webgpu::enqueue_axisymmetric_inverse(
                device_, inverse,
                [self, inverse](
                    std::string error,
                    cumes::webgpu::AxisymmetricInverseResult actual) {
                    if (!error.empty()) {
                        self->finish(false, std::move(error));
                        return;
                    }
                    self->finish_stage_inverse(
                        std::move(actual),
                        self->production_solve_
                            ? cumes::webgpu::ToroidalInverseResult{}
                            : cumes::webgpu::axisymmetric_inverse_reference(
                                  inverse));
                });
            return;
        }
        stage_toroidal_inverse_case_.ns = initialized_stage_.ns;
        stage_toroidal_inverse_case_.mpol = initialized_stage_.mpol;
        stage_toroidal_inverse_case_.ntor = initialized_stage_.ntor;
        stage_toroidal_inverse_case_.ntheta = initialized_stage_.ntheta;
        stage_toroidal_inverse_case_.nzeta = initialized_stage_.nzeta;
        stage_toroidal_inverse_case_.nfp = initialized_stage_.nfp;
        stage_toroidal_inverse_case_.double_single = double_single_solve_;
        stage_toroidal_inverse_case_.state = initialized_stage_.state;
        if (double_single_solve_) {
            stage_toroidal_inverse_case_.state_lo = stage_state_lo_;
        } else {
            stage_toroidal_inverse_case_.state_lo.clear();
        }
        cumes::webgpu::enqueue_toroidal_inverse(
            device_, stage_toroidal_inverse_case_,
            [self](std::string error,
                   cumes::webgpu::ToroidalInverseResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                self->finish_stage_inverse(
                    std::move(actual),
                    self->production_solve_
                        ? cumes::webgpu::ToroidalInverseResult{}
                        : cumes::webgpu::toroidal_inverse_reference(
                              self->stage_toroidal_inverse_case_));
            });
    }

    void finish_stage_inverse(
        cumes::webgpu::ToroidalInverseResult actual,
        const cumes::webgpu::ToroidalInverseResult& expected) {
        float max_error = 0.0F;
        bool valid = production_solve_ || !expected.geometry.empty();
        const auto compare = [&max_error, &valid](const auto& gpu,
                                                  const auto& cpu) {
            valid &= gpu.size() == cpu.size();
            if (gpu.size() != cpu.size()) return;
            for (std::size_t i = 0; i < gpu.size(); ++i) {
                max_error = std::max(max_error, std::abs(gpu[i] - cpu[i]));
                valid &= std::isfinite(gpu[i]);
            }
        };
        if (!production_solve_) {
            compare(actual.geometry, expected.geometry);
            compare(actual.r_con, expected.r_con);
            compare(actual.z_con, expected.z_con);
        } else {
            valid &=
                std::all_of(actual.geometry.begin(), actual.geometry.end(),
                            [](float value) { return std::isfinite(value); });
        }
        if (!valid || max_error > 5.0e-4F) {
            finish(false,
                   "iterative inverse mismatch: " + std::to_string(max_error));
            return;
        }
        if (!production_solve_) {
            std::printf("  %s pass %d inverse: PASS (max |GPU-CPU| = %.3e)\n",
                        active_case_name_.c_str(), completed_passes_ + 1,
                        static_cast<double>(max_error));
        }
        stage_r_con_ = std::move(actual.r_con);
        stage_z_con_ = std::move(actual.z_con);
        stage_geometry_lo_ = std::move(actual.geometry_lo);
        stage_r_con_lo_ = std::move(actual.r_con_lo);
        stage_z_con_lo_ = std::move(actual.z_con_lo);
        run_base_geometry(std::move(actual.geometry));
    }

    void run_base_geometry(std::vector<float> geometry) {
        base_geometry_case_.ns = initialized_stage_.ns;
        base_geometry_case_.ntheta = initialized_stage_.ntheta;
        base_geometry_case_.nzeta = initialized_stage_.nzeta;
        base_geometry_case_.delta_s = initialized_stage_.profiles.delta_s;
        base_geometry_case_.double_single = double_single_solve_;
        base_geometry_case_.geometry = std::move(geometry);
        if (double_single_solve_) {
            base_geometry_case_.geometry_lo = stage_geometry_lo_;
        } else {
            base_geometry_case_.geometry_lo.clear();
        }
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
                const auto expected =
                    self->production_solve_
                        ? cumes::webgpu::BaseGeometryResult{}
                        : cumes::webgpu::base_geometry_reference(
                              self->base_geometry_case_);
                if (!self->production_solve_ &&
                    actual.fields.size() != expected.fields.size()) {
                    self->finish(false, "base geometry result shape mismatch");
                    return;
                }
                float max_error = 0.0F;
                if (!self->production_solve_) {
                    for (std::size_t i = 0; i < actual.fields.size(); ++i) {
                        max_error = std::max(
                            max_error,
                            std::abs(actual.fields[i] - expected.fields[i]));
                    }
                }
                const std::size_t half_points =
                    static_cast<std::size_t>(self->base_geometry_case_.ns - 1) *
                    self->base_geometry_case_.ntheta *
                    self->base_geometry_case_.nzeta;
                const auto guv = actual.fields.begin() + 8 * half_points;
                const bool axisymmetric_guv_zero =
                    self->initialized_stage_.ntor != 0 ||
                    std::all_of(guv, guv + half_points,
                                [](float value) { return value == 0.0F; });
                const auto gsqrt = actual.fields.begin() + 6 * half_points;
                const bool finite_jacobian =
                    std::all_of(gsqrt, gsqrt + half_points, [](float value) {
                        return std::isfinite(value) && value != 0.0F;
                    });
                const bool precision_valid =
                    !self->double_single_solve_ ||
                    (actual.fields_lo.size() == actual.fields.size() &&
                     std::all_of(
                         actual.fields_lo.begin(), actual.fields_lo.end(),
                         [](float value) { return std::isfinite(value); }));
                if (max_error > 2.0e-4F || !axisymmetric_guv_zero ||
                    !finite_jacobian || !precision_valid) {
                    self->finish(
                        false, "base geometry mismatch: max_error=" +
                                   std::to_string(max_error) + " guv_zero=" +
                                   (axisymmetric_guv_zero ? "true" : "false") +
                                   " finite_jacobian=" +
                                   (finite_jacobian ? "true" : "false"));
                    return;
                }
                cumes::JacobianStatus<double> jacobian;
                jacobian.min_oriented = std::numeric_limits<double>::infinity();
                jacobian.max_abs = 0.0;
                jacobian.min_index = -1;
                jacobian.nonfinite_count = 0.0;
                for (std::size_t i = 0; i < half_points; ++i) {
                    const std::size_t gsqrt_index = 6 * half_points + i;
                    const double value = static_cast<double>(gsqrt[i]) +
                                         (self->double_single_solve_
                                              ? actual.fields_lo[gsqrt_index]
                                              : 0.0);
                    if (!std::isfinite(value)) {
                        jacobian.nonfinite_count += 1.0;
                        continue;
                    }
                    const double oriented = -value;
                    if (oriented < jacobian.min_oriented) {
                        jacobian.min_oriented = oriented;
                        jacobian.min_index = static_cast<int>(i);
                    }
                    jacobian.max_abs =
                        std::max(jacobian.max_abs, std::abs(value));
                }
                if (self->controller_->jacobian_invalid(
                        jacobian, self->initialized_stage_.ntheta)) {
                    self->restore_checkpoint();
                    std::printf(
                        "  invalid Jacobian restore: iter=%d min=%.3e "
                        "max=%.3e delta=%.3e\n",
                        self->controller_->effective_iteration(),
                        jacobian.min_oriented, jacobian.max_abs,
                        self->controller_->delta_t());
                    self->run_stage_inverse();
                    return;
                }
                if (!self->production_solve_) {
                    std::printf(
                        "  %s half-grid base geometry: PASS "
                        "(max |GPU-CPU| = %.3e)\n",
                        self->active_case_name_.c_str(),
                        static_cast<double>(max_error));
                }
                self->stage_base_geometry_lo_ = std::move(actual.fields_lo);
                self->run_magnetic_field(std::move(actual.fields));
            });
    }

    void run_magnetic_field(std::vector<float> base_geometry) {
        magnetic_field_case_.ns = initialized_stage_.ns;
        magnetic_field_case_.ntheta = initialized_stage_.ntheta;
        magnetic_field_case_.nzeta = initialized_stage_.nzeta;
        magnetic_field_case_.lamscale = initialized_stage_.profiles.lamscale;
        magnetic_field_case_.prescribed_current =
            initialized_stage_.prescribed_current;
        magnetic_field_case_.geometry = base_geometry_case_.geometry;
        magnetic_field_case_.base_geometry = std::move(base_geometry);
        magnetic_field_case_.sqrt_s_h = initialized_stage_.profiles.sqrt_s_h;
        magnetic_field_case_.phip_f = initialized_stage_.profiles.phip_f;
        magnetic_field_case_.chip_h = initialized_stage_.profiles.chip_h;
        magnetic_field_case_.pres_h = initialized_stage_.profiles.pres_h;
        magnetic_field_case_.curr_h = initialized_stage_.profiles.curr_h;
        magnetic_field_case_.phip_h = initialized_stage_.profiles.phip_h;
        magnetic_field_case_.iota_h = initialized_stage_.profiles.iota_h;
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_magnetic_field(
            device_, magnetic_field_case_,
            [self](std::string error,
                   cumes::webgpu::MagneticFieldResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected =
                    self->production_solve_
                        ? cumes::webgpu::MagneticFieldResult{}
                        : cumes::webgpu::magnetic_field_reference(
                              self->magnetic_field_case_);
                if (!self->production_solve_ &&
                    (actual.fields.size() != expected.fields.size() ||
                     actual.chip_h.size() != expected.chip_h.size() ||
                     actual.iota_h.size() != expected.iota_h.size())) {
                    self->finish(false, "magnetic field result shape mismatch");
                    return;
                }
                float max_error = 0.0F;
                bool finite = true;
                for (std::size_t i = 0; i < actual.fields.size(); ++i) {
                    if (!self->production_solve_) {
                        max_error = std::max(
                            max_error,
                            std::abs(actual.fields[i] - expected.fields[i]));
                    }
                    finite &= std::isfinite(actual.fields[i]);
                }
                for (std::size_t i = 0; i < actual.chip_h.size(); ++i) {
                    if (!self->production_solve_) {
                        max_error = std::max(
                            {max_error,
                             std::abs(actual.chip_h[i] - expected.chip_h[i]),
                             std::abs(actual.iota_h[i] - expected.iota_h[i])});
                    }
                    finite &= std::isfinite(actual.chip_h[i]) &&
                              std::isfinite(actual.iota_h[i]);
                }
                if (max_error > 2.0e-4F || !finite) {
                    self->finish(false,
                                 "magnetic field mismatch: max_error=" +
                                     std::to_string(max_error) +
                                     " finite=" + (finite ? "true" : "false"));
                    return;
                }
                if (!self->production_solve_) {
                    std::printf(
                        "  %s magnetic field+pressure: PASS "
                        "(max |GPU-CPU| = %.3e)\n",
                        self->active_case_name_.c_str(),
                        static_cast<double>(max_error));
                }
                self->initialized_stage_.profiles.chip_h = actual.chip_h;
                self->initialized_stage_.profiles.iota_h = actual.iota_h;
                if (self->initialized_stage_.prescribed_current) {
                    auto& profiles = self->initialized_stage_.profiles;
                    const int ns = self->initialized_stage_.ns;
                    profiles.iota_f[0] =
                        1.5F * profiles.iota_h[0] - 0.5F * profiles.iota_h[1];
                    for (int surface = 1; surface < ns - 1; ++surface) {
                        profiles.iota_f[surface] =
                            0.5F * (profiles.iota_h[surface] +
                                    profiles.iota_h[surface - 1]);
                        profiles.chi_f[surface] =
                            0.5F * (profiles.chip_h[surface] +
                                    profiles.chip_h[surface - 1]);
                    }
                    profiles.iota_f[ns - 1] = 1.5F * profiles.iota_h[ns - 2] -
                                              0.5F * profiles.iota_h[ns - 3];
                    profiles.chi_f[ns - 1] = 2.0F * profiles.chip_h[ns - 2] -
                                             profiles.chip_h[ns - 3];
                }
                self->run_axisymmetric_force(std::move(actual.fields));
            });
    }

    void run_axisymmetric_force(std::vector<float> magnetic_field) {
        force_case_.ns = initialized_stage_.ns;
        force_case_.ntheta = initialized_stage_.ntheta;
        force_case_.nzeta = initialized_stage_.nzeta;
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
                    self->production_solve_
                        ? cumes::webgpu::AxisymmetricForceResult{}
                        : cumes::webgpu::axisymmetric_force_reference(
                              self->force_case_);
                if (!self->production_solve_ &&
                    actual.fields.size() != expected.fields.size()) {
                    self->finish(false, "force result shape mismatch");
                    return;
                }
                float max_error = 0.0F;
                bool finite = true;
                for (std::size_t i = 0; i < actual.fields.size(); ++i) {
                    if (!self->production_solve_) {
                        max_error = std::max(
                            max_error,
                            std::abs(actual.fields[i] - expected.fields[i]));
                    }
                    finite &= std::isfinite(actual.fields[i]);
                }
                if (max_error > 5.0e-4F || !finite) {
                    self->finish(false, "axisymmetric force mismatch: " +
                                            std::to_string(max_error));
                    return;
                }
                if (!self->production_solve_) {
                    std::printf(
                        "  %s MHD force: PASS "
                        "(max |GPU-CPU| = %.3e)\n",
                        self->active_case_name_.c_str(),
                        static_cast<double>(max_error));
                }
                self->stage_force_fields_ = actual.fields;
                self->run_solovev_forward(std::move(actual.fields));
            });
    }

    void run_solovev_forward(std::vector<float> force_fields) {
        if (initialized_stage_.ntor != 0) {
            solver_toroidal_forward_case_.ns = initialized_stage_.ns;
            solver_toroidal_forward_case_.mpol = initialized_stage_.mpol;
            solver_toroidal_forward_case_.ntor = initialized_stage_.ntor;
            solver_toroidal_forward_case_.ntheta = initialized_stage_.ntheta;
            solver_toroidal_forward_case_.nzeta = initialized_stage_.nzeta;
            solver_toroidal_forward_case_.nfp = initialized_stage_.nfp;
            solver_toroidal_forward_case_.include_lcfs = false;
            const std::size_t points =
                static_cast<std::size_t>(initialized_stage_.ns) *
                initialized_stage_.ntheta * initialized_stage_.nzeta;
            solver_toroidal_forward_case_.fields.assign(
                cumes::webgpu::TOROIDAL_FORWARD_FIELD_COUNT * points, 0.0F);
            std::copy(force_fields.begin(), force_fields.end(),
                      solver_toroidal_forward_case_.fields.begin());
            const auto self = shared_from_this();
            cumes::webgpu::enqueue_toroidal_forward(
                device_, solver_toroidal_forward_case_,
                [self](std::string error,
                       cumes::webgpu::ToroidalForwardResult actual) {
                    if (!error.empty()) {
                        self->finish(false, std::move(error));
                        return;
                    }
                    self->finish_stage_forward(
                        std::move(actual.residual),
                        self->production_solve_
                            ? std::vector<float>{}
                            : cumes::webgpu::toroidal_forward_reference(
                                  self->solver_toroidal_forward_case_)
                                  .residual);
                });
            return;
        }
        solver_forward_case_.ns = initialized_stage_.ns;
        solver_forward_case_.mpol = initialized_stage_.mpol;
        solver_forward_case_.ntheta = initialized_stage_.ntheta;
        solver_forward_case_.include_lcfs = false;
        const std::size_t points =
            static_cast<std::size_t>(solver_forward_case_.ns) *
            solver_forward_case_.ntheta;
        solver_forward_case_.fields.assign(
            cumes::webgpu::FORWARD_INPUT_FIELD_COUNT * points, 0.0F);
        std::copy_n(force_fields.begin(),
                    cumes::webgpu::FORWARD_INPUT_FIELD_COUNT * points,
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
                self->finish_stage_forward(
                    std::move(actual.residual),
                    self->production_solve_
                        ? std::vector<float>{}
                        : cumes::webgpu::axisymmetric_forward_reference(
                              self->solver_forward_case_)
                              .residual);
            });
    }

    void finish_stage_forward(std::vector<float> residual,
                              const std::vector<float>& expected) {
        float max_error = 0.0F;
        bool finite = production_solve_ || residual.size() == expected.size();
        if (finite && !production_solve_) {
            for (std::size_t i = 0; i < residual.size(); ++i) {
                max_error =
                    std::max(max_error, std::abs(residual[i] - expected[i]));
                finite &= std::isfinite(residual[i]);
            }
        }
        if (production_solve_) {
            finite =
                std::all_of(residual.begin(), residual.end(),
                            [](float value) { return std::isfinite(value); });
        }
        const bool nonzero =
            std::any_of(residual.begin(), residual.end(),
                        [](float value) { return value != 0.0F; });
        if (max_error > 5.0e-4F || !finite || !nonzero) {
            finish(false, "stage forward residual mismatch: " +
                              std::to_string(max_error));
            return;
        }
        if (!production_solve_) {
            std::printf(
                "  %s spectral residual projection: PASS "
                "(max |GPU-CPU| = %.3e)\n",
                active_case_name_.c_str(), static_cast<double>(max_error));
        }
        run_residual_decomposition(std::move(residual));
    }

    void run_residual_decomposition(std::vector<float> residual) {
        residual_case_.ns = initialized_stage_.ns;
        residual_case_.mpol = initialized_stage_.mpol;
        residual_case_.ntor = initialized_stage_.ntor;
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
                    self->production_solve_
                        ? cumes::webgpu::ResidualDecompositionResult{}
                        : cumes::webgpu::residual_decomposition_reference(
                              self->residual_case_);
                float max_error = 0.0F;
                bool valid = self->production_solve_ ||
                             actual.residual.size() == expected.residual.size();
                if (valid && !self->production_solve_) {
                    for (std::size_t i = 0; i < actual.residual.size(); ++i) {
                        max_error =
                            std::max(max_error, std::abs(actual.residual[i] -
                                                         expected.residual[i]));
                    }
                }
                double max_norm_error = 0.0;
                for (int group = 0; group < 3; ++group) {
                    if (!self->production_solve_) {
                        max_norm_error = std::max(
                            max_norm_error,
                            std::abs(actual.raw_norm[group] -
                                     expected.raw_norm[group]) /
                                (1.0 + std::abs(expected.raw_norm[group])));
                    }
                    valid &= std::isfinite(actual.raw_norm[group]);
                }
                if (self->production_solve_) {
                    valid &= std::all_of(
                        actual.residual.begin(), actual.residual.end(),
                        [](float value) { return std::isfinite(value); });
                }
                if (!valid || max_error > 5.0e-4F || max_norm_error > 5.0e-4) {
                    self->finish(false, "residual decomposition mismatch: " +
                                            std::to_string(max_error));
                    return;
                }
                if (!self->production_solve_) {
                    std::printf(
                        "  decomposed residual+double norms: PASS "
                        "(max |GPU-CPU| = %.3e)\n",
                        static_cast<double>(max_error));
                }
                self->run_preconditioner_elements();
            });
    }

    void run_preconditioner_elements() {
        if (!controller_->refresh_preconditioner() &&
            !preconditioner_elements_.ard.empty()) {
            run_axisymmetric_constraint();
            return;
        }
        preconditioner_case_.ns = initialized_stage_.ns;
        preconditioner_case_.ntheta = initialized_stage_.ntheta;
        preconditioner_case_.nzeta = initialized_stage_.nzeta;
        preconditioner_case_.delta_s = initialized_stage_.profiles.delta_s;
        preconditioner_case_.free_boundary = false;
        preconditioner_case_.geometry = base_geometry_case_.geometry;
        preconditioner_case_.base_geometry = magnetic_field_case_.base_geometry;
        preconditioner_case_.magnetic_field = force_case_.magnetic_field;
        preconditioner_case_.sqrt_s_f = initialized_stage_.profiles.sqrt_s_f;
        preconditioner_case_.sqrt_s_h = initialized_stage_.profiles.sqrt_s_h;
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_axisymmetric_preconditioner_elements(
            device_, preconditioner_case_,
            [self](std::string error,
                   cumes::webgpu::AxisymmetricPreconditionerElements actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected =
                    self->production_solve_
                        ? cumes::webgpu::AxisymmetricPreconditionerElements{}
                        : cumes::webgpu::
                              axisymmetric_preconditioner_element_reference(
                                  self->preconditioner_case_);
                float max_scaled_error = 0.0F;
                bool valid = true;
                const auto compare = [&max_scaled_error, &valid](
                                         const auto& gpu, const auto& cpu) {
                    valid &= gpu.size() == cpu.size();
                    if (gpu.size() != cpu.size()) return;
                    for (std::size_t i = 0; i < gpu.size(); ++i) {
                        max_scaled_error = std::max(
                            max_scaled_error, std::abs(gpu[i] - cpu[i]) /
                                                  (1.0F + std::abs(cpu[i])));
                        valid &= std::isfinite(gpu[i]);
                    }
                };
                if (!self->production_solve_) {
                    compare(actual.ard, expected.ard);
                    compare(actual.brd, expected.brd);
                    compare(actual.azd, expected.azd);
                    compare(actual.bzd, expected.bzd);
                    compare(actual.cxd, expected.cxd);
                    compare(actual.arm, expected.arm);
                    compare(actual.brm, expected.brm);
                    compare(actual.azm, expected.azm);
                    compare(actual.bzm, expected.bzm);
                } else {
                    const auto finite = [](const auto& values) {
                        return std::all_of(
                            values.begin(), values.end(),
                            [](float value) { return std::isfinite(value); });
                    };
                    valid = finite(actual.ard) && finite(actual.brd) &&
                            finite(actual.azd) && finite(actual.bzd) &&
                            finite(actual.cxd) && finite(actual.arm) &&
                            finite(actual.brm) && finite(actual.azm) &&
                            finite(actual.bzm);
                }
                if (!valid || max_scaled_error > 5.0e-5F) {
                    self->finish(false, "preconditioner element mismatch: " +
                                            std::to_string(max_scaled_error));
                    return;
                }
                if (!self->production_solve_) {
                    std::printf(
                        "  %s radial preconditioner elements: PASS "
                        "(max scaled |GPU-CPU| = %.3e)\n",
                        self->active_case_name_.c_str(),
                        static_cast<double>(max_scaled_error));
                }
                self->preconditioner_elements_ = std::move(actual);
                self->run_preconditioner_matrix();
            });
    }

    void run_preconditioner_matrix() {
        preconditioner_matrix_case_.ns = initialized_stage_.ns;
        preconditioner_matrix_case_.mpol = initialized_stage_.mpol;
        preconditioner_matrix_case_.ntor = initialized_stage_.ntor;
        preconditioner_matrix_case_.ntheta = initialized_stage_.ntheta;
        preconditioner_matrix_case_.nzeta = initialized_stage_.nzeta;
        preconditioner_matrix_case_.nfp = initialized_stage_.nfp;
        preconditioner_matrix_case_.delta_s =
            initialized_stage_.profiles.delta_s;
        preconditioner_matrix_case_.free_boundary = false;
        preconditioner_matrix_case_.elements = preconditioner_elements_;
        preconditioner_matrix_case_.base_geometry =
            magnetic_field_case_.base_geometry;
        preconditioner_matrix_case_.sqrt_s_f =
            initialized_stage_.profiles.sqrt_s_f;
        preconditioner_matrix_case_.phip_h = initialized_stage_.profiles.phip_h;
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_axisymmetric_preconditioner_matrix(
            device_, preconditioner_matrix_case_,
            [self](std::string error,
                   cumes::webgpu::AxisymmetricPreconditionerMatrix actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected =
                    self->production_solve_
                        ? cumes::webgpu::AxisymmetricPreconditionerMatrix{}
                        : cumes::webgpu::
                              axisymmetric_preconditioner_matrix_reference(
                                  self->preconditioner_matrix_case_);
                float max_scaled_error = 0.0F;
                bool valid = self->production_solve_ ||
                             actual.first_surface == expected.first_surface;
                const auto compare = [&max_scaled_error, &valid](
                                         const auto& gpu, const auto& cpu) {
                    valid &= gpu.size() == cpu.size();
                    if (gpu.size() != cpu.size()) return;
                    for (std::size_t i = 0; i < gpu.size(); ++i) {
                        max_scaled_error = std::max(
                            max_scaled_error, std::abs(gpu[i] - cpu[i]) /
                                                  (1.0F + std::abs(cpu[i])));
                        valid &= std::isfinite(gpu[i]);
                    }
                };
                if (!self->production_solve_) {
                    compare(actual.upper_r, expected.upper_r);
                    compare(actual.diagonal_r, expected.diagonal_r);
                    compare(actual.lower_r, expected.lower_r);
                    compare(actual.upper_z, expected.upper_z);
                    compare(actual.diagonal_z, expected.diagonal_z);
                    compare(actual.lower_z, expected.lower_z);
                    compare(actual.lambda, expected.lambda);
                    compare(actual.scale, expected.scale);
                } else {
                    const auto finite = [](const auto& values) {
                        return std::all_of(
                            values.begin(), values.end(),
                            [](float value) { return std::isfinite(value); });
                    };
                    valid =
                        finite(actual.upper_r) && finite(actual.diagonal_r) &&
                        finite(actual.lower_r) && finite(actual.upper_z) &&
                        finite(actual.diagonal_z) && finite(actual.lower_z) &&
                        finite(actual.lambda) && finite(actual.scale);
                }
                if (!valid || max_scaled_error > 2.0e-4F) {
                    self->finish(false, "preconditioner matrix mismatch: " +
                                            std::to_string(max_scaled_error));
                    return;
                }
                if (!self->production_solve_) {
                    std::printf(
                        "  %s tridiagonal+lambda preconditioner: PASS "
                        "(max scaled |GPU-CPU| = %.3e)\n",
                        self->active_case_name_.c_str(),
                        static_cast<double>(max_scaled_error));
                }
                self->preconditioner_matrix_ = std::move(actual);
                self->run_axisymmetric_constraint();
            });
    }

    void run_axisymmetric_constraint() {
        constraint_case_.ns = initialized_stage_.ns;
        constraint_case_.mpol = initialized_stage_.mpol;
        constraint_case_.ntor = initialized_stage_.ntor;
        constraint_case_.ntheta = initialized_stage_.ntheta;
        constraint_case_.nzeta = initialized_stage_.nzeta;
        constraint_case_.delta_s = initialized_stage_.profiles.delta_s;
        constraint_case_.tcon0 = initialized_stage_.tcon0;
        constraint_case_.reset_reference =
            controller_->reset_constraint_reference();
        constraint_case_.refresh_preconditioner =
            controller_->refresh_preconditioner();
        constraint_case_.geometry = base_geometry_case_.geometry;
        constraint_case_.r_con = stage_r_con_;
        constraint_case_.z_con = stage_z_con_;
        const std::size_t points =
            static_cast<std::size_t>(constraint_case_.ns) *
            constraint_case_.ntheta * constraint_case_.nzeta;
        if (constraint_r_con0_.empty()) {
            constraint_r_con0_.assign(points, 0.0F);
            constraint_z_con0_.assign(points, 0.0F);
            constraint_tcon_.assign(constraint_case_.ns, 0.0F);
        }
        constraint_case_.r_con0 = constraint_r_con0_;
        constraint_case_.z_con0 = constraint_z_con0_;
        constraint_case_.tcon = constraint_tcon_;
        constraint_case_.ard = preconditioner_elements_.ard;
        constraint_case_.azd = preconditioner_elements_.azd;
        constraint_case_.sqrt_s_f = initialized_stage_.profiles.sqrt_s_f;
        const std::size_t force_field_count =
            initialized_stage_.ntor == 0 ? 10
                                         : cumes::webgpu::FORCE_FIELD_COUNT;
        constraint_case_.force_fields.assign(
            stage_force_fields_.begin(),
            stage_force_fields_.begin() + force_field_count * points);
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_axisymmetric_constraint(
            device_, constraint_case_,
            [self](std::string error,
                   cumes::webgpu::AxisymmetricConstraintResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                if (self->production_solve_) {
                    const bool finite = std::all_of(
                        actual.fields.begin(), actual.fields.end(),
                        [](float value) { return std::isfinite(value); });
                    if (!finite) {
                        self->finish(false,
                                     "constraint produced nonfinite fields");
                        return;
                    }
                    self->constraint_r_con0_ = actual.r_con0;
                    self->constraint_z_con0_ = actual.z_con0;
                    self->constraint_tcon_ = actual.tcon;
                    self->run_constraint_forward(std::move(actual.fields));
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
                            std::max(max_error, std::abs(gpu[i] - cpu[i]) /
                                                    (1.0F + std::abs(cpu[i])));
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
                    "  %s constraint refresh+force: PASS "
                    "(max scaled |GPU-CPU| = %.3e)\n",
                    self->active_case_name_.c_str(),
                    static_cast<double>(max_error));
                self->constraint_r_con0_ = actual.r_con0;
                self->constraint_z_con0_ = actual.z_con0;
                self->constraint_tcon_ = actual.tcon;
                self->run_constraint_forward(std::move(actual.fields));
            });
    }

    void run_constraint_forward(std::vector<float> fields) {
        if (initialized_stage_.ntor != 0) {
            constraint_toroidal_forward_case_.ns = initialized_stage_.ns;
            constraint_toroidal_forward_case_.mpol = initialized_stage_.mpol;
            constraint_toroidal_forward_case_.ntor = initialized_stage_.ntor;
            constraint_toroidal_forward_case_.ntheta =
                initialized_stage_.ntheta;
            constraint_toroidal_forward_case_.nzeta = initialized_stage_.nzeta;
            constraint_toroidal_forward_case_.nfp = initialized_stage_.nfp;
            constraint_toroidal_forward_case_.include_lcfs = false;
            constraint_toroidal_forward_case_.fields = std::move(fields);
            const auto self = shared_from_this();
            cumes::webgpu::enqueue_toroidal_forward(
                device_, constraint_toroidal_forward_case_,
                [self](std::string error,
                       cumes::webgpu::ToroidalForwardResult actual) {
                    if (!error.empty()) {
                        self->finish(false, std::move(error));
                        return;
                    }
                    self->finish_constraint_forward(
                        std::move(actual.residual),
                        self->production_solve_
                            ? std::vector<float>{}
                            : cumes::webgpu::toroidal_forward_reference(
                                  self->constraint_toroidal_forward_case_)
                                  .residual);
                });
            return;
        }
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
                self->finish_constraint_forward(
                    std::move(actual.residual),
                    self->production_solve_
                        ? std::vector<float>{}
                        : cumes::webgpu::axisymmetric_forward_reference(
                              self->constraint_forward_case_)
                              .residual);
            });
    }

    void finish_constraint_forward(std::vector<float> residual,
                                   const std::vector<float>& expected) {
        float max_error = 0.0F;
        bool valid = production_solve_ || residual.size() == expected.size();
        if (valid && !production_solve_) {
            for (std::size_t i = 0; i < residual.size(); ++i) {
                max_error =
                    std::max(max_error, std::abs(residual[i] - expected[i]));
                valid &= std::isfinite(residual[i]);
            }
        }
        if (production_solve_) {
            valid =
                std::all_of(residual.begin(), residual.end(),
                            [](float value) { return std::isfinite(value); });
        }
        if (!valid || max_error > 5.0e-4F) {
            finish(false, "constraint residual projection mismatch: " +
                              std::to_string(max_error));
            return;
        }
        if (!production_solve_) {
            std::printf(
                "  constrained spectral residual projection: PASS "
                "(max |GPU-CPU| = %.3e)\n",
                static_cast<double>(max_error));
        }
        run_constraint_residual_decomposition(std::move(residual));
    }

    void run_constraint_residual_decomposition(std::vector<float> residual) {
        constraint_residual_case_.ns = initialized_stage_.ns;
        constraint_residual_case_.mpol = initialized_stage_.mpol;
        constraint_residual_case_.ntor = initialized_stage_.ntor;
        constraint_residual_case_.include_edge_rz = false;
        constraint_residual_case_.zero_m1_z =
            controller_->effective_iteration() < 2 ||
            controller_->fsqz_prev() < 1.0e-6;
        constraint_residual_case_.residual = std::move(residual);
        constraint_residual_case_.sqrt_s_f =
            initialized_stage_.profiles.sqrt_s_f;
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_residual_decomposition(
            device_, constraint_residual_case_,
            [self](std::string error,
                   cumes::webgpu::ResidualDecompositionResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected =
                    self->production_solve_
                        ? cumes::webgpu::ResidualDecompositionResult{}
                        : cumes::webgpu::residual_decomposition_reference(
                              self->constraint_residual_case_);
                float max_error = 0.0F;
                bool valid = self->production_solve_ ||
                             actual.residual.size() == expected.residual.size();
                if (valid && !self->production_solve_) {
                    for (std::size_t i = 0; i < actual.residual.size(); ++i) {
                        max_error =
                            std::max(max_error, std::abs(actual.residual[i] -
                                                         expected.residual[i]));
                    }
                }
                if (!valid || max_error > 5.0e-4F) {
                    self->finish(
                        false, "constrained residual decomposition mismatch: " +
                                   std::to_string(max_error));
                    return;
                }
                if (self->production_solve_ &&
                    !std::all_of(
                        actual.residual.begin(), actual.residual.end(),
                        [](float value) { return std::isfinite(value); })) {
                    self->finish(false, "constrained residual is nonfinite");
                    return;
                }
                self->invariant_raw_ = actual.raw_norm;
                self->run_preconditioner_apply(std::move(actual.residual));
            });
    }

    void run_preconditioner_apply(std::vector<float> residual) {
        preconditioner_apply_case_.ns = initialized_stage_.ns;
        preconditioner_apply_case_.mpol = initialized_stage_.mpol;
        preconditioner_apply_case_.ntor = initialized_stage_.ntor;
        preconditioner_apply_case_.include_lcfs = false;
        preconditioner_apply_case_.elements = preconditioner_elements_;
        preconditioner_apply_case_.matrix = preconditioner_matrix_;
        preconditioner_apply_case_.residual = std::move(residual);
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_axisymmetric_preconditioner_apply(
            device_, preconditioner_apply_case_,
            [self](
                std::string error,
                cumes::webgpu::AxisymmetricPreconditionerApplyResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected =
                    self->production_solve_
                        ? cumes::webgpu::AxisymmetricPreconditionerApplyResult{}
                        : cumes::webgpu::
                              axisymmetric_preconditioner_apply_reference(
                                  self->preconditioner_apply_case_);
                float max_scaled_error = 0.0F;
                std::size_t max_error_index = 0;
                bool valid = self->production_solve_ ||
                             actual.residual.size() == expected.residual.size();
                if (!self->production_solve_) {
                    valid &= actual.breakdown_count == expected.breakdown_count;
                }
                if (valid && !self->production_solve_) {
                    for (std::size_t i = 0; i < actual.residual.size(); ++i) {
                        const float error =
                            std::abs(actual.residual[i] -
                                     expected.residual[i]) /
                            (1.0F + std::abs(expected.residual[i]));
                        if (error > max_scaled_error) {
                            max_scaled_error = error;
                            max_error_index = i;
                        }
                        valid &= std::isfinite(actual.residual[i]);
                    }
                } else if (self->production_solve_) {
                    valid = std::all_of(
                        actual.residual.begin(), actual.residual.end(),
                        [](float value) { return std::isfinite(value); });
                }
                if (!valid || actual.breakdown_count != 0 ||
                    max_scaled_error > 2.0e-4F) {
                    const float actual_value =
                        max_error_index < actual.residual.size()
                            ? actual.residual[max_error_index]
                            : 0.0F;
                    const float expected_value =
                        !self->production_solve_ &&
                                max_error_index < expected.residual.size()
                            ? expected.residual[max_error_index]
                            : 0.0F;
                    char detail[256];
                    std::snprintf(
                        detail, sizeof(detail),
                        "preconditioner apply mismatch: scaled=%.6g index=%zu "
                        "actual=%.9g expected=%.9g breakdown=%d/%d",
                        static_cast<double>(max_scaled_error), max_error_index,
                        static_cast<double>(actual_value),
                        static_cast<double>(expected_value),
                        actual.breakdown_count,
                        self->production_solve_ ? 0 : expected.breakdown_count);
                    self->finish(false, detail);
                    return;
                }
                if (!self->force_norm_ready_ ||
                    self->controller_->refresh_preconditioner()) {
                    cumes::webgpu::AxisymmetricForceNormalizationCase norm_case;
                    norm_case.ns = self->initialized_stage_.ns;
                    norm_case.mpol = self->initialized_stage_.mpol;
                    norm_case.ntor = self->initialized_stage_.ntor;
                    norm_case.ntheta = self->initialized_stage_.ntheta;
                    norm_case.nzeta = self->initialized_stage_.nzeta;
                    norm_case.delta_s =
                        self->initialized_stage_.profiles.delta_s;
                    norm_case.lamscale =
                        self->initialized_stage_.profiles.lamscale;
                    norm_case.state = self->initialized_stage_.state;
                    norm_case.base_geometry =
                        self->magnetic_field_case_.base_geometry;
                    norm_case.magnetic_field = self->force_case_.magnetic_field;
                    norm_case.pres_h = self->initialized_stage_.profiles.pres_h;
                    self->force_normalization_ =
                        cumes::webgpu::axisymmetric_force_normalization(
                            norm_case);
                    self->force_norm_ready_ = true;
                }
                const double plain =
                    static_cast<double>(self->initialized_stage_.ns) *
                    self->initialized_stage_.mpol *
                    (self->initialized_stage_.ntor + 1);
                self->invariant_normalized_ = {
                    self->invariant_raw_[0] * plain *
                        self->force_normalization_.f_norm_rz * 0.25,
                    self->invariant_raw_[1] * plain *
                        self->force_normalization_.f_norm_rz * 0.25,
                    self->invariant_raw_[2] * plain *
                        self->force_normalization_.f_norm_l};
                const auto preconditioned_raw =
                    cumes::webgpu::residual_raw_norms(
                        actual.residual, self->initialized_stage_.ns,
                        self->initialized_stage_.mpol,
                        self->initialized_stage_.ntor, true);
                self->preconditioned_normalized_ = {
                    preconditioned_raw[0] * plain *
                        self->force_normalization_.f_norm_1,
                    preconditioned_raw[1] * plain *
                        self->force_normalization_.f_norm_1,
                    preconditioned_raw[2] * plain *
                        self->initialized_stage_.profiles.delta_s};
                const auto verdict = self->controller_->classify_invariant(
                    self->invariant_normalized_.data());
                if (verdict.nonfinite) {
                    self->restore_checkpoint();
                    std::printf(
                        "  nonfinite residual restore: iter=%d delta=%.3e\n",
                        self->controller_->effective_iteration(),
                        self->controller_->delta_t());
                    self->run_stage_inverse();
                    return;
                }
                if (verdict.converged) {
                    self->complete_stage();
                    return;
                }
                self->pending_decision_ = self->controller_->decide_restart(
                    self->preconditioned_normalized_.data(),
                    self->invariant_normalized_.data());
                if (!self->production_solve_) {
                    std::printf(
                        "  %s preconditioned residual: PASS "
                        "(max scaled |GPU-CPU| = %.3e)\n",
                        self->active_case_name_.c_str(),
                        static_cast<double>(max_scaled_error));
                }
                self->run_descent(std::move(actual.residual));
            });
    }

    void run_descent(std::vector<float> residual) {
        descent_case_.ns = initialized_stage_.ns;
        descent_case_.mpol = initialized_stage_.mpol;
        descent_case_.ntor = initialized_stage_.ntor;
        descent_case_.move_lcfs = false;
        descent_case_.delta_t = static_cast<float>(controller_->delta_t());
        descent_case_.damping_b1 =
            static_cast<float>(pending_decision_.damping.b1);
        descent_case_.damping_fac =
            static_cast<float>(pending_decision_.damping.fac);
        descent_case_.double_single = double_single_solve_;
        descent_case_.state = initialized_stage_.state;
        descent_case_.velocity = stage_velocity_;
        if (double_single_solve_) {
            descent_case_.state_lo = stage_state_lo_;
            descent_case_.velocity_lo = stage_velocity_lo_;
        } else {
            descent_case_.state_lo.clear();
            descent_case_.velocity_lo.clear();
        }
        descent_case_.residual = std::move(residual);
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_axisymmetric_descent(
            device_, descent_case_,
            [self](std::string error,
                   cumes::webgpu::AxisymmetricDescentResult actual) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected =
                    self->production_solve_
                        ? cumes::webgpu::AxisymmetricDescentResult{}
                        : cumes::webgpu::axisymmetric_descent_reference(
                              self->descent_case_);
                float max_scaled_error = 0.0F;
                bool valid = true;
                const auto compare = [&max_scaled_error, &valid](
                                         const auto& gpu, const auto& cpu) {
                    valid &= gpu.size() == cpu.size();
                    if (gpu.size() != cpu.size()) return;
                    for (std::size_t i = 0; i < gpu.size(); ++i) {
                        max_scaled_error = std::max(
                            max_scaled_error, std::abs(gpu[i] - cpu[i]) /
                                                  (1.0F + std::abs(cpu[i])));
                        valid &= std::isfinite(gpu[i]);
                    }
                };
                if (!self->production_solve_) {
                    compare(actual.state, expected.state);
                    compare(actual.velocity, expected.velocity);
                } else {
                    valid =
                        std::all_of(
                            actual.state.begin(), actual.state.end(),
                            [](float value) { return std::isfinite(value); }) &&
                        std::all_of(
                            actual.velocity.begin(), actual.velocity.end(),
                            [](float value) { return std::isfinite(value); });
                    if (self->double_single_solve_) {
                        valid &=
                            actual.state_lo.size() == actual.state.size() &&
                            actual.velocity_lo.size() ==
                                actual.velocity.size() &&
                            std::all_of(actual.state_lo.begin(),
                                        actual.state_lo.end(),
                                        [](float value) {
                                            return std::isfinite(value);
                                        }) &&
                            std::all_of(actual.velocity_lo.begin(),
                                        actual.velocity_lo.end(),
                                        [](float value) {
                                            return std::isfinite(value);
                                        });
                    }
                }
                const std::size_t family_values =
                    static_cast<std::size_t>(self->descent_case_.ns) *
                    self->descent_case_.mpol * (self->descent_case_.ntor + 1);
                const int mode_count =
                    self->descent_case_.mpol * (self->descent_case_.ntor + 1);
                bool fixed_lcfs = true;
                for (const int component : {0, 1, 3, 4}) {
                    for (int mode = 0; mode < mode_count; ++mode) {
                        const std::size_t lcfs =
                            static_cast<std::size_t>(component) *
                                family_values +
                            static_cast<std::size_t>(mode) *
                                self->descent_case_.ns +
                            self->descent_case_.ns - 1;
                        fixed_lcfs &= actual.state[lcfs] ==
                                      self->descent_case_.state[lcfs];
                        if (self->double_single_solve_ &&
                            actual.state_lo.size() == actual.state.size()) {
                            fixed_lcfs &= actual.state_lo[lcfs] ==
                                          self->descent_case_.state_lo[lcfs];
                        }
                    }
                }
                if (!valid || !fixed_lcfs || max_scaled_error > 2.0e-5F) {
                    self->finish(false, "descent mismatch: " +
                                            std::to_string(max_scaled_error));
                    return;
                }
                if (!self->production_solve_) {
                    std::printf(
                        "  %s accelerated descent: PASS "
                        "(max scaled |GPU-CPU| = %.3e)\n",
                        self->active_case_name_.c_str(),
                        static_cast<double>(max_scaled_error));
                }
                if (self->pending_decision_.do_refresh) {
                    self->checkpoint_state_ = actual.state;
                    if (self->double_single_solve_) {
                        self->checkpoint_state_lo_ = actual.state_lo;
                    }
                }
                if (self->pending_decision_.reason !=
                    cumes::RestartReason::NONE) {
                    self->restore_checkpoint();
                } else {
                    self->initialized_stage_.state = std::move(actual.state);
                    self->stage_velocity_ = std::move(actual.velocity);
                    if (self->double_single_solve_) {
                        self->stage_state_lo_ = std::move(actual.state_lo);
                        self->stage_velocity_lo_ =
                            std::move(actual.velocity_lo);
                    }
                }
                self->controller_->after_descent(self->pending_decision_);
                ++self->completed_passes_;
                const int iteration = self->controller_->effective_iteration();
                if (!self->production_solve_ || iteration <= 3 ||
                    iteration % 25 == 0) {
                    std::printf(
                        "  host controller: PASS (iter=%d, FSQR=%.3e, "
                        "delta=%.3e)\n",
                        iteration, self->invariant_normalized_[0],
                        self->controller_->delta_t());
                }
                if (self->w7x_stage_slice_ && self->completed_passes_ == 2) {
                    std::printf(
                        "  W7-X controller-complete two-pass slice: PASS "
                        "(FSQR=%.3e)\n",
                        self->invariant_normalized_[0]);
                    self->w7x_stage_slice_ = false;
                    self->run_solovev_initialization();
                    return;
                }
                self->run_stage_inverse();
            });
    }

    void restore_checkpoint() {
        initialized_stage_.state = checkpoint_state_;
        stage_velocity_.assign(initialized_stage_.state.size(), 0.0F);
        if (double_single_solve_) {
            stage_state_lo_ = checkpoint_state_lo_;
            stage_velocity_lo_.assign(initialized_stage_.state.size(), 0.0F);
        }
    }

    void reset_stage_state() {
        controller_.emplace(cumes::IterationController<double>::Options{
            static_cast<double>(initialized_stage_.delta_t),
            initialized_stage_.tolerance, 0.0, false});
        stage_velocity_.assign(initialized_stage_.state.size(), 0.0F);
        checkpoint_state_ = initialized_stage_.state;
        if (double_single_solve_) {
            if (stage_state_lo_.size() != initialized_stage_.state.size()) {
                stage_state_lo_ = initialized_stage_.state_lo;
            }
            stage_velocity_lo_.assign(initialized_stage_.state.size(), 0.0F);
            checkpoint_state_lo_ = stage_state_lo_;
        } else {
            stage_state_lo_.clear();
            stage_velocity_lo_.clear();
            checkpoint_state_lo_.clear();
        }
        constraint_r_con0_.clear();
        constraint_z_con0_.clear();
        constraint_tcon_.clear();
        preconditioner_elements_ = {};
        preconditioner_matrix_ = {};
        force_normalization_ = {};
        force_norm_ready_ = false;
        completed_passes_ = 0;
        attempted_passes_ = 0;
    }

    void complete_stage() {
        const int stage_iterations = controller_->effective_iteration();
        stage_iterations_.push_back(stage_iterations);
        total_iterations_ += stage_iterations;
        cumes::StageReport stage_report;
        stage_report.ns = initialized_stage_.ns;
        stage_report.effective_iterations = stage_iterations;
        stage_report.converged = true;
        stage_report.final_residual = cumes::ResidualTriple{
            invariant_normalized_[0], invariant_normalized_[1],
            invariant_normalized_[2]};
        stage_report.restarts = controller_->restart_events();
        stage_reports_.push_back(std::move(stage_report));
        std::printf(
            "  %s stage %zu/%zu converged: iter=%d residual=(%.3e, "
            "%.3e, %.3e)\n",
            active_case_name_.c_str(), stage_index_ + 1,
            problem_->stage_shapes().size(), stage_iterations,
            invariant_normalized_[0], invariant_normalized_[1],
            invariant_normalized_[2]);
        if (stage_index_ + 1 == problem_->stage_shapes().size()) {
            const std::string output_error = publish_output();
            if (!output_error.empty()) {
                finish(false, output_error);
                return;
            }
            char detail[320];
            std::snprintf(detail, sizeof(detail),
                          "%s multigrid converged in %d effective iterations; "
                          "final residual=(%.3e, %.3e, %.3e)",
                          active_case_name_.c_str(), total_iterations_,
                          invariant_normalized_[0], invariant_normalized_[1],
                          invariant_normalized_[2]);
            finish(true, detail);
            return;
        }

        const std::size_t next_stage = stage_index_ + 1;
        const auto& next_shape = problem_->stage_shapes()[next_stage];
        cumes::webgpu::ProlongationCase transfer;
        transfer.ns_old = initialized_stage_.ns;
        transfer.ns_new = next_shape.ns;
        transfer.mnmax =
            initialized_stage_.mpol * (initialized_stage_.ntor + 1);
        transfer.ntor = initialized_stage_.ntor;
        transfer.interpolation = cumes::webgpu::RadialInterpolation::LINEAR;
        transfer.state = initialized_stage_.state;
        auto low_transfer = transfer;
        low_transfer.state = stage_state_lo_;
        const auto self = shared_from_this();
        cumes::webgpu::enqueue_prolongation(
            device_, transfer,
            [self, transfer, low_transfer, next_stage](
                std::string error,
                cumes::webgpu::ProlongationResult prolonged) {
                if (!error.empty()) {
                    self->finish(false, std::move(error));
                    return;
                }
                const auto expected =
                    cumes::webgpu::prolongation_reference(transfer);
                float max_error = 0.0F;
                bool valid =
                    prolonged.state.size() == expected.state.size() &&
                    prolonged.velocity.size() == expected.velocity.size();
                if (valid) {
                    for (std::size_t i = 0; i < prolonged.state.size(); ++i) {
                        max_error = std::max(
                            max_error,
                            std::abs(prolonged.state[i] - expected.state[i]));
                        valid &= prolonged.velocity[i] == 0.0F;
                    }
                }
                if (!valid || max_error > 4.0e-6F) {
                    self->finish(false, "stage prolongation mismatch: " +
                                            std::to_string(max_error));
                    return;
                }
                self->stage_index_ = next_stage;
                std::vector<float> prolonged_lo;
                if (self->double_single_solve_) {
                    const auto low =
                        cumes::webgpu::prolongation_reference(low_transfer);
                    prolonged_lo.resize(prolonged.state.size());
                    for (std::size_t i = 0; i < prolonged.state.size(); ++i) {
                        const auto combined = cumes::webgpu::add(
                            cumes::webgpu::FloatFloat{prolonged.state[i], 0.0F},
                            low.state[i]);
                        prolonged.state[i] = combined.hi;
                        prolonged_lo[i] = combined.lo;
                    }
                }
                self->initialized_stage_ = cumes::webgpu::initialize_stage(
                    *self->problem_, next_stage);
                self->initialized_stage_.state = std::move(prolonged.state);
                self->stage_state_lo_ = std::move(prolonged_lo);
                self->reset_stage_state();
                std::printf(
                    "  %s stage prolongation: PASS (ns=%d -> %d, max "
                    "|GPU-CPU| = %.3e)\n",
                    self->active_case_name_.c_str(), transfer.ns_old,
                    transfer.ns_new, static_cast<double>(max_error));
                self->run_stage_inverse();
            });
    }

    std::string publish_output() {
        cumes::EquilibriumSnapshot snapshot;
        snapshot.ns = initialized_stage_.ns;
        snapshot.mnmax =
            initialized_stage_.mpol * (initialized_stage_.ntor + 1);
        const std::size_t family_values =
            static_cast<std::size_t>(snapshot.ns) * snapshot.mnmax;
        for (std::size_t component = 0;
             component < cumes::EquilibriumSnapshot::COUNT; ++component) {
            const auto begin =
                initialized_stage_.state.begin() + component * family_values;
            snapshot.families[component].assign(begin, begin + family_values);
            if (double_single_solve_) {
                const auto low_begin =
                    stage_state_lo_.begin() + component * family_values;
                for (std::size_t i = 0; i < family_values; ++i) {
                    snapshot.families[component][i] +=
                        static_cast<double>(low_begin[i]);
                }
            }
        }

        cumes::DerivedFieldInputs fields;
        fields.ns = initialized_stage_.ns;
        fields.ntheta = initialized_stage_.ntheta;
        fields.nzeta = initialized_stage_.nzeta;
        fields.nfp = problem_->spec().nfp;
        fields.delta_s = initialized_stage_.profiles.delta_s;
        fields.mu0 = static_cast<double>(DeviceParams<float>::MU_0);
        fields.sqrt_s_full.assign(initialized_stage_.profiles.sqrt_s_f.begin(),
                                  initialized_stage_.profiles.sqrt_s_f.end());
        fields.sqrt_s_half.assign(initialized_stage_.profiles.sqrt_s_h.begin(),
                                  initialized_stage_.profiles.sqrt_s_h.end());
        const std::size_t full_points =
            static_cast<std::size_t>(fields.ns) * fields.ntheta * fields.nzeta;
        const std::size_t half_points =
            static_cast<std::size_t>(fields.ns - 1) * fields.ntheta *
            fields.nzeta;
        const auto copy_field = [](const std::vector<float>& source,
                                   std::size_t field, std::size_t count) {
            const auto begin = source.begin() + field * count;
            return std::vector<double>(begin, begin + count);
        };
        const auto& full = base_geometry_case_.geometry;
        fields.r_e = copy_field(full, 0, full_points);
        fields.z_e = copy_field(full, 1, full_points);
        fields.ru_e = copy_field(full, 3, full_points);
        fields.zu_e = copy_field(full, 4, full_points);
        fields.r_o = copy_field(full, 6, full_points);
        fields.z_o = copy_field(full, 7, full_points);
        fields.ru_o = copy_field(full, 9, full_points);
        fields.zu_o = copy_field(full, 10, full_points);
        fields.rv_e = copy_field(full, 12, full_points);
        fields.zv_e = copy_field(full, 13, full_points);
        fields.rv_o = copy_field(full, 15, full_points);
        fields.zv_o = copy_field(full, 16, full_points);
        const auto& base = magnetic_field_case_.base_geometry;
        fields.ru12 = copy_field(base, 1, half_points);
        fields.zu12 = copy_field(base, 2, half_points);
        fields.rs = copy_field(base, 3, half_points);
        fields.zs = copy_field(base, 4, half_points);
        fields.sqrtg = copy_field(base, 6, half_points);
        const auto& magnetic = force_case_.magnetic_field;
        fields.bsupu = copy_field(magnetic, 0, half_points);
        fields.bsupv = copy_field(magnetic, 1, half_points);
        fields.bsubu = copy_field(magnetic, 2, half_points);
        fields.bsubv = copy_field(magnetic, 3, half_points);
        const cumes::Status derived_status =
            cumes::populate_derived_fields(fields, snapshot);
        if (!derived_status.has_value()) {
            return "WebGPU derived fields failed: " + derived_status.error();
        }

        publish_equilibrium_plot();

        cumes::RunReport report;
        report.status = cumes::RunStatus::CONVERGED;
        report.total_effective_iterations = total_iterations_;
        report.stages = stage_reports_;
        report.build.revision = CUMES_GIT_REVISION;
        report.build.dirty = CUMES_GIT_DIRTY != 0;
        report.build.build_type = CUMES_BUILD_TYPE;
        report.build.scalar_type = "float";
        report.build.precision_policy = CUMES_PRECISION_POLICY_NAME;
        report.build.compile_flags = CUMES_PRECISION_FLAGS;
        report.input.source_path = active_input_path_;
        report.runtime.gpu_name = "WebGPU adapter";
        report.runtime.runtime = "emdawnwebgpu";
        report.runtime.toolkit = "Emscripten";
        report.input_params = cumes::make_input_params(*problem_);

        const cumes::OutputSpec spec{cumes::OutputFormat::BINARY,
                                     "/cumes-output.bin"};
        auto writer = cumes::make_binary_writer();
        if (!writer) return "WebGPU binary writer is unavailable";
        const cumes::Status status =
            writer->write_atomic(snapshot, report, spec, *problem_);
        if (!status.has_value()) {
            return "WebGPU output failed: " + status.error();
        }
        auto reader = cumes::make_binary_reader();
        if (!reader) return "WebGPU binary reader is unavailable";
        cumes::RunReport roundtrip_report;
        auto roundtrip = reader->read(spec.path, std::ref(roundtrip_report));
        if (!roundtrip.has_value()) {
            return "WebGPU output round-trip failed: " + roundtrip.error();
        }
        const auto& recovered = roundtrip.value();
        if (recovered.ns != snapshot.ns || recovered.mnmax != snapshot.mnmax ||
            recovered.families != snapshot.families ||
            !recovered.has_derived_fields() ||
            recovered.half_fields != snapshot.half_fields ||
            recovered.full_fields != snapshot.full_fields ||
            roundtrip_report.status != cumes::RunStatus::CONVERGED ||
            roundtrip_report.total_effective_iterations != total_iterations_ ||
            roundtrip_report.stages.size() != stage_reports_.size() ||
            !(roundtrip_report.input_params == report.input_params)) {
            return "WebGPU output round-trip contract mismatch";
        }
        const int output_bytes = publish_browser_output(spec.path.c_str());
        if (output_bytes <= 0) {
            return "WebGPU output could not be exposed for download";
        }
        std::printf("  published schema-v8 output: %d bytes\n", output_bytes);
        return {};
    }

    void publish_equilibrium_plot() const {
        const int ns = initialized_stage_.ns;
        const int mpol = initialized_stage_.mpol;
        const int ntor = initialized_stage_.ntor;
        const int mnmax = mpol * (ntor + 1);
        const auto& state = initialized_stage_.state;
        const std::size_t family_values = static_cast<std::size_t>(ns) * mnmax;
        if (ns <= 0 || mpol <= 0 || state.size() < 2 * family_values) return;

        std::ostringstream json;
        json << std::setprecision(9);
        json << "{\"name\":\"" << active_case_name_
             << "\",\"iterations\":" << total_iterations_ << ",\"residual\":["
             << invariant_normalized_[0] << ',' << invariant_normalized_[1]
             << ',' << invariant_normalized_[2] << "],\"surfaces\":[";
        const int stride = std::max(1, (ns - 1) / 18);
        bool first_surface = true;
        for (int surface = 0; surface < ns; ++surface) {
            if (surface != 0 && surface != ns - 1 && surface % stride != 0) {
                continue;
            }
            if (!first_surface) json << ',';
            first_surface = false;
            json << '[';
            constexpr int PLOT_SEGMENTS = 128;
            for (int theta_index = 0; theta_index <= PLOT_SEGMENTS;
                 ++theta_index) {
                if (theta_index != 0) json << ',';
                const double theta = 2.0 * std::numbers::pi_v<double> *
                                     theta_index / PLOT_SEGMENTS;
                double r = 0.0;
                double z = 0.0;
                for (int m = 0; m < mpol; ++m) {
                    if (surface == 0 && m % 2 == 1) continue;
                    const double cosine = std::cos(m * theta);
                    const double sine = std::sin(m * theta);
                    for (int n = 0; n <= ntor; ++n) {
                        const int mode = m * (ntor + 1) + n;
                        const std::size_t offset =
                            static_cast<std::size_t>(mode) * ns + surface;
                        r += state[offset] * cosine;
                        z += state[family_values + offset] * sine;
                    }
                }
                json << '[' << r << ',' << z << ']';
            }
            json << ']';
        }
        json << "]}";
        publish_browser_equilibrium(json.str().c_str());
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
    cumes::webgpu::ToroidalInverseCase toroidal_case_;
    cumes::webgpu::ToroidalForwardCase toroidal_forward_case_;
    cumes::webgpu::ToroidalDealiasCase toroidal_dealias_case_;
    cumes::webgpu::AxisymmetricConstraintCase toroidal_constraint_case_;
    cumes::webgpu::BaseGeometryCase toroidal_geometry_case_;
    cumes::webgpu::MagneticFieldCase toroidal_magnetic_case_;
    cumes::webgpu::AxisymmetricForceCase toroidal_force_case_;
    cumes::webgpu::AxisymmetricPreconditionerElementCase
        toroidal_preconditioner_case_;
    cumes::webgpu::AxisymmetricPreconditionerElements
        toroidal_preconditioner_elements_;
    cumes::webgpu::AxisymmetricPreconditionerMatrixCase
        toroidal_preconditioner_matrix_case_;
    cumes::webgpu::AxisymmetricPreconditionerMatrix
        toroidal_preconditioner_matrix_;
    cumes::webgpu::AxisymmetricPreconditionerApplyCase
        toroidal_preconditioner_apply_case_;
    cumes::webgpu::AxisymmetricStageData initialized_stage_;
    cumes::webgpu::ToroidalInverseCase stage_toroidal_inverse_case_;
    cumes::webgpu::BaseGeometryCase base_geometry_case_;
    cumes::webgpu::MagneticFieldCase magnetic_field_case_;
    cumes::webgpu::AxisymmetricForceCase force_case_;
    cumes::webgpu::AxisymmetricForwardCase solver_forward_case_;
    cumes::webgpu::ToroidalForwardCase solver_toroidal_forward_case_;
    cumes::webgpu::ResidualDecompositionCase residual_case_;
    cumes::webgpu::AxisymmetricConstraintCase constraint_case_;
    cumes::webgpu::AxisymmetricForwardCase constraint_forward_case_;
    cumes::webgpu::ToroidalForwardCase constraint_toroidal_forward_case_;
    cumes::webgpu::AxisymmetricPreconditionerElementCase preconditioner_case_;
    cumes::webgpu::AxisymmetricPreconditionerElements preconditioner_elements_;
    cumes::webgpu::AxisymmetricPreconditionerMatrixCase
        preconditioner_matrix_case_;
    cumes::webgpu::AxisymmetricPreconditionerMatrix preconditioner_matrix_;
    cumes::webgpu::ResidualDecompositionCase constraint_residual_case_;
    cumes::webgpu::AxisymmetricPreconditionerApplyCase
        preconditioner_apply_case_;
    cumes::webgpu::AxisymmetricDescentCase descent_case_;
    std::optional<cumes::IterationController<double>> controller_;
    std::optional<cumes::ValidatedProblem> problem_;
    std::optional<cumes::ValidatedProblem> w7x_problem_;
    cumes::webgpu::AxisymmetricStageData w7x_stage_;
    cumes::webgpu::ToroidalInverseCase w7x_inverse_case_;
    cumes::webgpu::BaseGeometryCase w7x_geometry_case_;
    cumes::webgpu::MagneticFieldCase w7x_magnetic_case_;
    cumes::webgpu::AxisymmetricForceCase w7x_force_case_;
    cumes::webgpu::ToroidalForwardCase w7x_forward_case_;
    cumes::webgpu::ResidualDecompositionCase w7x_residual_case_;
    cumes::webgpu::AxisymmetricDescentCase w7x_descent_case_;
    cumes::RestartDecision<double> pending_decision_;
    std::array<double, 3> invariant_raw_{};
    std::array<double, 3> invariant_normalized_{};
    std::array<double, 3> preconditioned_normalized_{};
    cumes::webgpu::ForceNormalizationResult force_normalization_;
    bool force_norm_ready_ = false;
    bool w7x_stage_slice_ = false;
    bool production_solve_ = false;
    bool double_single_solve_ = false;
    std::vector<float> stage_r_con_;
    std::vector<float> stage_z_con_;
    std::vector<float> stage_geometry_lo_;
    std::vector<float> stage_base_geometry_lo_;
    std::vector<float> stage_r_con_lo_;
    std::vector<float> stage_z_con_lo_;
    std::vector<float> stage_force_fields_;
    std::vector<float> toroidal_r_con_;
    std::vector<float> toroidal_z_con_;
    std::vector<float> w7x_r_con_;
    std::vector<float> w7x_z_con_;
    std::vector<float> w7x_geometry_lo_;
    std::vector<float> w7x_base_geometry_lo_;
    std::vector<float> constraint_r_con0_;
    std::vector<float> constraint_z_con0_;
    std::vector<float> constraint_tcon_;
    std::vector<float> stage_velocity_;
    std::vector<float> stage_state_lo_;
    std::vector<float> stage_velocity_lo_;
    std::vector<float> checkpoint_state_;
    std::vector<float> checkpoint_state_lo_;
    int completed_passes_ = 0;
    int attempted_passes_ = 0;
    std::size_t stage_index_ = 0;
    int total_iterations_ = 0;
    std::vector<int> stage_iterations_;
    std::vector<cumes::StageReport> stage_reports_;
    std::size_t case_index_ = 0;
    std::size_t forward_case_index_ = 0;
    std::string active_case_name_ = "Solovev";
    std::string active_input_path_ = "inputs/solovev.json";
    std::string gpu_error_;
};

}  // namespace

int main() {
    if (requested_app_mode() != 0 && requested_app_run() == 0) {
        publish_browser_ready();
        return 0;
    }
    std::make_shared<BrowserSelfTest>()->start();
    return 0;
}
