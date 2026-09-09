#include "float_geometry_tests.hpp"

#include "cumes/config/json_reader.hpp"
#include "cumes/config/validated_problem.hpp"
#include "cumes/webgpu/force.hpp"
#include "cumes/webgpu/geometry.hpp"
#include "cumes/webgpu/initialization.hpp"
#include "cumes/webgpu/toroidal.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <sstream>
#include <utility>

namespace cumes::webgpu {
namespace {

class FloatGeometryTest
    : public std::enable_shared_from_this<FloatGeometryTest> {
   public:
    wgpu::Device device;
    std::function<void(std::string)> callback;

    void start() {
        SolverOptions options;
        options.precision = PrecisionPolicy::VERIFY_DOUBLE;
        auto parsed = read_problem_spec("/inputs/w7x.json", options);
        auto validated = validate(std::move(parsed.spec), options);
        if (!validated.has_value()) {
            callback("float geometry seed test input failed");
            return;
        }
        const auto& problem = validated.value();
        for (std::size_t k = 0; k < problem.stage_shapes().size(); ++k) {
            const auto plain = initialize_stage(problem, k);
            const auto relative = initialize_stage(problem, k, true, true);
            const auto toroidal =
                initialize_stage(problem, k, true, true, true);
            bool valid =
                !plain.radius_reference && !plain.compensated_geometry &&
                !plain.compensated_toroidal_geometry &&
                relative.radius_reference && relative.compensated_geometry &&
                !relative.compensated_toroidal_geometry &&
                toroidal.compensated_geometry &&
                toroidal.compensated_toroidal_geometry &&
                toroidal.state == relative.state;
            for (int n = 0; n <= relative.ntor; ++n) {
                valid &= relative.radius_reference->coefficients[n] ==
                         problem.boundary().rbcc[n];
                for (int j = 0; j < relative.ns; ++j) {
                    const float expected = static_cast<float>(
                        (1.0 - double(j) / (relative.ns - 1)) *
                        (problem.spec().raxis_c[n] -
                         problem.boundary().rbcc[n]));
                    valid &= relative.state[n * relative.ns + j] == expected;
                }
                valid &= relative.state[(n + 1) * relative.ns - 1] == 0.0F;
            }
            const auto shifted = relative.ns * (relative.ntor + 1);
            valid &=
                std::equal(plain.state.begin() + shifted, plain.state.end(),
                           relative.state.begin() + shifted);
            if (!valid) {
                callback("float radius-reference seed/LCFS/isolation mismatch");
                return;
            }
        }
        input_.ns = 4;
        input_.mpol = 6;
        input_.ntor = 2;
        input_.ntheta = 16;
        input_.nzeta = 8;
        input_.nfp = 5;
        input_.state.assign(6 * 4 * 6 * 3, 0.0F);
        auto reference = std::make_shared<FloatRadiusReference>();
        reference->coefficients = {5.5, 0.125, -0.0625};
        reference->angular.resize(128);
        for (int k = 0; k < 8; ++k) {
            const float zeta = 2.0F * std::numbers::pi_v<float> * k / 8.0F;
            for (int t = 0; t < 16; ++t)
                reference->angular[16 * k + t] =
                    0.125F * std::cos(zeta) - 0.0625F * std::cos(2.0F * zeta);
        }
        input_.radius_reference = reference;
        // Sub-ULP radial variation must survive differentiation, including
        // with a nonconstant n>0 reference. Odd modes cancel at theta=0.
        for (int j = 0; j < 4; ++j) {
            input_.state[j] = float(j) * 0x1p-25F;
            put(0, 1, j, 0.75F);
            put(0, 3, j, -0.75F + 0x1p-24F);
            put(0, 5, j, 0x1p-27F);
            put(1, 1, j, 0.3F);
            put(1, 3, j, -0.3F);
            put(1, 5, j, 0x1p-26F);
        }
        const auto self = shared_from_this();
        enqueue_toroidal_inverse(
            device, input_,
            [self](std::string error, ToroidalInverseResult value) {
                if (!error.empty()) {
                    self->callback(error);
                    return;
                }
                self->native_ = std::move(value);
                self->compensated();
            });
    }

   private:
    ToroidalInverseCase input_;
    ToroidalInverseResult native_;
    BaseGeometryCase geometry_;
    AxisymmetricForceCase force_;

    void put(int family, int m, int surface, float value) {
        input_.state[(family * 18 + m * 3) * 4 + surface] = value;
    }

    void compensated() {
        input_.compensated_geometry = true;
        const auto self = shared_from_this();
        enqueue_toroidal_inverse(
            device, input_,
            [self](std::string error, ToroidalInverseResult value) {
                if (!error.empty()) {
                    self->callback(error);
                    return;
                }
                constexpr std::size_t POINTS = 4 * 128;
                bool valid = value.geometry.size() == 18 * POINTS;
                std::string mismatch;
                for (std::size_t field = 0; field < 18 && valid; ++field)
                    for (std::size_t p = 0; p < POINTS; ++p) {
                        const auto i = field * POINTS + p;
                        if (field != 6 && field != 7) {
                            if (value.geometry[i] !=
                                self->native_.geometry[i]) {
                                valid = false;
                                mismatch = "uncompensated field=" +
                                           std::to_string(field) +
                                           " point=" + std::to_string(p);
                                break;
                            }
                            continue;
                        }
                        const int j = int(p / 128), t = int(p % 16);
                        const float theta =
                            2.0F * std::numbers::pi_v<float> * t / 16.0F;
                        const float scale =
                            1.0F / std::max(std::sqrt(float(j) / 3.0F),
                                            std::sqrt(1.0F / 3.0F));
                        double sum = 0.0;
                        const int family = field == 6 ? 0 : 1;
                        for (int m = 1; m < 6; m += 2) {
                            const float basis =
                                field == 6 ? std::cos(float(m) * theta)
                                           : std::sin(float(m) * theta);
                            sum +=
                                double(
                                    self->input_
                                        .state[(family * 18 + m * 3) * 4 + j]) *
                                basis;
                        }
                        const float expected = static_cast<float>(sum * scale);
                        const float ulp = std::abs(
                            std::nextafter(expected, INFINITY) - expected);
                        if (!std::isfinite(value.geometry[i]) ||
                            std::abs(value.geometry[i] - expected) >
                                2.0F * ulp) {
                            valid = false;
                            std::ostringstream detail;
                            detail.precision(9);
                            detail << "field=" << field << " point=" << p
                                   << " actual=" << value.geometry[i]
                                   << " expected=" << expected
                                   << " ULP=" << ulp;
                            mismatch = detail.str();
                            break;
                        }
                    }
                valid &= value.r_con == self->native_.r_con &&
                         value.z_con == self->native_.z_con;
                // m=0 position stays displaced, but its zeta derivative is
                // physical.
                const auto expected = toroidal_inverse_reference(self->input_);
                for (std::size_t p = 0; p < POINTS; ++p)
                    valid &=
                        std::abs(value.geometry[12 * POINTS + p] -
                                 expected.geometry[12 * POINTS + p]) < 2.0e-6F;
                if (!valid) {
                    self->callback(
                        "float compensated inverse/reference isolation "
                        "mismatch: " +
                        mismatch);
                    return;
                }
                self->geometry_.ns = 4;
                self->geometry_.ntheta = 16;
                self->geometry_.nzeta = 8;
                self->geometry_.delta_s = 1.0F / 3.0F;
                self->geometry_.radius_reference =
                    self->input_.radius_reference;
                self->geometry_.geometry = std::move(value.geometry);
                // Isolate the even-radius derivative from odd radial terms.
                std::fill(self->geometry_.geometry.begin() + 6 * POINTS,
                          self->geometry_.geometry.begin() + 8 * POINTS, 0.0F);
                for (int j = 0; j < 4; ++j)
                    self->geometry_.sqrt_s_f.push_back(
                        std::sqrt(float(j) / 3.0F));
                for (int j = 0; j < 3; ++j)
                    self->geometry_.sqrt_s_h.push_back(
                        std::sqrt((float(j) + 0.5F) / 3.0F));
                self->geometry();
            });
    }

    void geometry() {
        const auto self = shared_from_this();
        enqueue_base_geometry(
            device, geometry_,
            [self](std::string error, BaseGeometryResult value) {
                if (!error.empty()) {
                    self->callback(error);
                    return;
                }
                const auto expected = base_geometry_reference(self->geometry_);
                bool valid = value.fields.size() == expected.fields.size();
                for (std::size_t i = 0; i < value.fields.size() && valid; ++i)
                    valid &=
                        std::isfinite(value.fields[i]) &&
                        std::abs(value.fields[i] - expected.fields[i]) <
                            3.0e-6F *
                                std::max(1.0F, std::abs(expected.fields[i]));
                for (int p = 0; p < 3 * 128; ++p)
                    valid &= value.fields[3 * 3 * 128 + p] == 3.0F * 0x1p-25F;
                if (!valid) {
                    self->callback(
                        "float radius-reference radial derivative/metric "
                        "mismatch");
                    return;
                }
                auto& force = self->force_;
                force.ns = 4;
                force.ntheta = 16;
                force.nzeta = 8;
                force.delta_s = 1.0F / 3.0F;
                force.lamscale = 1.0F;
                force.radius_reference = self->input_.radius_reference;
                force.geometry = self->geometry_.geometry;
                force.base_geometry = std::move(value.fields);
                // Nonzero gsqrt/Bzeta makes force depend on absolute radius.
                std::fill(force.base_geometry.begin() + 6 * 384,
                          force.base_geometry.begin() + 7 * 384, 1.0F);
                force.magnetic_field.assign(5 * 384, 0.125F);
                force.sqrt_s_f = self->geometry_.sqrt_s_f;
                force.sqrt_s_h = self->geometry_.sqrt_s_h;
                force.phip_f.assign(4, 1.0F);
                self->force();
            });
    }

    void force() {
        const auto self = shared_from_this();
        enqueue_axisymmetric_force(
            device, force_,
            [self](std::string error, AxisymmetricForceResult relative) {
                if (!error.empty()) {
                    self->callback(error);
                    return;
                }
                for (std::size_t p = 0; p < 512; ++p)
                    self->force_.geometry[p] =
                        self->force_.radius_reference->restore(
                            self->force_.geometry[p], p);
                self->force_.radius_reference.reset();
                enqueue_axisymmetric_force(
                    self->device, self->force_,
                    [self, relative = std::move(relative)](
                        std::string message, AxisymmetricForceResult absolute) {
                        if (message.empty() &&
                            relative.fields != absolute.fields)
                            message =
                                "float radius-reference absolute force "
                                "mismatch";
                        if (!message.empty())
                            self->callback(std::move(message));
                        else
                            self->toroidal_compensated();
                    });
            });
    }

    void toroidal_compensated(int ns = 4) {
        input_.ns = ns;
        input_.radius_reference.reset();
        input_.compensated_geometry = true;
        input_.compensated_toroidal_geometry = false;
        input_.state.assign(6 * 18 * std::size_t(ns), 0.0F);
        const auto coefficient = [this](int family, int m, int n, int surface,
                                        float value) {
            input_.state[(family * 18 + m * 3 + n) * input_.ns + surface] =
                value;
        };
        // Each m=1 toroidal sum carries a sub-ULP term that matters after
        // cancellation with m=3. Exercise all four R/Z harmonic families.
        for (int j = 0; j < ns; ++j) {
            coefficient(0, 1, 0, j, 0.75F);
            coefficient(0, 1, 1, j, 0x1p-27F);
            coefficient(0, 3, 0, j, -0.75F);
            coefficient(1, 1, 0, j, 0.5F);
            coefficient(1, 1, 1, j, 0x1p-26F);
            coefficient(1, 3, 0, j, 0.5F);
            coefficient(3, 1, 1, j, 0.375F);
            coefficient(3, 1, 2, j, 0x1p-28F);
            coefficient(3, 3, 1, j, 0.375F);
            coefficient(4, 1, 1, j, 0.625F);
            coefficient(4, 1, 2, j, 0x1p-27F);
            coefficient(4, 3, 1, j, -0.625F);
        }
        const auto self = shared_from_this();
        enqueue_toroidal_inverse(
            device, input_,
            [self](std::string error, ToroidalInverseResult baseline) {
                if (!error.empty()) {
                    self->callback(std::move(error));
                    return;
                }
                self->native_ = std::move(baseline);
                self->input_.compensated_toroidal_geometry = true;
                self->check_toroidal_compensated();
            });
    }

    void check_toroidal_compensated() {
        const auto self = shared_from_this();
        const auto expected = toroidal_inverse_reference(input_);
        enqueue_toroidal_inverse(
            device, input_,
            [self, expected](std::string error, ToroidalInverseResult value) {
                if (!error.empty()) {
                    self->callback(std::move(error));
                    return;
                }
                const auto points = std::size_t(self->input_.ns) * 128;
                const auto same_bits = [](float a, float b) {
                    return std::bit_cast<std::uint32_t>(a) ==
                           std::bit_cast<std::uint32_t>(b);
                };
                bool valid = value.geometry.size() == 18 * points &&
                             std::equal(value.r_con.begin(), value.r_con.end(),
                                        self->native_.r_con.begin(),
                                        self->native_.r_con.end(), same_bits) &&
                             std::equal(value.z_con.begin(), value.z_con.end(),
                                        self->native_.z_con.begin(),
                                        self->native_.z_con.end(), same_bits);
                if (!valid) {
                    self->callback(
                        "m=1 toroidal output shape/constraint mismatch");
                    return;
                }
                for (std::size_t field = 0; field < 18 && valid; ++field) {
                    for (std::size_t p = 0; p < points; ++p) {
                        const auto i = field * points + p;
                        if (field != 6 && field != 7) {
                            valid &= same_bits(value.geometry[i],
                                               self->native_.geometry[i]);
                        } else {
                            const float reference = expected.geometry[i];
                            const float ulp =
                                std::abs(std::nextafter(reference, INFINITY) -
                                         reference);
                            // Near angular zeros, cancellation exposes the
                            // rounding of both the double oracle and paired
                            // sums of unit-scale terms. Allow 2^-45 absolute
                            // there; the analytic case below stays exact.
                            valid &= std::isfinite(value.geometry[i]) &&
                                     std::abs(value.geometry[i] - reference) <=
                                         2.0F * ulp + 0x1p-45F;
                        }
                        if (!valid) {
                            std::ostringstream detail;
                            detail.precision(9);
                            detail << "m=1 toroidal compensation mismatch: "
                                   << "field=" << field << " point=" << p
                                   << " actual=" << value.geometry[i]
                                   << " expected=" << expected.geometry[i];
                            self->callback(detail.str());
                            return;
                        }
                    }
                }
                // At theta=zeta=0 the Rcc terms reduce analytically to
                // sqrt(ns-1) * 2^-27 on surface 1. The old scalar toroidal
                // intermediate loses this term before the m=3 cancellation.
                const auto index = 6 * points + 128;
                const float exact =
                    float(std::sqrt(double(self->input_.ns - 1)) * 0x1p-27);
                valid &= value.geometry[index] == exact &&
                         self->native_.geometry[index] == 0.0F;
                if (!valid)
                    self->callback(
                        "m=1 toroidal cancellation lost radial detail");
                else if (self->input_.ns == 4)
                    self->toroidal_compensated(2);
                else
                    self->callback({});
            });
    }
};
}  // namespace

void run_float_geometry_tests(const wgpu::Device& device,
                              std::function<void(std::string)> callback) {
    auto test = std::make_shared<FloatGeometryTest>();
    test->device = device;
    test->callback = std::move(callback);
    test->start();
}
}  // namespace cumes::webgpu
