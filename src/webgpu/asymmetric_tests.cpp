#include "asymmetric_tests.hpp"

#include "cumes/webgpu/descent.hpp"
#include "cumes/webgpu/float_float.hpp"
#include "cumes/webgpu/numerics.hpp"
#include "cumes/webgpu/preconditioner.hpp"
#include "cumes/webgpu/prolongation.hpp"
#include "cumes/webgpu/reduction.hpp"
#include "cumes/webgpu/toroidal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <utility>

namespace cumes::webgpu {
namespace {

class AsymmetricTest : public std::enable_shared_from_this<AsymmetricTest> {
   public:
    wgpu::Device device;
    std::function<void(std::string)> callback;

    void start() {
        if (variant_ == 4) {
            callback({});
            return;
        }
        inverse_ = {};
        inverse_.ns = 5;
        inverse_.mpol = 5;
        inverse_.ntor = variant_ / 2 == 0 ? 0 : 2;
        inverse_.ntheta = 24;
        inverse_.nzeta = inverse_.ntor == 0 ? 1 : 8;
        inverse_.nfp = 3;
        inverse_.lasym = true;
        inverse_.double_single = variant_ % 2 != 0;
        const int modes = inverse_.mpol * (inverse_.ntor + 1);
        inverse_.state.resize(12 * inverse_.ns * modes);
        inverse_.state_lo.resize(inverse_.state.size());
        for (std::size_t i = 0; i < inverse_.state.size(); ++i) {
            const auto value = split(.03 * std::sin(.13 * (i + 1)));
            inverse_.state[i] = value.hi;
            inverse_.state_lo[i] = value.lo;
        }
        const auto self = shared_from_this();
        enqueue_toroidal_inverse(
            device, inverse_,
            [self](std::string error, ToroidalInverseResult result) {
                if (self->failed(error)) return;
                const auto expected =
                    toroidal_inverse_reference(self->inverse_);
                if (!self->compare(result.geometry, result.geometry_lo,
                                   expected.geometry, expected.geometry_lo,
                                   "inverse") ||
                    !self->compare(result.r_con, result.r_con_lo,
                                   expected.r_con, expected.r_con_lo,
                                   "R constraint") ||
                    !self->compare(result.z_con, result.z_con_lo,
                                   expected.z_con, expected.z_con_lo,
                                   "Z constraint"))
                    return;
                self->forward();
            });
    }

   private:
    int variant_ = 0;
    ToroidalInverseCase inverse_;
    ToroidalForwardResult forward_;
    ResidualDecompositionResult decomposed_;

    bool failed(const std::string& error) {
        if (error.empty()) return false;
        callback("asymmetric variant " + std::to_string(variant_) + ": " +
                 error);
        return true;
    }

    bool compare(const std::vector<float>& hi,
                 const std::vector<float>& lo,
                 const std::vector<float>& expected_hi,
                 const std::vector<float>& expected_lo,
                 const char* name,
                 double tolerance = 0) {
        if (tolerance == 0) tolerance = inverse_.double_single ? 3e-11 : 4e-5;
        if (hi.size() != expected_hi.size() ||
            lo.size() != expected_lo.size() || hi.empty())
            return !failed(std::string(name) + " shape mismatch");
        double maximum = 0;
        for (std::size_t i = 0; i < hi.size(); ++i) {
            const double actual = double(hi[i]) + (lo.empty() ? 0 : lo[i]);
            const double expected = double(expected_hi[i]) +
                                    (expected_lo.empty() ? 0 : expected_lo[i]);
            if (!std::isfinite(actual) || !std::isfinite(expected))
                return !failed(std::string(name) + " nonfinite");
            maximum = std::max(maximum, std::abs(actual - expected) /
                                            (1 + std::abs(expected)));
        }
        return !failed(maximum <= tolerance ? ""
                                            : std::string(name) + " error=" +
                                                  std::to_string(maximum));
    }

    void forward(bool use_fft = false) {
        ToroidalForwardCase in;
        in.ns = inverse_.ns;
        in.mpol = inverse_.mpol;
        in.ntor = inverse_.ntor;
        in.ntheta = inverse_.ntheta;
        in.nzeta = inverse_.nzeta;
        in.nfp = inverse_.nfp;
        in.lasym = true;
        in.double_single = inverse_.double_single;
        in.include_lcfs = variant_ % 2 == 0;
        in.use_fft = use_fft;
        in.fields.resize(TOROIDAL_FORWARD_FIELD_COUNT * in.ns * in.ntheta *
                         in.nzeta);
        if (in.double_single) in.fields_lo.resize(in.fields.size());
        for (std::size_t i = 0; i < in.fields.size(); ++i) {
            const auto value = split(.1 * std::sin(.113 * (i + 1)));
            in.fields[i] = value.hi;
            if (in.double_single) in.fields_lo[i] = value.lo;
        }
        const auto self = shared_from_this();
        enqueue_toroidal_forward(
            device, in,
            [self, in](std::string error, ToroidalForwardResult result) {
                if (self->failed(error)) return;
                const auto expected = toroidal_forward_reference(in);
                if (!self->compare(result.residual, result.residual_lo,
                                   expected.residual, expected.residual_lo,
                                   "forward"))
                    return;
                self->forward_ = std::move(result);
                if (in.double_single && in.ntor != 0 && !in.use_fft)
                    self->forward(true);
                else
                    self->decompose();
            });
    }

    void decompose() {
        ResidualDecompositionCase in;
        in.ns = inverse_.ns;
        in.mpol = inverse_.mpol;
        in.ntor = inverse_.ntor;
        in.lasym = true;
        in.double_single = inverse_.double_single;
        in.zero_m1_z = variant_ % 2 == 0;
        in.residual = forward_.residual;
        in.residual_lo = forward_.residual_lo;
        for (int j = 0; j < in.ns; ++j) {
            const auto value = split(std::sqrt(double(j) / (in.ns - 1)));
            in.sqrt_s_f.push_back(value.hi);
            in.sqrt_s_f_lo.push_back(value.lo);
        }
        const auto self = shared_from_this();
        enqueue_residual_decomposition(
            device, in,
            [self, in](std::string error, ResidualDecompositionResult result) {
                if (self->failed(error)) return;
                const auto expected = residual_decomposition_reference(in);
                if (!self->compare(result.residual, result.residual_lo,
                                   expected.residual, expected.residual_lo,
                                   "decomposition"))
                    return;
                self->decomposed_ = std::move(result);
                self->norm();
            });
    }

    void norm(bool nonfinite = false) {
        const auto& hi = decomposed_.residual;
        const auto& lo = decomposed_.residual_lo;
        std::vector<float> values = hi;
        values.insert(values.end(), lo.begin(), lo.end());
        const std::size_t points = hi.size() / 12;
        if (nonfinite)
            values[9 * points + inverse_.ns - 1] =
                std::numeric_limits<float>::quiet_NaN();
        wgpu::BufferDescriptor descriptor{};
        descriptor.size = values.size() * sizeof(float);
        descriptor.usage =
            wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
        const auto buffer = device.CreateBuffer(&descriptor);
        device.GetQueue().WriteBuffer(buffer, 0, values.data(),
                                      descriptor.size);
        ResidualNormCase in;
        in.ns = inverse_.ns;
        in.lasym = true;
        in.paired = inverse_.double_single;
        in.include_edge_rz = variant_ % 2 == 0;
        in.residual = {buffer, hi.size(), 0, hi.size() * sizeof(float)};
        in.readback.batch = std::make_shared<ReadbackBatch>(device, 40);
        const auto result = std::make_shared<ResidualNormResult>();
        const auto error = std::make_shared<std::string>();
        enqueue_residual_norm(
            device, in,
            [result, error](std::string message, ResidualNormResult value) {
                *result = std::move(value);
                *error = std::move(message);
            });
        const auto self = shared_from_this();
        in.readback.batch->map([self, in, result, error, points,
                                nonfinite](std::string message) {
            if (self->failed(message) || self->failed(*error) ||
                self->failed(result->finite == !nonfinite ? ""
                                                          : "norm finite gate"))
                return;
            if (!nonfinite) {
                for (int group = 0; group < 3; ++group) {
                    double expected = 0;
                    for (int c = group; c < 12; c += 3)
                        for (std::size_t i = 0; i < points; ++i) {
                            if (group != 2 && !in.include_edge_rz &&
                                i % in.ns == std::size_t(in.ns - 1))
                                continue;
                            const auto index = c * points + i;
                            const double value =
                                double(self->decomposed_.residual[index]) +
                                (in.paired
                                     ? self->decomposed_.residual_lo[index]
                                     : 0);
                            expected += value * value;
                        }
                    expected /= double(points);
                    if (self->failed(std::abs(result->raw[group] - expected) <
                                             2e-11 * (1 + expected)
                                         ? ""
                                         : "norm lost a complementary family"))
                        return;
                }
                self->norm(true);
            } else
                self->precondition();
        });
    }

    void precondition() {
        AxisymmetricPreconditionerApplyCase in;
        in.ns = inverse_.ns;
        in.mpol = inverse_.mpol;
        in.ntor = inverse_.ntor;
        in.lasym = true;
        in.include_lcfs = variant_ % 2 == 0;
        in.residual = decomposed_.residual;
        auto& e = in.elements;
        e.ard.assign(2 * in.ns, 2);
        e.brd.assign(2 * in.ns, 1);
        e.azd.assign(2 * in.ns, 3);
        e.bzd.assign(2 * in.ns, 1);
        const int modes = in.mpol * (in.ntor + 1), points = modes * in.ns;
        auto& matrix = in.matrix;
        matrix.upper_r.assign(points, -.25F);
        matrix.lower_r = matrix.upper_r;
        matrix.upper_z.assign(points, -.5F);
        matrix.lower_z = matrix.upper_z;
        matrix.diagonal_r.assign(points, 2);
        matrix.diagonal_z.assign(points, 3);
        matrix.lambda.assign(points, .7F);
        matrix.scale.assign(modes, 3);
        for (int mode = 0; mode < modes; ++mode)
            matrix.first_surface.push_back(mode / (in.ntor + 1) == 0 ? 0 : 1);
        const auto self = shared_from_this();
        enqueue_axisymmetric_preconditioner_apply(
            device, in,
            [self, in](std::string error,
                       AxisymmetricPreconditionerApplyResult result) {
                if (self->failed(error)) return;
                const auto expected =
                    axisymmetric_preconditioner_apply_reference(in);
                if (!self->compare(result.residual, {}, expected.residual, {},
                                   "preconditioner", 3e-6) ||
                    self->failed(
                        result.breakdown_count == 0 ? "" : "radial breakdown"))
                    return;
                self->descent(result.residual);
            });
    }

    void descent(std::vector<float> residual) {
        AxisymmetricDescentCase in;
        in.ns = inverse_.ns;
        in.mpol = inverse_.mpol;
        in.ntor = inverse_.ntor;
        in.lasym = true;
        in.double_single = inverse_.double_single;
        in.move_lcfs = variant_ % 2 == 0;
        in.extrapolate_axis = true;
        in.delta_t = .7F;
        in.damping_b1 = .8F;
        in.damping_fac = .9F;
        in.state = inverse_.state;
        in.velocity.assign(in.state.size(), .001F);
        in.residual = std::move(residual);
        if (in.double_single) {
            in.state_lo = inverse_.state_lo;
            in.velocity_lo.assign(in.state.size(), 0);
            in.residual_is_f32 = true;
        }
        const auto self = shared_from_this();
        enqueue_axisymmetric_descent(
            device, in,
            [self, in](std::string error, AxisymmetricDescentResult result) {
                if (self->failed(error)) return;
                const auto expected = axisymmetric_descent_reference(in);
                if (!self->compare(result.state, result.state_lo,
                                   expected.state, expected.state_lo,
                                   "descent") ||
                    !self->compare(result.velocity, result.velocity_lo,
                                   expected.velocity, expected.velocity_lo,
                                   "velocity"))
                    return;
                self->bandpass();
            });
    }

    void bandpass() {
        ToroidalDealiasCase in;
        in.ns = inverse_.ns;
        in.mpol = inverse_.mpol;
        in.ntor = inverse_.ntor;
        in.ntheta = inverse_.ntheta;
        in.nzeta = inverse_.nzeta;
        in.lasym = true;
        in.tcon.assign(in.ns, 1);
        in.faccon.assign(in.mpol, 1);
        in.g_con_eff.resize(in.ns * in.ntheta * in.nzeta);
        for (std::size_t i = 0; i < in.g_con_eff.size(); ++i)
            in.g_con_eff[i] = .1F * std::cos(float(i) * .117F);
        const auto self = shared_from_this();
        enqueue_toroidal_dealias(
            device, in,
            [self, in](std::string error, ToroidalDealiasResult result) {
                if (self->failed(error)) return;
                const auto expected = toroidal_dealias_reference(in);
                if (!self->compare(result.g_con, {}, expected.g_con, {},
                                   "bandpass", 3e-6))
                    return;
                std::printf(
                    "  asymmetric "
                    "transforms/gauge/preconditioner/descent/bandpass "
                    "variant=%d: PASS\n",
                    self->variant_);
                ++self->variant_;
                self->start();
            });
    }
};
}  // namespace

void run_asymmetric_tests(const wgpu::Device& device,
                          std::function<void(std::string)> callback) {
    auto test = std::make_shared<AsymmetricTest>();
    test->device = device;
    test->callback = std::move(callback);
    test->start();
}
}  // namespace cumes::webgpu
