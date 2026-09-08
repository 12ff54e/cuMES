#include "cumes/webgpu/axisymmetric.hpp"

#include "cumes/webgpu/toroidal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <utility>

namespace cumes::webgpu {
namespace {

constexpr std::size_t RESULT_FIELD_COUNT = GEOMETRY_PARITY_FIELD_COUNT + 2;

std::string validate_case(const AxisymmetricInverseCase& input) {
    if (input.ns < 2 || input.mpol <= 0 || input.ntheta < 2 ||
        input.ntheta % 2 != 0) {
        return "axisymmetric inverse requires ns>=2, mpol>0, and even "
               "ntheta>=2";
    }
    const auto spectral_values = SPECTRAL_COMPONENT_COUNT *
                                 static_cast<std::size_t>(input.mpol) *
                                 static_cast<std::size_t>(input.ns);
    if (input.state.size() != spectral_values) {
        return "axisymmetric state size does not match 6*mpol*ns";
    }
    const auto points = static_cast<std::size_t>(input.ns) * input.ntheta;
    if (points > std::numeric_limits<std::uint32_t>::max() ||
        points > std::numeric_limits<std::size_t>::max() / RESULT_FIELD_COUNT) {
        return "axisymmetric output exceeds WebGPU indexing limits";
    }
    return {};
}

}  // namespace

AxisymmetricInverseResult axisymmetric_inverse_reference(
    const AxisymmetricInverseCase& input) {
    if (!validate_case(input).empty()) return {};

    const std::size_t points =
        static_cast<std::size_t>(input.ns) * input.ntheta;
    AxisymmetricInverseResult result;
    result.geometry.assign(GEOMETRY_PARITY_FIELD_COUNT * points, 0.0F);
    result.r_con.resize(points);
    result.z_con.resize(points);

    const auto coefficient = [&](int component, int mode, int surface) {
        return input
            .state[(static_cast<std::size_t>(component) * input.mpol + mode) *
                       input.ns +
                   surface];
    };
    const auto store = [&](int field, std::size_t point, float value) {
        result.geometry[static_cast<std::size_t>(field) * points + point] =
            value;
    };

    for (int surface = 0; surface < input.ns; ++surface) {
        const float maxsc =
            std::max(std::sqrt(static_cast<float>(surface) /
                               static_cast<float>(input.ns - 1)),
                     std::sqrt(1.0F / static_cast<float>(input.ns - 1)));
        for (int theta_index = 0; theta_index < input.ntheta; ++theta_index) {
            const std::size_t point =
                static_cast<std::size_t>(surface) * input.ntheta + theta_index;
            const float theta = 2.0F * std::numbers::pi_v<float> *
                                static_cast<float>(theta_index) /
                                static_cast<float>(input.ntheta);
            float r_e = 0.0F, z_e = 0.0F, l_e = 0.0F;
            float ru_e = 0.0F, zu_e = 0.0F, lu_e = 0.0F;
            float r_o = 0.0F, z_o = 0.0F, l_o = 0.0F;
            float ru_o = 0.0F, zu_o = 0.0F, lu_o = 0.0F;
            float r_con = 0.0F, z_con = 0.0F;
            for (int mode = 0; mode < input.mpol; ++mode) {
                const float mf = static_cast<float>(mode);
                const float cosine = std::cos(mf * theta);
                const float sine = std::sin(mf * theta);
                const float rc = coefficient(0, mode, surface);
                const float zs = coefficient(1, mode, surface);
                const float ls = coefficient(2, mode, surface);
                const bool odd = mode % 2 == 1;
                const float scale = odd ? 1.0F / maxsc : 1.0F;
                const float rv = scale * rc * cosine;
                const float zv = scale * zs * sine;
                const float lv = scale * ls * sine;
                const float ruv = scale * rc * (-mf * sine);
                const float zuv = scale * zs * (mf * cosine);
                const float luv = scale * ls * (mf * cosine);
                if (odd) {
                    r_o += rv;
                    z_o += zv;
                    l_o += lv;
                    ru_o += ruv;
                    zu_o += zuv;
                    lu_o += luv;
                } else {
                    r_e += rv;
                    z_e += zv;
                    l_e += lv;
                    ru_e += ruv;
                    zu_e += zuv;
                    lu_e += luv;
                }
                const float xmpq = mf * (mf - 1.0F);
                r_con += xmpq * rc * cosine;
                z_con += xmpq * zs * sine;
            }
            store(0, point, r_e);
            store(1, point, z_e);
            store(2, point, l_e);
            store(3, point, ru_e);
            store(4, point, zu_e);
            store(5, point, lu_e);
            store(6, point, r_o);
            store(7, point, z_o);
            store(8, point, l_o);
            store(9, point, ru_o);
            store(10, point, zu_o);
            store(11, point, lu_o);
            result.r_con[point] = r_con;
            result.z_con[point] = z_con;
        }
    }
    return result;
}

void enqueue_axisymmetric_inverse(const wgpu::Device& device,
                                  const AxisymmetricInverseCase& input,
                                  AxisymmetricInverseCallback callback) {
    const std::string validation_error = validate_case(input);
    if (!validation_error.empty()) {
        callback(validation_error, {});
        return;
    }
    ToroidalInverseCase shared;
    shared.ns = input.ns;
    shared.mpol = input.mpol;
    shared.ntheta = input.ntheta;
    shared.nzeta = 1;
    shared.nfp = 1;
    shared.state = input.state;
    enqueue_toroidal_inverse(device, shared, std::move(callback));
}

}  // namespace cumes::webgpu
