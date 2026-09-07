#include "cumes/config/json_writer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace cumes {
namespace {

std::string number(double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "cannot serialize a non-finite ProblemSpec value");
    }
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << value;
    return output.str();
}

std::string escape(std::string_view value) {
    std::string result;
    result.reserve(value.size());
    for (const char character : value) {
        switch (character) {
            case '\\':
                result += "\\\\";
                break;
            case '"':
                result += "\\\"";
                break;
            case '\n':
                result += "\\n";
                break;
            case '\r':
                result += "\\r";
                break;
            case '\t':
                result += "\\t";
                break;
            case '\b':
                result += "\\b";
                break;
            case '\f':
                result += "\\f";
                break;
            default:
                if (static_cast<unsigned char>(character) < 0x20) {
                    constexpr char HEX[] = "0123456789abcdef";
                    const unsigned char value =
                        static_cast<unsigned char>(character);
                    result += "\\u00";
                    result += HEX[value >> 4];
                    result += HEX[value & 0x0f];
                } else {
                    result += character;
                }
        }
    }
    return result;
}

template <typename Value, typename Emit>
void array(std::ostringstream& output,
           const std::vector<Value>& values,
           Emit emit) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) output << ',';
        emit(output, values[index]);
    }
    output << ']';
}

void doubles(std::ostringstream& output, const std::vector<double>& values) {
    array(output, values, [](std::ostringstream& stream, double value) {
        stream << number(value);
    });
}

void harmonics(std::ostringstream& output,
               std::vector<BoundaryHarmonic> values) {
    std::sort(values.begin(), values.end(),
              [](const auto& left, const auto& right) {
                  if (left.m != right.m) return left.m < right.m;
                  return left.n < right.n;
              });
    array(output, values,
          [](std::ostringstream& stream, const BoundaryHarmonic& harmonic) {
              stream << "{\"m\":" << harmonic.m << ",\"n\":" << harmonic.n
                     << ",\"value\":" << number(harmonic.value) << '}';
          });
}

void makegrid_parameters(std::ostringstream& output,
                         const MakegridParametersSpec& value) {
    output << "{\"normalize_by_currents\":"
           << (value.normalize_by_currents ? "true" : "false")
           << ",\"assume_stellarator_symmetry\":"
           << (value.assume_stellarator_symmetry ? "true" : "false")
           << ",\"number_of_field_periods\":" << value.number_of_field_periods
           << ",\"r_grid_minimum\":" << number(value.r_grid_minimum)
           << ",\"r_grid_maximum\":" << number(value.r_grid_maximum)
           << ",\"number_of_r_grid_points\":" << value.number_of_r_grid_points
           << ",\"z_grid_minimum\":" << number(value.z_grid_minimum)
           << ",\"z_grid_maximum\":" << number(value.z_grid_maximum)
           << ",\"number_of_z_grid_points\":" << value.number_of_z_grid_points
           << ",\"number_of_phi_grid_points\":"
           << value.number_of_phi_grid_points << '}';
}

}  // namespace

std::string problem_spec_to_json(const ProblemSpec& problem) {
    std::ostringstream output;
    output << "{\n"
           << "  \"mpol\":" << problem.mpol << ",\n"
           << "  \"ntor\":" << problem.ntor << ",\n"
           << "  \"nfp\":" << problem.nfp << ",\n"
           << "  \"ntheta\":" << problem.angular.ntheta << ",\n"
           << "  \"nzeta\":" << problem.angular.nzeta << ",\n"
           << "  \"ncurr\":"
           << (problem.current_model == CurrentModel::PRESCRIBED_CURRENT ? 1
                                                                         : 0)
           << ",\n"
           << "  \"delt\":" << number(problem.delt) << ",\n"
           << "  \"phiedge\":" << number(problem.physical.phiedge) << ",\n"
           << "  \"pres_scale\":" << number(problem.physical.pres_scale)
           << ",\n"
           << "  \"gamma\":" << number(problem.physical.adiabatic_index)
           << ",\n"
           << "  \"spres_ped\":" << number(problem.physical.spres_ped) << ",\n"
           << "  \"bloat\":" << number(problem.physical.bloat) << ",\n"
           << "  \"curtor\":" << number(problem.physical.curtor) << ",\n"
           << "  \"tcon0\":" << number(problem.physical.tcon0) << ",\n"
           << "  \"pmass_type\":\"" << profile_type_to_string(problem.mass.type)
           << "\",\n"
           << "  \"piota_type\":\"" << profile_type_to_string(problem.iota.type)
           << "\",\n"
           << "  \"pcurr_type\":\""
           << profile_type_to_string(problem.current.type) << "\",\n"
           << "  \"am\":";
    doubles(output, problem.mass.coefficients);
    output << ",\n  \"ai\":";
    doubles(output, problem.iota.coefficients);
    output << ",\n  \"ac\":";
    doubles(output, problem.current.coefficients);
    output << ",\n  \"aphi\":";
    doubles(output, problem.toroidal_flux.coefficients);
    if (problem.has_raxis_c) {
        output << ",\n  \"raxis_c\":";
        doubles(output, problem.raxis_c);
    }
    if (problem.has_zaxis_s) {
        output << ",\n  \"zaxis_s\":";
        doubles(output, problem.zaxis_s);
    }

    std::vector<std::size_t> radial_surfaces;
    std::vector<std::size_t> max_iterations;
    std::vector<double> tolerances;
    radial_surfaces.reserve(problem.stages.size());
    max_iterations.reserve(problem.stages.size());
    tolerances.reserve(problem.stages.size());
    for (const auto& stage : problem.stages) {
        radial_surfaces.push_back(stage.radial_surfaces);
        max_iterations.push_back(stage.max_iterations);
        tolerances.push_back(stage.tolerance);
    }
    if (!problem.stages.empty()) {
        output << ",\n  \"ns_array\":";
        array(output, radial_surfaces,
              [](std::ostringstream& stream, std::size_t value) {
                  stream << value;
              });
        output << ",\n  \"niter_array\":";
        array(output, max_iterations,
              [](std::ostringstream& stream, std::size_t value) {
                  stream << value;
              });
        output << ",\n  \"ftol_array\":";
        doubles(output, tolerances);
    }
    output << ",\n  \"rbc\":";
    harmonics(output, problem.rbc);
    output << ",\n  \"zbs\":";
    harmonics(output, problem.zbs);

    output << ",\n  \"lasym\":false,\n"
           << "  \"lfreeb\":"
           << (problem.free_boundary.lfreeb ? "true" : "false") << ",\n"
           << "  \"mgrid_file\":\"" << escape(problem.free_boundary.mgrid_file)
           << "\",\n"
           << "  \"coils_file\":\"" << escape(problem.free_boundary.coils_file)
           << "\",\n"
           << "  \"makegrid_parameters_file\":\""
           << escape(problem.free_boundary.makegrid_parameters_file) << "\",\n"
           << "  \"nvacskip\":" << problem.free_boundary.nvacskip
           << ",\n  \"extcur\":";
    doubles(output, problem.free_boundary.extcur);
    if (problem.free_boundary.embedded_makegrid_parameters.has_value()) {
        output << ",\n  \"makegrid_parameters\":";
        makegrid_parameters(
            output, *problem.free_boundary.embedded_makegrid_parameters);
    }
    output << "\n}\n";
    return output.str();
}

void write_problem_spec(const std::string& path, const ProblemSpec& problem) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("could not open ProblemSpec JSON: " + path);
    }
    output << problem_spec_to_json(problem);
    if (!output) {
        throw std::runtime_error("could not write ProblemSpec JSON: " + path);
    }
}

}  // namespace cumes
