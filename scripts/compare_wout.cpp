// Compare a cuMES state with vmecpp's wout HDF5 spectra.

#include "../include/clap.h"
#include "compare_common.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CUMES_COMPARE_HAVE_HDF5
#include <hdf5.h>
#endif

namespace {

struct CommandLine {
    std::string cumes_state;
    std::string vmecpp_h5;
    bool with_axis = false;
    double tolerance = 1.0e-9;
};

int parse_command_line(CommandLine& command, int argc, char** argv) {
    CLAP_BEGIN(CommandLine)
    CLAP_ADD_USAGE("CUMES_STATE VMEC_OUTPUT_H5 [--with-axis] [--tol X]")
    CLAP_ADD_DESCRIPTION(
        "Fold vmecpp signed-n wout modes and compare all six cuMES families.")
    CLAP_REGISTER_ARG(cumes_state)
    CLAP_REGISTER_ARG(vmecpp_h5)
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(with_axis, "--with-axis",
                                          "include the state axis row j=0")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        tolerance, "--tol", "maximum family absolute difference (default 1e-9)")
    CLAP_END(CommandLine)
    try {
        CLAP<CommandLine>::parse_input(command, argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what();
        return EINVAL;
    }
    if (!(command.tolerance >= 0.0) || !std::isfinite(command.tolerance)) {
        std::cerr << "error: --tol must be finite and nonnegative\n";
        return EINVAL;
    }
    return 0;
}

#ifdef CUMES_COMPARE_HAVE_HDF5

class H5Handle {
   public:
    using Closer = herr_t (*)(hid_t);

    H5Handle(hid_t value, Closer closer, const std::string& description)
        : value_(value), closer_(closer) {
        if (value_ < 0) throw std::runtime_error(description);
    }
    H5Handle(const H5Handle&) = delete;
    H5Handle& operator=(const H5Handle&) = delete;
    H5Handle(H5Handle&& other) noexcept
        : value_(other.value_), closer_(other.closer_) {
        other.value_ = -1;
    }
    ~H5Handle() {
        if (value_ >= 0) closer_(value_);
    }
    operator hid_t() const { return value_; }

   private:
    hid_t value_ = -1;
    Closer closer_ = nullptr;
};

int read_scalar_integer(hid_t group, const char* name) {
    H5Handle dataset(H5Dopen2(group, name, H5P_DEFAULT), H5Dclose,
                     std::string("missing wout/") + name);
    H5Handle space(H5Dget_space(dataset), H5Sclose,
                   std::string("could not inspect wout/") + name);
    const int rank = H5Sget_simple_extent_ndims(space);
    if (rank != 0) {
        throw std::runtime_error(std::string("wout/") + name +
                                 " is not scalar");
    }
    int value = 0;
    if (H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                &value) < 0) {
        throw std::runtime_error(std::string("could not read wout/") + name);
    }
    return value;
}

struct Matrix {
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::vector<double> values;

    double operator()(std::size_t row, std::size_t column) const {
        return values[row * columns + column];
    }
};

Matrix read_matrix(hid_t group, const char* name) {
    H5Handle dataset(H5Dopen2(group, name, H5P_DEFAULT), H5Dclose,
                     std::string("missing wout/") + name);
    H5Handle space(H5Dget_space(dataset), H5Sclose,
                   std::string("could not inspect wout/") + name);
    if (H5Sget_simple_extent_ndims(space) != 2) {
        throw std::runtime_error(std::string("wout/") + name +
                                 " must have rank 2");
    }
    std::array<hsize_t, 2> dimensions{};
    if (H5Sget_simple_extent_dims(space, dimensions.data(), nullptr) < 0) {
        throw std::runtime_error(std::string("could not size wout/") + name);
    }
    if (dimensions[0] == 0 || dimensions[1] == 0 ||
        dimensions[0] > std::numeric_limits<std::size_t>::max() ||
        dimensions[1] > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(std::string("invalid shape for wout/") + name);
    }
    Matrix matrix;
    matrix.rows = static_cast<std::size_t>(dimensions[0]);
    matrix.columns = static_cast<std::size_t>(dimensions[1]);
    if (matrix.rows >
        std::numeric_limits<std::size_t>::max() / matrix.columns) {
        throw std::runtime_error(std::string("oversized wout/") + name);
    }
    matrix.values.resize(matrix.rows * matrix.columns);
    if (H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                matrix.values.data()) < 0) {
        throw std::runtime_error(std::string("could not read wout/") + name);
    }
    return matrix;
}

struct VmecState {
    int ns = 0;
    int mnmax = 0;
    std::array<std::vector<double>, cumes::compare::FAMILY_NAMES.size()>
        families;
};

VmecState read_vmecpp(const std::string& path) {
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    H5Handle file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose,
                  "could not open vmecpp HDF5 file " + path);
    H5Handle group(H5Gopen2(file, "wout", H5P_DEFAULT), H5Gclose,
                   "missing HDF5 group wout");
    const int ntor = read_scalar_integer(group, "ntor");
    const int mpol = read_scalar_integer(group, "mpol");
    if (ntor < 0 || mpol < 1) {
        throw std::runtime_error("invalid vmecpp mpol/ntor");
    }
    const auto rmnc = read_matrix(group, "rmnc");
    const auto zmns = read_matrix(group, "zmns");
    const auto lmns = read_matrix(group, "lmns_full");
    if (rmnc.rows != zmns.rows || rmnc.rows != lmns.rows ||
        rmnc.columns != zmns.columns || rmnc.columns != lmns.columns) {
        throw std::runtime_error(
            "vmecpp spectral arrays have different shapes");
    }
    const auto signed_modes = static_cast<std::size_t>(ntor + 1) +
                              static_cast<std::size_t>(mpol - 1) *
                                  static_cast<std::size_t>(2 * ntor + 1);
    if (rmnc.rows != signed_modes ||
        rmnc.columns >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("vmecpp spectral array shape is inconsistent");
    }

    VmecState state;
    state.ns = static_cast<int>(rmnc.columns);
    state.mnmax = mpol * (ntor + 1);
    for (auto& family : state.families) {
        family.assign(static_cast<std::size_t>(state.ns) *
                          static_cast<std::size_t>(state.mnmax),
                      0.0);
    }
    const auto wmode = [ntor](int m, int n) -> std::size_t {
        if (m == 0) return static_cast<std::size_t>(n);
        return static_cast<std::size_t>(ntor + 1) +
               static_cast<std::size_t>(m - 1) *
                   static_cast<std::size_t>(2 * ntor + 1) +
               static_cast<std::size_t>(n + ntor);
    };
    const auto copy_surface = [&](std::size_t family, int mode, int surface,
                                  double value) {
        const auto index = static_cast<std::size_t>(mode) *
                               static_cast<std::size_t>(state.ns) +
                           static_cast<std::size_t>(surface);
        state.families[family][index] = value;
    };

    for (int m = 0; m < mpol; ++m) {
        for (int n = 0; n <= ntor; ++n) {
            const int mode = m * (ntor + 1) + n;
            for (int surface = 0; surface < state.ns; ++surface) {
                const auto j = static_cast<std::size_t>(surface);
                const double rp = rmnc(wmode(m, n), j);
                const double zp = zmns(wmode(m, n), j);
                const double lp = lmns(wmode(m, n), j);
                if (m == 0) {
                    copy_surface(0, mode, surface, rp);
                    copy_surface(4, mode, surface, -zp);
                    copy_surface(5, mode, surface, -lp);
                    continue;
                }
                const double rn = n > 0 ? rmnc(wmode(m, -n), j) : 0.0;
                const double zn = n > 0 ? zmns(wmode(m, -n), j) : 0.0;
                const double ln = n > 0 ? lmns(wmode(m, -n), j) : 0.0;
                copy_surface(0, mode, surface, rp + rn);
                copy_surface(1, mode, surface, zp + zn);
                copy_surface(2, mode, surface, lp + ln);
                if (n > 0) {
                    copy_surface(3, mode, surface, rp - rn);
                    copy_surface(4, mode, surface, -zp + zn);
                    copy_surface(5, mode, surface, -lp + ln);
                }
            }
        }
    }
    return state;
}

#endif  // CUMES_COMPARE_HAVE_HDF5

}  // namespace

int main(int argc, char** argv) {
    CommandLine command;
    if (const int status = parse_command_line(command, argc, argv); status) {
        return status;
    }

#ifndef CUMES_COMPARE_HAVE_HDF5
    (void)command;
    std::cerr << "error: compare_wout requires HDF5; compile it with h5c++ "
                 "or use scripts/build_compare_tools.sh\n";
    return 2;
#else
    try {
        const auto cumes =
            cumes::compare::read_state(command.cumes_state, false);
        const auto vmec = read_vmecpp(command.vmecpp_h5);
        if (cumes.ns != vmec.ns || cumes.mnmax != vmec.mnmax) {
            std::cerr << "error: grid mismatch: cuMES ns=" << cumes.ns
                      << " mnmax=" << cumes.mnmax << " vs vmecpp ns=" << vmec.ns
                      << " mnmax=" << vmec.mnmax << '\n';
            return 2;
        }
        const int first_surface = command.with_axis ? 0 : 1;
        if (first_surface >= cumes.ns) {
            throw std::runtime_error("comparison has no radial samples");
        }
        std::cout << "ns=" << cumes.ns << " mnmax=" << cumes.mnmax
                  << (command.with_axis ? " (axis row included)\n"
                                        : " (axis row j=0 skipped)\n");
        double overall_worst = 0.0;
        for (std::size_t family = 0;
             family < cumes::compare::FAMILY_NAMES.size(); ++family) {
            double max_absolute = -1.0;
            double max_relative = 0.0;
            double min_vmec_magnitude = std::numeric_limits<double>::infinity();
            double max_vmec_magnitude = 0.0;
            int max_mode = 0;
            int max_surface = first_surface;
            for (int mode = 0; mode < cumes.mnmax; ++mode) {
                for (int surface = first_surface; surface < cumes.ns;
                     ++surface) {
                    const auto index = static_cast<std::size_t>(mode) *
                                           static_cast<std::size_t>(cumes.ns) +
                                       static_cast<std::size_t>(surface);
                    const double reference = vmec.families[family][index];
                    const double difference =
                        std::abs(cumes.families[family][index] - reference);
                    const double relative =
                        difference / std::max(std::abs(reference), 1.0e-30);
                    if (difference > max_absolute) {
                        max_absolute = difference;
                        max_mode = mode;
                        max_surface = surface;
                    }
                    max_relative = std::max(max_relative, relative);
                    min_vmec_magnitude =
                        std::min(min_vmec_magnitude, std::abs(reference));
                    max_vmec_magnitude =
                        std::max(max_vmec_magnitude, std::abs(reference));
                }
            }
            overall_worst = std::max(overall_worst, max_absolute);
            std::cout << std::left << std::setw(6)
                      << cumes::compare::FAMILY_NAMES[family] << std::right
                      << " max|d| = " << std::scientific << std::setprecision(3)
                      << max_absolute << " at (mode " << max_mode << ", j "
                      << max_surface << ")   max rel = " << max_relative
                      << "   [vmecpp |x| range " << min_vmec_magnitude << " .. "
                      << max_vmec_magnitude << "]\n";
        }
        if (overall_worst > command.tolerance) {
            std::cout << "FAIL: worst family max|d| " << std::scientific
                      << std::setprecision(3) << overall_worst
                      << " exceeds --tol " << std::defaultfloat
                      << command.tolerance << '\n';
            return EXIT_FAILURE;
        }
        std::cout << "OK: worst family max|d| " << std::scientific
                  << std::setprecision(3) << overall_worst << " within --tol "
                  << std::defaultfloat << command.tolerance << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
#endif
}
