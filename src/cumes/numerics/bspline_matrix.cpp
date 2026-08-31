#include "cumes/numerics/bspline_matrix.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <bsintp/Interpolation.hpp>

namespace cumes {

std::vector<double> cubic_bspline_interpolation_matrix(int ns_old, int ns_new) {
    if (ns_old < 3 || ns_new <= ns_old) {
        throw std::invalid_argument(
            "cubic_bspline_interpolation_matrix: need ns_new > ns_old >= 3");
    }

    intp::InterpolationFunctionTemplate1D<3> interpolation_template(
        std::make_pair(0.0, 1.0), static_cast<std::size_t>(ns_old));
#ifdef INTP_CELL_LAYOUT
    using EvalProxy = decltype(interpolation_template.eval_proxy(0.0));
    std::vector<EvalProxy> evaluation;
    evaluation.reserve(static_cast<std::size_t>(ns_new));
    for (int j_new = 0; j_new < ns_new; ++j_new) {
        evaluation.push_back(interpolation_template.eval_proxy(
            static_cast<double>(j_new) / static_cast<double>(ns_new - 1)));
    }
#endif

    std::vector<double> basis(static_cast<std::size_t>(ns_old), 0.0);
    std::vector<double> matrix(static_cast<std::size_t>(ns_new) * ns_old);
    for (int j_old = 0; j_old < ns_old; ++j_old) {
        basis[static_cast<std::size_t>(j_old)] = 1.0;
        const auto interpolation = interpolation_template.interpolate(
            std::make_pair(basis.cbegin(), basis.cend()));
        for (int j_new = 0; j_new < ns_new; ++j_new) {
#ifdef INTP_CELL_LAYOUT
            const double weight =
                evaluation[static_cast<std::size_t>(j_new)](interpolation);
#else
            const double weight = interpolation(
                static_cast<double>(j_new) / static_cast<double>(ns_new - 1));
#endif
            matrix[static_cast<std::size_t>(j_new) * ns_old + j_old] = weight;
        }
        basis[static_cast<std::size_t>(j_old)] = 0.0;
    }
    return matrix;
}

}  // namespace cumes
