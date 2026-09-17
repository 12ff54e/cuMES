#include "cumes/config/json_reader.hpp"
#include "cumes/core/error.hpp"
#include "cumes/io/checkpoint.hpp"
#include "cumes/io/reader.hpp"
#include "cumes/io/writer.hpp"
#include "cumes/solver/equilibrium_solver.hpp"
#include "cumes_test.h"

#include <cmath>
#include <filesystem>
#include <string>

#include <unistd.h>

int main() {
    using namespace cumes;
    using test::check;
    SolverOptions options;
#ifdef CUMES_USE_FLOAT
    options.precision = PrecisionPolicy::MIXED_FLOAT;
#endif
    auto parsed = read_problem_spec("inputs/asymmetric_tokamak.json", options);
#ifdef CUMES_USE_FLOAT
    for (auto& stage : parsed.spec.stages) stage.tolerance = 1e-6;
#endif
    for (int ntor : {0, 1}) {
        auto spec = parsed.spec;
        spec.ntor = ntor;
        if (ntor) {
            spec.nfp = 3;
            spec.raxis_c.push_back(0);
            spec.raxis_s.push_back(.001);
            spec.zaxis_c.push_back(0);
            spec.zaxis_s.push_back(0);
            spec.rbc.push_back({1, 1, .008});
            spec.zbs.push_back({1, -1, .006});
            spec.rbs.push_back({1, 1, .005});
            spec.rbs.push_back({2, -1, .003});
            spec.zbc.push_back({0, 1, .003});
            spec.zbc.push_back({2, -1, .004});
        }
        auto problem = validate(spec, options);
        check(problem.has_value(), "asymmetric problem validates");
        if (!problem.has_value()) continue;
        EquilibriumSolver solver;
        auto result = solver.solve(problem.value());
#ifdef CUMES_USE_FLOAT
        result.report.build.scalar_type = "float";
#else
        result.report.build.scalar_type = "double";
#endif
        const double tolerance = spec.stages.back().tolerance;
        check(result.converged && result.fsqr < tolerance &&
                  result.fsqz < tolerance && result.fsql < tolerance,
              "all asymmetric residuals converge");
        check(result.has_complete_equilibrium() && result.equilibrium.lasym(),
              "all twelve families and scientific fields are present");
        const auto& state = result.equilibrium;
        const auto& boundary = problem.value().boundary();
        for (int c : {0, 1, 3, 4, 6, 7, 9, 10}) {
            const auto& edge = c == 0   ? boundary.rbcc
                               : c == 1 ? boundary.zbsc
                               : c == 3 ? boundary.rbss
                               : c == 4 ? boundary.zbcs
                               : c == 6 ? boundary.rbsc
                               : c == 7 ? boundary.zbcc
                               : c == 9 ? boundary.rbcs
                                        : boundary.zbss;
            for (int mode = 0; mode < state.mnmax; ++mode) {
                const double actual =
                    state.families[c][mode * state.ns + state.ns - 1];
                check(std::abs(actual - edge[mode]) < 1e-6,
                      "all fixed LCFS coefficients are preserved");
            }
        }
        bool finite = true, oriented = true;
        for (const auto& field : state.half_fields)
            for (double value : field) finite = finite && std::isfinite(value);
        for (double value : state.half_fields[EquilibriumSnapshot::SQRTG])
            oriented = oriented && value < 0;
        check(finite && oriented, "finite fields and valid oriented Jacobian");

        const auto base = (std::filesystem::temp_directory_path() /
                           ("cumes-asymmetric-" + std::to_string(getpid()) +
                            "-" + std::to_string(ntor)))
                              .string();
        for (auto format :
             {OutputFormat::BINARY, OutputFormat::NETCDF, OutputFormat::HDF5}) {
            if (!output_format_available(format)) continue;
            OutputSpec output{format,
                              base + std::string(output_suffix(format))};
            auto written = make_writer(format)->write_atomic(
                state, result.report, output, problem.value());
            check(written.has_value(), "write asymmetric result");
            if (!written.has_value()) continue;
            RunReport report;
            auto read =
                make_reader(format)->read(output.path, std::ref(report));
            check(read.has_value(), "read asymmetric result");
            if (read.has_value()) {
                check(read.value().families == state.families,
                      "twelve families round trip exactly");
                check(report.input_params == result.report.input_params,
                      "asymmetric input provenance round trip");
            }
            std::filesystem::remove(output.path);
        }
        const auto path = base + ".ckpt";
        check(write_checkpoint(state, result.report.input_params, path)
                  .has_value(),
              "write asymmetric checkpoint");
        InputParams input;
        auto restored = read_checkpoint(path, std::ref(input));
        check(restored.has_value(), "read asymmetric checkpoint");
        if (restored.has_value()) {
            check(restored.value().families == state.families &&
                      input == result.report.input_params,
                  "checkpoint preserves asymmetric state and provenance");
            spec.stages = {{std::size_t(state.ns), 100, tolerance}};
            auto replay_problem = validate(spec, options);
            SolveRequest request;
            request.restart = std::cref(restored.value());
            auto replay = solver.solve(replay_problem.value(), request);
            check(replay.converged && replay.iterations <= 2,
                  "asymmetric checkpoint is a fixed point");
        }
        std::filesystem::remove(path);

        // An axis outside the LCFS gives an invalid initial Jacobian. There
        // is no valid backup yet, so the solver must diagnose it immediately
        // instead of exhausting the iteration cap restoring the same state.
        auto invalid_spec = spec;
        invalid_spec.raxis_c[0] = 10.0;
        invalid_spec.stages = {{11, 3, tolerance}};
        bool rejected_initial_geometry = false;
        try {
            solver.solve(validate(invalid_spec, options).value());
        } catch (const CumesError& error) {
            rejected_initial_geometry =
                std::string(error.what()).find("Invalid initial geometry") !=
                std::string::npos;
        }
        check(rejected_initial_geometry,
              "invalid initial axis fails without futile restore retries");
    }
    // A toroidal phase shift breaks stellarator symmetry about zeta=0 but
    // leaves the physical equilibrium unchanged. Use the symmetric solver's
    // converged state as an independent fixed-point reference for all twelve
    // asymmetric force components, without another cold-start trajectory.
    auto symmetric = parsed.spec;
    symmetric.lasym = false;
    symmetric.rbs.clear();
    symmetric.zbc.clear();
    symmetric.raxis_s.clear();
    symmetric.zaxis_c.clear();
    symmetric.has_raxis_s = symmetric.has_zaxis_c = false;
    symmetric.ntor = 1;
    symmetric.nfp = 3;
    symmetric.raxis_c.push_back(0);
    symmetric.zaxis_s.push_back(0);
    symmetric.rbc.push_back({1, 1, .008});
    symmetric.zbs.push_back({1, -1, .006});
    EquilibriumSolver solver;
    auto reference = solver.solve(validate(symmetric, options).value());
    check(reference.converged, "rotation reference converges");
    auto rotated = symmetric;
    rotated.lasym = true;
    rotated.raxis_s = rotated.zaxis_c = {0, 0};
    constexpr double PHASE = .7;
    for (auto& h : rotated.rbc) {
        rotated.rbs.push_back({h.m, h.n, h.value * std::sin(h.n * PHASE)});
        h.value *= std::cos(h.n * PHASE);
    }
    for (auto& h : rotated.zbs) {
        rotated.zbc.push_back({h.m, h.n, -h.value * std::sin(h.n * PHASE)});
        h.value *= std::cos(h.n * PHASE);
    }
    auto state = reference.equilibrium;
    state.families.resize(EquilibriumSnapshot::ASYMMETRIC_COUNT);
    for (auto& family : state.families) family.resize(state.family_size());
    constexpr int COMPLEMENT[] = {9, 10, 11, 6, 7, 8};
    for (int c = 0; c < EquilibriumSnapshot::COUNT; ++c)
        for (int mode = 0; mode < state.mnmax; ++mode)
            for (int j = 0; j < state.ns; ++j) {
                const int i = mode * state.ns + j;
                const double value = reference.equilibrium.families[c][i];
                const double phase = (mode % 2) * PHASE;
                state.families[c][i] = value * std::cos(phase);
                state.families[COMPLEMENT[c]][i] =
                    (c < 3 ? -value : value) * std::sin(phase);
            }
    rotated.stages = {
        {std::size_t(state.ns), 100, symmetric.stages.back().tolerance}};
    SolveRequest request;
    request.restart = std::cref(state);
    auto replay = solver.solve(validate(rotated, options).value(), request);
    check(replay.converged && replay.iterations <= 2,
          "toroidal rotation preserves the converged physical equilibrium");
    return test::summary();
}
