// validated_problem.hpp — immutable validated host model (blueprint §6.1).
//
// ValidatedProblem is the "third stage" of the four-stage config pipeline
// (ProblemSpec -> ValidationReport -> ValidatedProblem -> DeviceParams<T>).
// Construction via validate() proves the invariants the solver needs:
//   - scalar/angular/radial ranges, resolution defaults applied (even ntheta);
//   - a non-empty, strictly increasing stage schedule (no fixed capacity — the
//     v1 output records active dimensions);
//   - boundary harmonics folded into the stellarator-symmetric product basis;
//   - the per-mode table (physical_n, mn_scale, xmpq, parity, first_surface);
//   - precision/tolerance compatibility.
//
// It is host-only and owns no device pointers. The solver consumes it directly
// (the legacy InputParams fixed-capacity bridge was deleted in migration step
// 13.2).
#pragma once

#include "cumes/config/precision_policy.hpp"
#include "cumes/config/problem_spec.hpp"
#include "cumes/config/solver_options.hpp"
#include "cumes/config/validation_report.hpp"
#include "cumes/core/grid_shape.hpp"
#include "cumes/core/mode_table.hpp"
#include "cumes/core/result.hpp"

#include <string>
#include <vector>

namespace cumes {

// Folded stellarator-symmetric boundary coefficients (blueprint §4.2), stored
// flattened [mode] = m*(ntor+1)+n (surface-agnostic; the surface dependence is
// reintroduced by the initial-state builder). Length modes() each.
struct FoldedBoundary {
    std::vector<double> rbcc;  // R: cos(mθ)cos(nζ)
    std::vector<double> rbss;  // R: sin(mθ)sin(nζ)
    std::vector<double> zbsc;  // Z: sin(mθ)cos(nζ)
    std::vector<double> zbcs;  // Z: cos(mθ)sin(nζ)

    std::size_t size() const { return rbcc.size(); }
};

class ValidatedProblem {
   public:
    const ProblemSpec& spec() const { return spec_; }
    const SolverOptions& options() const { return options_; }
    PrecisionPolicy precision() const { return options_.precision; }

    // Resolved shape of each stage (ns varies; angles/modes are common).
    const std::vector<GridShape>& stage_shapes() const { return stage_shapes_; }
    // Stage-0 shape (angles/modes common to all stages).
    const GridShape& shape() const { return stage_shapes_.front(); }
    const ModeTable<double>& mode_table() const { return mode_table_; }
    const FoldedBoundary& boundary() const { return boundary_; }

    // Non-fatal findings (e.g. a skipped out-of-range boundary harmonic, an
    // unknown-key in compatibility mode). Collected end-to-end so the CLI can
    // report them; a warning never makes the problem invalid.
    const ValidationReport& warnings() const { return warnings_; }
    void add_warning(std::string key, std::string message) {
        warnings_.warn(std::move(key), std::move(message));
    }

    // Canonical, deterministic JSON representation (for configuration goldens).
    std::string normalize_to_json() const;

   private:
    friend BasicResult<ValidatedProblem, ValidationReport> validate(
        ProblemSpec spec,
        const SolverOptions& options);

    ProblemSpec spec_;
    SolverOptions options_;
    std::vector<GridShape> stage_shapes_;
    ModeTable<double> mode_table_;
    FoldedBoundary boundary_;
    ValidationReport warnings_;
};

using ValidationResult = BasicResult<ValidatedProblem, ValidationReport>;

// Four-stage parse->validate->fold->resolve. Returns the immutable model, or
// the collected ValidationReport on any error. RuntimeCapabilities is deferred
// to Phase 3 (no device probing yet); shape-only validation lives here.
ValidationResult validate(ProblemSpec spec, const SolverOptions& options);

}  // namespace cumes
