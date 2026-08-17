// writer.hpp — host-memory writer interface (blueprint §6.13).
//
// Writers consume host memory only (no CUDA calls) and publish atomically:
// write to a same-directory temp file, flush + fsync + close, then rename over
// the target. A reader never sees a half-written file, and any failure leaves
// the target untouched.
#pragma once

#include "cumes/config/validated_problem.hpp"
#include "cumes/core/result.hpp"
#include "cumes/io/equilibrium_snapshot.hpp"
#include "cumes/io/output_spec.hpp"
#include "cumes/io/run_report.hpp"

#include <memory>

namespace cumes {

// Final-stage scalar pack the legacy-v0 layouts record verbatim (the v0 files
// document the run through the final stage DeviceParams + SolverResult).
struct LegacyRunScalars {
    int mpol = 0, ntor = 0, nfp = 0, ntheta = 0, nzeta = 0, ns = 0, mnmax = 0,
        nZnT = 0, ncurr = 0, max_iter = 0;
    double delt = 0.0, ftol = 0.0, lamscale = 0.0;
    int iterations = 0;
    bool converged = false;
    double fsqr = 0.0, fsqz = 0.0, fsql = 0.0;
};

class Writer {
 public:
    virtual ~Writer() = default;
    // Writers consume host memory only (no CUDA calls) and publish atomically:
    // write to a same-directory temp file, flush + fsync + close, then rename
    // over the target. `problem` supplies the validated input model (v0 fixed-
    // capacity provenance, v1 raw/folded boundary harmonics); `scalars` the
    // final-stage scalar pack the v0 layout records.
    virtual Status write_atomic(const EquilibriumSnapshot& snapshot,
                                const RunReport& report,
                                const OutputSpec& spec,
                                const ValidatedProblem& problem,
                                const LegacyRunScalars& scalars) = 0;
};

// Factory. Returns nullptr for a (format, schema) combination this build does
// not produce. Defined in the ADAPTER library (cumes_io); the host-only
// binary subset is make_binary_writer below.
std::unique_ptr<Writer> make_writer(OutputFormat format, OutputSchema schema);

// Binary-only factories (host I/O library; no backend defines required).
std::unique_ptr<Writer> make_binary_writer(OutputFormat format,
                                           OutputSchema schema);

}  // namespace cumes
