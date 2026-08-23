// writer.hpp — host-memory writer interface (blueprint §6.13).
//
// Writers consume host memory only (no CUDA calls) and publish atomically:
// write to a same-directory temp file, flush + fsync + close, then rename over
// the target. A reader never sees a half-written file, and any failure leaves
// the target untouched.
#ifndef CUMES_INCLUDE_CUMES_IO_WRITER_HPP_
#define CUMES_INCLUDE_CUMES_IO_WRITER_HPP_

#include "cumes/config/validated_problem.hpp"
#include "cumes/core/result.hpp"
#include "cumes/io/equilibrium_snapshot.hpp"
#include "cumes/io/output_spec.hpp"
#include "cumes/io/run_report.hpp"

#include <memory>

namespace cumes {

class Writer {
   public:
    virtual ~Writer() = default;
    // Writers consume host memory only (no CUDA calls) and publish atomically:
    // write to a same-directory temp file, flush + fsync + close, then rename
    // over the target. `problem` supplies the validated input model (the raw
    // and folded boundary harmonics).
    virtual Status write_atomic(const EquilibriumSnapshot& snapshot,
                                const RunReport& report,
                                const OutputSpec& spec,
                                const ValidatedProblem& problem) = 0;
};

// Factory. Returns nullptr for a format this build does not produce. Defined
// in the ADAPTER library (cumes_io); the host-only binary subset is
// make_binary_writer below.
std::unique_ptr<Writer> make_writer(OutputFormat format);

// Binary-only factory (host I/O library; no backend defines required).
std::unique_ptr<Writer> make_binary_writer();

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_IO_WRITER_HPP_
