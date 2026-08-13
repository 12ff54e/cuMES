// writer.hpp — host-memory writer interface (blueprint §6.13).
//
// Writers consume host memory only (no CUDA calls) and publish atomically:
// write to a same-directory temp file, flush + fsync + close, then rename over
// the target. A reader never sees a half-written file, and any failure leaves
// the target untouched.
#pragma once

#include "cumes/core/result.hpp"
#include "cumes/io/equilibrium_snapshot.hpp"
#include "cumes/io/output_spec.hpp"
#include "cumes/io/run_report.hpp"

#include <memory>

namespace cumes {

class Writer {
 public:
    virtual ~Writer() = default;
    virtual Status write_atomic(const EquilibriumSnapshot& snapshot,
                                const RunReport& report,
                                const OutputSpec& spec) = 0;
};

// Factory. Returns nullptr for a (format, schema) combination this build does
// not produce yet (e.g. the NetCDF/HDF5 host adapters, which are deferred until
// the Phase 3 snapshot bridge).
std::unique_ptr<Writer> make_writer(OutputFormat format, OutputSchema schema);

}  // namespace cumes
