// reader.hpp — host-memory reader interface.
#ifndef CUMES_INCLUDE_CUMES_IO_READER_HPP_
#define CUMES_INCLUDE_CUMES_IO_READER_HPP_

#include "cumes/core/result.hpp"
#include "cumes/io/equilibrium_snapshot.hpp"
#include "cumes/io/output_spec.hpp"
#include "cumes/io/run_report.hpp"

#include <memory>

namespace cumes {

class Reader {
   public:
    virtual ~Reader() = default;
    // Read a state file into a host snapshot. Returns an error on a missing
    // file, truncation, trailing data, or a header/dimension mismatch.
    // report, when non-null, additionally receives the recorded RunReport
    // (provenance + per-stage outcomes + restart history) - the schema-v1
    // round-trip contract (completion plan step 2.3).
    virtual Result<EquilibriumSnapshot> read(const std::string& path,
                                             RunReport* report = nullptr) = 0;
};

// Factory. Returns nullptr for a format this build does not read. Defined in
// the ADAPTER library (cumes_io); the host-only binary subset is
// make_binary_reader below.
std::unique_ptr<Reader> make_reader(OutputFormat format);

// Binary-only factory (host I/O library; no backend defines required).
std::unique_ptr<Reader> make_binary_reader();

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_IO_READER_HPP_
