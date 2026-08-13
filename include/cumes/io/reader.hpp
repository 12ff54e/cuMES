// reader.hpp — host-memory reader interface.
#pragma once

#include "cumes/core/result.hpp"
#include "cumes/io/equilibrium_snapshot.hpp"
#include "cumes/io/output_spec.hpp"

#include <memory>

namespace cumes {

class Reader {
 public:
    virtual ~Reader() = default;
    // Read a state file into a host snapshot. Returns an error on a missing
    // file, truncation, trailing data, or a header/dimension mismatch.
    virtual Result<EquilibriumSnapshot> read(const std::string& path) = 0;
};

// Factory. Returns nullptr for a (format, schema) combination this build does
// not read yet.
std::unique_ptr<Reader> make_reader(OutputFormat format, OutputSchema schema);

}  // namespace cumes
