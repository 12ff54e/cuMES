// checkpoint.hpp — versioned checkpoint + legacy init converter.
//
// Replaces the environment-only CUMES_LOAD_INIT / vmecpp_init.bin path with an
// explicit, versioned restart artifact: a self-describing checkpoint (magic,
// version, precision, dimensions) that a reader validates before accepting.
// convert_legacy_init() reads the legacy six-family double payload (the exact
// CUMES_LOAD_INIT format) into the same host snapshot, enabling a conversion
// tool or a Phase 3 --restart option.
#pragma once

#include "cumes/core/result.hpp"
#include "cumes/io/equilibrium_snapshot.hpp"

#include <string>

namespace cumes {

// Write a versioned (schema v1) checkpoint.
Status write_checkpoint(const EquilibriumSnapshot& snapshot,
                        const std::string& path);

// Read a versioned checkpoint, validating the magic, version, and dimensions.
Result<EquilibriumSnapshot> read_checkpoint(const std::string& path);

// Convert a legacy six-family vmecpp_init.bin payload into a snapshot,
// validating the (ns, mnmax) header against the expected shape.
Result<EquilibriumSnapshot> convert_legacy_init(const std::string& path,
                                                int expected_ns,
                                                int expected_mnmax);

}  // namespace cumes
