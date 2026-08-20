// checkpoint.hpp — versioned checkpoint.
//
// An explicit, versioned restart artifact: a self-describing checkpoint
// (magic, version, precision, dimensions) that a reader validates before
// accepting.
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

}  // namespace cumes
