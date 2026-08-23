// checkpoint.hpp — versioned checkpoint.
//
// An explicit, versioned restart artifact: a self-describing checkpoint
// (magic, version, precision, dimensions) that a reader validates before
// accepting.
#ifndef CUMES_INCLUDE_CUMES_IO_CHECKPOINT_HPP_
#define CUMES_INCLUDE_CUMES_IO_CHECKPOINT_HPP_

#include "cumes/core/result.hpp"
#include "cumes/io/equilibrium_snapshot.hpp"
#include "cumes/io/input_params.hpp"

#include <string>

namespace cumes {

// Write a versioned (schema v1) checkpoint. Version 2 appends the embedded
// normalized-input record after the state families.
Status write_checkpoint(const EquilibriumSnapshot& snapshot,
                        const InputParams& input_params,
                        const std::string& path);

// Read a versioned checkpoint, validating the magic, version, and dimensions.
// The restart path consumes the state only; when `input_params` is non-null
// and the checkpoint carries the version-2 record, it is decoded into it
// (left default-empty for version-1 checkpoints).
Result<EquilibriumSnapshot> read_checkpoint(
    const std::string& path,
    InputParams* input_params = nullptr);

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_IO_CHECKPOINT_HPP_
