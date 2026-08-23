// output_spec.hpp — typed output format selection (blueprint §6.13).
//
// Replaces the suffix-string dispatcher in src/output.cpp with a validated,
// typed contract. resolve_output_spec() maps a path suffix (case-insensitive)
// to an OutputFormat; a known suffix whose backend is not linked is a preflight
// error (before any CUDA work), and an unknown suffix is always an error.
#ifndef CUMES_INCLUDE_CUMES_IO_OUTPUT_SPEC_HPP_
#define CUMES_INCLUDE_CUMES_IO_OUTPUT_SPEC_HPP_

#include "cumes/core/result.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace cumes {

enum class OutputFormat : std::uint8_t { BINARY = 0, NETCDF = 1, HDF5 = 2 };

struct OutputSpec {
    OutputFormat format = OutputFormat::BINARY;
    std::string path;
};

// Is `fmt` produced by this build (backend compiled in)?
bool output_format_available(OutputFormat fmt);

// Resolve a path's suffix to an OutputSpec. An unknown suffix is rejected; a
// known-but-unlinked backend is rejected separately by the
// output_format_available() preflight.
Result<OutputSpec> resolve_output_spec(const std::string& path);

// The suffix (lowercased, with leading '.') implied by a format, e.g. ".bin".
std::string_view output_suffix(OutputFormat fmt);

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_IO_OUTPUT_SPEC_HPP_
