// output_spec.hpp — typed output format/schema selection (blueprint §6.13).
//
// Replaces the suffix-string dispatcher in src/output.cpp with a validated,
// typed contract. resolve_output_spec() maps a path suffix (case-insensitive)
// to an OutputFormat; a known suffix whose backend is not linked is a preflight
// error (before any CUDA work), and an unknown suffix is an error unless
// compatibility mode requests the legacy binary fallback.
#pragma once

#include "cumes/core/result.hpp"

#include <cstdint>
#include <string>

namespace cumes {

enum class OutputFormat : std::uint8_t { kBinary = 0, kNetCdf = 1, kHdf5 = 2 };

enum class OutputSchema : std::uint8_t {
    kLegacyV0 = 0,  // padded-capacity, [surface,mode] logical mapping
    kV1 = 1,        // active dimensions + full provenance
};

struct OutputSpec {
    OutputFormat format = OutputFormat::kBinary;
    OutputSchema schema = OutputSchema::kLegacyV0;
    std::string path;
};

// Is `fmt` produced by this build (backend compiled in)?
bool output_format_available(OutputFormat fmt);

// Resolve a path's suffix to an OutputSpec. `compatibility_mode` permits the
// legacy behavior of falling back to binary on an unknown suffix; otherwise an
// unknown suffix is rejected. A known-but-unlinked backend is always rejected.
Result<OutputSpec> resolve_output_spec(const std::string& path,
                                       bool compatibility_mode);

// The suffix (lowercased, with leading '.') implied by a format, e.g. ".bin".
const char* output_suffix(OutputFormat fmt);

}  // namespace cumes
