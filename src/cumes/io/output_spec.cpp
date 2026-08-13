// output_spec.cpp — suffix -> OutputSpec dispatch and backend availability.
#include "cumes/io/output_spec.hpp"

#include <strings.h>  // strcasecmp

#include <optional>
#include <string>

namespace cumes {

bool output_format_available(OutputFormat fmt) {
    switch (fmt) {
        case OutputFormat::kBinary:
            return true;
        case OutputFormat::kNetCdf:
#ifdef CUMES_HAVE_NETCDF
            return true;
#else
            return false;
#endif
        case OutputFormat::kHdf5:
#ifdef CUMES_HAVE_HDF5
            return true;
#else
            return false;
#endif
    }
    return false;
}

const char* output_suffix(OutputFormat fmt) {
    switch (fmt) {
        case OutputFormat::kBinary: return ".bin";
        case OutputFormat::kNetCdf: return ".nc";
        case OutputFormat::kHdf5: return ".h5";
    }
    return "";
}

namespace {

// Map a path suffix (case-insensitive) to a format, or nullopt if unknown.
std::optional<OutputFormat> suffix_format(const std::string& path) {
    const std::size_t dot = path.find_last_of('.');
    const std::string ext = (dot == std::string::npos) ? "" : path.substr(dot);
    if (strcasecmp(ext.c_str(), ".bin") == 0) return OutputFormat::kBinary;
    if (strcasecmp(ext.c_str(), ".nc") == 0) return OutputFormat::kNetCdf;
    if (strcasecmp(ext.c_str(), ".h5") == 0 ||
        strcasecmp(ext.c_str(), ".hdf5") == 0) {
        return OutputFormat::kHdf5;
    }
    return std::nullopt;
}

}  // namespace

Result<OutputSpec> resolve_output_spec(const std::string& path,
                                       bool compatibility_mode) {
    // Pure suffix -> format dispatch; backend availability is the separate
    // output_format_available() preflight (mirrors the legacy
    // outputFormatAvailable vs outputSave split).
    const auto fmt = suffix_format(path);
    if (!fmt.has_value()) {
        if (compatibility_mode) {
            OutputSpec spec;
            spec.format = OutputFormat::kBinary;
            spec.schema = OutputSchema::kLegacyV0;
            spec.path = "cumes_state.bin";  // legacy fallback target
            return spec;
        }
        return Result<OutputSpec>("'" + path +
                                  "': unrecognized output suffix; pass "
                                  "--output-schema or a recognized suffix");
    }
    OutputSpec spec;
    spec.format = *fmt;
    spec.schema = OutputSchema::kLegacyV0;  // caller upgrades to v1 explicitly
    spec.path = path;
    return spec;
}

}  // namespace cumes
