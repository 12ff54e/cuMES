// output_spec.cpp — suffix -> OutputSpec dispatch and backend availability.
#include "cumes/io/output_spec.hpp"

#include <optional>
#include <string>
#include <string_view>

#include <strings.h>  // strcasecmp

namespace cumes {

// output_format_available() is deliberately NOT defined here: it depends on
// the optional-backend availability defines
// (CUMES_HAVE_NETCDF/CUMES_HAVE_HDF5), which are confined to the adapter
// library per completion plan step 2.5. The definition lives in src/output.cpp
// (cumes_io); this host library stays free of NetCDF/HDF5 headers and
// availability defines.

std::string_view output_suffix(OutputFormat fmt) {
    switch (fmt) {
        case OutputFormat::BINARY:
            return ".bin";
        case OutputFormat::NETCDF:
            return ".nc";
        case OutputFormat::HDF5:
            return ".h5";
    }
    return "";
}

namespace {

// Map a path suffix (case-insensitive) to a format, or nullopt if unknown.
std::optional<OutputFormat> suffix_format(const std::string& path) {
    const std::size_t dot = path.find_last_of('.');
    const std::string ext = (dot == std::string::npos) ? "" : path.substr(dot);
    if (strcasecmp(ext.c_str(), ".bin") == 0) return OutputFormat::BINARY;
    if (strcasecmp(ext.c_str(), ".nc") == 0) return OutputFormat::NETCDF;
    if (strcasecmp(ext.c_str(), ".h5") == 0 ||
        strcasecmp(ext.c_str(), ".hdf5") == 0) {
        return OutputFormat::HDF5;
    }
    return std::nullopt;
}

}  // namespace

Result<OutputSpec> resolve_output_spec(const std::string& path) {
    // Pure suffix -> format dispatch; backend availability is the separate
    // output_format_available() preflight (mirrors the legacy
    // outputFormatAvailable vs outputSave split).
    const auto fmt = suffix_format(path);
    if (!fmt.has_value()) {
        return Result<OutputSpec>("'" + path +
                                  "': unrecognized output suffix; pass a "
                                  "recognized suffix (.bin, .nc, .h5)");
    }
    OutputSpec spec;
    spec.format = *fmt;
    spec.path = path;
    return spec;
}

}  // namespace cumes
