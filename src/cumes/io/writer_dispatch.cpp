// writer_dispatch.cpp — the full Writer/Reader factory dispatch, compiled into
// the ADAPTER library (cumes_io — the only target with the CUMES_HAVE_NETCDF/
// CUMES_HAVE_HDF5 availability defines and the NetCDF/HDF5 headers; completion
// plan step 2.5).
//
// make_writer/make_reader here reference the optional-backend factories
// STRONGLY, which is what makes the static-library linker extract the adapter
// TUs (netcdf_writer.o / hdf5_writer.o) when the backends are compiled in. The
// host-only library keeps the binary factories (make_binary_writer /
// make_binary_reader in legacy_binary_v0.cpp) for consumers that need no
// optional backends.
#include "cumes/io/reader.hpp"
#include "cumes/io/writer.hpp"
#include "internal_factories.hpp"

#include <memory>

namespace cumes {

std::unique_ptr<Writer> make_writer(OutputFormat format, OutputSchema schema) {
    if (format == OutputFormat::kBinary) {
        return make_binary_writer(format, schema);
    }
#ifdef CUMES_HAVE_NETCDF
    if (format == OutputFormat::kNetCdf && schema == OutputSchema::kLegacyV0) {
        return make_netcdf_v0_writer();  // defined in netcdf_writer.cpp
    }
    if (format == OutputFormat::kNetCdf && schema == OutputSchema::kV1) {
        return make_netcdf_v1_writer();
    }
#endif
#ifdef CUMES_HAVE_HDF5
    if (format == OutputFormat::kHdf5 && schema == OutputSchema::kLegacyV0) {
        return make_hdf5_v0_writer();  // defined in hdf5_writer.cpp
    }
    if (format == OutputFormat::kHdf5 && schema == OutputSchema::kV1) {
        return make_hdf5_v1_writer();
    }
#endif
    return nullptr;
}

std::unique_ptr<Reader> make_reader(OutputFormat format, OutputSchema schema) {
    if (format == OutputFormat::kBinary) {
        return make_binary_reader(format, schema);
    }
#ifdef CUMES_HAVE_NETCDF
    if (format == OutputFormat::kNetCdf && schema == OutputSchema::kV1) {
        return make_netcdf_v1_reader();  // defined in netcdf_writer.cpp
    }
#endif
#ifdef CUMES_HAVE_HDF5
    if (format == OutputFormat::kHdf5 && schema == OutputSchema::kV1) {
        return make_hdf5_v1_reader();  // defined in hdf5_writer.cpp
    }
#endif
    return nullptr;
}

}  // namespace cumes
