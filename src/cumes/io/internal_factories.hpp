// internal_factories.hpp — cross-TU declarations for the v1 NetCDF/HDF5
// backend factories. Not installed.
#ifndef CUMES_SRC_CUMES_IO_INTERNAL_FACTORIES_HPP_
#define CUMES_SRC_CUMES_IO_INTERNAL_FACTORIES_HPP_

#include "cumes/io/reader.hpp"
#include "cumes/io/writer.hpp"

#include <memory>

namespace cumes {

// NetCDF/HDF5 host adapters (completion plan step 2.2/2.3): STRONG definitions
// live in src/cumes/io/netcdf_writer.cpp / hdf5_writer.cpp, which compile only
// under their CUMES_HAVE_* defines and are the only TUs including
// netcdf.h/hdf5.h. The full make_writer/make_reader dispatch lives in the
// adapter library (src/cumes/io/writer_dispatch.cpp, compiled into cumes_io)
// so the strong references force the linker to extract these adapter members
// from the archive; the host-only library exposes binary-only factories
// (make_binary_writer/make_binary_reader in versioned_binary.cpp) that need
// no backend defines.
std::unique_ptr<Writer> make_netcdf_v1_writer();
std::unique_ptr<Reader> make_netcdf_v1_reader();
std::unique_ptr<Writer> make_hdf5_v1_writer();
std::unique_ptr<Reader> make_hdf5_v1_reader();

}  // namespace cumes

#endif  // CUMES_SRC_CUMES_IO_INTERNAL_FACTORIES_HPP_
