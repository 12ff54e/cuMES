# CumesDependencies.cmake — required and optional third-party dependencies.
#
# Defines:
#   CUDAToolkit          (required; imported targets CUDA::cufft / CUDA::cudart)
#   Threads              (required by asynchronous multigrid setup)
#   CUMES_HAVE_NETCDF    (bool) — .nc output compiled in
#   CUMES_HAVE_HDF5      (bool) — .h5/.hdf5 output compiled in
#   netCDF::netCDF / PkgConfig::netcdf / netCDF_*     (NetCDF link target)
#   HDF5::HDF5                                        (HDF5 link target)
#
# The optional backends are host-only: their `.cpp` sources are compiled by the
# C++ compiler and linked only against the writer's own library, never the CUDA
# solver target.

find_package(CUDAToolkit 11 REQUIRED)
find_package(Threads REQUIRED)

# ---- NetCDF ---------------------------------------------------------------
# Debian's libnetcdf-dev ships neither FindnetCDF.cmake nor a netCDFConfig
# package, so probe for a package first and fall back to pkg-config (the vmecpp
# pattern). find_package and pkg_check_modules share the netCDF_* variable
# namespace, so the fallback cleanly overwrites the failed probe.
set(CUMES_HAVE_NETCDF FALSE)
if(CUMES_USE_NETCDF)
  find_package(netCDF QUIET)
  if(NOT netCDF_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(netCDF QUIET IMPORTED_TARGET netcdf)
    endif()
  endif()
  if(netCDF_FOUND)
    set(CUMES_HAVE_NETCDF TRUE)
    message(STATUS "cuMES: NetCDF output enabled (netCDF ${netCDF_VERSION})")
  else()
    message(STATUS "cuMES: NetCDF not found; continuing without NetCDF support")
  endif()
endif()

# ---- HDF5 ------------------------------------------------------------------
# CMake's FindHDF5 handles the Debian serial layout (interrogates h5cc, resolves
# libhdf5.so under .../hdf5/serial, headers in /usr/include/hdf5/serial) and
# provides the HDF5::HDF5 imported target.
set(CUMES_HAVE_HDF5 FALSE)
if(CUMES_USE_HDF5)
  find_package(HDF5 QUIET COMPONENTS C)
  if(HDF5_FOUND)
    set(CUMES_HAVE_HDF5 TRUE)
    message(STATUS "cuMES: HDF5 output enabled (HDF5 ${HDF5_VERSION})")
  else()
    message(STATUS "cuMES: HDF5 not found; continuing without HDF5 support")
  endif()
endif()

# ---- helper: attach the optional writer backends to a target ----------------
# Adds the .cpp backend sources and their library/defines to `target`. The
# writer TUs are guarded by CUMES_HAVE_NETCDF / CUMES_HAVE_HDF5 in output.cuh,
# so the defines must be PUBLIC (the dispatcher in output.cpp reads them too).
function(cumes_link_output_backends target)
  if(CUMES_HAVE_NETCDF)
    target_sources(${target} PRIVATE ${PROJECT_SOURCE_DIR}/src/cumes/io/netcdf_writer.cpp)
    target_compile_definitions(${target} PUBLIC CUMES_HAVE_NETCDF)
    if(TARGET netCDF::netCDF)
      target_link_libraries(${target} PUBLIC netCDF::netCDF)
    elseif(TARGET PkgConfig::netcdf)
      target_link_libraries(${target} PUBLIC PkgConfig::netcdf)
    else()
      target_link_libraries(${target} PUBLIC ${netCDF_LIBRARIES})
      target_include_directories(${target} PUBLIC ${netCDF_INCLUDE_DIRS})
    endif()
  endif()
  if(CUMES_HAVE_HDF5)
    target_sources(${target} PRIVATE ${PROJECT_SOURCE_DIR}/src/cumes/io/hdf5_writer.cpp)
    target_compile_definitions(${target} PUBLIC CUMES_HAVE_HDF5)
    target_link_libraries(${target} PUBLIC HDF5::HDF5)
  endif()
endfunction()
