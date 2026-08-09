cmake_minimum_required(VERSION 3.20)

# Use g++-12 as CUDA host compiler (CUDA 12.1 doesn't support gcc > 12).
# Must be set BEFORE project() or as a cache variable.
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12 CACHE FILEPATH "CUDA host compiler")

project(cuMES VERSION 0.1.0 LANGUAGES CXX CUDA)

# C++20: JsonParser.h (JSON input, src/input_json.cu) uses requires-clauses
# and concepts; CUDA 12.1 nvcc compiles the whole project at C++20.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "61;75;80;86;89")

# Optimisation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")

# Precision: all computation is templated on T (double or float); the
# executable's `Real` type is chosen here (the tests instantiate both
# explicitly and are unaffected).
option(CUMES_USE_FLOAT "Build cuMES in single precision (float)" OFF)

# Optional state-output backends. Each is compiled in only if the option is
# ON and the library is found; otherwise the run falls back to binary
# cumes_state.bin with a stderr warning.
option(CUMES_USE_NETCDF "Enable NetCDF output of the solved state (.nc)" ON)
option(CUMES_USE_HDF5  "Enable HDF5  output of the solved state (.h5/.hdf5)" ON)

find_package(CUDAToolkit 11 REQUIRED)

# ---- executable -------------------------------------------------------
add_executable(cuMES
    src/main.cu
    src/input_json.cu
    src/fourier.cu
    src/geometry.cu
    src/forces.cu
    src/solver.cu
    src/output.cu
    src/profiles.cu
    src/precon.cu
    src/constraint.cu
    src/refine.cu
)

target_link_libraries(cuMES PRIVATE
    CUDA::cublas
    CUDA::cufft
)

target_include_directories(cuMES PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- optional output backends (NetCDF / HDF5) ---------------------------
# Each backend TU is added only when the option is ON and the library is
# found; the suffix dispatcher in src/output.cu falls back to binary
# cumes_state.bin when a requested format is not compiled in.
set(CUMES_OUTPUT_BACKENDS "")   # backend TUs, appended when a library is found

# NetCDF: Debian's libnetcdf-dev ships neither FindnetCDF.cmake nor a
# netCDFConfig package, so probe for a package first and fall back to
# pkg-config (the vmecpp pattern). find_package and pkg_check_modules share
# the netCDF_* variable namespace, so the fallback cleanly overwrites the
# failed probe.
if(CUMES_USE_NETCDF)
    find_package(netCDF QUIET)
    if(NOT netCDF_FOUND)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(netCDF QUIET IMPORTED_TARGET netcdf)
        endif()
    endif()
    if(netCDF_FOUND)
        list(APPEND CUMES_OUTPUT_BACKENDS src/output_netcdf.cu)
        target_compile_definitions(cuMES PRIVATE CUMES_HAVE_NETCDF)
        if(TARGET netCDF::netCDF)
            target_link_libraries(cuMES PRIVATE netCDF::netCDF)
        elseif(TARGET PkgConfig::netcdf)
            target_link_libraries(cuMES PRIVATE PkgConfig::netcdf)
        else()
            target_link_libraries(cuMES PRIVATE ${netCDF_LIBRARIES})
            target_include_directories(cuMES PRIVATE ${netCDF_INCLUDE_DIRS})
        endif()
        message(STATUS "cuMES: NetCDF output enabled (netCDF ${netCDF_VERSION})")
    else()
        message(WARNING "cuMES: NetCDF not found - .nc output disabled "
                        "(install libnetcdf-dev or pass -DCUMES_USE_NETCDF=OFF)")
    endif()
endif()

# HDF5: CMake's FindHDF5 handles the Debian serial layout (interrogates
# h5cc, resolves libhdf5.so under .../hdf5/serial, headers in
# /usr/include/hdf5/serial) and provides the HDF5::HDF5 imported target.
if(CUMES_USE_HDF5)
    find_package(HDF5 QUIET COMPONENTS C)
    if(HDF5_FOUND)
        list(APPEND CUMES_OUTPUT_BACKENDS src/output_hdf5.cu)
        target_compile_definitions(cuMES PRIVATE CUMES_HAVE_HDF5)
        target_link_libraries(cuMES PRIVATE HDF5::HDF5)
        message(STATUS "cuMES: HDF5 output enabled (HDF5 ${HDF5_VERSION})")
    else()
        message(WARNING "cuMES: HDF5 not found - .h5 output disabled "
                        "(install libhdf5-dev or pass -DCUMES_USE_HDF5=OFF)")
    endif()
endif()

if(CUMES_OUTPUT_BACKENDS)
    target_sources(cuMES PRIVATE ${CUMES_OUTPUT_BACKENDS})
endif()

if(CUMES_USE_FLOAT)
    target_compile_definitions(cuMES PRIVATE CUMES_USE_FLOAT)
endif()

# ---- tests (standalone executables, no framework) ----------------------
add_executable(test_fourier
    tests/test_fourier.cu
    src/fourier.cu
)

add_executable(test_input_json
    tests/test_input_json.cu
    src/input_json.cu
)
target_include_directories(test_input_json PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_fourier PRIVATE CUDA::cublas CUDA::cufft)
target_include_directories(test_fourier PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_forces
    tests/test_forces.cu
    src/input_json.cu
    src/fourier.cu
    src/geometry.cu
    src/forces.cu
    src/profiles.cu
)
target_link_libraries(test_forces PRIVATE CUDA::cublas CUDA::cufft)
target_include_directories(test_forces PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_force_verify
    tests/test_force_verify.cu
    src/input_json.cu
    src/fourier.cu
    src/geometry.cu
    src/forces.cu
    src/profiles.cu
)

add_executable(test_geometry_iso
    tests/test_geometry_iso.cu
    src/input_json.cu
    src/fourier.cu
    src/geometry.cu
    src/profiles.cu
)
target_link_libraries(test_geometry_iso PRIVATE CUDA::cublas CUDA::cufft)
target_include_directories(test_geometry_iso PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_force_verify PRIVATE CUDA::cublas CUDA::cufft)
target_include_directories(test_force_verify PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
