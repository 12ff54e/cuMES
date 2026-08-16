cmake_minimum_required(VERSION 3.20)

# Use g++-12 as CUDA host compiler (CUDA 12.1 doesn't support gcc > 12).
# Must be set BEFORE project() or as a cache variable. Override with
# -DCMAKE_CUDA_HOST_COMPILER=<path> if your toolchain differs.
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12 CACHE FILEPATH "CUDA host compiler")

# ---- cmake modules ---------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# CUDA architectures must be defaulted BEFORE project()/enable_language(CUDA):
# CMake otherwise substitutes its own fallback (sm_52, which lacks 64-bit
# atomics) into the cache, and a later non-FORCE set(... CACHE ...) cannot
# override an already-cached value. CumesCudaArchitectures.cmake sets the
# default only when CMAKE_CUDA_ARCHITECTURES was not already provided.
include(CumesCudaArchitectures)

project(cuMES VERSION 0.1.0 LANGUAGES CXX CUDA)

include(CumesOptions)
include(CumesDependencies)
include(CumesWarnings)
include(CumesTest)
include(CumesSanitizers)

# ---- language standards (project-wide) -------------------------------------
# C++20: JsonParser.h uses requires-clauses and concepts; CUDA 12.1 nvcc
# compiles the whole project at C++20.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- global optimisation flags ---------------------------------------------
# Warnings are target-scoped (CumesWarnings.cmake); optimisation stays global
# so the exact -O3/--use_fast_math/-march=native codegen the baseline was
# frozen with is preserved. Fast-math is the pre-existing default; precision
# policies (verify/fast/float/debug) are a Phase 2 PrecisionPolicy concern.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# ---- device modules --------------------------------------------------------
# Eight templated operator modules, each split into a *_double.cu and a
# *_float.cu explicit-instantiation TU (one scalar type per TU). The CLI links
# only the TU matching `Real`; the tests link both (they instantiate both).
set(CUMES_CUDA_MODULES fourier geometry forces solver profiles precon constraint refine axisymmetric)
set(CUMES_CUDA_DOUBLE_SOURCES "")
set(CUMES_CUDA_FLOAT_SOURCES "")
foreach(mod IN LISTS CUMES_CUDA_MODULES)
  list(APPEND CUMES_CUDA_DOUBLE_SOURCES src/${mod}_double.cu)
  list(APPEND CUMES_CUDA_FLOAT_SOURCES src/${mod}_float.cu)
endforeach()

# ---- host-only library targets ---------------------------------------------

# JsonParser implementation (header-only library, compiled once). Shared by the
# legacy parser and the new host-model reader.
add_library(cumes_json STATIC src/json_parser.cpp)
target_include_directories(cumes_json PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
cumes_enable_warnings(cumes_json)

# JSON input parser (no CUDA; compiled by the C++ compiler).
add_library(cumes_config STATIC src/input_json.cpp)
target_include_directories(cumes_config PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cumes_config PUBLIC cumes_json)
cumes_enable_warnings(cumes_config)

# ---- Phase 2: validated host model (namespace cumes) -----------------------
# Core host model: result/status, checked arithmetic, grid shape, mode table.
# Header-mostly; the two non-trivial TU implementations live in src/cumes/core.
add_library(cumes_core STATIC
  src/cumes/core/grid_shape.cpp
  src/cumes/core/mode_table.cpp)
target_include_directories(cumes_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
cumes_enable_warnings(cumes_core)

# Config model + JSON reader: dynamic ProblemSpec -> ValidatedProblem, the
# legacy InputParams bridge, and the normalization emitter. Host-only C++.
add_library(cumes_config_json STATIC
  src/cumes/config/validation_report.cpp
  src/cumes/config/validated_problem.cpp
  src/cumes/config/json_reader.cpp)
target_include_directories(cumes_config_json PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cumes_config_json PUBLIC cumes_core cumes_json)
cumes_enable_warnings(cumes_config_json)

# Versioned + legacy host I/O: output spec, run report, snapshot, binary v0
# (byte-exact) and v1 writers/readers, and the versioned checkpoint + legacy
# init converter. Host-only C++ (no CUDA); NetCDF/HDF5 host adapters are
# deferred to Phase 3.
add_library(cumes_io_host STATIC
  src/cumes/io/output_spec.cpp
  src/cumes/io/legacy_binary_v0.cpp
  src/cumes/io/versioned_binary.cpp
  src/cumes/io/checkpoint.cpp)
target_include_directories(cumes_io_host PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cumes_io_host PUBLIC cumes_core)
cumes_enable_warnings(cumes_io_host)

# Output writers: binary dispatcher + optional NetCDF/HDF5 backends (host-only
# C++, using only the CUDA runtime API for D2H copies — no device code).
add_library(cumes_io STATIC src/output.cpp)
target_include_directories(cumes_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cumes_io PUBLIC CUDA::cudart CUDA::cufft)
cumes_enable_warnings(cumes_io)
cumes_link_output_backends(cumes_io)

# Header-only shared test helpers (checkCuda/cc). Tests include
# cumes_test_support.cuh; this target only exposes its include dir.
add_library(cumes_test_support INTERFACE)
target_include_directories(cumes_test_support INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/tests/support)

# ---- CUDA runtime RAII (host-only CUDA runtime API) ------------------------
# DeviceBuffer/PinnedBuffer/Stream/Event/DeviceContext and the centralized
# check_cuda/check_cufft error boundary. No device code; links cudart+cufft.
add_library(cumes_cuda_runtime STATIC src/cumes/runtime/device_context.cpp)
target_include_directories(cumes_cuda_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cumes_cuda_runtime PUBLIC CUDA::cudart CUDA::cufft)
cumes_enable_warnings(cumes_cuda_runtime)

# ---- CUDA operator libraries (one scalar type each) ------------------------

add_library(cumes_cuda_double STATIC ${CUMES_CUDA_DOUBLE_SOURCES})
target_include_directories(cumes_cuda_double PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cumes_cuda_double PUBLIC CUDA::cufft cumes_cuda_runtime)
cumes_enable_warnings(cumes_cuda_double)

add_library(cumes_cuda_float STATIC ${CUMES_CUDA_FLOAT_SOURCES})
target_include_directories(cumes_cuda_float PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cumes_cuda_float PUBLIC CUDA::cufft cumes_cuda_runtime)
cumes_enable_warnings(cumes_cuda_float)

# ---- CLI executable --------------------------------------------------------
if(CUMES_USE_FLOAT)
  set(CUMES_SELECTED_CUDA cumes_cuda_float)
else()
  set(CUMES_SELECTED_CUDA cumes_cuda_double)
endif()

add_executable(cuMES src/main.cu)
target_include_directories(cuMES PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cuMES PRIVATE cumes_config_json cumes_io cumes_io_host ${CUMES_SELECTED_CUDA})
cumes_enable_warnings(cuMES)
if(CUMES_USE_FLOAT)
  target_compile_definitions(cuMES PRIVATE CUMES_USE_FLOAT)
endif()

# ---- build provenance (recorded in the v1 output schema) --------------------
# Configure-time git revision/dirty flag; empty when git is unavailable. These
# are best-effort provenance, not a build gate.
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE CUMES_GIT_REVISION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
execute_process(
  COMMAND git status --porcelain
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE CUMES_GIT_STATUS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(CUMES_GIT_STATUS STREQUAL "")
  set(CUMES_GIT_DIRTY 0)
else()
  set(CUMES_GIT_DIRTY 1)
endif()
target_compile_definitions(cuMES PRIVATE
  CUMES_GIT_REVISION="${CUMES_GIT_REVISION}"
  CUMES_GIT_DIRTY=${CUMES_GIT_DIRTY}
  CUMES_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

# ---- tests (standalone executables, no framework) --------------------------
# Each test links the full device libraries (both scalar types) so every test
# can instantiate double and float; the linker pulls in only the objects it
# needs. Tests run from the source dir (they read inputs/*.json via CWD).

add_executable(test_fourier tests/test_fourier.cu)
target_include_directories(test_fourier PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_fourier PRIVATE cumes_test_support cumes_cuda_double cumes_cuda_float)
cumes_enable_warnings(test_fourier)

add_executable(test_input_json tests/test_input_json.cu)
target_include_directories(test_input_json PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_input_json PRIVATE cumes_config)
cumes_enable_warnings(test_input_json)

foreach(t IN ITEMS test_forces test_force_verify test_geometry_iso
                   test_geometry_ncurr test_regression_kernels test_constraint_tcon
                   test_force_reference test_axisym_backend test_rzcon_fusion
                   test_tridiagonal)
  add_executable(${t} tests/${t}.cu)
  target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(${t} PRIVATE cumes_test_support cumes_config cumes_cuda_double cumes_cuda_float)
  cumes_enable_warnings(${t})
endforeach()

add_executable(test_output_failure tests/test_output_failure.cu)
target_include_directories(test_output_failure PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_output_failure PRIVATE cumes_test_support cumes_io)
cumes_enable_warnings(test_output_failure)

add_executable(test_host_config tests/test_host_config.cpp)
target_include_directories(test_host_config PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_host_config PRIVATE cumes_config_json cumes_config)
cumes_enable_warnings(test_host_config)

add_executable(test_host_io tests/test_host_io.cpp)
target_include_directories(test_host_io PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_host_io PRIVATE cumes_io_host)
cumes_enable_warnings(test_host_io)

add_executable(test_checkpoint tests/test_checkpoint.cpp)
target_include_directories(test_checkpoint PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_checkpoint PRIVATE cumes_io_host)
cumes_enable_warnings(test_checkpoint)

add_executable(test_io_golden tests/test_io_golden.cu)
target_include_directories(test_io_golden PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_io_golden PRIVATE cumes_test_support cumes_io cumes_io_host)
cumes_enable_warnings(test_io_golden)

add_executable(test_runtime tests/test_runtime.cu)
target_include_directories(test_runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_runtime PRIVATE cumes_test_support cumes_cuda_runtime)
cumes_enable_warnings(test_runtime)

add_executable(test_controller tests/test_controller.cpp)
target_include_directories(test_controller PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
cumes_enable_warnings(test_controller)

add_executable(test_arena tests/test_arena.cu)
target_include_directories(test_arena PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_arena PRIVATE cumes_test_support cumes_cuda_runtime)
cumes_enable_warnings(test_arena)

add_executable(test_operator_views tests/test_operator_views.cu)
target_include_directories(test_operator_views PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_operator_views PRIVATE cumes_test_support cumes_cuda_runtime)
cumes_enable_warnings(test_operator_views)

# ---- ctest registration ----------------------------------------------------
enable_testing()
cumes_add_test(test_fourier          unit;fourier)
cumes_add_test(test_input_json       unit;input)
cumes_add_test(test_host_config      unit;config)
cumes_add_test(test_host_io          unit;io)
cumes_add_test(test_checkpoint       unit;io)
cumes_add_test(test_io_golden        unit;io)
cumes_add_test(test_runtime          unit;runtime)
cumes_add_test(test_controller       unit;solver)
cumes_add_test(test_arena            unit;runtime)
cumes_add_test(test_operator_views   unit;runtime)
cumes_add_test(test_forces           unit;forces)
cumes_add_test(test_output_failure   unit;output)
cumes_add_test(test_geometry_ncurr   unit;geometry)
cumes_add_test(test_constraint_tcon  unit;constraint)
cumes_add_test(test_regression_kernels unit;constraint;precon)
cumes_add_test(test_geometry_iso     unit;geometry)
cumes_add_test(test_force_verify     unit;forces)
cumes_add_test(test_force_reference  unit;forces)
cumes_add_test(test_axisym_backend   unit;fourier)
cumes_add_test(test_rzcon_fusion     unit;fourier;constraint)
cumes_add_test(test_tridiagonal      unit;precon)

# Compute Sanitizer memcheck variants of the kernel-driving tests only
# (test_input_json is host-only and compute-sanitizer refuses to profile it).
cumes_register_sanitizer_variants(
    test_fourier test_forces test_geometry_ncurr test_constraint_tcon
    test_regression_kernels test_runtime test_arena test_operator_views
    test_axisym_backend test_rzcon_fusion test_tridiagonal)

# ---- benchmarks (not part of the default CTest unit gate) --------------------
# The fixed-iteration harness (§8.1). Always double (verify-double is the
# benchmark reference config), regardless of CUMES_USE_FLOAT.
add_executable(cumes_benchmark_fixed_iteration benchmarks/fixed_iteration.cu)
target_include_directories(cumes_benchmark_fixed_iteration PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cumes_benchmark_fixed_iteration PRIVATE cumes_config_json cumes_io_host cumes_cuda_double)
cumes_enable_warnings(cumes_benchmark_fixed_iteration)

# Smoke gate: runs two passes and exits 0 on success (verifies the harness
# builds, seeds, drives a stage, and emits JSON — not a timing acceptance).
add_test(NAME cumes_benchmark_smoke
         COMMAND cumes_benchmark_fixed_iteration --config solovev --passes 2 --warmup 0)
set_tests_properties(cumes_benchmark_smoke PROPERTIES
    LABELS "performance;smoke" TIMEOUT 120
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
