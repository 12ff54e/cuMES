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
set(CUMES_CUDA_MODULES fourier geometry forces solver profiles precon constraint prolongation axisymmetric)
set(CUMES_CUDA_DOUBLE_SOURCES "")
set(CUMES_CUDA_FLOAT_SOURCES "")
foreach(mod IN LISTS CUMES_CUDA_MODULES)
  list(APPEND CUMES_CUDA_DOUBLE_SOURCES src/${mod}_double.cu)
  list(APPEND CUMES_CUDA_FLOAT_SOURCES src/${mod}_float.cu)
endforeach()

# ---- host-only library targets ---------------------------------------------
# Source lists are defined once here and reused verbatim by the sanitizer
# twins below (CUMES_HOST_SANITIZERS): adding a TU to one of these libraries
# must not require editing a second, divergent list (blueprint §10.5).
set(CUMES_JSON_SOURCES src/json_parser.cpp)
set(CUMES_CORE_SOURCES
  src/cumes/core/grid_shape.cpp
  src/cumes/core/mode_table.cpp)
set(CUMES_CONFIG_JSON_SOURCES
  src/cumes/config/validation_report.cpp
  src/cumes/config/validated_problem.cpp
  src/cumes/config/json_reader.cpp)
set(CUMES_IO_HOST_SOURCES
  src/cumes/io/output_spec.cpp
  src/cumes/io/legacy_binary_v0.cpp
  src/cumes/io/versioned_binary.cpp
  src/cumes/io/checkpoint.cpp)

# Define a host-only static library from a source list, with the common
# include-dir/warnings setup. `asan_twin` additionally applies the ASan/UBSan
# flags via cumes_apply_host_sanitizers (a no-op unless CUMES_HOST_SANITIZERS
# is ON) — used by the *_asan twins below.
function(cumes_add_host_library name asan_twin)
  add_library(${name} STATIC ${ARGN})
  target_include_directories(${name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  cumes_enable_warnings(${name})
  if(asan_twin)
    cumes_apply_host_sanitizers(${name})
  endif()
endfunction()

# Host-only ASan test twin: links the _asan library directly (PRIVATE) so the
# sanitized runtime stays FIRST in the link line (never propagated through
# INTERFACE — see CumesSanitizers.cmake), then registers it under CTest label
# `sanitizer;asan;<label>`.
function(cumes_add_host_asan_test name source lib label)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(${name} PRIVATE ${lib})
  cumes_enable_warnings(${name})
  cumes_apply_host_sanitizers(${name})
  cumes_add_test(${name} "sanitizer;asan;${label}")
endfunction()

# JsonParser implementation (header-only library, compiled once), shared by the
# host-model reader.
cumes_add_host_library(cumes_json FALSE ${CUMES_JSON_SOURCES})

# ---- Phase 2: validated host model (namespace cumes) -----------------------
# Core host model: result/status, checked arithmetic, grid shape, mode table.
# Header-mostly; the two non-trivial TU implementations live in src/cumes/core.
cumes_add_host_library(cumes_core FALSE ${CUMES_CORE_SOURCES})

# Config model + JSON reader: dynamic ProblemSpec -> ValidatedProblem, the
# legacy InputParams bridge, and the normalization emitter. Host-only C++.
cumes_add_host_library(cumes_config_json FALSE ${CUMES_CONFIG_JSON_SOURCES})
target_link_libraries(cumes_config_json PUBLIC cumes_core cumes_json)

# Versioned + legacy host I/O: output spec, run report, snapshot, binary v0
# (byte-exact) and v1 writers/readers, and the versioned checkpoint + legacy
# init converter. Host-only C++ (no CUDA); NetCDF/HDF5 host adapters are
# deferred to Phase 3.
cumes_add_host_library(cumes_io_host FALSE ${CUMES_IO_HOST_SOURCES})
target_link_libraries(cumes_io_host PUBLIC cumes_core)

# Output writers: binary dispatcher + optional NetCDF/HDF5 backends (host-only
# C++, using only the CUDA runtime API for D2H copies — no device code).
add_library(cumes_io STATIC src/output.cpp)
target_include_directories(cumes_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cumes_io PUBLIC CUDA::cudart CUDA::cufft)
cumes_enable_warnings(cumes_io)
cumes_link_output_backends(cumes_io)

# cumes_io_host compiles output_spec.cpp SEPARATELY from the writer TUs, and
# its output_format_available() preflight gates the .nc/.h5 dispatcher on these
# defines. cumes_link_output_backends() above gives them to cumes_io only — a
# stock build with the backends found must not silently report them
# unavailable, so mirror the availability defines onto cumes_io_host (and its
# sanitizer twin, same sources).
function(cumes_apply_backend_avail_defines target)
  if(CUMES_HAVE_NETCDF)
    target_compile_definitions(${target} PUBLIC CUMES_HAVE_NETCDF)
  endif()
  if(CUMES_HAVE_HDF5)
    target_compile_definitions(${target} PUBLIC CUMES_HAVE_HDF5)
  endif()
endfunction()
cumes_apply_backend_avail_defines(cumes_io_host)

# Header-only shared test helpers (checkCuda/cc). Tests include
# cumes_test_support.cuh; this target only exposes its include dir.
add_library(cumes_test_support INTERFACE)
target_include_directories(cumes_test_support INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/tests/support)

# ---- CUDA runtime RAII (host-only CUDA runtime API) ------------------------
# DeviceBuffer/PinnedBuffer/Stream/Event and the centralized check_cuda/
# check_cufft error boundary — header-only since the DeviceContext removal
# (2026-08-17), so the target carries no sources and is INTERFACE (propagates
# the include dir + cudart/cufft links). No device code.
add_library(cumes_cuda_runtime INTERFACE)
target_include_directories(cumes_cuda_runtime INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cumes_cuda_runtime INTERFACE CUDA::cudart CUDA::cufft)

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
target_link_libraries(test_input_json PRIVATE cumes_config_json)
cumes_enable_warnings(test_input_json)

foreach(t IN ITEMS test_forces test_force_verify test_geometry_iso
                   test_geometry_ncurr test_regression_kernels test_constraint_tcon
                   test_force_reference test_axisym_backend
                   test_tridiagonal test_cuda_graph test_safety_predicates)
  add_executable(${t} tests/${t}.cu)
  target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(${t} PRIVATE cumes_test_support cumes_config_json cumes_cuda_double cumes_cuda_float)
  cumes_enable_warnings(${t})
endforeach()

add_executable(test_output_failure tests/test_output_failure.cu)
target_include_directories(test_output_failure PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_output_failure PRIVATE cumes_test_support cumes_config_json cumes_io)
cumes_enable_warnings(test_output_failure)

add_executable(test_host_config tests/test_host_config.cpp)
target_include_directories(test_host_config PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_host_config PRIVATE cumes_config_json)
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

add_executable(test_accumulation tests/test_accumulation.cu)
target_include_directories(test_accumulation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_accumulation PRIVATE cumes_test_support cumes_cuda_runtime)
cumes_enable_warnings(test_accumulation)

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

# ---- host sanitizer twins (blueprint §10.5; CUMES_HOST_SANITIZERS=ON) -------
# Dedicated ASan+UBSan copies of the host-only libraries and their tests, so
# the sanitized runtime is FIRST in each asan executable's library list and
# never leaks into the CUDA targets (whose link order would break it). The
# normal targets above are untouched. Source lists are shared with the normal
# targets (CUMES_*_SOURCES above); cumes_add_host_library/cumes_add_host_asan_test
# apply the identical include-dir/warning setup plus the sanitizer flags.
if(CUMES_HOST_SANITIZERS)
  cumes_add_host_library(cumes_json_asan TRUE ${CUMES_JSON_SOURCES})

  cumes_add_host_library(cumes_core_asan TRUE ${CUMES_CORE_SOURCES})

  cumes_add_host_library(cumes_config_json_asan TRUE ${CUMES_CONFIG_JSON_SOURCES})
  target_link_libraries(cumes_config_json_asan PUBLIC cumes_core_asan cumes_json_asan)

  cumes_add_host_library(cumes_io_host_asan TRUE ${CUMES_IO_HOST_SOURCES})
  target_link_libraries(cumes_io_host_asan PUBLIC cumes_core_asan)
  cumes_apply_backend_avail_defines(cumes_io_host_asan)

  cumes_add_host_asan_test(asan_test_host_config tests/test_host_config.cpp
                           cumes_config_json_asan config)
  cumes_add_host_asan_test(asan_test_host_io tests/test_host_io.cpp
                           cumes_io_host_asan io)
  cumes_add_host_asan_test(asan_test_checkpoint tests/test_checkpoint.cpp
                           cumes_io_host_asan io)
  cumes_add_host_asan_test(asan_test_controller tests/test_controller.cpp
                           cumes_core_asan solver)
endif()
cumes_add_test(test_arena            unit;runtime)
cumes_add_test(test_operator_views   unit;runtime)
cumes_add_test(test_accumulation     unit;numerics)
cumes_add_test(test_forces           unit;forces)
cumes_add_test(test_output_failure   unit;output)
cumes_add_test(test_geometry_ncurr   unit;geometry)
cumes_add_test(test_constraint_tcon  unit;constraint)
cumes_add_test(test_regression_kernels unit;constraint;precon)
cumes_add_test(test_safety_predicates unit;safety)
cumes_add_test(test_geometry_iso     unit;geometry)
cumes_add_test(test_force_verify     unit;forces)
cumes_add_test(test_force_reference  unit;forces)
cumes_add_test(test_axisym_backend   unit;fourier)
cumes_add_test(test_tridiagonal      unit;precon)
cumes_add_test(test_cuda_graph       unit;runtime)

# Compute Sanitizer memcheck variants of the kernel-driving tests only
# (test_input_json is host-only and compute-sanitizer refuses to profile it).
cumes_register_sanitizer_variants(
    test_fourier test_forces test_geometry_ncurr test_constraint_tcon
    test_regression_kernels test_runtime test_arena test_operator_views
    test_accumulation test_axisym_backend test_tridiagonal
    test_cuda_graph test_safety_predicates)

# ---- benchmarks (not part of the default CTest unit gate) --------------------
# The fixed-iteration harness (§8.1). Always double (verify-double is the
# benchmark reference config), regardless of CUMES_USE_FLOAT.
add_executable(cumes_benchmark_fixed_iteration benchmarks/fixed_iteration.cu)
target_include_directories(cumes_benchmark_fixed_iteration PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cumes_benchmark_fixed_iteration PRIVATE cumes_config_json cumes_io_host cumes_cuda_double)
cumes_enable_warnings(cumes_benchmark_fixed_iteration)

add_executable(cumes_benchmark_graph_overhead benchmarks/graph_overhead.cu)
target_include_directories(cumes_benchmark_graph_overhead PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/tests/support)
target_link_libraries(cumes_benchmark_graph_overhead PRIVATE cumes_cuda_runtime)
cumes_enable_warnings(cumes_benchmark_graph_overhead)

# ADR-0003 re-measure: CUDA Graph capture of the REAL per-iteration DAG
# (builds the production operator stack and captures EquilibriumOperator::
# enqueue; the empty-kernel overhead benchmark's upper bound is replaced by
# real-pass submission numbers). Always double, like the fixed-iteration
# harness (verify-double is the benchmark reference config).
add_executable(cumes_benchmark_graph_realpass benchmarks/graph_realpass.cu)
target_include_directories(cumes_benchmark_graph_realpass PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cumes_benchmark_graph_realpass PRIVATE cumes_config_json cumes_io_host cumes_cuda_double)
cumes_enable_warnings(cumes_benchmark_graph_realpass)

# Smoke gate: runs two passes and exits 0 on success (verifies the harness
# builds, seeds, drives a stage, and emits JSON — not a timing acceptance).
add_test(NAME cumes_benchmark_smoke
         COMMAND cumes_benchmark_fixed_iteration --config solovev --passes 2 --warmup 0)
set_tests_properties(cumes_benchmark_smoke PROPERTIES
    LABELS "performance;smoke" TIMEOUT 120
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
