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

# ---- language standards -----------------------------------------------------
# C++20 for host C++ and CUDA TUs alike: the JSON parser's requires-clauses
# and concepts need it, and the operator, I/O, and test code is written
# against it. Stated explicitly here so a toolchain default can never
# silently regress the standard; GNU extensions are off (strict -std=c++20,
# never -std=gnu++20).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# ---- precision policy -> flags (completion plan step 3.1) -------------------
# No global --use_fast_math: fast intrinsics appear only under the named
# fast-double policy; the verification configuration (verify-double) is
# precise double math. The policy and its effective flags are recorded in the
# build provenance of every v1 output (CUMES_PRECISION_POLICY_NAME /
# CUMES_PRECISION_FLAGS compile definitions).
# The flag sets are LISTS (one argv token per element): target_compile_options
# passes each element separately, unlike the old global CMAKE_CUDA_FLAGS string
# which CMake split on whitespace.
set(CUMES_FAST_MATH_FLAGS)
set(CUMES_DEVICE_CHECK_FLAGS)
if(CUMES_PRECISION_POLICY STREQUAL "fast-double")
  set(CUMES_FAST_MATH_FLAGS "--use_fast_math")
elseif(CUMES_PRECISION_POLICY STREQUAL "mixed-float")
  set(CUMES_USE_FLOAT ON CACHE BOOL "implied by CUMES_PRECISION_POLICY=mixed-float" FORCE)
elseif(CUMES_PRECISION_POLICY STREQUAL "debug-double")
  set(CUMES_DEVICE_CHECK_FLAGS "-G" "-lineinfo")
elseif(NOT CUMES_PRECISION_POLICY STREQUAL "verify-double")
  message(FATAL_ERROR "unknown CUMES_PRECISION_POLICY '${CUMES_PRECISION_POLICY}' (expected verify-double|fast-double|mixed-float|debug-double)")
endif()
string(JOIN " " CUMES_PRECISION_FLAGS_STR ${CUMES_FAST_MATH_FLAGS}
                                  ${CUMES_DEVICE_CHECK_FLAGS})
# A policy/path disagreement is a configure error, not a silent rebuild.
if(CUMES_PRECISION_POLICY STREQUAL "mixed-float" AND NOT CUMES_USE_FLOAT)
  message(FATAL_ERROR "CUMES_PRECISION_POLICY=mixed-float requires CUMES_USE_FLOAT=ON")
endif()
if(CUMES_USE_FLOAT AND NOT CUMES_PRECISION_POLICY STREQUAL "mixed-float")
  message(FATAL_ERROR "CUMES_USE_FLOAT=ON requires CUMES_PRECISION_POLICY=mixed-float")
endif()

# Policy flags are TARGET-SCOPED (completion-plan follow-up §3.4): no global
# CMAKE_CUDA_FLAGS / CMAKE_CXX_FLAGS appends. Only the solver targets (the
# CUDA operator libraries, the CLI, and the two real-DAG benchmarks) receive
# the named policy's device flags; the parser, config, I/O, test, and adapter
# targets get neither. Optimization levels come from CMAKE_BUILD_TYPE
# (Release = -O3), which also removes the sanitizer configuration's -O2/-O3
# redefinition warning.
function(cumes_apply_policy_flags target)
  # DELIBERATELY no host tuning here: the frozen Class A baseline never
  # passed CMAKE_CXX_FLAGS into nvcc's host pass (CMake does not forward it),
  # so the solver TUs' host code must keep compiling WITHOUT -march=native.
  # Adding it enables FMA contraction in the controller host math and
  # diverges the trajectory bitwise (measured on the Solovev baseline).
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUMES_FAST_MATH_FLAGS} ${CUMES_DEVICE_CHECK_FLAGS}>)
endfunction()

# Host tuning for the production HOST-only libraries (previously the global
# CMAKE_CXX_FLAGS -march=native). Applied explicitly — never through
# cumes_add_host_library — so the ASan twins and tests stay free of it.
function(cumes_apply_host_tuning target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endfunction()

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
set(CUMES_JSON_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/json_parser.cpp)
set(CUMES_CORE_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cumes/core/grid_shape.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cumes/core/mode_table.cpp)
set(CUMES_CONFIG_JSON_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cumes/config/validation_report.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cumes/config/validated_problem.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cumes/config/json_reader.cpp)
set(CUMES_IO_HOST_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cumes/io/output_spec.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cumes/io/versioned_binary.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cumes/io/checkpoint.cpp)

# Define a host-only static library from a source list, with the common
# include-dir/warnings setup. `asan_twin` additionally applies the ASan/UBSan
# flags via cumes_apply_host_sanitizers (a no-op unless CUMES_HOST_SANITIZERS
# is ON) — used by the *_asan twins below.
function(cumes_add_host_library name asan_twin)
  add_library(${name} STATIC ${ARGN})
  target_include_directories(${name} PUBLIC ${PROJECT_SOURCE_DIR}/include)
  cumes_enable_warnings(${name})
  if(asan_twin)
    cumes_apply_host_sanitizers(${name})
  endif()
endfunction()

# JsonParser implementation (header-only library, compiled once), shared by the
# host-model reader.
cumes_add_host_library(cumes_json FALSE ${CUMES_JSON_SOURCES})
cumes_apply_host_tuning(cumes_json)

# ---- Phase 2: validated host model (namespace cumes) -----------------------
# Core host model: result/status, checked arithmetic, grid shape, mode table.
# Header-mostly; the two non-trivial TU implementations live in src/cumes/core.
cumes_add_host_library(cumes_core FALSE ${CUMES_CORE_SOURCES})
cumes_apply_host_tuning(cumes_core)

# Config model + JSON reader: dynamic ProblemSpec -> ValidatedProblem, the
# legacy InputParams bridge, and the normalization emitter. Host-only C++.
cumes_add_host_library(cumes_config_json FALSE ${CUMES_CONFIG_JSON_SOURCES})
target_link_libraries(cumes_config_json PUBLIC cumes_core cumes_json)
cumes_apply_host_tuning(cumes_config_json)

# Versioned host I/O: output spec, run report, snapshot, the v1 binary
# writer/reader, and the versioned checkpoint. Host-only C++ (no CUDA);
# NetCDF/HDF5 host adapters are deferred to Phase 3.
cumes_add_host_library(cumes_io_host FALSE ${CUMES_IO_HOST_SOURCES})
target_link_libraries(cumes_io_host PUBLIC cumes_core)
cumes_apply_host_tuning(cumes_io_host)

# Output writers: binary dispatcher + optional NetCDF/HDF5 backends (host-only
# C++, using only the CUDA runtime API for the output printout's D2H copy — no
# device code).
# The adapter library: the console output printout + format availability
# preflight (output.cpp), the full Writer/Reader dispatch, and the optional
# NetCDF/HDF5 host adapters (added by cumes_link_output_backends below under
# their availability defines). Links cumes_io_host for the binary factories
# (completion plan step 2.5: backend headers and defines stay inside this
# target).
add_library(cumes_io STATIC src/output.cpp src/cumes/io/writer_dispatch.cpp)
target_include_directories(cumes_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cumes_io PUBLIC CUDA::cudart CUDA::cufft cumes_io_host)
cumes_enable_warnings(cumes_io)
cumes_apply_host_tuning(cumes_io)
cumes_link_output_backends(cumes_io)

# The CUMES_HAVE_NETCDF/HDF5 availability defines are confined to the adapter
# library (cumes_io, via cumes_link_output_backends above) — completion plan
# step 2.5. The output_format_available() preflight lives in the adapter
# library (src/output.cpp), so the host I/O library needs no backend defines
# and no NetCDF/HDF5 headers.

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
cumes_apply_policy_flags(cumes_cuda_double)
if(CUMES_ENABLE_VERIFY_DUMP)
  target_compile_definitions(cumes_cuda_double PRIVATE DUMP_CUMES_VERIFY)
endif()

add_library(cumes_cuda_float STATIC ${CUMES_CUDA_FLOAT_SOURCES})
target_include_directories(cumes_cuda_float PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cumes_cuda_float PUBLIC CUDA::cufft cumes_cuda_runtime)
cumes_enable_warnings(cumes_cuda_float)
cumes_apply_policy_flags(cumes_cuda_float)
if(CUMES_ENABLE_VERIFY_DUMP)
  target_compile_definitions(cumes_cuda_float PRIVATE DUMP_CUMES_VERIFY)
endif()

# ---- CLI executable --------------------------------------------------------
if(CUMES_USE_FLOAT)
  set(CUMES_SELECTED_CUDA cumes_cuda_float)
else()
  set(CUMES_SELECTED_CUDA cumes_cuda_double)
endif()

add_executable(cuMES src/main.cu)
target_include_directories(cuMES PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# Link order matters: cumes_io_host BEFORE cumes_io so the adapter library
# is scanned while the backend-factory references are still undefined — the
# linker then extracts the strong NetCDF/HDF5 factory definitions (completion
# plan step 2.5).
target_link_libraries(cuMES PRIVATE cumes_config_json cumes_io_host cumes_io ${CUMES_SELECTED_CUDA})
cumes_enable_warnings(cuMES)
cumes_apply_policy_flags(cuMES)
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

# Precision-policy provenance (completion plan step 3.1): the named policy and
# its effective flags (space-joined) travel with the binary into every v1 output.
function(cumes_add_precision_provenance target)
  target_compile_definitions(${target} PRIVATE
    CUMES_PRECISION_POLICY_NAME="${CUMES_PRECISION_POLICY}"
    CUMES_PRECISION_FLAGS="${CUMES_PRECISION_FLAGS_STR}")
endfunction()
cumes_add_precision_provenance(cuMES)

# ---- tests (standalone executables, no framework) --------------------------
enable_testing()
add_subdirectory(tests)

# ---- benchmarks (not part of the default CTest unit gate) --------------------
# The fixed-iteration harness (§8.1). Always double (verify-double is the
# benchmark reference config), regardless of CUMES_USE_FLOAT.
add_executable(cumes_benchmark_fixed_iteration benchmarks/fixed_iteration.cu)
target_include_directories(cumes_benchmark_fixed_iteration PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cumes_benchmark_fixed_iteration PRIVATE cumes_config_json cumes_io_host cumes_cuda_double)
cumes_enable_warnings(cumes_benchmark_fixed_iteration)
cumes_apply_policy_flags(cumes_benchmark_fixed_iteration)
cumes_add_precision_provenance(cumes_benchmark_fixed_iteration)

add_executable(cumes_benchmark_graph_overhead benchmarks/graph_overhead.cu)
target_include_directories(cumes_benchmark_graph_overhead PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
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
cumes_apply_policy_flags(cumes_benchmark_graph_realpass)
cumes_add_precision_provenance(cumes_benchmark_graph_realpass)

# CLI policy gate (completion plan step 2.1): strict-vs-compatibility schema
# and output-path/suffix policy, exercised end to end against the real binary
# with one-iteration tiny configs (validation and preflight failures exit
# before any meaningful GPU work).
add_test(NAME cli_policy
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_policy_test.sh
                 $<TARGET_FILE:cuMES>)
set_tests_properties(cli_policy PROPERTIES
    LABELS "unit;cli" TIMEOUT 300
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# End-to-end durable-publication contract (completion-plan follow-up §3): a
# converged run whose output cannot be published must exit nonzero and must
# never damage an existing destination. The float build skips the converged
# cases (its double-tuned ftols are unreachable; the protocol is
# precision-independent and covered by test_output_failure at both
# precisions).
add_test(NAME output_publication
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/output_publication_test.sh
                 $<TARGET_FILE:cuMES>
                 ${CMAKE_CURRENT_SOURCE_DIR}/inputs/solovev.json)
set_tests_properties(output_publication PROPERTIES
    LABELS "unit;output" TIMEOUT 600
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Smoke gate: runs two passes and exits 0 on success (verifies the harness
# builds, seeds, drives a stage, and emits JSON — not a timing acceptance).
add_test(NAME cumes_benchmark_smoke
         COMMAND cumes_benchmark_fixed_iteration --config solovev --passes 2 --warmup 0)
set_tests_properties(cumes_benchmark_smoke PROPERTIES
    LABELS "performance;smoke" TIMEOUT 120
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
