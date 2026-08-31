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

project(cuMES VERSION 1.2.0 LANGUAGES CXX CUDA)

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

# ---- precision policy -> flags --------------------------------------------
# No global --use_fast_math: fast intrinsics appear only under the named
# fast-double policy; the verification configuration (verify-double) is
# precise double math. The policy and its effective flags are recorded in the
# build provenance of every v1 output (CUMES_PRECISION_POLICY_NAME /
# CUMES_PRECISION_FLAGS compile definitions).
# The flag sets are LISTS (one argv token per element): target_compile_options
# passes each element separately.
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

# Policy flags are TARGET-SCOPED: no global
# CMAKE_CUDA_FLAGS / CMAKE_CXX_FLAGS appends. Only the solver targets (the
# CUDA operator libraries, the CLI, and the two real-DAG benchmarks) receive
# the named policy's device flags; the parser, config, I/O, test, and adapter
# targets get neither. Optimization levels come from CMAKE_BUILD_TYPE
# (Release = -O3), which also removes the sanitizer configuration's -O2/-O3
# redefinition warning.
function(cumes_apply_policy_flags target)
  # DELIBERATELY no host tuning here: the solver TUs' host code must compile
  # WITHOUT -march=native. Adding it enables FMA contraction in the
  # controller host math and diverges the trajectory bitwise from the
  # Class A baseline (measured on Solovev).
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUMES_FAST_MATH_FLAGS} ${CUMES_DEVICE_CHECK_FLAGS}>)
endfunction()

# Host tuning (-march=native) for the production HOST-only libraries.
# Applied explicitly — never through cumes_add_host_library — so the ASan
# twins and tests stay free of it.
function(cumes_apply_host_tuning target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endfunction()

# ---- free-boundary vacuum library (deps/vacuum-field) ----------------------
# See docs/adr/0005-vacuum-field-submodule.md. Auto-disabled when the submodule
# is not checked out (hosted CI builds without submodules); the module list
# below picks the real free_boundary module or the stub accordingly.
option(CUMES_USE_VACUUM_FIELD
       "Enable free-boundary support via optional deps/vacuum-field" ON)
if(CUMES_USE_VACUUM_FIELD AND NOT EXISTS
   ${CMAKE_CURRENT_SOURCE_DIR}/deps/vacuum-field/CMakeLists.txt)
  message(WARNING "cuMES: deps/vacuum-field not checked out "
                  "(run: git submodule update --init); "
                  "vacuum-field support disabled")
  set(CUMES_USE_VACUUM_FIELD OFF)
endif()
if(CUMES_USE_VACUUM_FIELD)
  message(STATUS "cuMES: optional vacuum-field support enabled")
else()
  message(STATUS "cuMES: optional vacuum-field support disabled (fixed boundary only)")
endif()

# ---- device modules --------------------------------------------------------
# The templated operator modules, each split into a *_double.cu and a
# *_float.cu explicit-instantiation TU (one scalar type per TU). The CLI links
# only the TU matching `Real`; the tests link both (they instantiate both).
set(CUMES_CUDA_MODULES fourier geometry forces solver profiles precon constraint prolongation axisymmetric)
if(CUMES_USE_VACUUM_FIELD)
  list(APPEND CUMES_CUDA_MODULES free_boundary)
else()
  list(APPEND CUMES_CUDA_MODULES free_boundary_stub)
endif()
set(CUMES_CUDA_DOUBLE_SOURCES "")
set(CUMES_CUDA_FLOAT_SOURCES "")
foreach(mod IN LISTS CUMES_CUDA_MODULES)
  list(APPEND CUMES_CUDA_DOUBLE_SOURCES src/${mod}_double.cu)
  list(APPEND CUMES_CUDA_FLOAT_SOURCES src/${mod}_float.cu)
endforeach()

# ---- host-only library targets ---------------------------------------------
# Source lists are defined once here and reused verbatim by the sanitizer
# twins below (CUMES_HOST_SANITIZERS): adding a TU to one of these libraries
# must not require editing a second, divergent list.
set(CUMES_JSON_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/json_parser.cpp)
set(CUMES_CORE_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cumes/core/grid_shape.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cumes/core/mode_table.cpp)
set(CUMES_CONFIG_JSON_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cumes/config/validation_report.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cumes/config/validated_problem.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cumes/config/json_reader.cpp)
set(CUMES_IO_HOST_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cumes/io/derived_fields.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cumes/io/output_spec.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cumes/io/versioned_binary.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cumes/io/checkpoint.cpp)

# Define a host-only static library from a source list, with the common
# include-dir/warnings setup. `asan_twin` additionally applies the ASan/UBSan
# flags via cumes_apply_host_sanitizers (a no-op unless CUMES_HOST_SANITIZERS
# is ON) — used by the *_asan twins below.
function(cumes_add_host_library name asan_twin)
  add_library(${name} STATIC ${ARGN})
  target_include_directories(${name} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
  cumes_enable_warnings(${name})
  if(asan_twin)
    cumes_apply_host_sanitizers(${name})
  endif()
endfunction()

# JsonParser implementation (header-only library, compiled once), shared by the
# host-model reader.
cumes_add_host_library(cumes_json FALSE ${CUMES_JSON_SOURCES})
cumes_apply_host_tuning(cumes_json)

# ---- validated host model (namespace cumes) --------------------------------
# Core host model: result/status, checked arithmetic, grid shape, mode table.
# Header-mostly; the two non-trivial TU implementations live in src/cumes/core.
cumes_add_host_library(cumes_core FALSE ${CUMES_CORE_SOURCES})
cumes_apply_host_tuning(cumes_core)
add_library(cumes::core ALIAS cumes_core)

# Config model + JSON reader: dynamic ProblemSpec -> ValidatedProblem, the
# legacy InputParams bridge, and the normalization emitter. Host-only C++.
cumes_add_host_library(cumes_config_json FALSE ${CUMES_CONFIG_JSON_SOURCES})
target_link_libraries(cumes_config_json PUBLIC cumes_core cumes_json)
if(NOT CUMES_USE_VACUUM_FIELD)
  target_compile_definitions(cumes_config_json
    PUBLIC CUMES_VACUUM_FIELD_DISABLED)
endif()
cumes_apply_host_tuning(cumes_config_json)
add_library(cumes::config_json ALIAS cumes_config_json)

# Versioned host I/O: output spec, run report, snapshot, the v1 binary
# writer/reader, and the versioned checkpoint. Host-only C++ (no CUDA); the
# optional NetCDF/HDF5 writers live on the cumes_io adapter library below.
cumes_add_host_library(cumes_io_host FALSE ${CUMES_IO_HOST_SOURCES})
target_link_libraries(cumes_io_host PUBLIC cumes_core)
cumes_apply_host_tuning(cumes_io_host)

# Output writers: binary dispatcher + optional NetCDF/HDF5 backends. Console
# output also consumes the facade's host snapshot, so this target is entirely
# host-only and contains no CUDA state transfer.
# The adapter library: the console output printout + format availability
# preflight (output.cpp), the full Writer/Reader dispatch, and the optional
# NetCDF/HDF5 host adapters (added by cumes_link_output_backends below under
# their availability defines). Links cumes_io_host for the binary factories;
# the backend headers and defines stay inside this target.
add_library(cumes_io STATIC src/output.cpp src/cumes/io/writer_dispatch.cpp)
target_include_directories(cumes_io PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(cumes_io PUBLIC cumes_io_host)
cumes_enable_warnings(cumes_io)
cumes_apply_host_tuning(cumes_io)
cumes_link_output_backends(cumes_io)
add_library(cumes::io ALIAS cumes_io)

# The CUMES_HAVE_NETCDF/HDF5 availability defines are confined to the adapter
# library (cumes_io, via cumes_link_output_backends above). The
# output_format_available() preflight lives in the adapter library
# (src/output.cpp), so the host I/O library needs no backend defines and no
# NetCDF/HDF5 headers.

# ---- CUDA runtime RAII (host-only CUDA runtime API) ------------------------
# DeviceBuffer/PinnedBuffer/Stream/Event and the centralized check_cuda/
# check_cufft error boundary — header-only, so the target carries no sources
# and is INTERFACE (propagates the include dir + cudart/cufft links). No
# device code.
add_library(cumes_cuda_runtime INTERFACE)
target_include_directories(cumes_cuda_runtime INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(cumes_cuda_runtime INTERFACE CUDA::cudart CUDA::cufft)

# Reuse the direct header-only BSplineInterpolation submodule to construct a
# small radial transfer matrix on the host. The spectral state remains
# device-resident; the matrix is uploaded and applied in one batched CUDA
# kernel. Submodule-free builds retain the Catmull-Rom fallback. Do not call
# the dependency's add_subdirectory(): it enables standalone tests and changes
# its directory-wide language standard, neither of which cuMES needs.
set(CUMES_HAVE_BSPLINE_PROLONGATION OFF)
if(CUMES_USE_BSPLINE_PROLONGATION AND EXISTS
   ${CMAKE_CURRENT_SOURCE_DIR}/deps/BSplineInterpolation/include/bsintp/Interpolation.hpp)
  cumes_add_host_library(cumes_bspline_prolongation FALSE
    src/cumes/numerics/bspline_matrix.cpp)
  target_include_directories(cumes_bspline_prolongation SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/deps/BSplineInterpolation/include)
  target_compile_definitions(cumes_bspline_prolongation PRIVATE
    INTP_CELL_LAYOUT)
  # GCC 12/13 reports a false positive while inlining the dependency's
  # std::variant-held band matrix; confine the suppression to this adapter.
  target_compile_options(cumes_bspline_prolongation PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wno-maybe-uninitialized>)
  cumes_apply_host_tuning(cumes_bspline_prolongation)
  set(CUMES_HAVE_BSPLINE_PROLONGATION ON)
  message(STATUS "cuMES: B-spline multigrid transfer enabled")
elseif(CUMES_USE_BSPLINE_PROLONGATION)
  message(STATUS "cuMES: B-spline multigrid transfer unavailable; using Catmull-Rom")
else()
  message(STATUS "cuMES: B-spline multigrid transfer disabled; using Catmull-Rom")
endif()

# ---- CUDA operator libraries (one scalar type each) ------------------------

add_library(cumes_cuda_double STATIC ${CUMES_CUDA_DOUBLE_SOURCES})
target_include_directories(cumes_cuda_double PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(cumes_cuda_double PUBLIC CUDA::cufft cumes_cuda_runtime)
if(CUMES_HAVE_BSPLINE_PROLONGATION)
  target_link_libraries(cumes_cuda_double PUBLIC cumes_bspline_prolongation)
  target_compile_definitions(cumes_cuda_double PUBLIC
    CUMES_HAVE_BSPLINE_PROLONGATION)
endif()
cumes_enable_warnings(cumes_cuda_double)
cumes_apply_policy_flags(cumes_cuda_double)
if(CUMES_ENABLE_VERIFY_DUMP)
  target_compile_definitions(cumes_cuda_double PRIVATE DUMP_CUMES_VERIFY)
endif()

add_library(cumes_cuda_float STATIC ${CUMES_CUDA_FLOAT_SOURCES})
target_include_directories(cumes_cuda_float PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(cumes_cuda_float PUBLIC CUDA::cufft cumes_cuda_runtime)
if(CUMES_HAVE_BSPLINE_PROLONGATION)
  target_link_libraries(cumes_cuda_float PUBLIC cumes_bspline_prolongation)
  target_compile_definitions(cumes_cuda_float PUBLIC
    CUMES_HAVE_BSPLINE_PROLONGATION)
endif()
cumes_enable_warnings(cumes_cuda_float)
cumes_apply_policy_flags(cumes_cuda_float)
if(CUMES_ENABLE_VERIFY_DUMP)
  target_compile_definitions(cumes_cuda_float PRIVATE DUMP_CUMES_VERIFY)
endif()

# ---- free-boundary vacuum library (deps/vacuum-field) ----------------------
# The library target vfield::vfield exists only when the submodule is present;
# the free_boundary module TUs include its headers, so the operator libraries
# link it (PUBLIC — tests linking both operator libraries get it transitively).
if(CUMES_USE_VACUUM_FIELD)
  # Propagate the scalar switch: each scalar library picks the matching
  # instantiation. Unless explicitly overridden, propagate the top-level
  # NetCDF availability too: this avoids a duplicate dependency probe/warning
  # and keeps a plain `cmake -B build` usable when NetCDF is absent.
  set(VFIELD_USE_FLOAT ${CUMES_USE_FLOAT} CACHE BOOL
      "vacuum-field scalar switch (propagated from CUMES_USE_FLOAT)" FORCE)
  if(NOT DEFINED VFIELD_USE_NETCDF)
    set(VFIELD_USE_NETCDF ${CUMES_HAVE_NETCDF})
  endif()
  set(VFIELD_BUILD_MAKEGRID ON CACHE BOOL
      "build vacuum-field Makegrid support for cuMES" FORCE)
  add_subdirectory(deps/vacuum-field)
  # vacuum-field is also a standalone project and therefore exposes source
  # include paths directly. Replace only its exported interface here so the
  # cuMES install tree is relocatable without modifying the submodule.
  foreach(vfield_target IN ITEMS vfield_host vfield_makegrid
                                 vfield_cuda_double vfield_cuda_float)
    set_property(TARGET ${vfield_target} PROPERTY
      INTERFACE_INCLUDE_DIRECTORIES
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/deps/vacuum-field/include>;$<INSTALL_INTERFACE:include>")
  endforeach()
  target_link_libraries(cumes_cuda_double PUBLIC
    vfield_cuda_double vfield::makegrid)
  target_link_libraries(cumes_cuda_float PUBLIC
    vfield_cuda_float vfield::makegrid)
endif()

# ---- magnetic-coordinate library and postprocessor -------------------------
# A normal build provides both the standalone cumes-boozer converter and an
# in-process library path from the converged cuMES snapshot.
if(CUMES_BUILD_MAGNETIC_COORDINATE AND NOT EXISTS
   ${CMAKE_CURRENT_SOURCE_DIR}/deps/magnetic-coordinate/CMakeLists.txt)
  message(WARNING "cuMES: deps/magnetic-coordinate not checked out "
                  "(run: git submodule update --init); "
                  "Boozer post-processing disabled")
  set(CUMES_BUILD_MAGNETIC_COORDINATE OFF)
endif()
if(CUMES_BUILD_MAGNETIC_COORDINATE)
  message(STATUS "cuMES: magnetic-coordinate post-processing enabled")
  set(MAGNETIC_COORDINATE_BUILD_TESTING OFF CACHE BOOL
      "standalone magnetic-coordinate tests are not duplicated in cuMES"
      FORCE)
  add_subdirectory(deps/magnetic-coordinate)
else()
  message(STATUS "cuMES: magnetic-coordinate post-processing disabled")
endif()

# ---- public in-process solver facade --------------------------------------
if(CUMES_USE_FLOAT)
  set(CUMES_SELECTED_CUDA cumes_cuda_float)
else()
  set(CUMES_SELECTED_CUDA cumes_cuda_double)
endif()

add_library(cumes_solver STATIC src/cumes/solver/equilibrium_solver.cu)
target_include_directories(cumes_solver PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(cumes_solver PUBLIC
  cumes_config_json cumes_io_host ${CUMES_SELECTED_CUDA} Threads::Threads)
if(CUMES_USE_VACUUM_FIELD)
  target_link_libraries(cumes_solver PUBLIC vfield::vfield)
endif()
cumes_enable_warnings(cumes_solver)
cumes_apply_policy_flags(cumes_solver)
if(CUMES_USE_FLOAT)
  target_compile_definitions(cumes_solver PUBLIC CUMES_USE_FLOAT)
endif()
add_library(cumes::solver ALIAS cumes_solver)
set_target_properties(cumes_core PROPERTIES EXPORT_NAME core)
set_target_properties(cumes_config_json PROPERTIES EXPORT_NAME config_json)
set_target_properties(cumes_io PROPERTIES EXPORT_NAME io)
set_target_properties(cumes_solver PROPERTIES EXPORT_NAME solver)

# Optional integration application: dependency direction stays at the app
# layer (example -> cuMES + meow); neither reusable library knows the other.
if(CUMES_MEOW_SOURCE_DIR)
  if(NOT EXISTS ${CUMES_MEOW_SOURCE_DIR}/CMakeLists.txt)
    message(FATAL_ERROR
      "CUMES_MEOW_SOURCE_DIR does not contain a CMake project: "
      "${CUMES_MEOW_SOURCE_DIR}")
  endif()
  set(MEOW_BUILD_TESTS OFF CACHE BOOL
      "do not duplicate meow tests in the cuMES integration build" FORCE)
  set(MEOW_BUILD_CUMES_INTEGRATION ON CACHE BOOL
      "build the meow-owned cuMES integration applications" FORCE)
  add_subdirectory(${CUMES_MEOW_SOURCE_DIR}
                   ${CMAKE_CURRENT_BINARY_DIR}/meow EXCLUDE_FROM_ALL)
endif()

# Precision-policy provenance: the named policy and its effective flags
# (space-joined) travel with the binary into every v1 output.
function(cumes_add_precision_provenance target)
  target_compile_definitions(${target} PRIVATE
    CUMES_PRECISION_POLICY_NAME="${CUMES_PRECISION_POLICY}"
    CUMES_PRECISION_FLAGS="${CUMES_PRECISION_FLAGS_STR}")
endfunction()

# ---- CLI executable --------------------------------------------------------
if(CUMES_BUILD_CLI)
  add_executable(cumes src/main.cu)
  target_include_directories(cumes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # Link order matters: cumes_io_host BEFORE cumes_io so the adapter library
  # is scanned while the backend-factory references are still undefined — the
  # linker then extracts the strong NetCDF/HDF5 factory definitions.
  target_link_libraries(cumes PRIVATE cumes_config_json cumes_io_host cumes_io
    cumes_solver Threads::Threads)
  if(CUMES_BUILD_MAGNETIC_COORDINATE)
    target_link_libraries(cumes PRIVATE
      magnetic-coordinate::magnetic_coordinate)
    target_compile_definitions(cumes PRIVATE CUMES_HAVE_MAGNETIC_COORDINATE)
  endif()
  if(CUMES_USE_VACUUM_FIELD)
    target_link_libraries(cumes PRIVATE vfield::vfield)
  endif()
  cumes_enable_warnings(cumes)
  cumes_apply_policy_flags(cumes)
  if(CUMES_USE_FLOAT)
    target_compile_definitions(cumes PRIVATE CUMES_USE_FLOAT)
  endif()

  # Configure-time git revision/dirty flag; empty when git is unavailable.
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
  target_compile_definitions(cumes PRIVATE
    CUMES_GIT_REVISION="${CUMES_GIT_REVISION}"
    CUMES_GIT_DIRTY=${CUMES_GIT_DIRTY}
    CUMES_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
  cumes_add_precision_provenance(cumes)
endif()

# ---- standalone verification tools ----------------------------------------
# These remain ordinary one-file host programs under scripts/: each can also
# be compiled directly with `g++ -std=c++20 scripts/compare_<name>.cpp -o ...`.
# compare_wout uses the already-optional HDF5 dependency when available and
# otherwise builds an explanatory runtime stub.
if(CUMES_BUILD_TESTS)
  foreach(tool IN ITEMS compare_states compare_runs compare_bitwise compare_wout)
    add_executable(${tool} scripts/${tool}.cpp)
    cumes_enable_warnings(${tool})
  endforeach()
  if(CUMES_HAVE_HDF5)
    target_compile_definitions(compare_wout PRIVATE CUMES_COMPARE_HAVE_HDF5=1)
    target_link_libraries(compare_wout PRIVATE HDF5::HDF5)
  endif()

  # ---- tests (standalone executables, no framework) ------------------------
  enable_testing()
  add_test(NAME compare_tools
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_tools_test.sh
                   $<TARGET_FILE:compare_states>
                   $<TARGET_FILE:compare_runs>
                   $<TARGET_FILE:compare_bitwise>
                   $<TARGET_FILE:compare_wout>)
  set_tests_properties(compare_tools PROPERTIES
      LABELS "unit;verification" TIMEOUT 60
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_subdirectory(tests)
endif()

# ---- benchmarks (not part of the default CTest unit gate) --------------------
# The fixed-iteration harness. Always double (verify-double is the
# benchmark reference config), regardless of CUMES_USE_FLOAT.
if(CUMES_BUILD_BENCHMARKS)
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

  # CUDA Graph capture of the real per-iteration DAG.
  add_executable(cumes_benchmark_graph_realpass benchmarks/graph_realpass.cu)
  target_include_directories(cumes_benchmark_graph_realpass PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(cumes_benchmark_graph_realpass PRIVATE cumes_config_json cumes_io_host cumes_cuda_double)
  cumes_enable_warnings(cumes_benchmark_graph_realpass)
  cumes_apply_policy_flags(cumes_benchmark_graph_realpass)
  cumes_add_precision_provenance(cumes_benchmark_graph_realpass)
endif()

# CLI policy gate: strict-vs-compatibility schema
# and output-path/suffix policy, exercised end to end against the real binary
# with one-iteration tiny configs (validation and preflight failures exit
# before any meaningful GPU work).
if(CUMES_BUILD_TESTS AND CUMES_BUILD_CLI)
add_test(NAME cli_policy
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_policy_test.sh
                 $<TARGET_FILE:cumes>)
set_tests_properties(cli_policy PROPERTIES
    LABELS "unit;cli" TIMEOUT 300
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# End-to-end durable-publication contract: a
# converged run whose output cannot be published must exit nonzero and must
# never damage an existing destination. The float build skips the converged
# cases (its double-tuned ftols are unreachable; the protocol is
# precision-independent and covered by test_output_failure at both
# precisions).
add_test(NAME output_publication
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/output_publication_test.sh
                 $<TARGET_FILE:cumes>
                 ${CMAKE_CURRENT_SOURCE_DIR}/inputs/solovev.json)
set_tests_properties(output_publication PROPERTIES
    LABELS "unit;output" TIMEOUT 600
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# Smoke gate: runs two passes and exits 0 on success (verifies the harness
# builds, seeds, drives a stage, and emits JSON — not a timing acceptance).
if(CUMES_BUILD_TESTS AND CUMES_BUILD_BENCHMARKS)
add_test(NAME cumes_benchmark_smoke
         COMMAND cumes_benchmark_fixed_iteration --config solovev --passes 2 --warmup 0)
set_tests_properties(cumes_benchmark_smoke PROPERTIES
    LABELS "performance;smoke" TIMEOUT 120
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# ---- installable C++ package ----------------------------------------------
# Export the facade and its static-library closure. Optional third-party
# libraries remain ordinary package dependencies reconstructed by
# cuMESConfig.cmake; implementation-only cuMES/vacuum targets are exported
# under the same namespace so consumers link only cumes::solver.
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CUMES_INSTALL_TARGETS
  cumes_solver cumes_config_json cumes_core cumes_json cumes_io_host cumes_io
  cumes_cuda_runtime ${CUMES_SELECTED_CUDA})
if(CUMES_HAVE_BSPLINE_PROLONGATION)
  list(APPEND CUMES_INSTALL_TARGETS cumes_bspline_prolongation)
endif()
if(CUMES_USE_VACUUM_FIELD)
  if(CUMES_USE_FLOAT)
    set(CUMES_SELECTED_VFIELD vfield_cuda_float)
  else()
    set(CUMES_SELECTED_VFIELD vfield_cuda_double)
  endif()
  list(APPEND CUMES_INSTALL_TARGETS
    vfield_host vfield_makegrid ${CUMES_SELECTED_VFIELD})
endif()

set(CUMES_PACKAGE_NETCDF_TARGET "")
foreach(candidate IN ITEMS netCDF::netCDF netCDF::netcdf
                           PkgConfig::netCDF PkgConfig::netcdf)
  if(TARGET ${candidate})
    set(CUMES_PACKAGE_NETCDF_TARGET ${candidate})
    break()
  endif()
endforeach()

install(TARGETS ${CUMES_INSTALL_TARGETS}
  EXPORT cuMESTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(CUMES_USE_VACUUM_FIELD)
  install(DIRECTORY deps/vacuum-field/include/
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
install(EXPORT cuMESTargets
  FILE cuMESTargets.cmake
  NAMESPACE cumes::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuMES)

configure_package_config_file(
  cmake/cuMESConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cuMESConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuMES)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/cuMESConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/cuMESConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cuMESConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuMES)

if(CUMES_BUILD_TESTS)
add_test(NAME package_consumer
  COMMAND ${CMAKE_COMMAND}
          -DCUMES_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
          -DCUMES_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/package_consumer_test.cmake)
set_tests_properties(package_consumer PROPERTIES
  LABELS "unit;api;package" TIMEOUT 300)
endif()
