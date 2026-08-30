# tests/CMakeLists.txt — standalone test executables (no framework) and their
# CTest / compute-sanitizer / ASan registrations. Kept out of the root
# CMakeLists (which does add_subdirectory(tests)) so the library/CLI build and
# the test tree stay separate.
#
# Every registered test runs from the project source directory (they read
# inputs/*.json relative to the CWD) — see cmake/CumesTest.cmake.
#
# The shared test headers (cumes_test.h + cumes_test_cuda_helper.cuh) live in
# tests/include. The project-wide language standard is C++20 (set in the root
# CMakeLists.txt), so nothing needs to be set here.

include_directories(${PROJECT_SOURCE_DIR}/tests/include)

# Backend-availability defines for TEST targets only: the production targets
# keep the defines confined to the adapter library; a test asserting
# build-specific on-disk behavior gets the compile-time flags, not the
# backend sources or headers.
function(cumes_apply_backend_test_defines target)
  if(CUMES_HAVE_NETCDF)
    target_compile_definitions(${target} PRIVATE CUMES_HAVE_NETCDF)
  endif()
  if(CUMES_HAVE_HDF5)
    target_compile_definitions(${target} PRIVATE CUMES_HAVE_HDF5)
  endif()
endfunction()

# Host-only ASan test twin: links the _asan library directly (PRIVATE) so the
# sanitized runtime stays FIRST in the link line (never propagated through
# INTERFACE — see CumesSanitizers.cmake), then registers it under CTest label
# `sanitizer;asan;<label>`.
function(cumes_add_host_asan_test name source lib label)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(${name} PRIVATE ${lib})
  cumes_enable_warnings(${name})
  cumes_apply_host_sanitizers(${name})
  cumes_add_test(${name} "sanitizer;asan;${label}")
endfunction()

# ---- test executables ------------------------------------------------------
# Each test links the full device libraries (both scalar types) so every test
# can instantiate double and float; the linker pulls in only the objects it
# needs.

add_executable(test_fourier test_fourier.cu)
target_include_directories(test_fourier PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_fourier PRIVATE cumes_cuda_double cumes_cuda_float)
cumes_enable_warnings(test_fourier)

add_executable(test_input_json test_input_json.cu)
target_include_directories(test_input_json PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_input_json PRIVATE cumes_config_json)
cumes_enable_warnings(test_input_json)

foreach(t IN ITEMS test_forces test_force_verify test_geometry_iso
                   test_geometry_ncurr test_regression_kernels test_constraint_tcon
                   test_force_reference test_axisym_backend
                   test_tridiagonal test_cuda_graph test_safety_predicates
                   test_event_dag test_free_boundary_seed)
  add_executable(${t} ${t}.cu)
  target_include_directories(${t} PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(${t} PRIVATE cumes_config_json cumes_cuda_double cumes_cuda_float)
  cumes_enable_warnings(${t})
endforeach()

add_executable(test_output_failure test_output_failure.cu)
target_include_directories(test_output_failure PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_output_failure PRIVATE cumes_config_json cumes_io_host cumes_io)
cumes_enable_warnings(test_output_failure)

add_executable(test_host_config test_host_config.cpp)
target_include_directories(test_host_config PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_host_config PRIVATE cumes_config_json)
cumes_enable_warnings(test_host_config)

add_executable(test_host_io test_host_io.cpp)
target_include_directories(test_host_io PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_host_io PRIVATE cumes_io_host cumes_config_json)
cumes_enable_warnings(test_host_io)

add_executable(test_derived_fields test_derived_fields.cpp)
target_include_directories(test_derived_fields PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_derived_fields PRIVATE cumes_io_host)
cumes_enable_warnings(test_derived_fields)

add_executable(test_checkpoint test_checkpoint.cpp)
target_include_directories(test_checkpoint PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_checkpoint PRIVATE cumes_io_host)
cumes_enable_warnings(test_checkpoint)

if(CUMES_BUILD_MAGNETIC_COORDINATE)
  add_executable(test_magnetic_coordinate_bridge
                 test_magnetic_coordinate_bridge.cpp)
  target_include_directories(test_magnetic_coordinate_bridge PRIVATE
                             ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(test_magnetic_coordinate_bridge PRIVATE
                        magnetic-coordinate::magnetic_coordinate)
  cumes_enable_warnings(test_magnetic_coordinate_bridge)
endif()

add_executable(test_io_golden test_io_golden.cu)
target_include_directories(test_io_golden PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_io_golden PRIVATE cumes_config_json cumes_io_host cumes_io)
cumes_apply_backend_test_defines(test_io_golden)
cumes_enable_warnings(test_io_golden)

# Corrupted v1 restart-metadata fixtures:
# host-only so the ASan/UBSan twin runs the exact source under the host
# sanitizers.
add_executable(test_io_restart_offsets test_io_restart_offsets.cpp)
target_include_directories(test_io_restart_offsets PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_io_restart_offsets PRIVATE cumes_io_host cumes_io)
cumes_apply_backend_test_defines(test_io_restart_offsets)
cumes_enable_warnings(test_io_restart_offsets)

# Malformed v1 rank/type/extent fixtures:
# host-only so the ASan/UBSan twin runs the exact source under the host
# sanitizers.
add_executable(test_io_malformed_shapes test_io_malformed_shapes.cpp)
target_include_directories(test_io_malformed_shapes PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_io_malformed_shapes PRIVATE cumes_io_host cumes_io)
cumes_apply_backend_test_defines(test_io_malformed_shapes)
cumes_enable_warnings(test_io_malformed_shapes)

add_executable(test_runtime test_runtime.cu)
target_include_directories(test_runtime PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_runtime PRIVATE cumes_cuda_runtime)
cumes_enable_warnings(test_runtime)

add_executable(test_controller test_controller.cpp)
target_include_directories(test_controller PRIVATE ${PROJECT_SOURCE_DIR}/include)
cumes_enable_warnings(test_controller)

add_executable(test_seed_envelope test_seed_envelope.cpp)
target_include_directories(test_seed_envelope PRIVATE ${PROJECT_SOURCE_DIR}/include)
cumes_enable_warnings(test_seed_envelope)

add_executable(test_arena test_arena.cu)
target_include_directories(test_arena PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_arena PRIVATE cumes_cuda_runtime)
cumes_enable_warnings(test_arena)

add_executable(test_operator_views test_operator_views.cu)
target_include_directories(test_operator_views PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_operator_views PRIVATE cumes_cuda_runtime)
cumes_enable_warnings(test_operator_views)

add_executable(test_accumulation test_accumulation.cu)
target_include_directories(test_accumulation PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_accumulation PRIVATE cumes_cuda_runtime)
cumes_enable_warnings(test_accumulation)

# ---- ctest registration ----------------------------------------------------
cumes_add_test(test_fourier          unit;fourier)
cumes_add_test(test_input_json       unit;input)
cumes_add_test(test_host_config      unit;config)
cumes_add_test(test_host_io          unit;io)
cumes_add_test(test_derived_fields   unit;io;physics)
cumes_add_test(test_checkpoint       unit;io)
if(CUMES_BUILD_MAGNETIC_COORDINATE)
  cumes_add_test(test_magnetic_coordinate_bridge unit;io;boozer)
endif()
cumes_add_test(test_io_golden        unit;io)
cumes_add_test(test_io_restart_offsets unit;io)
cumes_add_test(test_io_malformed_shapes unit;io)
cumes_add_test(test_runtime          unit;runtime)
cumes_add_test(test_controller       unit;solver)
cumes_add_test(test_seed_envelope    unit;solver)

# ---- host sanitizer twins (CUMES_HOST_SANITIZERS=ON) ------------------------
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
  if(NOT CUMES_USE_VACUUM_FIELD)
    target_compile_definitions(cumes_config_json_asan
      PUBLIC CUMES_VACUUM_FIELD_DISABLED)
  endif()

  cumes_add_host_library(cumes_io_host_asan TRUE ${CUMES_IO_HOST_SOURCES})
  target_link_libraries(cumes_io_host_asan PUBLIC cumes_core_asan)

  # Reader-only ASan twin of the adapter dispatch + optional backends (the
  # device-staging output.cpp is excluded; the corrupted-fixture test needs
  # only the host readers).
  cumes_add_host_library(cumes_io_readers_asan TRUE
                         ${PROJECT_SOURCE_DIR}/src/cumes/io/writer_dispatch.cpp)
  target_link_libraries(cumes_io_readers_asan PUBLIC cumes_io_host_asan)
  cumes_link_output_backends(cumes_io_readers_asan)
  cumes_add_host_asan_test(asan_test_io_restart_offsets
                           test_io_restart_offsets.cpp
                           cumes_io_readers_asan io)
  cumes_apply_backend_test_defines(asan_test_io_restart_offsets)
  cumes_add_host_asan_test(asan_test_io_malformed_shapes
                           test_io_malformed_shapes.cpp
                           cumes_io_readers_asan io)
  cumes_apply_backend_test_defines(asan_test_io_malformed_shapes)

  cumes_add_host_asan_test(asan_test_host_config test_host_config.cpp
                           cumes_config_json_asan config)
  cumes_add_host_asan_test(asan_test_host_io test_host_io.cpp
                           cumes_io_host_asan io)
  target_link_libraries(asan_test_host_io PRIVATE cumes_config_json_asan)
  cumes_add_host_asan_test(asan_test_derived_fields test_derived_fields.cpp
                           cumes_io_host_asan io)
  cumes_add_host_asan_test(asan_test_checkpoint test_checkpoint.cpp
                           cumes_io_host_asan io)
  cumes_add_host_asan_test(asan_test_controller test_controller.cpp
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
cumes_add_test(test_event_dag unit;runtime)
# The stream-poison case runs in its own process (the trap faults the context
# for further work); not registered under compute-sanitizer.
add_test(NAME event_dag_poison
         COMMAND test_event_dag --poison)
set_tests_properties(event_dag_poison PROPERTIES
    LABELS "unit;runtime" TIMEOUT 60
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
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
# test_event_dag registers its sanitizer variants manually: the deliberate
# pending-event API-error probes are skipped under the sanitizers (they count
# the intended cudaErrorNotReady return as a sanitizer error).
if(CUMES_ENABLE_SANITIZER_TESTS AND CUMES_COMPUTE_SANITIZER)
  add_test(NAME sanitizer_test_event_dag
           COMMAND ${CUMES_COMPUTE_SANITIZER}
                   --tool memcheck --error-exitcode 3
                   $<TARGET_FILE:test_event_dag> --no-notready-probe)
  set_tests_properties(sanitizer_test_event_dag PROPERTIES
      LABELS "sanitizer" TIMEOUT 600 RUN_SERIAL TRUE
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  add_test(NAME initcheck_test_event_dag
           COMMAND ${CUMES_COMPUTE_SANITIZER}
                   --tool initcheck --error-exitcode 3
                   $<TARGET_FILE:test_event_dag> --no-notready-probe)
  set_tests_properties(initcheck_test_event_dag PROPERTIES
      LABELS "sanitizer;initcheck" TIMEOUT 600 RUN_SERIAL TRUE
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()
