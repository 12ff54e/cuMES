# Browser-facing WebGPU milestone. This target intentionally has no CUDA
# includes or libraries; emdawnwebgpu is enabled for both compilation and the
# final link as required by the Emscripten port contract.
add_compile_options(-pthread)
add_link_options(-pthread)
add_executable(cumes_webgpu
  ${PROJECT_SOURCE_DIR}/src/json_parser.cpp
  ${PROJECT_SOURCE_DIR}/src/cumes/config/json_reader.cpp
  ${PROJECT_SOURCE_DIR}/src/cumes/config/validated_problem.cpp
  ${PROJECT_SOURCE_DIR}/src/cumes/config/validation_report.cpp
  ${PROJECT_SOURCE_DIR}/src/cumes/core/grid_shape.cpp
  ${PROJECT_SOURCE_DIR}/src/cumes/core/mode_table.cpp
  ${PROJECT_SOURCE_DIR}/src/cumes/io/derived_fields.cpp
  ${PROJECT_SOURCE_DIR}/src/cumes/io/versioned_binary.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/axisymmetric.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/axisymmetric_forward.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/constraint.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/descent.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/geometry.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/force.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/float_geometry_tests.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/rounding_tests.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/initialization.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/iteration.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/main.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/vacuum.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/magnetic_field.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/numerics.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/preconditioner.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/preconditioner_apply.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/preconditioner_matrix.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/prolongation.cpp)
target_sources(cumes_webgpu PRIVATE
  ${PROJECT_SOURCE_DIR}/src/webgpu/toroidal.cpp)
target_sources(cumes_webgpu PRIVATE ${PROJECT_SOURCE_DIR}/src/webgpu/reduction.cpp)
target_link_options(cumes_webgpu PRIVATE
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/geometry_control.wgsl@/shaders/geometry_control.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/field_finite.wgsl@/shaders/field_finite.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/residual_norm.wgsl@/shaders/residual_norm.wgsl")

target_include_directories(cumes_webgpu PRIVATE ${PROJECT_SOURCE_DIR}/include)
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/webgpu-fft/CMakeLists.txt")
  message(FATAL_ERROR "Missing webgpu-fft: run git submodule update --init deps/webgpu-fft")
endif()
add_subdirectory(${PROJECT_SOURCE_DIR}/deps/webgpu-fft ${CMAKE_CURRENT_BINARY_DIR}/fft)
target_link_libraries(cumes_webgpu PRIVATE webgpu_fft)
set(VFIELD_BACKEND HOST CACHE STRING "vacuum-field execution backend" FORCE)
set(VFIELD_USE_NETCDF OFF CACHE BOOL "Browser uses in-memory coil grids" FORCE)
set(VFIELD_BUILD_TESTS OFF CACHE BOOL "Test vacuum-field separately" FORCE)
set(VFIELD_BUILD_TOOLS OFF CACHE BOOL "No vacuum command-line tools" FORCE)
set(VFIELD_BUILD_MAKEGRID ON CACHE BOOL "Generate browser coil grids in memory" FORCE)
set(VFIELD_WARNINGS_AS_ERRORS ${CUMES_WARNINGS_AS_ERRORS} CACHE BOOL "Vacuum warnings policy" FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/deps/vacuum-field ${CMAKE_CURRENT_BINARY_DIR}/vacuum)
target_link_libraries(cumes_webgpu PRIVATE vfield_cuda_double vfield::makegrid)
target_compile_definitions(cumes_webgpu PRIVATE CUMES_VACUUM_HOST)
target_compile_definitions(cumes_webgpu PRIVATE
  CUMES_GIT_REVISION="${CUMES_GIT_REVISION}"
  CUMES_GIT_DIRTY=${CUMES_GIT_DIRTY}
  CUMES_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  CUMES_PRECISION_POLICY_NAME="${CUMES_PRECISION_POLICY}"
  CUMES_PRECISION_FLAGS="${CUMES_PRECISION_FLAGS_STR}")
cumes_enable_warnings(cumes_webgpu)

target_compile_options(cumes_webgpu PRIVATE --use-port=emdawnwebgpu)
target_link_options(cumes_webgpu PRIVATE
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/toroidal_fft_transfer.wgsl@/shaders/toroidal_fft_transfer.wgsl"
  --use-port=emdawnwebgpu
  -sASYNCIFY=1
  -sALLOW_MEMORY_GROWTH=1
  -sEXIT_RUNTIME=1
  -sASSERTIONS=1
  "-sPTHREAD_POOL_SIZE=cumesThreadCount()-1"
  -sPTHREAD_POOL_SIZE_STRICT=2
  "-sINCOMING_MODULE_JS_API=['preRun','locateFile','print','printErr','onAbort','mainScriptUrlOrBlob']"
  "--shell-file=${CMAKE_CURRENT_SOURCE_DIR}/shell.html"
  "--js-library=${CMAKE_CURRENT_SOURCE_DIR}/browser_bridge.js"
  "--pre-js=${CMAKE_CURRENT_SOURCE_DIR}/iteration_timing.js"
  "--pre-js=${CMAKE_CURRENT_SOURCE_DIR}/runtime_threads.js"
  "--embed-file=${PROJECT_SOURCE_DIR}/inputs/solovev.json@/inputs/solovev.json"
  "--embed-file=${PROJECT_SOURCE_DIR}/inputs/w7x.json@/inputs/w7x.json"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/base_geometry.wgsl@/shaders/base_geometry.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/base_geometry_double_single.wgsl@/shaders/base_geometry_double_single.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_force.wgsl@/shaders/axisymmetric_force.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/force_double_single.wgsl@/shaders/force_double_single.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_constraint_head.wgsl@/shaders/axisymmetric_constraint_head.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/constraint_head_double_single.wgsl@/shaders/constraint_head_double_single.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_constraint_tail.wgsl@/shaders/axisymmetric_constraint_tail.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/constraint_tail_double_single.wgsl@/shaders/constraint_tail_double_single.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_descent.wgsl@/shaders/axisymmetric_descent.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_descent_double_single.wgsl@/shaders/axisymmetric_descent_double_single.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/float_float.wgsl@/shaders/float_float.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/magnetic_field.wgsl@/shaders/magnetic_field.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/magnetic_field_double_single.wgsl@/shaders/magnetic_field_double_single.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/residual_decompose.wgsl@/shaders/residual_decompose.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/residual_decompose_double_single.wgsl@/shaders/residual_decompose_double_single.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_preconditioner_elements.wgsl@/shaders/axisymmetric_preconditioner_elements.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_preconditioner_apply.wgsl@/shaders/axisymmetric_preconditioner_apply.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_preconditioner_matrix.wgsl@/shaders/axisymmetric_preconditioner_matrix.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_dealias.wgsl@/shaders/axisymmetric_dealias.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_forward.wgsl@/shaders/axisymmetric_forward.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_inverse.wgsl@/shaders/axisymmetric_inverse.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/prolongation.wgsl@/shaders/prolongation.wgsl")
target_link_options(cumes_webgpu PRIVATE
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/toroidal_inverse.wgsl@/shaders/toroidal_inverse.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/toroidal_inverse_double_single.wgsl@/shaders/toroidal_inverse_double_single.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/toroidal_forward.wgsl@/shaders/toroidal_forward.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/toroidal_forward_double_single.wgsl@/shaders/toroidal_forward_double_single.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/toroidal_dealias.wgsl@/shaders/toroidal_dealias.wgsl")

set_target_properties(cumes_webgpu PROPERTIES SUFFIX ".html")
add_custom_command(
  TARGET cumes_webgpu POST_BUILD
  COMMAND ${CMAKE_COMMAND}
          -DARTIFACT_BASE=$<TARGET_FILE_DIR:cumes_webgpu>/cumes_webgpu
          -DFRONTEND_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cache_bust_artifacts.cmake
  COMMENT "Adding content versions to the WebGPU runtime artifacts")
set_property(TARGET cumes_webgpu APPEND PROPERTY LINK_DEPENDS
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/geometry_control.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/field_finite.wgsl
  ${CMAKE_CURRENT_SOURCE_DIR}/shell.html
  ${CMAKE_CURRENT_SOURCE_DIR}/browser_ui.js
  ${CMAKE_CURRENT_SOURCE_DIR}/equilibrium_view.js
  ${CMAKE_CURRENT_SOURCE_DIR}/boundary_editor.js
  ${CMAKE_CURRENT_SOURCE_DIR}/free_boundary.js
  ${CMAKE_CURRENT_SOURCE_DIR}/presets/solovev.json
  ${CMAKE_CURRENT_SOURCE_DIR}/presets/w7x.json
  ${CMAKE_CURRENT_SOURCE_DIR}/presets/cth_like.json
  ${CMAKE_CURRENT_SOURCE_DIR}/residual_plot.js
  ${CMAKE_CURRENT_SOURCE_DIR}/verification_worker.js
  ${CMAKE_CURRENT_SOURCE_DIR}/browser_bridge.js
  ${CMAKE_CURRENT_SOURCE_DIR}/iteration_timing.js
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime_threads.js
  ${CMAKE_CURRENT_SOURCE_DIR}/cache_bust_artifacts.cmake
  ${PROJECT_SOURCE_DIR}/inputs/solovev.json
  ${PROJECT_SOURCE_DIR}/inputs/w7x.json
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_dealias.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_forward.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_inverse.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_force.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/force_double_single.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_constraint_head.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/constraint_head_double_single.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_constraint_tail.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/constraint_tail_double_single.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_descent.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_descent_double_single.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/float_float.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/residual_decompose.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/residual_decompose_double_single.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_preconditioner_elements.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_preconditioner_apply.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_preconditioner_matrix.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/base_geometry.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/base_geometry_double_single.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/magnetic_field.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/magnetic_field_double_single.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/prolongation.wgsl)
set_property(TARGET cumes_webgpu APPEND PROPERTY LINK_DEPENDS
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/toroidal_inverse.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/toroidal_inverse_double_single.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/toroidal_forward.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/toroidal_forward_double_single.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/toroidal_fft_transfer.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/toroidal_dealias.wgsl)

# CTest cannot provide a browser WebGPU implementation, but it can guard the
# complete browser artifact set. Emscripten 6 embeds the preloaded shader in
# the Wasm module, so there is no separate .data file. Numerical validation
# runs inside the page and exits nonzero on a mismatch.
add_test(
  NAME webgpu_artifacts
  COMMAND ${CMAKE_COMMAND}
          -DARTIFACT_BASE=$<TARGET_FILE_DIR:cumes_webgpu>/cumes_webgpu
          -P ${CMAKE_CURRENT_SOURCE_DIR}/check_artifacts.cmake)
set_tests_properties(webgpu_artifacts PROPERTIES LABELS "unit;webgpu")
find_program(CUMES_WEBGPU_NODE NAMES node nodejs)
if(CUMES_WEBGPU_NODE)
  add_test(NAME webgpu_threads
    COMMAND ${CUMES_WEBGPU_NODE}
            ${PROJECT_SOURCE_DIR}/scripts/test_webgpu_threads.mjs)
  set_tests_properties(webgpu_threads PROPERTIES LABELS "unit;webgpu")
  add_test(NAME webgpu_boundary_geometry
    COMMAND ${CUMES_WEBGPU_NODE}
            ${PROJECT_SOURCE_DIR}/scripts/test_webgpu_boundary_geometry.mjs)
  set_tests_properties(webgpu_boundary_geometry PROPERTIES LABELS "unit;webgpu")
  add_test(NAME webgpu_residual_plot
    COMMAND ${CUMES_WEBGPU_NODE}
            ${PROJECT_SOURCE_DIR}/scripts/test_webgpu_residual_plot.mjs)
  set_tests_properties(webgpu_residual_plot PROPERTIES LABELS "unit;webgpu")
  add_test(NAME webgpu_precision_switch
    COMMAND ${CUMES_WEBGPU_NODE}
            ${PROJECT_SOURCE_DIR}/scripts/test_webgpu_precision_switch.mjs)
  set_tests_properties(webgpu_precision_switch PROPERTIES LABELS "unit;webgpu")
  add_test(NAME webgpu_log_buffer
    COMMAND ${CUMES_WEBGPU_NODE}
            ${PROJECT_SOURCE_DIR}/scripts/test_webgpu_log_buffer.mjs)
  set_tests_properties(webgpu_log_buffer PROPERTIES LABELS "unit;webgpu")
  add_test(NAME webgpu_free_boundary
    COMMAND ${CUMES_WEBGPU_NODE}
            ${PROJECT_SOURCE_DIR}/scripts/test_webgpu_free_boundary.mjs)
  set_tests_properties(webgpu_free_boundary PROPERTIES LABELS "unit;webgpu")
  add_test(NAME webgpu_worker
    COMMAND ${CUMES_WEBGPU_NODE}
            ${PROJECT_SOURCE_DIR}/scripts/test_webgpu_worker.mjs)
  set_tests_properties(webgpu_worker PROPERTIES LABELS "unit;webgpu")
  add_test(NAME webgpu_runtime_loader
    COMMAND ${CUMES_WEBGPU_NODE}
            ${PROJECT_SOURCE_DIR}/scripts/test_webgpu_runtime_loader.mjs)
  set_tests_properties(webgpu_runtime_loader PROPERTIES LABELS "unit;webgpu")
  add_test(NAME webgpu_verification_log
    COMMAND ${CUMES_WEBGPU_NODE}
            ${PROJECT_SOURCE_DIR}/scripts/test_webgpu_verification_log.mjs)
  set_tests_properties(webgpu_verification_log PROPERTIES LABELS "unit;webgpu")
endif()

# The CUDA and Wasm vacuum bridges share these analytic contracts.
add_executable(test_webgpu_vacuum_bridge
  ${PROJECT_SOURCE_DIR}/tests/test_vacuum_bridge.cpp)
target_include_directories(test_webgpu_vacuum_bridge PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(test_webgpu_vacuum_bridge PRIVATE CUMES_VACUUM_HOST)
target_link_libraries(test_webgpu_vacuum_bridge PRIVATE vfield_cuda_double vfield::makegrid)
cumes_enable_warnings(test_webgpu_vacuum_bridge)
add_test(NAME webgpu_vacuum_bridge COMMAND test_webgpu_vacuum_bridge)
set_tests_properties(webgpu_vacuum_bridge PROPERTIES LABELS "unit;webgpu")
