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
  ${PROJECT_SOURCE_DIR}/src/webgpu/vacuum_force.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/vacuum_force_tests.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/magnetic_field.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/numerics.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/newton.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/newton_tests.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/preconditioner.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/preconditioner_apply.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/preconditioner_matrix.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/prolongation.cpp)
target_sources(cumes_webgpu PRIVATE
  ${PROJECT_SOURCE_DIR}/src/webgpu/toroidal.cpp)
target_sources(cumes_webgpu PRIVATE ${PROJECT_SOURCE_DIR}/src/webgpu/reduction.cpp)

# Preprocess shared precision templates with Node before Emscripten embeds the
# generated WGSL. Browser compilation receives ordinary WGSL, with no runtime
# substitution or precision branches. Output names preserve the C++ bindings.
find_program(CUMES_WEBGPU_NODE NAMES node nodejs REQUIRED)
set(CUMES_SHADER_COMPILER ${PROJECT_SOURCE_DIR}/scripts/compile_webgpu_shaders.mjs)
set(CUMES_SHADER_DIR ${PROJECT_SOURCE_DIR}/src/webgpu/shaders)
set(CUMES_GENERATED_SHADERS ${CMAKE_CURRENT_BINARY_DIR}/shaders)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${CUMES_SHADER_COMPILER} ${CUMES_SHADER_DIR}/templates.json)
file(GLOB_RECURSE CUMES_SHADER_SOURCES CONFIGURE_DEPENDS
  ${CUMES_SHADER_DIR}/*.wgsl ${CUMES_SHADER_DIR}/*.wgsl.in)
execute_process(COMMAND ${CUMES_WEBGPU_NODE} ${CUMES_SHADER_COMPILER} --list
  OUTPUT_VARIABLE CUMES_SHADER_NAMES COMMAND_ERROR_IS_FATAL ANY)
set(CUMES_SHADER_OUTPUTS)
foreach(shader IN LISTS CUMES_SHADER_NAMES)
  list(APPEND CUMES_SHADER_OUTPUTS ${CUMES_GENERATED_SHADERS}/${shader})
endforeach()
add_custom_command(OUTPUT ${CUMES_SHADER_OUTPUTS}
  COMMAND ${CUMES_WEBGPU_NODE} ${CUMES_SHADER_COMPILER} ${CUMES_GENERATED_SHADERS}
  DEPENDS ${CUMES_SHADER_SOURCES} ${CUMES_SHADER_COMPILER}
          ${PROJECT_SOURCE_DIR}/webgpu/shader_template.mjs
          ${CUMES_SHADER_DIR}/templates.json
  COMMENT "Generating single and pair-single WGSL from shared templates"
  VERBATIM)
add_custom_target(cumes_webgpu_shaders DEPENDS ${CUMES_SHADER_OUTPUTS})
add_dependencies(cumes_webgpu cumes_webgpu_shaders)
target_link_options(cumes_webgpu PRIVATE
  "--embed-file=${CUMES_GENERATED_SHADERS}@/shaders")
set_property(TARGET cumes_webgpu APPEND PROPERTY LINK_DEPENDS ${CUMES_SHADER_OUTPUTS})

target_include_directories(cumes_webgpu PRIVATE ${PROJECT_SOURCE_DIR}/include)
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/webgpu-fft/CMakeLists.txt")
  message(FATAL_ERROR "Missing webgpu-fft: run git submodule update --init deps/webgpu-fft")
endif()
add_subdirectory(${PROJECT_SOURCE_DIR}/deps/webgpu-fft ${CMAKE_CURRENT_BINARY_DIR}/fft)
target_link_libraries(cumes_webgpu PRIVATE webgpu_fft)
set(VFIELD_BACKEND WEBGPU CACHE STRING "vacuum-field execution backend" FORCE)
set(VFIELD_USE_NETCDF OFF CACHE BOOL "Browser uses in-memory coil grids" FORCE)
set(VFIELD_BUILD_TESTS OFF CACHE BOOL "Test vacuum-field separately" FORCE)
set(VFIELD_BUILD_TOOLS OFF CACHE BOOL "No vacuum command-line tools" FORCE)
set(VFIELD_BUILD_MAKEGRID ON CACHE BOOL "Generate browser coil grids in memory" FORCE)
set(VFIELD_WARNINGS_AS_ERRORS ${CUMES_WARNINGS_AS_ERRORS} CACHE BOOL "Vacuum warnings policy" FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/deps/vacuum-field ${CMAKE_CURRENT_BINARY_DIR}/vacuum)
target_link_libraries(cumes_webgpu PRIVATE vfield_cuda_double vfield::makegrid vfield::webgpu)

# Reuse vacuum-field's converter for previews without loading the solver or
# maintaining another coils-dot/JSON parser in JavaScript.
add_executable(cumes_coils
  ${PROJECT_SOURCE_DIR}/deps/vacuum-field/tools/coils_convert.cpp)
target_link_libraries(cumes_coils PRIVATE vfield::makegrid)
target_link_options(cumes_coils PRIVATE
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sINVOKE_RUN=0 -sEXIT_RUNTIME=0
  -sPTHREAD_POOL_SIZE=0
  "-sEXPORTED_RUNTIME_METHODS=['FS','callMain']"
  "-sINCOMING_MODULE_JS_API=['locateFile','print','printErr']")
set_target_properties(cumes_coils PROPERTIES SUFFIX ".mjs")
add_dependencies(cumes_webgpu cumes_coils)
set_property(TARGET cumes_webgpu APPEND PROPERTY LINK_DEPENDS
  $<TARGET_FILE:cumes_coils>)
target_compile_definitions(cumes_webgpu PRIVATE CUMES_VACUUM_HOST CUMES_VACUUM_WEBGPU)
target_compile_definitions(cumes_webgpu PRIVATE
  CUMES_GIT_REVISION="${CUMES_GIT_REVISION}"
  CUMES_GIT_DIRTY=${CUMES_GIT_DIRTY}
  CUMES_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  CUMES_PRECISION_POLICY_NAME="${CUMES_PRECISION_POLICY}"
  CUMES_PRECISION_FLAGS="${CUMES_PRECISION_FLAGS_STR}")
cumes_enable_warnings(cumes_webgpu)

target_compile_options(cumes_webgpu PRIVATE --use-port=emdawnwebgpu)
target_link_options(cumes_webgpu PRIVATE
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
  "--pre-js=${CMAKE_CURRENT_SOURCE_DIR}/timestamp_capture.js"
  "--pre-js=${CMAKE_CURRENT_SOURCE_DIR}/iteration_timing.js"
  "--pre-js=${CMAKE_CURRENT_SOURCE_DIR}/runtime_threads.js"
  "--embed-file=${PROJECT_SOURCE_DIR}/inputs/solovev.json@/inputs/solovev.json"
  "--embed-file=${PROJECT_SOURCE_DIR}/inputs/w7x.json@/inputs/w7x.json")

set_target_properties(cumes_webgpu PROPERTIES SUFFIX ".html")
add_custom_command(
  TARGET cumes_webgpu POST_BUILD
  COMMAND ${CMAKE_COMMAND}
          -DARTIFACT_BASE=$<TARGET_FILE_DIR:cumes_webgpu>/cumes_webgpu
          -DFRONTEND_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cache_bust_artifacts.cmake
  COMMENT "Adding content versions to the WebGPU runtime artifacts")
set_property(TARGET cumes_webgpu APPEND PROPERTY LINK_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/shell.html
  ${CMAKE_CURRENT_SOURCE_DIR}/browser_ui.js
  ${CMAKE_CURRENT_SOURCE_DIR}/equilibrium_view.js
  ${CMAKE_CURRENT_SOURCE_DIR}/orbit_renderer.js
  ${CMAKE_CURRENT_SOURCE_DIR}/boundary_editor.js
  ${CMAKE_CURRENT_SOURCE_DIR}/free_boundary.js
  ${CMAKE_CURRENT_SOURCE_DIR}/coil_geometry.js
  ${CMAKE_CURRENT_SOURCE_DIR}/presets/solovev.json
  ${CMAKE_CURRENT_SOURCE_DIR}/presets/w7x.json
  ${CMAKE_CURRENT_SOURCE_DIR}/presets/cth_like.json
  ${CMAKE_CURRENT_SOURCE_DIR}/residual_plot.js
  ${CMAKE_CURRENT_SOURCE_DIR}/verification_worker.js
  ${CMAKE_CURRENT_SOURCE_DIR}/browser_bridge.js
  ${CMAKE_CURRENT_SOURCE_DIR}/timestamp_capture.js
  ${CMAKE_CURRENT_SOURCE_DIR}/iteration_timing.js
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime_threads.js
  ${CMAKE_CURRENT_SOURCE_DIR}/cache_bust_artifacts.cmake
  ${PROJECT_SOURCE_DIR}/inputs/solovev.json
  ${PROJECT_SOURCE_DIR}/inputs/w7x.json)

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
foreach(test IN ITEMS shader_templates threads pages boundary_geometry
                      residual_plot iteration_timing timestamps precision_switch log_buffer
                      free_boundary worker runtime_loader verification_log cdp)
  add_test(NAME webgpu_${test}
    COMMAND ${CUMES_WEBGPU_NODE}
            ${PROJECT_SOURCE_DIR}/scripts/test_webgpu_${test}.mjs)
  set_tests_properties(webgpu_${test} PROPERTIES LABELS "unit;webgpu")
endforeach()
add_test(NAME webgpu_coils
  COMMAND ${CUMES_WEBGPU_NODE}
          ${PROJECT_SOURCE_DIR}/scripts/test_webgpu_coils.mjs
          $<TARGET_FILE:cumes_coils>)
set_tests_properties(webgpu_coils PROPERTIES LABELS "unit;webgpu")

# The CUDA and Wasm vacuum bridges share these analytic contracts.
add_executable(test_webgpu_vacuum_bridge
  ${PROJECT_SOURCE_DIR}/tests/test_vacuum_bridge.cpp)
target_include_directories(test_webgpu_vacuum_bridge PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(test_webgpu_vacuum_bridge PRIVATE CUMES_VACUUM_HOST)
target_link_libraries(test_webgpu_vacuum_bridge PRIVATE vfield_cuda_double vfield::makegrid)
cumes_enable_warnings(test_webgpu_vacuum_bridge)
add_test(NAME webgpu_vacuum_bridge COMMAND test_webgpu_vacuum_bridge)
set_tests_properties(webgpu_vacuum_bridge PROPERTIES LABELS "unit;webgpu")
