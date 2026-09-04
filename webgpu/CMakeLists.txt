# Browser-facing WebGPU milestone. This target intentionally has no CUDA
# includes or libraries; emdawnwebgpu is enabled for both compilation and the
# final link as required by the Emscripten port contract.
add_executable(cumes_webgpu
  ${PROJECT_SOURCE_DIR}/src/json_parser.cpp
  ${PROJECT_SOURCE_DIR}/src/cumes/config/json_reader.cpp
  ${PROJECT_SOURCE_DIR}/src/cumes/config/validated_problem.cpp
  ${PROJECT_SOURCE_DIR}/src/cumes/config/validation_report.cpp
  ${PROJECT_SOURCE_DIR}/src/cumes/core/grid_shape.cpp
  ${PROJECT_SOURCE_DIR}/src/cumes/core/mode_table.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/axisymmetric.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/axisymmetric_forward.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/constraint.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/geometry.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/force.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/initialization.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/main.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/magnetic_field.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/numerics.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/preconditioner.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/preconditioner_apply.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/preconditioner_matrix.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/prolongation.cpp)

target_include_directories(cumes_webgpu PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(cumes_webgpu PRIVATE CUMES_VACUUM_FIELD_DISABLED)
cumes_enable_warnings(cumes_webgpu)

target_compile_options(cumes_webgpu PRIVATE --use-port=emdawnwebgpu)
target_link_options(cumes_webgpu PRIVATE
  --use-port=emdawnwebgpu
  -sASYNCIFY=1
  -sALLOW_MEMORY_GROWTH=1
  -sEXIT_RUNTIME=1
  -sASSERTIONS=1
  "--shell-file=${CMAKE_CURRENT_SOURCE_DIR}/shell.html"
  "--embed-file=${PROJECT_SOURCE_DIR}/inputs/solovev.json@/inputs/solovev.json"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/base_geometry.wgsl@/shaders/base_geometry.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_force.wgsl@/shaders/axisymmetric_force.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_constraint_head.wgsl@/shaders/axisymmetric_constraint_head.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_constraint_tail.wgsl@/shaders/axisymmetric_constraint_tail.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/magnetic_field.wgsl@/shaders/magnetic_field.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/residual_decompose.wgsl@/shaders/residual_decompose.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_preconditioner_elements.wgsl@/shaders/axisymmetric_preconditioner_elements.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_preconditioner_apply.wgsl@/shaders/axisymmetric_preconditioner_apply.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_preconditioner_matrix.wgsl@/shaders/axisymmetric_preconditioner_matrix.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_dealias.wgsl@/shaders/axisymmetric_dealias.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_forward.wgsl@/shaders/axisymmetric_forward.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_inverse.wgsl@/shaders/axisymmetric_inverse.wgsl"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/prolongation.wgsl@/shaders/prolongation.wgsl")

set_target_properties(cumes_webgpu PROPERTIES SUFFIX ".html")
set_property(TARGET cumes_webgpu APPEND PROPERTY LINK_DEPENDS
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_dealias.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_forward.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_inverse.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_force.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_constraint_head.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_constraint_tail.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/residual_decompose.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_preconditioner_elements.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_preconditioner_apply.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/axisymmetric_preconditioner_matrix.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/base_geometry.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/magnetic_field.wgsl
  ${PROJECT_SOURCE_DIR}/src/webgpu/shaders/prolongation.wgsl)

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
