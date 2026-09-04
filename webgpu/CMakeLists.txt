# Browser-facing WebGPU milestone. This target intentionally has no CUDA
# includes or libraries; emdawnwebgpu is enabled for both compilation and the
# final link as required by the Emscripten port contract.
add_executable(cumes_webgpu
  ${PROJECT_SOURCE_DIR}/src/webgpu/main.cpp
  ${PROJECT_SOURCE_DIR}/src/webgpu/prolongation.cpp)

target_include_directories(cumes_webgpu PRIVATE ${PROJECT_SOURCE_DIR}/include)
cumes_enable_warnings(cumes_webgpu)

target_compile_options(cumes_webgpu PRIVATE --use-port=emdawnwebgpu)
target_link_options(cumes_webgpu PRIVATE
  --use-port=emdawnwebgpu
  -sASYNCIFY=1
  -sALLOW_MEMORY_GROWTH=1
  -sEXIT_RUNTIME=1
  -sASSERTIONS=1
  "--shell-file=${CMAKE_CURRENT_SOURCE_DIR}/shell.html"
  "--embed-file=${PROJECT_SOURCE_DIR}/src/webgpu/shaders/prolongation.wgsl@/shaders/prolongation.wgsl")

set_target_properties(cumes_webgpu PROPERTIES SUFFIX ".html")

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
