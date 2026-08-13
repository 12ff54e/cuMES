# CumesWarnings.cmake — target-scoped warning configuration.
#
# `cumes_enable_warnings(<target>)` applies the project warning set to a single
# target. Keeping this per-target (rather than in CMAKE_CXX_FLAGS) is what lets
# host-only `.cpp` targets, CUDA `.cu` targets, and vendored/dependency code use
# different warning levels.

function(cumes_enable_warnings target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wall -Wextra>)
  if(CUMES_WARNINGS_AS_ERRORS)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Werror>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
  endif()
endfunction()
