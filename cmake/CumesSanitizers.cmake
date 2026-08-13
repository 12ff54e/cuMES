# CumesSanitizers.cmake — optional Compute Sanitizer memcheck pass.
#
# When CUMES_ENABLE_SANITIZER_TESTS=ON and compute-sanitizer is on PATH,
# `cumes_register_sanitizer_variants(<test>...)` adds a `sanitizer_<test>`
# CTest entry that runs the given executable under `compute-sanitizer --tool
# memcheck --error-exitcode 3`. Host-only tests (no kernel launches) are
# excluded by the caller — compute-sanitizer refuses to profile them.

find_program(CUMES_COMPUTE_SANITIZER compute-sanitizer)

function(cumes_register_sanitizer_variants)
  if(NOT CUMES_ENABLE_SANITIZER_TESTS)
    return()
  endif()
  if(NOT CUMES_COMPUTE_SANITIZER)
    message(WARNING "cuMES: CUMES_ENABLE_SANITIZER_TESTS=ON but "
                    "compute-sanitizer not found on PATH")
    return()
  endif()
  foreach(t IN LISTS ARGN)
    add_test(NAME sanitizer_${t}
             COMMAND ${CUMES_COMPUTE_SANITIZER}
                     --tool memcheck --error-exitcode 3 $<TARGET_FILE:${t}>)
    set_tests_properties(sanitizer_${t} PROPERTIES
        LABELS "sanitizer" TIMEOUT 600
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endforeach()
endfunction()
