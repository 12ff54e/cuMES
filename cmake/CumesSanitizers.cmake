# CumesSanitizers.cmake — optional sanitizer passes.
#
# Device side (Compute Sanitizer):
#   When CUMES_ENABLE_SANITIZER_TESTS=ON and compute-sanitizer is on PATH,
#   `cumes_register_sanitizer_variants(<test>...)` adds a `sanitizer_<test>`
#   CTest entry that runs the given executable under `compute-sanitizer --tool
#   memcheck --error-exitcode 3`. Host-only tests (no kernel launches) are
#   excluded by the caller — compute-sanitizer refuses to profile them.
#   With CUMES_ENABLE_EXTRA_SANITIZER_TOOLS=ON the same tests additionally get
#   `racecheck_<test>` / `synccheck_<test>` entries (intra-kernel shared-memory
#   and barrier hazards); racecheck instruments every memory access and is far
#   slower, so these stay out of the default verify gate.
#
# Host side (ASan + UBSan):
#   `cumes_apply_host_sanitizers(<target>)` adds -fsanitize=address,undefined
#   when CUMES_HOST_SANITIZERS=ON. Apply it only to host-only C++ targets;
#   CUDA targets are excluded (nvcc + ASan do not mix).

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
        LABELS "sanitizer" TIMEOUT 600 RUN_SERIAL TRUE
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    # initcheck: uninitialized-device-memory access. initcheck flags benign
    # reads of never-written scratch as errors on some driver versions, so
    # the guarded kernels MUST write (or deterministically skip) every
    # consumed buffer — that is exactly the invariant under test.
    add_test(NAME initcheck_${t}
             COMMAND ${CUMES_COMPUTE_SANITIZER}
                     --tool initcheck --error-exitcode 3 $<TARGET_FILE:${t}>)
    set_tests_properties(initcheck_${t} PROPERTIES
        LABELS "sanitizer;initcheck" TIMEOUT 600 RUN_SERIAL TRUE
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    if(CUMES_ENABLE_EXTRA_SANITIZER_TOOLS)
      add_test(NAME racecheck_${t}
               COMMAND ${CUMES_COMPUTE_SANITIZER}
                       --tool racecheck --error-exitcode 3 $<TARGET_FILE:${t}>)
      set_tests_properties(racecheck_${t} PROPERTIES
          LABELS "sanitizer;racecheck" TIMEOUT 1800 RUN_SERIAL TRUE
          WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      add_test(NAME synccheck_${t}
               COMMAND ${CUMES_COMPUTE_SANITIZER}
                       --tool synccheck --error-exitcode 3 $<TARGET_FILE:${t}>)
      set_tests_properties(synccheck_${t} PROPERTIES
          LABELS "sanitizer;synccheck" TIMEOUT 1800 RUN_SERIAL TRUE
          WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    endif()
  endforeach()
endfunction()

function(cumes_apply_host_sanitizers target)
  if(NOT CUMES_HOST_SANITIZERS)
    return()
  endif()
  # Apply to the `_asan` twin targets only (see CMakeLists.txt): the ASan
  # runtime must be FIRST in the initial library list, which only holds when
  # the sanitized library is linked directly by the sanitized executable —
  # propagating it INTERFACE-wide breaks every CUDA consumer at startup
  # ("ASan runtime does not come first").
  # -fno-sanitize-recover=undefined: UBSan failures must fail the test, not
  # merely print and continue.
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined>)
  target_link_options(${target} PRIVATE -fsanitize=address,undefined)
endfunction()
