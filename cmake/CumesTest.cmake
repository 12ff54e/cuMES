# CumesTest.cmake — CTest registration helper.
#
# `cumes_add_test(<name> <labels>)` registers a test executable target by that
# name with the given semicolon-separated labels. Every registered test runs
# from the source directory (they read inputs/*.json relative to the CWD) and
# exits nonzero on failure, so a ctest run distinguishes a verified result from
# a diagnostic.

function(cumes_add_test name)
  # labels = ARGN (semicolon-joined)
  string(JOIN ";" labels ${ARGN})
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES
      LABELS "${labels}" TIMEOUT 300
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endfunction()
