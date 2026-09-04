if(NOT DEFINED CUMES_BINARY_DIR OR NOT DEFINED CUMES_SOURCE_DIR)
  message(FATAL_ERROR "package consumer test requires cuMES source/binary dirs")
endif()

set(test_root "${CUMES_BINARY_DIR}/package-consumer-test")
set(install_prefix "${test_root}/install")
set(consumer_build "${test_root}/build")
file(REMOVE_RECURSE "${test_root}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${CUMES_BINARY_DIR}"
          --prefix "${install_prefix}"
  RESULT_VARIABLE install_result)
if(NOT install_result EQUAL 0)
  message(FATAL_ERROR "cuMES package installation failed")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
          -S "${CUMES_SOURCE_DIR}/tests/package_consumer"
          -B "${consumer_build}"
          -DCMAKE_PREFIX_PATH=${install_prefix}
  RESULT_VARIABLE configure_result)
if(NOT configure_result EQUAL 0)
  message(FATAL_ERROR "cuMES package consumer configuration failed")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${consumer_build}"
  RESULT_VARIABLE build_result)
if(NOT build_result EQUAL 0)
  message(FATAL_ERROR "cuMES package consumer build failed")
endif()
