cmake_minimum_required(VERSION 3.20)

project(cumes-package-consumer LANGUAGES C CXX)

find_package(cuMES 1.3 CONFIG REQUIRED)

add_executable(cumes_package_consumer main.cpp)
target_compile_features(cumes_package_consumer PRIVATE cxx_std_20)
target_link_libraries(cumes_package_consumer PRIVATE
  cumes::solver cumes::config_json)
