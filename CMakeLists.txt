cmake_minimum_required(VERSION 3.20)

# Use g++-12 as CUDA host compiler (CUDA 12.1 doesn't support gcc > 12).
# Must be set BEFORE project() or as a cache variable.
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12 CACHE FILEPATH "CUDA host compiler")

project(cuMES VERSION 0.1.0 LANGUAGES CXX CUDA)

# C++20: JsonParser.h (JSON input, src/input_json.cu) uses requires-clauses
# and concepts; CUDA 12.1 nvcc compiles the whole project at C++20.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "61;75;80;86;89")

# Optimisation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")

# Precision: all computation is templated on T (double or float); the
# executable's `Real` type is chosen here (the tests instantiate both
# explicitly and are unaffected).
option(CUMES_USE_FLOAT "Build cuMES in single precision (float)" OFF)

find_package(CUDAToolkit 11 REQUIRED)

# ---- executable -------------------------------------------------------
add_executable(cuMES
    src/main.cu
    src/input_json.cu
    src/fourier.cu
    src/geometry.cu
    src/forces.cu
    src/solver.cu
    src/output.cu
    src/profiles.cu
    src/precon.cu
    src/constraint.cu
    src/refine.cu
)

target_link_libraries(cuMES PRIVATE
    CUDA::cublas
    CUDA::cufft
)

target_include_directories(cuMES PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CUMES_USE_FLOAT)
    target_compile_definitions(cuMES PRIVATE CUMES_USE_FLOAT)
endif()

# ---- tests (standalone executables, no framework) ----------------------
add_executable(test_fourier
    tests/test_fourier.cu
    src/fourier.cu
)

add_executable(test_input_json
    tests/test_input_json.cu
    src/input_json.cu
)
target_include_directories(test_input_json PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_fourier PRIVATE CUDA::cublas CUDA::cufft)
target_include_directories(test_fourier PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_forces
    tests/test_forces.cu
    src/input_json.cu
    src/fourier.cu
    src/geometry.cu
    src/forces.cu
    src/profiles.cu
)
target_link_libraries(test_forces PRIVATE CUDA::cublas CUDA::cufft)
target_include_directories(test_forces PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_force_verify
    tests/test_force_verify.cu
    src/input_json.cu
    src/fourier.cu
    src/geometry.cu
    src/forces.cu
    src/profiles.cu
)

add_executable(test_geometry_iso
    tests/test_geometry_iso.cu
    src/input_json.cu
    src/fourier.cu
    src/geometry.cu
    src/profiles.cu
)
target_link_libraries(test_geometry_iso PRIVATE CUDA::cublas CUDA::cufft)
target_include_directories(test_geometry_iso PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_force_verify PRIVATE CUDA::cublas CUDA::cufft)
target_include_directories(test_force_verify PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
