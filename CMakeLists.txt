cmake_minimum_required(VERSION 3.20)

# Use g++-12 as CUDA host compiler (CUDA 12.1 doesn't support gcc > 12).
# Must be set BEFORE project() or as a cache variable.
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12 CACHE FILEPATH "CUDA host compiler")

project(cuMES VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "61;75;80;86;89")

# Optimisation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

find_package(CUDAToolkit 11 REQUIRED)

# ---- executable -------------------------------------------------------
add_executable(cuMES
    src/main.cu
    src/fourier.cu
    src/geometry.cu
    src/forces.cu
    src/solver.cu
    src/output.cu
    src/profiles.cu
    src/precon.cu
    src/constraint.cu
)

target_link_libraries(cuMES PRIVATE
    CUDA::cublas
)

target_include_directories(cuMES PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- tests (standalone executables, no framework) ----------------------
add_executable(test_fourier
    tests/test_fourier.cu
    src/fourier.cu
)
target_link_libraries(test_fourier PRIVATE CUDA::cublas)
target_include_directories(test_fourier PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_forces
    tests/test_forces.cu
    src/fourier.cu
    src/geometry.cu
    src/forces.cu
    src/profiles.cu
)
target_link_libraries(test_forces PRIVATE CUDA::cublas)
target_include_directories(test_forces PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_force_verify
    tests/test_force_verify.cu
    src/fourier.cu
    src/geometry.cu
    src/forces.cu
    src/profiles.cu
)
target_link_libraries(test_force_verify PRIVATE CUDA::cublas)
target_include_directories(test_force_verify PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
