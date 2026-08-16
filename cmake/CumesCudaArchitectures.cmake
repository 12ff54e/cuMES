# CumesCudaArchitectures.cmake — CUDA architecture selection.
#
# Defaults to the Pascal/Turing/Ampere/Ada set the project has historically
# shipped (61/75/80/86/89). The value is a cache variable so a developer or CI
# preset can override it (e.g. -DCMAKE_CUDA_ARCHITECTURES=80 for a single-arch
# native build, or a legacy CUDA 11 profile retaining only sm_61). Compatibility
# policy (which toolkit supports which arch) is documented in docs/performance.md
# rather than asserted here.
#
# This module must be included BEFORE project()/enable_language(CUDA). CMake's
# CUDA support populates CMAKE_CUDA_ARCHITECTURES with its own fallback (sm_52)
# during enable_language if it is unset, and a post-project non-FORCE cache set
# cannot override that already-cached value — so the default below would be
# silently ignored. The guard honours an explicit -DCMAKE_CUDA_ARCHITECTURES
# (or the CMAKE_CUDA_ARCHITECTURES environment variable).

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CMAKE_CUDA_ARCHITECTURES})
  set(CMAKE_CUDA_ARCHITECTURES "61;75;80;86;89" CACHE STRING
      "CUDA architectures to build for (semicolon-separated compute capabilities)")
endif()
set_property(CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY STRINGS
    "61;75;80;86;89" "61" "75" "80" "86" "89")
