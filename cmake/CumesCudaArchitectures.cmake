# CumesCudaArchitectures.cmake — CUDA architecture selection.
#
# Defaults to the Pascal/Turing/Ampere/Ada set the project has historically
# shipped (61/75/80/86/89), trimmed to what the CONFIGURED toolkit can actually
# compile. The project's floor is CUDA >= 11.0 (CumesDependencies.cmake), and
# nvcc hard-errors on newer archs than it knows:
#   sm_61 / sm_75 / sm_80  supported since CUDA 11.0  (the project minimum)
#   sm_86                  supported since CUDA 11.1
#   sm_89                  supported since CUDA 11.8
# so sm_86/sm_89 are added only when the toolkit version allows. The toolkit
# version is probed here via find_package(CUDAToolkit) — this module runs
# BEFORE project()/enable_language(CUDA) because CMake's CUDA support
# populates CMAKE_CUDA_ARCHITECTURES with its own fallback (sm_52) during
# enable_language if it is unset, and a post-project non-FORCE cache set
# cannot override that already-cached value — so the default below would be
# silently ignored. The value is a cache variable so a developer or CI preset
# can override it (e.g. -DCMAKE_CUDA_ARCHITECTURES=80 for a single-arch native
# build). The guard honours an explicit -DCMAKE_CUDA_ARCHITECTURES (or the
# CMAKE_CUDA_ARCHITECTURES environment variable).

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CMAKE_CUDA_ARCHITECTURES})
  # Base set: every architecture supported by the project's CUDA 11.0 floor.
  set(CUMES_DEFAULT_ARCHS "61;75;80")
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.1")
      list(APPEND CUMES_DEFAULT_ARCHS 86)
    endif()
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8")
      list(APPEND CUMES_DEFAULT_ARCHS 89)
    endif()
  else()
    # The toolkit is not locatable yet (CumesDependencies.cmake's REQUIRED
    # find_package runs after project()). Fall back to the conservative
    # CUDA 11.0-compatible set and say so: silently dropping 86/89 on a
    # nonstandard toolkit layout would be a confusing behaviour change.
    message(STATUS "cuMES: CUDA toolkit not yet locatable; defaulting "
                   "CMAKE_CUDA_ARCHITECTURES to the CUDA 11.0-compatible set "
                   "${CUMES_DEFAULT_ARCHS}. Pass -DCMAKE_CUDA_ARCHITECTURES="
                   "61;75;80;86;89 to include Ampere/Ada on a newer toolkit.")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES "${CUMES_DEFAULT_ARCHS}" CACHE STRING
      "CUDA architectures to build for (semicolon-separated compute capabilities)")
endif()
set_property(CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY STRINGS
    "61;75;80;86;89" "61" "75" "80" "86" "89")
