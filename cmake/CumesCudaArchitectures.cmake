# CumesCudaArchitectures.cmake — CUDA architecture selection.
#
# Defaults to the Pascal/Turing/Ampere/Ada set the project has historically
# shipped (61/75/80/86/89). The value is a cache variable so a developer or CI
# preset can override it (e.g. -DCMAKE_CUDA_ARCHITECTURES=80 for a single-arch
# native build, or a legacy CUDA 11 profile retaining only sm_61). Compatibility
# policy (which toolkit supports which arch) is documented in docs/performance.md
# rather than asserted here.

set(CMAKE_CUDA_ARCHITECTURES "61;75;80;86;89" CACHE STRING
    "CUDA architectures to build for (semicolon-separated compute capabilities)")
set_property(CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY STRINGS
    "61;75;80;86;89" "61" "75" "80" "86" "89")
