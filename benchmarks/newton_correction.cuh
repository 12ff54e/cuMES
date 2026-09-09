// Compatibility aliases for the shared production correction operators.
#ifndef CUMES_BENCHMARKS_NEWTON_CORRECTION_CUH_
#define CUMES_BENCHMARKS_NEWTON_CORRECTION_CUH_

#include "correction_coordinates.cuh"
#include "cumes/numerics/newton_correction.hpp"
#include "device_gmres.cuh"

namespace block_bench {
using cumes::DifferenceScheme;
using cumes::NewtonCorrection;
}  // namespace block_bench

#endif  // CUMES_BENCHMARKS_NEWTON_CORRECTION_CUH_
