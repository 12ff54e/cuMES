// cuda_status.hpp — centralized CUDA/cuFFT status handling (blueprint §6.4).
//
// One throwing boundary for the whole runtime: `check_cuda`/`check_cufft`
// convert a non-success status into a `CumesError` (a std::runtime_error the
// application boundary catches), replacing the per-file `checkCuda`/`cc`/`ccf`
// helpers that used to `exit(1)`. `cuda_status` is the non-throwing `Result`
// form for code that wants to propagate status instead of throwing.
#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

#include <stdexcept>
#include <string>

#include "cumes/core/result.hpp"

namespace cumes {

class CumesError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

inline std::string cuda_error_string(cudaError_t err) {
  return std::string(cudaGetErrorString(err));
}

inline std::string cufft_error_string(cufftResult result) {
  switch (result) {
    case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
    case CUFFT_PARSE_ERROR: return "CUFFT_PARSE_ERROR";
    case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_LICENSE_ERROR: return "CUFFT_LICENSE_ERROR";
    case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
    default: return "CUFFT_UNKNOWN(" + std::to_string(static_cast<int>(result)) + ")";
  }
}

// Throws CumesError when `err != cudaSuccess`; returns otherwise.
inline void check_cuda(cudaError_t err, const char* tag) {
  if (err != cudaSuccess) {
    throw CumesError(std::string(tag) + ": CUDA error: " +
                     cuda_error_string(err));
  }
}

// Throws CumesError when `result != CUFFT_SUCCESS`; returns otherwise.
inline void check_cufft(cufftResult result, const char* tag) {
  if (result != CUFFT_SUCCESS) {
    throw CumesError(std::string(tag) + ": cuFFT error: " +
                     cufft_error_string(result));
  }
}

// Non-throwing status form for the Result boundary.
inline Status cuda_status(cudaError_t err, const char* tag) {
  if (err != cudaSuccess) {
    return Status(std::string(tag) + ": CUDA error: " + cuda_error_string(err));
  }
  return Status();
}

}  // namespace cumes
