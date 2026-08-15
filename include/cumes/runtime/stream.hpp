// stream.hpp — nonblocking CUDA stream RAII.
//
// Owns one `cudaStream_t` created with `cudaStreamNonBlocking`. Introduced in
// Phase 3; the solver still runs on the legacy default stream until the Phase
// 6A control-path performance work wires explicit streams into cuFFT/kernels.
#pragma once

#include <cuda_runtime.h>

#include "cumes/runtime/cuda_status.hpp"

namespace cumes {

class Stream {
 public:
  Stream() {
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "Stream::Stream");
  }

  ~Stream() {
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  Stream(Stream&& other) noexcept : stream_(other.stream_) {
    other.stream_ = nullptr;
  }

  Stream& operator=(Stream&& other) noexcept {
    if (this != &other) {
      if (stream_ != nullptr) cudaStreamDestroy(stream_);
      stream_ = other.stream_;
      other.stream_ = nullptr;
    }
    return *this;
  }

  cudaStream_t get() const { return stream_; }

  void synchronize() const {
    check_cuda(cudaStreamSynchronize(stream_), "Stream::synchronize");
  }

 private:
  cudaStream_t stream_ = nullptr;
};

}  // namespace cumes
