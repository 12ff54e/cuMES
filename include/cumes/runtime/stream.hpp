// stream.hpp — nonblocking CUDA stream RAII.
//
// Owns one `cudaStream_t` created with `cudaStreamNonBlocking`. The solver's
// hot loop runs on one such stream (Phase 6A); main.cu creates its own.
#ifndef CUMES_INCLUDE_CUMES_RUNTIME_STREAM_HPP_
#define CUMES_INCLUDE_CUMES_RUNTIME_STREAM_HPP_

#include "cumes/runtime/cuda_status.hpp"

#include <cuda_runtime.h>

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
        // A moved-from Stream must not silently sync the legacy default stream:
        // cudaStreamSynchronize(0) would report success without waiting for any
        // of the (nonblocking) compute-stream work. Fail loudly instead.
        if (stream_ == nullptr) {
            throw CumesError(
                "Stream::synchronize: stream is null (moved-from)");
        }
        check_cuda(cudaStreamSynchronize(stream_), "Stream::synchronize");
    }

   private:
    cudaStream_t stream_ = nullptr;
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_RUNTIME_STREAM_HPP_
