// pinned_buffer.hpp — movable, non-copyable pinned host allocation.
//
// Host mirror for async D2H control records. `cudaMallocHost` page-locks the
// span so `cudaMemcpyAsync` to/from it stays asynchronous; the buffer is freed
// with `cudaFreeHost` on destruction.
#ifndef CUMES_INCLUDE_CUMES_RUNTIME_PINNED_BUFFER_HPP_
#define CUMES_INCLUDE_CUMES_RUNTIME_PINNED_BUFFER_HPP_

#include "cumes/runtime/cuda_status.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace cumes {

template <class T>
class PinnedBuffer {
   public:
    using val_type = T;

    PinnedBuffer() = default;

    explicit PinnedBuffer(std::size_t count) { allocate(count); }

    ~PinnedBuffer() { release(); }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    PinnedBuffer(PinnedBuffer&& other) noexcept
        : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }

    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void allocate(std::size_t count) {
        release();
        if (count == 0) return;
        check_cuda(cudaMallocHost(&data_, count * sizeof(T)),
                   "PinnedBuffer::allocate");
        count_ = count;
    }

    void release() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
            data_ = nullptr;
        }
        count_ = 0;
    }

    T* data() const { return data_; }
    std::size_t size() const { return count_; }

   private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_RUNTIME_PINNED_BUFFER_HPP_
