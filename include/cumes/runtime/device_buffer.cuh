// device_buffer.cuh — movable, non-copyable RAII device allocation.
//
// Owns a cudaMalloc'd span of `T`. Move-only: moving transfers the raw pointer
// and nulls the source, so assigning a freshly-allocated buffer over a live one
// frees the old allocation exactly once. No allocation happens in the hot loop:
// a DeviceBuffer is constructed once per stage and its `data()` is passed to
// the kernels unchanged (same layout, same pointer arithmetic).
#ifndef CUMES_INCLUDE_CUMES_RUNTIME_DEVICE_BUFFER_CUH_
#define CUMES_INCLUDE_CUMES_RUNTIME_DEVICE_BUFFER_CUH_

#include "cumes/runtime/cuda_status.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace cumes {

template <class T>
class DeviceBuffer {
   public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) { allocate(count); }

    // Non-owning view over externally-managed device memory (an arena-carved
    // span): release() leaves the memory untouched. `data` may be nullptr only
    // when `count == 0`.
    DeviceBuffer(T* data, std::size_t count) : data_(data), count_(count) {
        owned_ = false;
    }

    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(other.data_), count_(other.count_), owned_(other.owned_) {
        other.data_ = nullptr;
        other.count_ = 0;
        other.owned_ = true;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            count_ = other.count_;
            owned_ = other.owned_;
            other.data_ = nullptr;
            other.count_ = 0;
            other.owned_ = true;
        }
        return *this;
    }

    void allocate(std::size_t count) {
        release();
        if (count == 0) return;
        check_cuda(cudaMalloc(&data_, count * sizeof(T)),
                   "DeviceBuffer::allocate");
        count_ = count;
        owned_ = true;
    }

    void release() {
        // A free failure is not recoverable and must not throw from a
        // destructor path; the allocation/transfer/copy entry points are the
        // checked boundary.
        if (data_ != nullptr && owned_) { cudaFree(data_); }
        data_ = nullptr;
        count_ = 0;
        owned_ = true;
    }

    void zero() {
        if (count_ != 0) {
            check_cuda(cudaMemset(data_, 0, count_ * sizeof(T)),
                       "DeviceBuffer::zero");
        }
    }

    // One device-to-device copy; requires identical element counts.
    void copy_from(const DeviceBuffer& other) {
        if (other.count_ != count_) {
            throw CumesError("DeviceBuffer::copy_from: size mismatch");
        }
        if (count_ == 0) return;
        check_cuda(cudaMemcpy(data_, other.data_, count_ * sizeof(T),
                              cudaMemcpyDeviceToDevice),
                   "DeviceBuffer::copy_from");
    }

    // Stream-ordered variants (Phase 6A): the solver's state checkpoint/restore
    // and velocity zero run after the descent kernel on the same compute
    // stream, so they must be enqueued asynchronously rather than as blocking
    // default- stream operations (which would race with the nonblocking
    // stream).
    void zero_async(cudaStream_t stream) {
        if (count_ != 0) {
            check_cuda(cudaMemsetAsync(data_, 0, count_ * sizeof(T), stream),
                       "DeviceBuffer::zero_async");
        }
    }

    void copy_from_async(const DeviceBuffer& other, cudaStream_t stream) {
        if (other.count_ != count_) {
            throw CumesError("DeviceBuffer::copy_from_async: size mismatch");
        }
        if (count_ == 0) return;
        check_cuda(cudaMemcpyAsync(data_, other.data_, count_ * sizeof(T),
                                   cudaMemcpyDeviceToDevice, stream),
                   "DeviceBuffer::copy_from_async");
    }

    T* data() const { return data_; }
    std::size_t size() const { return count_; }
    std::size_t byte_size() const { return count_ * sizeof(T); }
    bool empty() const { return count_ == 0; }

   private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
    bool owned_ = true;
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_RUNTIME_DEVICE_BUFFER_CUH_
