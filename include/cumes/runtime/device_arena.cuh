// device_arena.cuh — one aligned stage allocation with named subspans and a
// liveness/peak report (blueprint §6.4, §6.5).
//
// A DeviceArena owns a single contiguous cudaMalloc'd byte span and carves
// named, aligned subspans out of it. It replaces the per-array cudaMalloc calls
// that the legacy `*Create` workspace constructors make, so a stage allocates
// exactly once up front (no allocator calls in the hot loop) and can report its
// peak memory by category. Subspans are plain `T*` — they carry no ownership —
// and the arena frees everything in one cudaFree when it is released.
//
// This is a host-side primitive: the arena is allocated and carved on the host
// at stage setup, and its raw pointers are passed to kernels unchanged. It never
// synchronizes and adds no device code.
//
// The backing store comes from cudaMalloc (256-byte aligned on every supported
// platform), so aligning each subspan to `align` (a power of two <= 256, the
// default is alignof(T)) keeps every typed pointer correctly aligned.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "cumes/runtime/cuda_status.hpp"
#include "cumes/runtime/device_buffer.cuh"

namespace cumes {

class DeviceArena {
 public:
    // One named subspan: where it lives and how big it is. The report groups
    // these by name so peak bytes can be attributed per category.
    struct SpanInfo {
        std::string name;
        std::size_t offset = 0;  // byte offset into the backing store
        std::size_t bytes = 0;
    };

    DeviceArena() = default;

    virtual ~DeviceArena() = default;

    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    // Move transfers the store, the span table, and the byte counters; the
    // source is reset to a valid empty arena (the scalar counters are plain
    // size_t, which a defaulted move would copy rather than null).
    DeviceArena(DeviceArena&& other) noexcept
        : storage_(std::move(other.storage_)),
          spans_(std::move(other.spans_)),
          total_bytes_(other.total_bytes_),
          used_bytes_(other.used_bytes_),
          peak_bytes_(other.peak_bytes_) {
        other.total_bytes_ = 0;
        other.used_bytes_ = 0;
        other.peak_bytes_ = 0;
    }

    DeviceArena& operator=(DeviceArena&& other) noexcept {
        if (this != &other) {
            release();
            storage_ = std::move(other.storage_);
            spans_ = std::move(other.spans_);
            total_bytes_ = other.total_bytes_;
            used_bytes_ = other.used_bytes_;
            peak_bytes_ = other.peak_bytes_;
            other.total_bytes_ = 0;
            other.used_bytes_ = 0;
            other.peak_bytes_ = 0;
        }
        return *this;
    }

    // Allocate the backing store. `bytes` may be zero (empty arena).
    void allocate(std::size_t bytes) {
        release();
        storage_.allocate(bytes);  // DeviceBuffer<char>: bytes elements of 1
        total_bytes_ = bytes;
        used_bytes_ = 0;
    }

    void release() {
        storage_.release();
        spans_.clear();
        total_bytes_ = 0;
        used_bytes_ = 0;
        peak_bytes_ = 0;
    }

    // Carve a typed subspan of `count` elements, registered under `name`,
    // aligned to `align` bytes (default alignof(T)). Returns a non-owning `T*`
    // into the backing store. Throws CumesError if the arena is exhausted.
    template <class T>
    T* alloc_span(const char* name, std::size_t count,
                  std::size_t align = alignof(T)) {
        if (count == 0) return nullptr;
        return reinterpret_cast<T*>(carve_span(name, count * sizeof(T), align));
    }

    // Zero a previously-carved span (the arena does not know element counts,
    // so the caller passes them). Mirrors DeviceBuffer::zero for a subspan.
    template <class T>
    void zero_span(T* span, std::size_t count) const {
        if (count != 0) {
            check_cuda(cudaMemset(span, 0, count * sizeof(T)),
                       "DeviceArena::zero_span");
        }
    }

    char* data() const { return storage_.data(); }

    // Total backing-store bytes reserved.
    std::size_t total_bytes() const { return total_bytes_; }
    // Bytes carved so far (the high-water mark of a linear arena).
    std::size_t used_bytes() const { return used_bytes_; }
    // Peak live bytes. For a linear arena this equals used_bytes(); it is a
    // distinct accessor so a future reuse-aware arena can lower it.
    std::size_t peak_bytes() const { return peak_bytes_; }
    std::size_t span_count() const { return spans_.size(); }
    const std::vector<SpanInfo>& spans() const { return spans_; }
    bool empty() const { return total_bytes_ == 0; }

 protected:
    // Non-template carve seam: `alloc_span` forwards here, and subclasses may
    // intercept it (the stage-solver's measuring arena throws a typed
    // overflow carrying the byte requirement instead of the generic
    // CumesError). The base implementation is exactly the pre-seam logic —
    // same offsets, same span table, same exception message.
    virtual void* carve_span(const char* name, std::size_t bytes,
                             std::size_t align) {
        std::size_t off = 0;
        if (!carve_offsets(bytes, align, off)) {
            throw CumesError(std::string("DeviceArena::alloc_span: '") + name +
                             "' of " + std::to_string(bytes) +
                             " bytes overflows arena (used=" +
                             std::to_string(used_bytes_) +
                             ", total=" + std::to_string(total_bytes_) + ")");
        }
        spans_.push_back(SpanInfo{std::string(name), off, bytes});
        used_bytes_ = off + bytes;
        if (used_bytes_ > peak_bytes_) peak_bytes_ = used_bytes_;
        return storage_.data() + off;
    }

    // Validates `align` and computes the aligned offset for a `bytes` span at
    // the current used offset; returns false when it would overflow.
    bool carve_offsets(std::size_t bytes, std::size_t align,
                       std::size_t& off) const {
        if (align == 0 || (align & (align - 1)) != 0) {
            throw CumesError("DeviceArena: alignment must be a non-zero power of two");
        }
        off = (used_bytes_ + align - 1) & ~(align - 1);
        return !(off > total_bytes_ || bytes > total_bytes_ - off);
    }

 private:
    DeviceBuffer<char> storage_;
    std::vector<SpanInfo> spans_;
    std::size_t total_bytes_ = 0;
    std::size_t used_bytes_ = 0;
    std::size_t peak_bytes_ = 0;
};

}  // namespace cumes
