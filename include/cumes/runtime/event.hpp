// event.hpp — CUDA event RAII.
//
// Owns one `cudaEvent_t`. Events record/compute dependencies and timings; they
// never implicitly synchronize. Introduced in Phase 3 and not yet wired into
// the solver loop (the current timing pair is a Phase 6A removal).
#pragma once

#include <cuda_runtime.h>

#include "cumes/runtime/cuda_status.hpp"

namespace cumes {

class Event {
 public:
  Event() {
    check_cuda(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming),
               "Event::Event");
  }

  ~Event() {
    if (event_ != nullptr) cudaEventDestroy(event_);
  }

  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;

  Event(Event&& other) noexcept : event_(other.event_) {
    other.event_ = nullptr;
  }

  Event& operator=(Event&& other) noexcept {
    if (this != &other) {
      if (event_ != nullptr) cudaEventDestroy(event_);
      event_ = other.event_;
      other.event_ = nullptr;
    }
    return *this;
  }

  cudaEvent_t get() const { return event_; }

  void record(cudaStream_t stream) const {
    check_cuda(cudaEventRecord(event_, stream), "Event::record");
  }

  void synchronize() const {
    check_cuda(cudaEventSynchronize(event_), "Event::synchronize");
  }

 private:
  cudaEvent_t event_ = nullptr;
};

}  // namespace cumes
