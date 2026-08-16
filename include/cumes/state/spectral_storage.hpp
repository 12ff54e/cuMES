// spectral_storage.hpp — owning contiguous state/velocity slabs (blueprint
// §6.5).
//
// The legacy SpectralState<T> (a non-owning bundle of 12 raw pointers, each a
// separate `[mode][surface]` array) is gone (migration step 13.3). SpectralStorage
// co-locates those twelve arrays into two component-major slabs — `state`
// (6*mnmax*ns) and `velocity` (6*mnmax*ns) — in the exact order
// Rcc Zsc Lsc Rss Zcs Lcs; family_ptr()/velocity_family_ptr() return the 12
// per-family slab pointers, so every existing consumer and kernel keeps its
// indexing and arithmetic unchanged (bitwise identical). The whole
// state/velocity is one contiguous span, which turns the solver's six D2D
// backup copies into one and its six velocity memsets into one.
#pragma once

#include <cstddef>

#include "cumes/core/tensor_view.cuh"
#include "cumes/runtime/device_buffer.cuh"
#include "vmec_types.h"

namespace cumes {

template <class T>
class SpectralStorage {
 public:
  SpectralStorage() = default;

  explicit SpectralStorage(int ns, int mnmax) { allocate(ns, mnmax); }

  SpectralStorage(SpectralStorage&&) noexcept = default;
  SpectralStorage& operator=(SpectralStorage&&) noexcept = default;
  SpectralStorage(const SpectralStorage&) = delete;
  SpectralStorage& operator=(const SpectralStorage&) = delete;
  ~SpectralStorage() = default;

  void allocate(int ns, int mnmax) {
    ns_ = ns;
    mnmax_ = mnmax;
    const std::size_t count = static_cast<std::size_t>(6) * ns * mnmax;
    state_.allocate(count);
    velocity_.allocate(count);
    // Both slabs start zeroed: matches the legacy `new T[n]()` cold start and
    // the zeroed velocities of Prolongation::enqueue.
    state_.zero();
    velocity_.zero();
  }

  int ns() const { return ns_; }
  int mnmax() const { return mnmax_; }
  bool empty() const { return state_.empty(); }

  T* state_slab() const { return state_.data(); }
  T* velocity_slab() const { return velocity_.data(); }

  // Owning-buffer access for the single-copy checkpoint/restore in the solver.
  DeviceBuffer<T>& state_buffer() { return state_; }
  DeviceBuffer<T>& velocity_buffer() { return velocity_; }
  const DeviceBuffer<T>& state_buffer() const { return state_; }
  const DeviceBuffer<T>& velocity_buffer() const { return velocity_; }

  // The six per-family slab pointers (component-major order
  // Rcc Zsc Lsc Rss Zcs Lcs — the legacy SpectralState field order and
  // EquilibriumSnapshot::Component). Each family is the contiguous
  // [mnmax][ns] block at slab + c*ns*mnmax, indexed [mode*ns + j] exactly as
  // the legacy separate arrays were.
  T* family_ptr(SpectralComponent c) const {
    return state_.data() + static_cast<std::size_t>(static_cast<int>(c)) * ns_ *
                               mnmax_;
  }
  T* velocity_family_ptr(SpectralComponent c) const {
    return velocity_.data() +
           static_cast<std::size_t>(static_cast<int>(c)) * ns_ * mnmax_;
  }

  SpectralView<T, PhysicalStateDomain> physical() const {
    return SpectralView<T, PhysicalStateDomain>(state_.data(), ns_, mnmax_);
  }

  SpectralView<T, DecomposedVelocityDomain> velocity() const {
    return SpectralView<T, DecomposedVelocityDomain>(velocity_.data(), ns_,
                                                     mnmax_);
  }

  // Read-only spectral views (the const-input side of the operator boundaries).
  // `data()` returns T* even on a const DeviceBuffer, so these bind T* to the
  // const view's `const T*` constructor directly.
  SpectralView<const T, PhysicalStateDomain> physical_const() const {
    return SpectralView<const T, PhysicalStateDomain>(state_.data(), ns_,
                                                      mnmax_);
  }

  SpectralView<const T, DecomposedVelocityDomain> velocity_const() const {
    return SpectralView<const T, DecomposedVelocityDomain>(velocity_.data(), ns_,
                                                           mnmax_);
  }

 private:
  DeviceBuffer<T> state_;
  DeviceBuffer<T> velocity_;
  int ns_ = 0;
  int mnmax_ = 0;
};

}  // namespace cumes
