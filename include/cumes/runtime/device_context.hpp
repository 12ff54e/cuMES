// device_context.hpp — selected device, compute/auxiliary streams, capabilities
// (blueprint §6.4).
//
// Owns the CUDA device selection and two nonblocking streams. It is introduced
// and unit-tested in Phase 3 but NOT yet consumed by the solver: the solver
// keeps the legacy default stream until the Phase 6A scheduling work. The
// capability record is a stub populated with device properties; graph support
// is filled in when CUDA Graphs land (Phase 9).
#pragma once

#include <cuda_runtime.h>

#include <memory>

#include "cumes/runtime/cuda_status.hpp"
#include "cumes/runtime/stream.hpp"

namespace cumes {

struct RuntimeCapabilities {
  int device = -1;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
};

class DeviceContext {
 public:
  DeviceContext() { init(-1); }             // current device
  explicit DeviceContext(int device_index) { init(device_index); }

  ~DeviceContext() = default;

  DeviceContext(const DeviceContext&) = delete;
  DeviceContext& operator=(const DeviceContext&) = delete;
  DeviceContext(DeviceContext&&) noexcept = default;
  DeviceContext& operator=(DeviceContext&&) noexcept = default;

  cudaStream_t compute_stream() const noexcept { return compute_->get(); }
  cudaStream_t auxiliary_stream() const noexcept { return aux_->get(); }
  const RuntimeCapabilities& capabilities() const noexcept { return caps_; }

 private:
  void init(int device_index);

  std::unique_ptr<Stream> compute_;
  std::unique_ptr<Stream> aux_;
  RuntimeCapabilities caps_;
};

}  // namespace cumes
