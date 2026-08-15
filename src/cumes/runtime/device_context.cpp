// device_context.cpp — DeviceContext implementation (non-template TU).
#include "cumes/runtime/device_context.hpp"

namespace cumes {

void DeviceContext::init(int device_index) {
  if (device_index < 0) {
    check_cuda(cudaGetDevice(&device_index), "DeviceContext::init");
  } else {
    check_cuda(cudaSetDevice(device_index), "DeviceContext::init");
  }
  caps_.device = device_index;

  cudaDeviceProp prop{};
  check_cuda(cudaGetDeviceProperties(&prop, device_index),
             "DeviceContext::init");
  caps_.compute_capability_major = prop.major;
  caps_.compute_capability_minor = prop.minor;

  // Streams are created after the device is selected so they belong to the
  // requested context, not whatever device was current at construction time.
  compute_ = std::make_unique<Stream>();
  aux_ = std::make_unique<Stream>();
}

}  // namespace cumes
