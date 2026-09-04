// cuda_capture_coordination.hpp — process-wide coordination for CUDA runtime
// operations that are illegal while another thread captures a stream.
#ifndef CUMES_INCLUDE_CUMES_RUNTIME_CUDA_CAPTURE_COORDINATION_HPP_
#define CUMES_INCLUDE_CUMES_RUNTIME_CUDA_CAPTURE_COORDINATION_HPP_

#include <mutex>

namespace cumes {

inline std::mutex& cuda_capture_coordination_mutex() {
    static std::mutex mutex;
    return mutex;
}

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_RUNTIME_CUDA_CAPTURE_COORDINATION_HPP_
