// Restarted GMRES with device Arnoldi, reorthogonalization and Givens QR.
#ifndef CUMES_INCLUDE_CUMES_NUMERICS_DEVICE_GMRES_HPP_
#define CUMES_INCLUDE_CUMES_NUMERICS_DEVICE_GMRES_HPP_

#include "cumes/runtime/device_buffer.cuh"

#include <algorithm>
#include <cmath>

namespace cumes {

template <class T>
struct GmresControl {
    using val_type = T;
    T residual_norm = T(0), norm_squared = T(0), target_squared = T(0);
    T rhs_norm = T(0), target_norm = T(0);
    int steps = 0, cycles = 0;
    int active = 0, converged = 0;
    // 0: no breakdown; 1: singular projected system; 2: nonfinite arithmetic.
    int breakdown = 0;
    int cycle_active = 0, cycle_steps = 0, cycle_started = 0;
    int update_ready = 0;
    int reserved = 0;  // keep the double record free of tail padding
};

template <class T>
class DeviceGmres {
   public:
    using val_type = T;

    explicit DeviceGmres(int size, int max_basis = 32);

    // apply(d_input,d_output,stream) is a linear map without a host fence.
    // b and x must be nonoverlapping size-element vectors. x starts at zero.
    // The fixed submission contains iterations Arnoldi maps and one true
    // residual map per restart cycle, including inactive maps after stopping.
    // Check converged/breakdown and residual_norm, not active alone. All maps
    // and this object's lifetime must use the same ordered compute stream.
    template <class Apply>
    void enqueue(Apply&& apply,
                 const T* d_b,
                 T* d_x,
                 int iterations,
                 T tolerance,
                 cudaStream_t stream) {
        if (!d_b || !d_x || d_b == d_x || iterations < 1 ||
            !(tolerance > T(0)) || !std::isfinite(tolerance) ||
            tolerance >= T(1))
            throw CumesError("GMRES: invalid vectors, budget or tolerance");
        enqueue_initialize(d_b, d_x, tolerance, stream);
        for (int completed = 0; completed < iterations;) {
            const int count = std::min(max_basis_, iterations - completed);
            enqueue_begin_cycle(stream);
            for (int j = 0; j < count; ++j) {
                apply(d_basis_.data() + static_cast<std::size_t>(j) * size_,
                      d_work_.data(), stream);
                enqueue_arnoldi(j, stream);
            }
            enqueue_update(d_x, stream);
            apply(d_x, d_work_.data(), stream);
            enqueue_finish_cycle(d_b, stream);
            completed += count;
        }
        check_cuda(cudaGetLastError(), "GMRES");
    }

    const GmresControl<T>* control_device() const { return d_control_.data(); }

   private:
    static int check_size(int size);
    static int check_basis(int basis);
    static std::size_t basis_count(int size, int basis);

    void enqueue_initialize(const T* d_b,
                            T* d_x,
                            T tolerance,
                            cudaStream_t stream);
    void enqueue_begin_cycle(cudaStream_t stream);
    void enqueue_arnoldi(int column, cudaStream_t stream);
    void enqueue_update(T* d_x, cudaStream_t stream);
    void enqueue_finish_cycle(const T* d_b, cudaStream_t stream);

    int size_, max_basis_;
    DeviceBuffer<T> d_basis_, d_work_, d_residual_, d_hessenberg_;
    DeviceBuffer<T> d_projection_, d_cosine_, d_sine_, d_g_, d_y_;
    DeviceBuffer<GmresControl<T>> d_control_;
};

}  // namespace cumes
#endif  // CUMES_INCLUDE_CUMES_NUMERICS_DEVICE_GMRES_HPP_
