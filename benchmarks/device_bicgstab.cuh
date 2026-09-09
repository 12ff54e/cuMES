// Experimental matrix-free inner solve. Vectors, reductions and scalar
// recurrence stay on device; the host only submits a fixed iteration budget.
#ifndef CUMES_BENCHMARKS_DEVICE_BICGSTAB_CUH_
#define CUMES_BENCHMARKS_DEVICE_BICGSTAB_CUH_

#include "cumes/runtime/device_buffer.cuh"

namespace block_bench {

template <class T>
struct BicgControl {
    using val_type = T;
    T rho = T(1), previous_rho = T(1);
    T alpha = T(1), omega = T(1), beta = T(0);
    T norm_squared = T(0), target_squared = T(0);
    int active = 1;
    int iterations = 0;
};

template <class T>
__device__ T block_sum(T value) {
    __shared__ T values[256];
    values[threadIdx.x] = value;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride)
            values[threadIdx.x] += values[threadIdx.x + stride];
        __syncthreads();
    }
    return values[0];
}

template <class T>
__global__ void bicg_initialize(int size,
                                const T* d_b,
                                T* d_x,
                                T* d_r,
                                T* d_shadow,
                                T* d_p,
                                T* d_v,
                                BicgControl<T>* d_control,
                                T tolerance) {
    T norm_squared = T(0);
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        const T b = d_b[i];
        d_x[i] = d_p[i] = d_v[i] = T(0);
        d_r[i] = d_shadow[i] = b;
        norm_squared += b * b;
    }
    norm_squared = block_sum(norm_squared);
    if (threadIdx.x == 0) {
        *d_control = BicgControl<T>{};
        d_control->norm_squared = norm_squared;
        d_control->target_squared = norm_squared * tolerance * tolerance;
        d_control->active = norm_squared > T(0);
    }
}

template <class T>
__global__ void bicg_rho(int size,
                         const T* d_r,
                         const T* d_shadow,
                         BicgControl<T>* d_control) {
    if (!d_control->active) return;
    T rho = T(0);
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        rho += d_r[i] * d_shadow[i];
    rho = block_sum(rho);
    if (threadIdx.x == 0) {
        if (!isfinite(rho) || rho == T(0) || d_control->omega == T(0)) {
            d_control->active = 0;
            return;
        }
        d_control->rho = rho;
        d_control->beta =
            rho / d_control->previous_rho * d_control->alpha / d_control->omega;
    }
}

template <class T>
__global__ void bicg_direction(int size,
                               const T* d_r,
                               T* d_p,
                               const T* d_v,
                               const BicgControl<T>* d_control) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size && d_control->active)
        d_p[i] =
            d_r[i] + d_control->beta * (d_p[i] - d_control->omega * d_v[i]);
}

template <class T>
__global__ void bicg_alpha(int size,
                           const T* d_shadow,
                           const T* d_v,
                           BicgControl<T>* d_control) {
    if (!d_control->active) return;
    T denominator = T(0);
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        denominator += d_shadow[i] * d_v[i];
    denominator = block_sum(denominator);
    if (threadIdx.x == 0) {
        if (!isfinite(denominator) || denominator == T(0))
            d_control->active = 0;
        else
            d_control->alpha = d_control->rho / denominator;
    }
}

template <class T>
__global__ void bicg_intermediate(int size,
                                  const T* d_r,
                                  const T* d_v,
                                  T* d_s,
                                  const BicgControl<T>* d_control) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        d_s[i] = d_control->active ? d_r[i] - d_control->alpha * d_v[i] : T(0);
}

template <class T>
__global__ void bicg_omega(int size,
                           const T* d_s,
                           const T* d_t,
                           BicgControl<T>* d_control) {
    if (!d_control->active) return;
    T numerator = T(0), denominator = T(0);
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        numerator += d_s[i] * d_t[i];
        denominator += d_t[i] * d_t[i];
    }
    numerator = block_sum(numerator);
    __syncthreads();
    denominator = block_sum(denominator);
    if (threadIdx.x == 0) {
        // A zero intermediate residual finishes with the alpha update.
        d_control->omega = denominator > T(0) ? numerator / denominator : T(0);
        if (!isfinite(d_control->omega)) d_control->active = 0;
    }
}

template <class T>
__global__ void bicg_update(int size,
                            T* d_x,
                            T* d_r,
                            const T* d_p,
                            const T* d_s,
                            const T* d_t,
                            const BicgControl<T>* d_control) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size && d_control->active) {
        d_x[i] += d_control->alpha * d_p[i] + d_control->omega * d_s[i];
        d_r[i] = d_s[i] - d_control->omega * d_t[i];
    }
}

template <class T>
__global__ void bicg_finish(int size, const T* d_r, BicgControl<T>* d_control) {
    if (!d_control->active) return;
    T norm_squared = T(0);
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        norm_squared += d_r[i] * d_r[i];
    norm_squared = block_sum(norm_squared);
    if (threadIdx.x == 0) {
        d_control->previous_rho = d_control->rho;
        d_control->norm_squared = norm_squared;
        ++d_control->iterations;
        d_control->active =
            isfinite(norm_squared) && norm_squared > d_control->target_squared;
    }
}

template <class T>
class DeviceBicgstab {
   public:
    using val_type = T;

    explicit DeviceBicgstab(int size)
        : size_(size), d_vectors_(6 * size), d_control_(1) {}

    // apply(d_input, d_output, stream) must implement a linear map with the
    // same physical-vector convention as b and x, without host fences.
    template <class Apply>
    void enqueue(Apply&& apply,
                 const T* d_b,
                 T* d_x,
                 int iterations,
                 T tolerance,
                 cudaStream_t stream) {
        T* d_r = d_vectors_.data();
        T* d_shadow = d_r + size_;
        T* d_p = d_shadow + size_;
        T* d_v = d_p + size_;
        T* d_s = d_v + size_;
        T* d_t = d_s + size_;
        auto* d_ctl = d_control_.data();
        const int blocks = (size_ + 255) / 256;
        bicg_initialize<<<1, 256, 0, stream>>>(size_, d_b, d_x, d_r, d_shadow,
                                               d_p, d_v, d_ctl, tolerance);
        for (int i = 0; i < iterations; ++i) {
            bicg_rho<<<1, 256, 0, stream>>>(size_, d_r, d_shadow, d_ctl);
            bicg_direction<<<blocks, 256, 0, stream>>>(size_, d_r, d_p, d_v,
                                                       d_ctl);
            apply(d_p, d_v, stream);
            bicg_alpha<<<1, 256, 0, stream>>>(size_, d_shadow, d_v, d_ctl);
            bicg_intermediate<<<blocks, 256, 0, stream>>>(size_, d_r, d_v, d_s,
                                                          d_ctl);
            apply(d_s, d_t, stream);
            bicg_omega<<<1, 256, 0, stream>>>(size_, d_s, d_t, d_ctl);
            bicg_update<<<blocks, 256, 0, stream>>>(size_, d_x, d_r, d_p, d_s,
                                                    d_t, d_ctl);
            bicg_finish<<<1, 256, 0, stream>>>(size_, d_r, d_ctl);
        }
        cumes::check_cuda(cudaGetLastError(), "experimental BiCGStab");
    }

    const BicgControl<T>* control_device() const { return d_control_.data(); }

   private:
    int size_;
    cumes::DeviceBuffer<T> d_vectors_;
    cumes::DeviceBuffer<BicgControl<T>> d_control_;
};

}  // namespace block_bench

#endif  // CUMES_BENCHMARKS_DEVICE_BICGSTAB_CUH_
