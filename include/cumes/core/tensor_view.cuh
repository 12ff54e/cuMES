// tensor_view.cuh — non-owning typed views over the device layouts (blueprint
// §6.3).
//
// A view carries a raw pointer plus the extents needed to index it on host or
// device; it never allocates or frees. The domain tags (PhysicalStateDomain,
// DecomposedResidualDomain, DecomposedVelocityDomain, CheckpointDomain) make a
// view's mathematical domain part of its type, so a kernel that expects
// physical coefficients cannot silently receive decomposed residuals.
//
// Layouts (blueprint §4.1) — these match the legacy code bit-for-bit:
//   spectral: [component][mode][surface], surface contiguous
//             (component-major; the same order as d_f_spec and
//              EquilibriumSnapshot::Component).
//   real full: [surface][zeta][theta], theta contiguous
//              (point = zeta*ntheta + theta; idx = surface*nZnT + point).
//   real half: the same with `surfaces = ns-1`.
#pragma once

#include <cuda_runtime.h>  // __host__/__device__ (this header is included by
                           // host-only .cpp TUs, e.g. the output writers)
#include <cstdint>

namespace cumes {

enum class SpectralComponent : std::uint8_t {
    Rcc = 0,  // R: cos(mθ)cos(nζ)
    Zsc = 1,  // Z: sin(mθ)cos(nζ)
    Lsc = 2,  // λ: sin(mθ)cos(nζ)
    Rss = 3,  // R: sin(mθ)sin(nζ)
    Zcs = 4,  // Z: cos(mθ)sin(nζ)
    Lcs = 5,  // λ: cos(mθ)sin(nζ)
    Count = 6,
};

inline constexpr int kSpectralComponentCount = 6;

// Mathematical-domain tags. The same six-component layout is reused for the
// physical state, decomposed residuals, decomposed velocities, and the
// state-only checkpoint; the tag prevents cross-domain misuse.
struct PhysicalStateDomain {};
struct DecomposedResidualDomain {};
struct DecomposedVelocityDomain {};
struct CheckpointDomain {};

// Component-major spectral view: operator()(c, mode, surface).
template <class T, class Domain>
class SpectralView {
   public:
    __host__ __device__ SpectralView() = default;
    __host__ __device__ SpectralView(T* data, int ns, int mnmax)
        : data_(data), ns_(ns), mnmax_(mnmax) {}

    __host__ __device__ T& operator()(SpectralComponent c,
                                      int mode,
                                      int surface) {
        return data_[static_cast<int>(c) * mnmax_ * ns_ + mode * ns_ + surface];
    }
    __host__ __device__ const T& operator()(SpectralComponent c,
                                            int mode,
                                            int surface) const {
        return data_[static_cast<int>(c) * mnmax_ * ns_ + mode * ns_ + surface];
    }

    __host__ __device__ T* data() const { return data_; }
    __host__ __device__ int ns() const { return ns_; }
    __host__ __device__ int mnmax() const { return mnmax_; }

   private:
    T* data_ = nullptr;
    int ns_ = 0;
    int mnmax_ = 0;
};

// Real-space view over [surface][zeta][theta] (theta contiguous). `surfaces` is
// ns for a full-grid view and ns-1 for a half-grid view.
template <class T>
class RealFieldView {
   public:
    __host__ __device__ RealFieldView() = default;
    __host__ __device__
    RealFieldView(T* data, int surfaces, int ntheta, int nzeta)
        : data_(data), surfaces_(surfaces), ntheta_(ntheta), nzeta_(nzeta) {}

    __host__ __device__ T& operator()(int surface, int zeta, int theta) {
        return data_[surface * ntheta_ * nzeta_ + zeta * ntheta_ + theta];
    }
    __host__ __device__ const T& operator()(int surface,
                                            int zeta,
                                            int theta) const {
        return data_[surface * ntheta_ * nzeta_ + zeta * ntheta_ + theta];
    }

    __host__ __device__ T* data() const { return data_; }
    __host__ __device__ int surfaces() const { return surfaces_; }
    __host__ __device__ int ntheta() const { return ntheta_; }
    __host__ __device__ int nzeta() const { return nzeta_; }

   private:
    T* data_ = nullptr;
    int surfaces_ = 0;
    int ntheta_ = 0;
    int nzeta_ = 0;
};

}  // namespace cumes
