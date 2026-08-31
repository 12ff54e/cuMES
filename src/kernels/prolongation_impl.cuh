// kernels/prolongation_impl.cuh — template definitions for prolongation.hpp.
// Included once per scalar type by prolongation_double.cu /
// prolongation_float.cu; see the explicit-instantiation split
// (cumes_cuda_double / cumes_cuda_float).
#ifndef CUMES_SRC_PROLONGATION_IMPL_CUH_
#define CUMES_SRC_PROLONGATION_IMPL_CUH_
// prolongation.cu — grid-sequencing state interpolation (multi-radial-grid).
//
// Mirrors vmecpp's Vmec::InterpolateToNextMultigridStep (vmec.cc:1795-2042),
// LINEAR scheme: each spectral coefficient is a function of the radial
// index s = j/(ns-1), interpolated 2-point linearly in s. Odd-m modes are
// interpolated in scalxc-decomposed space (xc = c * scalxc, scalxc =
// 1/max(sqrt(s), sqrt(1/(ns-1)))) so the s^(m/2) near-axis behaviour is
// regularized; the result is un-decomposed by the NEW grid's scalxc. The
// odd-m axis value is extrapolated 2*x[1]-x[2] (decomposed) and feeds the
// first interior rows, then ALL odd-m values are zeroed at the new axis.
// At s=1 the stencil weight is {1,0} with scalxc=1, so the LCFS coefficients
// are copied exactly and the fixed boundary stays pinned.
//
// The stored state is physical (unscaled) on both grids, matching cuMES's
// SpectralStorage convention (the odd-m decomposition factor appears only
// transiently in the real-space DFT slots).
#include "cumes/numerics/prolongation.hpp"
#include "cumes/runtime/cuda_status.hpp"

#include <cmath>
#include <cstdio>
#include <string>

// vmecpp decomposeInto scalxc for odd-m: 1/max(sqrt(s), sqrt(1/(ns-1))).
// The max() clamp makes this identical to cuMES's sqrtS_F for every j
// (profiles.cu stores sqrt(s + 1e-12); the clamped max is the same value).
template <typename T>
__device__ T scalxc_of(int j, int ns, T sqrtS1) {
    return T(1.0) / fmax(sqrt(T(j) / T(ns - 1)), sqrtS1);
}

template <typename T>
__global__ void interpolate_state_kernel(T* __restrict__ o_cc,
                                         T* __restrict__ o_zsc,
                                         T* __restrict__ o_lsc,
                                         T* __restrict__ o_ss,
                                         T* __restrict__ o_zcs,
                                         T* __restrict__ o_lcs,
                                         const T* __restrict__ i_cc,
                                         const T* __restrict__ i_zsc,
                                         const T* __restrict__ i_lsc,
                                         const T* __restrict__ i_ss,
                                         const T* __restrict__ i_zcs,
                                         const T* __restrict__ i_lcs,
                                         int ns_new,
                                         int ns_old,
                                         int mnmax,
                                         int ntorp1,
                                         bool cubic) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, total = mnmax * ns_new;
    if (i >= total) return;
    int mode = i / ns_new, jNew = i % ns_new;
    // Parity on the POLOIDAL m (vmecpp), not the mode index (n odd flips
    // the mode-index parity) — same test as scalxc_apply_kernel.
    int m = mode / ntorp1;
    bool odd = (m % 2 == 1);

    T sj = T(jNew) / T(ns_new - 1);
    // vmecpp uses INTEGER division here: which old rows bracket sj.
    int js1 = (jNew * (ns_old - 1)) / (ns_new - 1);
    int js2 = min(js1 + 1, ns_old - 1);
    T xint = sj * T(ns_old - 1) - T(js1);
    xint = fmin(T(1.0), fmax(T(0.0), xint));
    T wl = T(1.0) - xint, wr = xint;

    T sqrtS1o = sqrt(T(1.0) / T(ns_old - 1));
    T sqrtS1n = sqrt(T(1.0) / T(ns_new - 1));

    const T* ins[6] = {i_cc, i_zsc, i_lsc, i_ss, i_zcs, i_lcs};
    T* outs[6] = {o_cc, o_zsc, o_lsc, o_ss, o_zcs, o_lcs};
    for (int f = 0; f < 6; ++f) {
        const T* in = ins[f];
        T* out = outs[f];
        // Stencil value at old grid point j in DECOMPOSED space for odd-m
        // (phys * scalxc; even-m scalxc = 1).
        auto xs = [&](int j) -> T {
            T v = in[mode * ns_old + j];
            return odd ? v * scalxc_of(j, ns_old, sqrtS1o) : v;
        };
        // vmecpp extrapolates the odd-m axis to 2*x[1] - x[2] (decomposed),
        // for ALL odd m — it feeds the first interior rows (js1 == 0, e.g.
        // jNew = 1, 2 for 33 -> 66) which straddle the old axis.
        auto axis_regular_xs = [&](int j) -> T {
            return (odd && j == 0) ? T(2.0) * xs(1) - xs(2) : xs(j);
        };
        T xsl = axis_regular_xs(js1);
        T xsr = axis_regular_xs(js2);
        T interpolated = wl * xsl + wr * xsr;
        if (cubic && js1 != js2) {
            // Uniform-grid Catmull-Rom interpolation in the same decomposed
            // coordinate as the legacy linear transfer. Linear endpoint
            // extrapolation supplies the missing neighbor at either edge;
            // exact old-grid nodes and the LCFS remain exact.
            const T xm1 =
                js1 > 0 ? axis_regular_xs(js1 - 1) : T(2.0) * xsl - xsr;
            const T xp2 = js2 + 1 < ns_old ? axis_regular_xs(js2 + 1)
                                           : T(2.0) * xsr - xsl;
            interpolated =
                xsl +
                T(0.5) * xint *
                    (xsr - xm1 +
                     xint * (T(2.0) * xm1 - T(5.0) * xsl + T(4.0) * xsr - xp2 +
                             xint * (T(3.0) * (xsl - xsr) + xp2 - xm1)));
        }
        // Unscale: physical = decomposed / scalxc = decomposed * max(...).
        // (scalxc = 1/max(sqrt(s), sqrtS1); scaling twice with scalxc would
        // divide by max(...)^2 and inflate the interior odd-m coefficients —
        // a 10x error near the axis, invisible at the LCFS where scalxc=1.)
        T val = interpolated * (odd ? fmax(sqrt(sj), sqrtS1n) : T(1.0));
        if (odd && jNew == 0) val = T(0.0);  // vmecpp: all odd-m zeroed at axis
        out[mode * ns_new + jNew] = val;
    }
}

template <typename T>
cumes::SpectralStorage<T> cumes::Prolongation<T>::enqueue(
    const DeviceParams<T>& p_new,
    const cumes::SpectralStorage<T>& st_old,
    const DeviceParams<T>& p_old,
    cudaStream_t stream,
    bool cubic) const {
    if (p_new.ns <= p_old.ns || p_new.mnmax != p_old.mnmax || p_old.ns < 3) {
        // Library contract: never exit() here — the RAII device buffers, the
        // caller's --checkpoint write, and the CLI run-report mapping must
        // unwind. Validation guarantees this precondition for validated
        // problems (ns_array strictly increasing, common mode counts); throw
        // the library error boundary so main's run-report mapping reports it.
        throw cumes::CumesError(
            "interpolateState: need ns_new > ns_old >= 3 and equal mnmax "
            "(ns_old=" +
            std::to_string(p_old.ns) + " ns_new=" + std::to_string(p_new.ns) +
            " mnmax " + std::to_string(p_old.mnmax) + "/" +
            std::to_string(p_new.mnmax) + ")");
    }

    // New grid's contiguous slabs; the ctor zeroes both (velocities are never
    // interpolated — vmecpp zeroes them per stage).
    cumes::SpectralStorage<T> st_new(p_new.ns, p_new.mnmax);

    dim3 bd(256), gd((p_new.ns * p_new.mnmax + 255) / 256);
    // The coarse state (st_old) is written by the previous stage's kernels on
    // the nonblocking compute stream. A `cudaStreamSynchronize(stream)` at the
    // previous stage's exit does NOT make those writes visible to this kernel —
    // the batched ζ-transforms and the stage's synchronous default-stream
    // memsets/memcpys leave the coarse slab in a state only a full device sync
    // orders (observed as an all-zero ns=55 stage on the axisymmetric path).
    // Full-sync once per grid stage (never in the hot loop): this is a
    // stage-boundary point, so the cost is one fence per stage, not per pass.
    cumes::check_cuda(cudaDeviceSynchronize(), "interpolateState pre-sync");
    interpolate_state_kernel<T><<<gd, bd, 0, stream>>>(
        st_new.family_ptr(cumes::SpectralComponent::Rcc),
        st_new.family_ptr(cumes::SpectralComponent::Zsc),
        st_new.family_ptr(cumes::SpectralComponent::Lsc),
        st_new.family_ptr(cumes::SpectralComponent::Rss),
        st_new.family_ptr(cumes::SpectralComponent::Zcs),
        st_new.family_ptr(cumes::SpectralComponent::Lcs),
        st_old.family_ptr(cumes::SpectralComponent::Rcc),
        st_old.family_ptr(cumes::SpectralComponent::Zsc),
        st_old.family_ptr(cumes::SpectralComponent::Lsc),
        st_old.family_ptr(cumes::SpectralComponent::Rss),
        st_old.family_ptr(cumes::SpectralComponent::Zcs),
        st_old.family_ptr(cumes::SpectralComponent::Lcs), p_new.ns, p_old.ns,
        p_new.mnmax, p_new.ntor + 1, cubic);
    cumes::check_cuda(cudaGetLastError(), "interpolateState");
    // The kernel reads the OLD (coarse) state asynchronously; the caller frees
    // that state (via move-assignment of the returned SpectralStorage) as soon
    // as this returns. Wait for the kernel so the old buffer is not released
    // while still being read.
    cumes::check_cuda(cudaStreamSynchronize(stream), "interpolateState sync");
    printf(
        "  interpolateState: ns=%d -> ns=%d (%s in s, scalxc-scaled "
        "odd-m)\n",
        p_old.ns, p_new.ns, cubic ? "cubic" : "linear");
    return st_new;
}

#endif  // CUMES_SRC_PROLONGATION_IMPL_CUH_
