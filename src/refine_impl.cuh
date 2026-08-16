// refine_impl.cuh — template definitions for refine.cuh.
// Included once per scalar type by refine_double.cu / refine_float.cu; see the
// explicit-instantiation split (cumes_cuda_double / cumes_cuda_float).
#pragma once
// refine.cu — grid-sequencing state interpolation (multi-radial-grid).
//
// Mirrors vmecpp's Vmec::InterpolateToNextMultigridStep (vmec.cc:1795-2042),
// kLinear scheme: each spectral coefficient is a function of the radial
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
// SpectralState convention (the odd-m decomposition factor appears only
// transiently in the real-space DFT slots).
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "refine.cuh"

#include "cumes/runtime/cuda_status.hpp"

// vmecpp decomposeInto scalxc for odd-m: 1/max(sqrt(s), sqrt(1/(ns-1))).
// The max() clamp makes this identical to cuMES's sqrtS_F for every j
// (profiles.cu stores sqrt(s + 1e-12); the clamped max is the same value).
template <typename T>
__device__ T scalxcOf(int j, int ns, T sqrtS1) {
    return T(1.0) / fmax(sqrt(T(j) / T(ns - 1)), sqrtS1);
}

template <typename T>
__global__ void interpolateStateKernel(
    T* __restrict__ o_cc, T* __restrict__ o_zsc, T* __restrict__ o_lsc,
    T* __restrict__ o_ss, T* __restrict__ o_zcs, T* __restrict__ o_lcs,
    const T* __restrict__ i_cc, const T* __restrict__ i_zsc, const T* __restrict__ i_lsc,
    const T* __restrict__ i_ss, const T* __restrict__ i_zcs, const T* __restrict__ i_lcs,
    int ns_new, int ns_old, int mnmax, int ntorp1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x, total = mnmax * ns_new;
    if (i >= total) return;
    int mode = i / ns_new, jNew = i % ns_new;
    // Parity on the POLOIDAL m (vmecpp), not the mode index (n odd flips
    // the mode-index parity) — same test as scalxcApplyKernel.
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
            return odd ? v * scalxcOf(j, ns_old, sqrtS1o) : v;
        };
        // vmecpp extrapolates the odd-m axis to 2*x[1] - x[2] (decomposed),
        // for ALL odd m — it feeds the first interior rows (js1 == 0, e.g.
        // jNew = 1, 2 for 33 -> 66) which straddle the old axis.
        T xsl = (odd && js1 == 0) ? T(2.0) * xs(1) - xs(2) : xs(js1);
        T xsr = xs(js2);
        // Unscale: physical = decomposed / scalxc = decomposed * max(...).
        // (scalxc = 1/max(sqrt(s), sqrtS1); scaling twice with scalxc would
        // divide by max(...)^2 and inflate the interior odd-m coefficients —
        // a 10x error near the axis, invisible at the LCFS where scalxc=1.)
        T val = (wl * xsl + wr * xsr) *
                (odd ? fmax(sqrt(sj), sqrtS1n) : T(1.0));
        if (odd && jNew == 0) val = T(0.0);  // vmecpp: all odd-m zeroed at axis
        out[mode * ns_new + jNew] = val;
    }
}

template <typename T>
cumes::SpectralStorage<T> interpolateState(const DeviceParams<T>& p_new,
                                           const cumes::SpectralStorage<T>& st_old,
                                           const DeviceParams<T>& p_old,
                                           cudaStream_t stream) {
    if (p_new.ns <= p_old.ns || p_new.mnmax != p_old.mnmax || p_old.ns < 3) {
        fprintf(stderr, "interpolateState: need ns_new > ns_old >= 3 and equal "
                "mnmax (ns_old=%d ns_new=%d mnmax %d/%d)\n",
                p_old.ns, p_new.ns, p_old.mnmax, p_new.mnmax);
        exit(EXIT_FAILURE);
    }

    // New grid's contiguous slabs; the ctor zeroes both (velocities are never
    // interpolated — vmecpp zeroes them per stage).
    cumes::SpectralStorage<T> st_new(p_new.ns, p_new.mnmax);
    SpectralState<T> v_new = st_new.legacy_view();
    SpectralState<T> v_old = st_old.legacy_view();

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
    interpolateStateKernel<T><<<gd, bd, 0, stream>>>(
        v_new.d_rmncc, v_new.d_zmnsc, v_new.d_lmnsc,
        v_new.d_rmnss, v_new.d_zmncs, v_new.d_lmncs,
        v_old.d_rmncc, v_old.d_zmnsc, v_old.d_lmnsc,
        v_old.d_rmnss, v_old.d_zmncs, v_old.d_lmncs,
        p_new.ns, p_old.ns, p_new.mnmax, p_new.ntor + 1);
    cumes::check_cuda(cudaGetLastError(), "interpolateState");
    // The kernel reads the OLD (coarse) state asynchronously; the caller frees
    // that state (via move-assignment of the returned SpectralStorage) as soon
    // as this returns. Wait for the kernel so the old buffer is not released
    // while still being read.
    cumes::check_cuda(cudaStreamSynchronize(stream), "interpolateState sync");
    printf("  interpolateState: ns=%d -> ns=%d (linear in s, scalxc-scaled odd-m)\n",
           p_old.ns, p_new.ns);
    return st_new;
}

