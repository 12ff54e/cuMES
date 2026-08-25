// solver_impl.cuh — template definitions for solver.cuh.
// Included once per scalar type by solver_double.cu / solver_float.cu; see the
// explicit-instantiation split (cumes_cuda_double / cumes_cuda_float).
#pragma once
// solver.cu — fixed-point iteration with Garabedian acceleration.
// The dump/debug machinery below (DUMP_CUMES_VERIFY blocks) is compiled in
// but RUNTIME-GATED: nothing is written and no debug output is produced
// unless the CUMES_DUMP=1 environment variable is set (see dumpEnabled()).
//
// All computation is templated on the scalar type T (double or float).
// Dump files are T-native (read back by same-build tooling only); the
// per_iter_residuals record stays double.
// DUMP_CUMES_VERIFY is now a BUILD OPTION (completion plan step 3.3): CMake
// defines it on the solver TUs when CUMES_ENABLE_VERIFY_DUMP=ON (the
// verify/sanitizer/float presets); production-style builds compile the dump
// machinery out entirely. The machinery stays runtime-gated (CUMES_DUMP=1)
// when compiled in.
#include "cumes/numerics/accumulation.hpp"
#include "cumes/numerics/descent_operator.hpp"
#include "cumes/numerics/device_predicates.cuh"
#include "cumes/numerics/preconditioner.hpp"
#include "cumes/numerics/residual_operator.hpp"
#include "cumes/physics/constraint_operator.hpp"
#include "cumes/physics/force_operator.hpp"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/magnetic_field_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/runtime/device_arena.cuh"
#include "cumes/runtime/device_buffer.cuh"
#include "cumes/runtime/pinned_buffer.hpp"
#include "cumes/solver/equilibrium_operator.hpp"
#include "cumes/solver/iteration_controller.hpp"
#include "cumes/solver/pass_record.hpp"
#include "cumes/solver/solver_bench.hpp"
#include "cumes/transforms/axisymmetric_operator.hpp"
#include "cumes/transforms/toroidal_fft_operator.hpp"
#include "solver.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef DUMP_CUMES_VERIFY
static bool dumpEnabled();  // defined below with the dump machinery
template <typename T>
static void dumpDeviceArray(const char* filename,
                            const T* d_data,
                            size_t nelem);  // defined below
#endif

template <typename T>
__global__ void rzNormKernel(  // defined below (before computeResidualsKernel)
    cumes::SpectralView<const T, cumes::PhysicalStateDomain> st,
    const int* __restrict__ xm,
    const int* __restrict__ xn,
    int ns,
    int mnmax,
    cumes::ControlRecord* __restrict__ rec);

// ---- vmecpp force-norm assembly (ideal_mhd_model.cc computeForceNorms) ---
// Combines the per-surface partial sums (computeForceNormPartials) with the
// decomposed R/Z coefficient norm (rzNorm) into the residual normalization
// factors used by the control:
//   E_mag  = |Σ gsqrt·|B|²/2·wInt|·deltaS      (magnetic energy)
//   E_therm = Σ presH·dVdsH·deltaS             (thermal energy; gamma=0 →
//              presH = mass profile, which is what cuMES stores in d_pres_H)
//   V      = Σ dVdsH·deltaS                    (plasma volume)
//   energyDensity = max(E_mag, E_therm)/V
//   fNormRZ = 1/(Σ guu·r12²·wInt · energyDensity²)
//   fNormL = 1/(Σ (bsubu²+bsubv²)·wInt · lamscale²)
//   fNorm1 = 1/rzNorm
// with the wInt trapezoid over the reduced poloidal grid [0, pi].
//
// Phase 6B splits this into a device reduction (enqueueForceNorms, writing six
// scalars into the combined control record) and a host finalize
// (finalizeForceNorms, called after the single control fence). The old path
// D2H-copied psum/dVds/pres and summed them sequentially behind a
// cudaDeviceSynchronize; the device reduction removes that fence and those
// three copies, at the cost of a ULP-level (Class B) change in summation order.

// Device reduction over the half-grid surfaces of the force-norm partials:
// out[0..4] = {sRZ, sL, sMag, eTherm, vol} (before the deltaS scaling).
template <typename T>
__global__ void forceNormReduceKernel(
    const T* __restrict__ psum,   // 4*(ns-1): sRZ sL sMag sG per surface
    const T* __restrict__ dVdsH,  // ns-1
    const T* __restrict__ presH,  // ns-1
    int nH,
    cumes::ControlRecord* __restrict__ rec) {
    using A = typename cumes::NormAccum<T>::type;  // double for mixed-float
    int tid = threadIdx.x;
    A sRZ = A(0), sL = A(0), sMag = A(0), eTherm = A(0), vol = A(0);
    for (int j = tid; j < nH; j += blockDim.x) {
        sRZ += psum[4 * j + 0];
        sL += psum[4 * j + 1];
        sMag += psum[4 * j + 2];
        eTherm += presH[j] * dVdsH[j];
        vol += dVdsH[j];
    }
    __shared__ A s_buf[5][256];
    s_buf[0][tid] = sRZ;
    s_buf[1][tid] = sL;
    s_buf[2][tid] = sMag;
    s_buf[3][tid] = eTherm;
    s_buf[4][tid] = vol;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_buf[0][tid] += s_buf[0][tid + s];
            s_buf[1][tid] += s_buf[1][tid + s];
            s_buf[2][tid] += s_buf[2][tid + s];
            s_buf[3][tid] += s_buf[3][tid + s];
            s_buf[4][tid] += s_buf[4][tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        // Status guard (completion plan step 1.4): on an invalid-Jacobian pass
        // the force norms are NOT evaluated — store the deterministic zero
        // sentinel and leave the evaluated bit clear (the host gate restores
        // before reading these slots anyway; the zeros make the record
        // deterministic instead of stale).
        if (rec->status.jacobian_valid) {
            rec->force_norms[0] = s_buf[0][0];
            rec->force_norms[1] = s_buf[1][0];
            rec->force_norms[2] = s_buf[2][0];
            rec->force_norms[3] = s_buf[3][0];
            rec->force_norms[4] = s_buf[4][0];
            rec->status.force_norms_evaluated = 1;
        } else {
            rec->force_norms[0] = rec->force_norms[1] = rec->force_norms[2] =
                rec->force_norms[3] = rec->force_norms[4] =
                    rec->force_norms[5] = 0.0;
        }
    }
}

// Device-only force-norm reduction (no host copy or fence): writes the six
// scalars {sRZ, sL, sMag, eTherm, vol, rzNorm} into d_out[0..5], which the
// solver folds into the combined control record and transfers at the single
// control fence. Called on the preconditioner-refresh cadence.
template <typename T>
static void enqueueForceNorms(
    cumes::SpectralView<const T, cumes::PhysicalStateDomain> st,
    const int* xm,
    const int* xn,
    const DeviceParams<T>& p,
    const cumes::RadialProfileViews<T>& rpv,
    const cumes::GeometryOperator<T>& geometry,
    T* d_psum,
    cumes::ControlRecord* rec,
    cudaStream_t stream) {
    geometry.force_norm_partials(p, rpv.dVds_H, d_psum, stream);
    {
        dim3 b1(256), g1(1);
        forceNormReduceKernel<T><<<g1, b1, 0, stream>>>(
            d_psum, rpv.dVds_H, rpv.pres_H, p.ns - 1, rec);
    }
    {
        dim3 b2(256), g2(1);
        rzNormKernel<T><<<g2, b2, 0, stream>>>(st, xm, xn, p.ns, p.mnmax, rec);
    }
    cumes::check_cuda(cudaGetLastError(), "force norms");
}

// Refresh-pass force-norm dump (dump-only telemetry). The six device scalars
// hc[0..5] = {sRZ, sL, sMag, eTherm, vol, rzNorm} are printed together with
// the fNorm factors ACTUALLY used for the convergence decision — those are the
// record's device-finalized final_f_norm_* fields (completion-plan follow-up
// §2.3), consumed by the caller, not recomputed here.
static void dumpForceNorms(const double* hc,
                           double delta_s,
                           int iter2,
                           double fNormRZ,
                           double fNormL,
                           double fNorm1) {
#ifndef DUMP_CUMES_VERIFY
    (void)hc;
    (void)delta_s;
    (void)iter2;
    (void)fNormRZ;
    (void)fNormL;
    (void)fNorm1;
#endif
#ifdef DUMP_CUMES_VERIFY
    if (dumpEnabled()) {
        double sRZ = hc[0], sL = hc[1], sMag = hc[2], eTherm = hc[3],
               vol = hc[4], h_rz = hc[5];
        double deltaS = delta_s;
        double eMag =
            fabs(sMag) * deltaS;  // vmecpp: fabs(localMagneticEnergy)*deltaS
        eTherm *= deltaS;
        vol *= deltaS;
        double energyDensity = std::max(eMag, eTherm) / vol;
        // Same format as vmecpp's dump/vmecpp/force_norms_iter_<iter2>.txt
        char fn[128];
        snprintf(fn, sizeof fn, "dump/cuMES/force_norms_iter_%d.txt", iter2);
        FILE* fp2 = fopen(fn, "w");
        if (fp2) {
            fprintf(fp2,
                    "magneticEnergy %.17e\n"
                    "thermalEnergy %.17e\n"
                    "plasmaVolume %.17e\n"
                    "energyDensity %.17e\n"
                    "forceNormSumRZ %.17e\n"
                    "forceNormSumL %.17e\n"
                    "rzNorm %.17e\n"
                    "fNormRZ %.17e\n"
                    "fNormL %.17e\n"
                    "fNorm1 %.17e\n",
                    (double)eMag, (double)eTherm, (double)vol,
                    (double)energyDensity, (double)sRZ, (double)sL,
                    (double)h_rz, (double)fNormRZ, (double)fNormL,
                    (double)fNorm1);
            fclose(fp2);
        }
    }
#endif
}

// extrapolateTowardsAxis: copy m=1 coefficients from first interior
// surface (j=1) to the magnetic axis (j=0), matching vmecpp's
// extrapolateTowardsAxis(). Only m=1 has a finite value at the axis
// for stellarator-symmetric equilibria. Mode table: mode = m*(ntor+1)+n.
template <typename T>
__global__ void extrapolateAxisKernel(
    cumes::SpectralView<T, cumes::PhysicalStateDomain> st,
    int ns,
    int mnmax,
    int ntorp1) {
    int mode = blockIdx.x * blockDim.x + threadIdx.x;
    if (mode >= mnmax) return;
    int m = mode / ntorp1;  // poloidal mode number
    if (m == 0) {
        // m=0: only the lambda sin(n zeta) component is extrapolated —
        // vmecpp's extrapolateTowardsAxis ("m=0 component of lambda
        // leftover from chi-force"). This makes the even-parity lambda_zeta
        // at the axis match vmecpp, which feeds the axis-adjacent B^theta
        // through the half-grid average (FIXED 2026-08-02: without it the
        // jH=0 bsupu was off by ~30% on the first lambda != 0 pass, seeding
        // a 1e-4-level drift of the lambda channel).
        st(cumes::SpectralComponent::Lcs, mode, 0) =
            st(cumes::SpectralComponent::Lcs, mode, 1);
        return;
    }
    if (m != 1) return;  // only m=1 needs extrapolation
    // Copy from j=1 to j=0
    st(cumes::SpectralComponent::Rcc, mode, 0) =
        st(cumes::SpectralComponent::Rcc, mode, 1);
    st(cumes::SpectralComponent::Zsc, mode, 0) =
        st(cumes::SpectralComponent::Zsc, mode, 1);
    st(cumes::SpectralComponent::Lsc, mode, 0) =
        st(cumes::SpectralComponent::Lsc, mode, 1);
    st(cumes::SpectralComponent::Rss, mode, 0) =
        st(cumes::SpectralComponent::Rss, mode, 1);
    st(cumes::SpectralComponent::Zcs, mode, 0) =
        st(cumes::SpectralComponent::Zcs, mode, 1);
    st(cumes::SpectralComponent::Lcs, mode, 0) =
        st(cumes::SpectralComponent::Lcs, mode, 1);
}

// Apply vmecpp's even/odd-m decomposition scaling (decomposeInto) to the
// spectral forces. vmecpp decomposes its FORCES: odd-m force coefficients
// carry an extra 1/max(sqrt(s_F), sqrt(1/(ns-1))) factor (Eqn. 8c in
// Hirshman, Schwenn & Nuehrenberg 1990), which it then evolves through the
// residual, preconditioned-solve, and descent pipeline (m_decomposed_f).
// max(..., sqrt(s1)) clamps the factor at the axis to the innermost
// surface's sqrt(s), keeping it finite at s=0 (constant extrapolation,
// the same treatment as extrapolateAxisKernel above).
//
// IMPORTANT: only the FORCES are transformed here. The spectral state
// (d_rmncc/...) stays in plain physical coefficients throughout — initState
// builds s^(m/2) profiles, and the dumped state matches vmecpp's physical
// wout coefficients exactly. cuMES's real-space odd arrays likewise carry
// the decomposed form only transiently (inverseDFT divides by max(...) to
// feed the vmecpp-formula kernels). Consequently the odd-m factor must
// NEVER be re-applied when post-processing the state (e.g. flux-surface
// plots): doing so double-scales and flattens the m=1 shift profile into
// a linear-in-s one. Even-m modes: scalxc = 1 (no change).
template <typename T>
__global__ void scalxcApplyKernel(
    cumes::SpectralView<T, cumes::DecomposedResidualDomain> f_spec,
    const T* __restrict__ sqrtS_F,
    const int* __restrict__ xm,
    int ns,
    int mnmax,
    T sqrtS1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, total = mnmax * ns;
    if (i >= total) return;
    int mode = i / ns;
    // Parity test on the POLOIDAL m (xm[mode]), NOT the mode index
    // (m*(ntor+1)+n): for odd n the mode index parity is flipped.
    // (FIXED 2026-08-02: `int m = i/ns` tested the mode index, so all
    // n=1,3,5... entries got scalxc applied to the wrong parity --
    // even-m scaled by 1/sqrt(s), odd-m left unscaled.)
    int m = xm[mode];
    if (m % 2 == 0) return;  // even-m: scalxc = 1
    int j = i % ns;
    T scal = T(1.0) / fmax(sqrtS_F[j], sqrtS1);
    for (int c = 0; c < 6; ++c)
        f_spec(static_cast<cumes::SpectralComponent>(c), mode, j) *= scal;
}

// vmecpp's m1 gauge on the decomposed forces, applied every iteration after
// the forward DFT:
//   1. m1Constraint: rss = (rss+zcs)/sqrt(2), zcs = (rss-zcs)/sqrt(2)
//   2. zeroZForceForM1 (conditional): fzcs = 0  (m=1 Z force constrained
//      to zero)
// The zeroing mirrors vmecpp's `fix_m1_gauge` flag in IdealMhdModel::update:
// with always_fix_m1_gauge = false (the vmec_standalone default) the m=1
// fzcs is zeroed ONLY on the first pass (iter2 < 2) or once the invariant
// Z-residual dropped below 1e-6. In between, the mixed fzcs stays nonzero
// and the m=1 mixed zcs' velocity evolves freely -- cuMES previously zeroed
// it unconditionally, which froze the mixed gauge and drifted the physical
// m=1 R/Z coefficients ~1e-3 off the reference.
// Applied to components 3 (frss) and 4 (fzcs) of the 6-component layout.
template <typename T>
__global__ void m1ConstraintKernel(
    cumes::SpectralView<T, cumes::DecomposedResidualDomain> f_spec,
    int ns,
    int mnmax,
    int ntor,
    int zeroZ) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ns) return;
    const T s = T(1.0) / std::sqrt(T(2.0));
    int m1base = ntor + 1;  // mode index of (m=1, n=0)
    for (int n = 0; n < ntor + 1; ++n) {
        int mn = m1base + n;
        T old_rss = f_spec(cumes::SpectralComponent::Rss, mn, j);
        T old_zcs = f_spec(cumes::SpectralComponent::Zcs, mn, j);
        f_spec(cumes::SpectralComponent::Rss, mn, j) = (old_rss + old_zcs) * s;
        if (zeroZ) {
            f_spec(cumes::SpectralComponent::Zcs, mn, j) =
                T(0.0);  // zeroZForceForM1
        } else {
            f_spec(cumes::SpectralComponent::Zcs, mn, j) =
                (old_rss - old_zcs) * s;  // mixed zcs
        }
    }
}

// ---- rzNorm (vmecpp FourierCoeffs::rzNorm) -------------------------------
// Sum of squared decomposed R/Z coefficients over all ns rows (axis through
// LCFS), excluding the rcc (m=0,n=0) offset. cuMES stores the state as plain
// PHYSICAL coefficients (state = vmecpp-decomposed * ms*ns), so each term is
// scaled by 1/(ms*ns)^2. For m=1 the decomposed (rmnss, zmncs) pair is
// stored M(1/2)-mixed in vmecpp, giving
//   rss_d^2 + zcs_d^2 = (rss_p^2 + zcs_p^2) / (2 * (ms*ns)^2)
// (the 0.5 factor below); all other modes use the unmixed square.
template <typename T>
__global__ void rzNormKernel(
    cumes::SpectralView<const T, cumes::PhysicalStateDomain> st,
    const int* __restrict__ xm,
    const int* __restrict__ xn,
    int ns,
    int mnmax,
    cumes::ControlRecord* __restrict__ rec) {
    using A = typename cumes::NormAccum<T>::type;  // double for mixed-float
    A sum = A(0.0);
    int total = mnmax * ns;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int m = i / ns, j = i % ns, mm = xm[m], nn = xn[m];
        if (j == 0 && mm > 0)
            continue;  // vmecpp keeps the stored axis m>0
                       // at 0 (extrapolated only in real
                       // space); the state-file axis row
                       // therefore contributes nothing to
                       // rzNorm
        T mfac = (mm == 0) ? T(1.0) : std::sqrt(T(2.0));
        T nfac = (nn == 0) ? T(1.0) : std::sqrt(T(2.0));
        // decomposed = physical/(ms*ns): the squared term picks up 1/(ms*ns)^2
        T inv2 = T(1.0) / (mfac * nfac * mfac * nfac);
        T rcc = st(cumes::SpectralComponent::Rcc, m, j);
        T zsc = st(cumes::SpectralComponent::Zsc, m, j);
        T rss = st(cumes::SpectralComponent::Rss, m, j);
        T zcs = st(cumes::SpectralComponent::Zcs, m, j);
        if (mm > 0 || nn > 0) sum += rcc * rcc * inv2;
        sum += zsc * zsc * inv2;
        if (mm == 1) {
            // decomposed pair is mixed: (rss_d^2 + zcs_d^2) = (rss_p^2 +
            // zcs_p^2) / (2 * (ms*ns)^2)
            sum += T(0.5) * (rss * rss + zcs * zcs) * inv2;
        } else {
            sum += (rss * rss + zcs * zcs) * inv2;
        }
    }
    __shared__ A s_sum[256];
    int tid = threadIdx.x;
    s_sum[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    if (tid == 0 && rec->status.jacobian_valid) rec->force_norms[5] = s_sum[0];
}

// Residual groups match vmecpp's FourierForces::residuals (folded basis):
//   fsqr = Σ frcc² + frss²,  fsqz = Σ fzsc² + fzcs²,  fsql = Σ flsc² + flcs²
// (components 0..5 of f_spec: frcc fzsc flsc frss fzcs flcs).
template <typename T>
__global__ void computeResidualsKernel(
    cumes::SpectralView<const T, cumes::DecomposedResidualDomain> f_spec,
    int ns,
    int mnmax,
    bool include_edge_rz,
    double* __restrict__ sq_out) {
    using A = typename cumes::NormAccum<T>::type;  // double for mixed-float
    int comp = blockIdx.x;
    if (comp >= 3) return;
    A sum = A(0);
    int total = mnmax * ns;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int mode = i / ns, j = i % ns;
        if (comp < 2 && !include_edge_rz && j == ns - 1) continue;
        T a = f_spec(static_cast<cumes::SpectralComponent>(comp), mode, j);
        T b = f_spec(static_cast<cumes::SpectralComponent>(comp + 3), mode, j);
        sum += a * a + b * b;
    }
    __shared__ A s_sum[256];
    int tid = threadIdx.x;
    s_sum[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    if (tid == 0) sq_out[comp] = s_sum[0] / (mnmax * ns);
}

template <typename T>
__global__ void descentStepKernel(
    cumes::SpectralView<T, cumes::PhysicalStateDomain> x,
    cumes::SpectralView<T, cumes::DecomposedVelocityDomain> v,
    cumes::SpectralView<const T, cumes::DecomposedResidualDomain> f_spec,
    const int* __restrict__ xm,
    const int* __restrict__ xn,
    int ns,
    int mnmax,
    T delt,
    T b1,
    T fac,
    int j_max) {
    using cumes::SpectralComponent;
    int i = blockIdx.x * blockDim.x + threadIdx.x, total = mnmax * ns;
    if (i >= total) return;
    int m = i / ns, j = i % ns, mm = xm[m];
    // Skip m>0 modes at axis: coordinate singularity at s=0. The forces
    // there are zeroed by the preconditioners (jMin identity rows for R/Z,
    // sqrt(s)^pwr for lambda), so vmecpp never moves these coefficients.
    if (j == 0 && mm > 0) return;

    // State/velocity representations: the spectral STATE carries the
    // mscale*nscale basis factors (state = vmecpp-decomposed * ms*ns) while
    // the FORCES and VELOCITIES are vmecpp-decomposed. The state increment
    // must therefore be scaled by ms*ns per mode. (FIXED 2026-08-02: the
    // odd-n/odd-m increments were ms*ns too small, e.g. sqrt(2) for m=1n=0,
    // making the iter-2+ states drift ~0.7%/step.)
    T mfac = (mm == 0) ? T(1.0) : std::sqrt(T(2.0));
    T nfac = (xn[m] == 0) ? T(1.0) : std::sqrt(T(2.0));
    T f = mfac * nfac;

    // R/Z components (0,1,3,4): the LCFS is fixed in fixed-boundary runs —
    // the force was zeroed by the forward DFT and the coefficient must not
    // move; free-boundary passes j_max = ns and the boundary evolves.
    // Lambda (comps 2,5) is free on all surfaces including the LCFS,
    // matching vmecpp.
    if (j < j_max) {
        T vr = v(SpectralComponent::Rcc, m, j);
        vr = fac * (b1 * vr + delt * f_spec(SpectralComponent::Rcc, m, j));
        v(SpectralComponent::Rcc, m, j) = vr;
        x(SpectralComponent::Rcc, m, j) += delt * vr * f;
        T vz = v(SpectralComponent::Zsc, m, j);
        vz = fac * (b1 * vz + delt * f_spec(SpectralComponent::Zsc, m, j));
        v(SpectralComponent::Zsc, m, j) = vz;
        x(SpectralComponent::Zsc, m, j) += delt * vz * f;
        T vs = v(SpectralComponent::Rss, m, j);
        vs = fac * (b1 * vs + delt * f_spec(SpectralComponent::Rss, m, j));
        v(SpectralComponent::Rss, m, j) = vs;
        T vzc = v(SpectralComponent::Zcs, m, j);
        vzc = fac * (b1 * vzc + delt * f_spec(SpectralComponent::Zcs, m, j));
        v(SpectralComponent::Zcs, m, j) = vzc;
        if (mm == 1) {
            // m1 gauge: the state is stored in the UNDONE gauge while the
            // velocities/forces are vmecpp-decomposed (mixed gauge). vmecpp's
            // state evolves in the mixed gauge and is undone each update, so
            // the undone state must increment with the UNDONE velocity:
            //   rmnss += (vrss+vzcs), zmncs += (vrss-vzcs)
            // (FIXED 2026-08-02: without the mixing the iter-2+ m=1 states
            // drifted from vmecpp by ~0.07, corrupting the real-space.)
            x(SpectralComponent::Rss, m, j) += delt * (vs + vzc) * f;
            x(SpectralComponent::Zcs, m, j) += delt * (vs - vzc) * f;
        } else {
            x(SpectralComponent::Rss, m, j) += delt * vs * f;
            x(SpectralComponent::Zcs, m, j) += delt * vzc * f;
        }
    }
    T vl = v(SpectralComponent::Lsc, m, j);
    vl = fac * (b1 * vl + delt * f_spec(SpectralComponent::Lsc, m, j));
    v(SpectralComponent::Lsc, m, j) = vl;
    x(SpectralComponent::Lsc, m, j) += delt * vl * f;
    T vlc = v(SpectralComponent::Lcs, m, j);
    vlc = fac * (b1 * vlc + delt * f_spec(SpectralComponent::Lcs, m, j));
    v(SpectralComponent::Lcs, m, j) = vlc;
    x(SpectralComponent::Lcs, m, j) += delt * vlc * f;
}

// ---------------------------------------------------------------------------
// Stateless operators (migration steps 8/10): thin wrappers over the solver's
// kernels. The solver drives these instead of the raw kernel launches; the
// launch configs are bit-identical.
// ---------------------------------------------------------------------------
template <typename T>
void cumes::ResidualOperator<T>::enqueue(
    cumes::SpectralView<const T, cumes::DecomposedResidualDomain> residual,
    int ns,
    int mnmax,
    bool include_edge_rz,
    double* sq_out,
    cudaStream_t stream) const {
    dim3 b3(256), g3(3);
    computeResidualsKernel<T>
        <<<g3, b3, 0, stream>>>(residual, ns, mnmax, include_edge_rz, sq_out);
}

template <typename T>
void cumes::ResidualOperator<T>::enqueue_preconditioned(
    cumes::SpectralView<const T, cumes::DecomposedResidualDomain> residual,
    int ns,
    int mnmax,
    cumes::ControlRecord* rec,
    cudaStream_t stream) const {
    dim3 b3(256), g3(3);
    computeResidualsPreconditionedKernel<T>
        <<<g3, b3, 0, stream>>>(residual, ns, mnmax, rec);
}

template <typename T>
void cumes::DescentOperator<T>::enqueue(
    cumes::SpectralView<T, cumes::PhysicalStateDomain> state,
    cumes::SpectralView<T, cumes::DecomposedVelocityDomain> velocity,
    cumes::SpectralView<const T, cumes::DecomposedResidualDomain> residual,
    const int* xm,
    const int* xn,
    int ns,
    int mnmax,
    const cumes::DescentAction& action,
    cudaStream_t stream) const {
    if (!action.perform_descent) return;
    dim3 bd(256), gd((ns * mnmax + 255) / 256);
    descentStepKernel<T><<<gd, bd, 0, stream>>>(
        state, velocity, residual, xm, xn, ns, mnmax, T(action.delta_t),
        T(action.damping_b1), T(action.damping_fac),
        action.move_lcfs ? ns : ns - 1);
    cumes::check_cuda(cudaGetLastError(), "descent");
}

#ifdef DUMP_CUMES_VERIFY
// Master switch for the dump/debug machinery: off unless CUMES_DUMP=1.
// All dump output routes through dumpEnsureDir/dumpDeviceArray, which no-op
// when disabled, so default runs write nothing to dump/ and print no debug
// noise. (The CUMES_DUMP_ITER knob below still selects WHICH iterations the
// windowed dumps fire on; CUMES_DUMP is the master enable.)
static bool dumpEnabled() {
    // Read the env var once per process (the old form re-read it at every
    // one of the ~6 dump-window entry points per iteration).
    static const bool enabled = [] {
        const char* e = getenv("CUMES_DUMP");
        return e != nullptr && atoi(e) != 0;
    }();
    return enabled;
}

static void dumpEnsureDir() {
    if (!dumpEnabled()) return;
    int rc = system("mkdir -p dump/cuMES");
    if (rc != 0)
        fprintf(stderr, "dumpEnsureDir: mkdir -p failed (rc=%d)\n", rc);
}

// T-native dump: written as sizeof(T) elements; only read back by same-build
// tooling (e.g. tests/test_geometry_iso.cu, which is double-build-only).
template <typename T>
static void dumpDeviceArray(const char* filename,
                            const T* d_data,
                            size_t nelem) {
    if (!dumpEnabled()) return;
    // The dump machinery reads device data on the (synchronous) default stream
    // while the hot loop produces it on the nonblocking compute stream. Sync
    // everything first so a dump never reads a stale/in-flight buffer. This is
    // compile- and runtime-gated observability, so the extra fence is free on
    // the production path.
    cudaDeviceSynchronize();
    T* h_tmp = new T[nelem];
    cudaError_t err =
        cudaMemcpy(h_tmp, d_data, nelem * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "dumpDeviceArray cudaMemcpy failed for %s: %s\n",
                filename, cudaGetErrorString(err));
    }
    FILE* fp = fopen(filename, "wb");
    if (fp) {
        uint64_t n = nelem;
        fwrite(&n, sizeof(uint64_t), 1, fp);
        fwrite(h_tmp, sizeof(T), nelem, fp);
        fclose(fp);
    }
    delete[] h_tmp;
}

// ---- per-iteration dump windows (7.1) ------------------------------------
// Extracted from EquilibriumOperator::enqueue so the DAG body reads as the
// arithmetic pipeline it is. Each helper reproduces its observation point's
// exact dump sequence (same files, same order, same contents) and is
// runtime-gated by dumpEnabled(). The one stream side effect the windows
// carry — combine_parity at iter 0 (the *_real snapshot materialization) —
// is preserved in dumpStepA so the dump-disabled path keeps the frozen
// trajectory's launch sequence bit-for-bit.

// Iter-0 loop diagnostic: print + dump the LCFS real-space R right after the
// first inverse DFT (2.1: the transform ran on the nonblocking compute
// stream, so dumpDeviceArray's device sync is what makes the read valid).
template <typename T>
static void dumpIter0LoopDiag(int iter,
                              const DeviceParams<T>& p,
                              const cumes::RealSpaceStorage<T>& rs) {
    if (iter != 0 || !dumpEnabled()) return;
    dumpDeviceArray("dump/cuMES/debug_r_e.bin", rs.d_r_e,
                    (size_t)p.nZnT * (size_t)p.ns);
    auto* h_test = new T[p.nZnT * p.ns];
    cumes::check_cuda(cudaMemcpy(h_test, rs.d_r_e, p.nZnT * p.ns * sizeof(T),
                                 cudaMemcpyDeviceToHost),
                      "loop test");
    int jB = p.ns - 1;
    printf("  [loop diag] LCFS theta=0: r_e=%.4f (expect ~3.93)\n",
           (double)h_test[0 + jB * p.nZnT]);
    delete[] h_test;
}

// step_A window (post-inverse, pre-geometry): lambda derivatives and, at
// iter 0, the full R/Z/λ real-space snapshot.
template <typename T>
static void dumpStepA(int iter,
                      int iter2,
                      const DeviceParams<T>& p,
                      const cumes::RealSpaceStorage<T>& rs,
                      cumes::ToroidalFftOperator<T>& transform,
                      cudaStream_t stream) {
    if (!dumpEnabled()) {
        // combine_parity runs at iter 0 even when dumping is off — the
        // frozen launch sequence (the *_real buffers are not read by the hot
        // loop, so this is launch-order fidelity only).
        if (iter == 0) transform.combine_parity(stream);
        return;
    }
    if (iter == 0 || iter2 == 2) {
        // iter 2 = first pass with lambda != 0: dump the real-space
        // lambda derivatives for the basis-convention check.
        size_t n_real = (size_t)p.ns * (size_t)p.nZnT;
        char fn[128];
        snprintf(fn, sizeof fn, "dump/cuMES/step_A_lu_e_iter_%d.bin",
                 iter == 0 ? 1 : 2);
        dumpDeviceArray(fn, rs.d_lu_e, n_real);
        snprintf(fn, sizeof fn, "dump/cuMES/step_A_lu_o_iter_%d.bin",
                 iter == 0 ? 1 : 2);
        dumpDeviceArray(fn, rs.d_lu_o, n_real);
        snprintf(fn, sizeof fn, "dump/cuMES/step_A_l_real_iter_%d.bin",
                 iter == 0 ? 1 : 2);
        dumpDeviceArray(fn, rs.d_l_real, n_real);
    }
    if (iter == 0 || iter2 == 2) {
        size_t n_real = (size_t)p.ns * (size_t)p.nZnT;
        char fn[128];
        snprintf(fn, sizeof fn, "dump/cuMES/step_A_lv_e_iter_%d.bin",
                 iter == 0 ? 1 : 2);
        dumpDeviceArray(fn, rs.d_lv_e, n_real);
        snprintf(fn, sizeof fn, "dump/cuMES/step_A_lv_o_iter_%d.bin",
                 iter == 0 ? 1 : 2);
        dumpDeviceArray(fn, rs.d_lv_o, n_real);
    }
    if (iter == 0) {
        size_t n_real = (size_t)p.ns * (size_t)p.nZnT;
        // The combined *_real arrays are NOT refreshed by the hot loop
        // (inverseDFT runs with do_combine=false); materialize a fresh
        // snapshot from the current parity arrays before dumping them.
        transform.combine_parity(stream);
        // Full R, Z, lambda (even+odd)
        dumpDeviceArray("dump/cuMES/step_A_r_real_iter_1.bin", rs.d_r_real,
                        n_real);
        dumpDeviceArray("dump/cuMES/step_A_z_real_iter_1.bin", rs.d_z_real,
                        n_real);
        // Even-m parity
        dumpDeviceArray("dump/cuMES/step_A_r_e_iter_1.bin", rs.d_r_e, n_real);
        dumpDeviceArray("dump/cuMES/step_A_z_e_iter_1.bin", rs.d_z_e, n_real);
        dumpDeviceArray("dump/cuMES/step_A_l_e_iter_1.bin", rs.d_l_e, n_real);
        // Odd-m parity
        dumpDeviceArray("dump/cuMES/step_A_r_o_iter_1.bin", rs.d_r_o, n_real);
        dumpDeviceArray("dump/cuMES/step_A_z_o_iter_1.bin", rs.d_z_o, n_real);
        dumpDeviceArray("dump/cuMES/step_A_l_o_iter_1.bin", rs.d_l_o, n_real);
        // Poloidal derivatives
        dumpDeviceArray("dump/cuMES/step_A_ru_real_iter_1.bin", rs.d_ru_real,
                        n_real);
        dumpDeviceArray("dump/cuMES/step_A_zu_real_iter_1.bin", rs.d_zu_real,
                        n_real);
        dumpDeviceArray("dump/cuMES/step_A_lu_real_iter_1.bin", rs.d_lu_real,
                        n_real);
        // Toroidal derivatives
        dumpDeviceArray("dump/cuMES/step_A_rv_real_iter_1.bin", rs.d_rv_real,
                        n_real);
        dumpDeviceArray("dump/cuMES/step_A_zv_real_iter_1.bin", rs.d_zv_real,
                        n_real);
        dumpDeviceArray("dump/cuMES/step_A_lv_real_iter_1.bin", rs.d_lv_real,
                        n_real);
        // Even-m poloidal derivatives
        dumpDeviceArray("dump/cuMES/step_A_ru_e_iter_1.bin", rs.d_ru_e, n_real);
        dumpDeviceArray("dump/cuMES/step_A_zu_e_iter_1.bin", rs.d_zu_e, n_real);
        dumpDeviceArray("dump/cuMES/step_A_lu_e_iter_1.bin", rs.d_lu_e, n_real);
        // Odd-m poloidal derivatives
        dumpDeviceArray("dump/cuMES/step_A_ru_o_iter_1.bin", rs.d_ru_o, n_real);
        dumpDeviceArray("dump/cuMES/step_A_zu_o_iter_1.bin", rs.d_zu_o, n_real);
        dumpDeviceArray("dump/cuMES/step_A_lu_o_iter_1.bin", rs.d_lu_o, n_real);
    }
}

// step_B/C/D window (post-field): metric + contravariant-B half-grid dump.
template <typename T>
static void dumpStepD(int iter,
                      int iter2,
                      const DeviceParams<T>& p,
                      const cumes::BaseGeometryHalfViews<T>& base,
                      const cumes::MagneticFieldViews<T>& field) {
    if (!dumpEnabled()) return;
    if (iter == 0 || iter2 == 2) {
        // iter 2 = first pass with lambda != 0 (E3-D bsupv check)
        size_t n_half = (size_t)(p.ns - 1) * (size_t)p.nZnT;
        char fn[128];
        snprintf(fn, sizeof fn, "dump/cuMES/step_D_bsupu_iter_%d.bin",
                 iter == 0 ? 1 : 2);
        dumpDeviceArray(fn, field.bsupu.data(), n_half);
        snprintf(fn, sizeof fn, "dump/cuMES/step_D_bsupv_iter_%d.bin",
                 iter == 0 ? 1 : 2);
        dumpDeviceArray(fn, field.bsupv.data(), n_half);
    }
    if (iter == 0) {
        size_t n_half = (size_t)(p.ns - 1) * (size_t)p.nZnT;
        dumpDeviceArray("dump/cuMES/step_B_gsqrt_iter_1.bin", base.gsqrt.data(),
                        n_half);
        dumpDeviceArray("dump/cuMES/step_C_guu_iter_1.bin", base.guu.data(),
                        n_half);
        dumpDeviceArray("dump/cuMES/step_C_guv_iter_1.bin", base.guv.data(),
                        n_half);
        dumpDeviceArray("dump/cuMES/step_C_gvv_iter_1.bin", base.gvv.data(),
                        n_half);
    }
}

// step_precon window (iter 0, on the refresh pass): tridiagonal matrices,
// jMin, intermediates, sizes.
template <typename T>
static void dumpStepPrecon(int iter,
                           const DeviceParams<T>& p,
                           const cumes::Preconditioner<T>& precon) {
    // The extracted helper gates the whole window behind dumpEnabled();
    // the pre-extraction body additionally ran the jMin D2H copy and the
    // jMin/sizes fopen writes unconditionally (harmless reads/failed opens
    // on the default path — no numeric effect, Class A verified).
    if (iter != 0 || !dumpEnabled()) return;
    // Dump tridiagonal preconditioner matrices for comparison
    // with vmecpp. cuMES layout: ar[mode * ns + jF] (mode-major).
    size_t n_tri = (size_t)p.mnmax * (size_t)p.ns;
    size_t n_half_2 = (size_t)2 * (size_t)(p.ns - 1);
    size_t n_full_2 = (size_t)2 * (size_t)p.ns;
    size_t n_full_1 = (size_t)p.ns;

    // Tridiagonal matrix elements (mode-major: [mode, jF])
    dumpDeviceArray("dump/cuMES/step_precon_ar_iter_1.bin", precon.ar(), n_tri);
    dumpDeviceArray("dump/cuMES/step_precon_dr_iter_1.bin", precon.dr(), n_tri);
    dumpDeviceArray("dump/cuMES/step_precon_br_iter_1.bin", precon.br(), n_tri);
    dumpDeviceArray("dump/cuMES/step_precon_az_iter_1.bin", precon.az(), n_tri);
    dumpDeviceArray("dump/cuMES/step_precon_dz_iter_1.bin", precon.dz(), n_tri);
    dumpDeviceArray("dump/cuMES/step_precon_bz_iter_1.bin", precon.bz(), n_tri);

    // jMin per mode (stored as int, convert to double for dump)
    {
        int* h_jMin = new int[p.mnmax];
        cudaMemcpy(h_jMin, precon.jmin(), p.mnmax * sizeof(int),
                   cudaMemcpyDeviceToHost);
        double* h_jMin_dbl = new double[p.mnmax];
        for (int i = 0; i < p.mnmax; ++i) h_jMin_dbl[i] = (double)h_jMin[i];
        FILE* fj = fopen("dump/cuMES/step_precon_jMin_iter_1.bin", "wb");
        if (fj) {
            uint64_t n = p.mnmax;
            fwrite(&n, sizeof(uint64_t), 1, fj);
            fwrite(h_jMin_dbl, sizeof(double), p.mnmax, fj);
            fclose(fj);
        }
        delete[] h_jMin;
        delete[] h_jMin_dbl;
    }

    // Intermediate arrays
    dumpDeviceArray("dump/cuMES/step_precon_arm_iter_1.bin", precon.arm(),
                    n_half_2);
    dumpDeviceArray("dump/cuMES/step_precon_ard_iter_1.bin", precon.ard(),
                    n_full_2);
    dumpDeviceArray("dump/cuMES/step_precon_brm_iter_1.bin", precon.brm(),
                    n_half_2);
    dumpDeviceArray("dump/cuMES/step_precon_brd_iter_1.bin", precon.brd(),
                    n_full_2);
    dumpDeviceArray("dump/cuMES/step_precon_azm_iter_1.bin", precon.azm(),
                    n_half_2);
    dumpDeviceArray("dump/cuMES/step_precon_azd_iter_1.bin", precon.azd(),
                    n_full_2);
    dumpDeviceArray("dump/cuMES/step_precon_bzm_iter_1.bin", precon.bzm(),
                    n_half_2);
    dumpDeviceArray("dump/cuMES/step_precon_bzd_iter_1.bin", precon.bzd(),
                    n_full_2);
    dumpDeviceArray("dump/cuMES/step_precon_cxd_iter_1.bin", precon.cxd(),
                    n_full_1);

    // Sizes for comparison script
    double sizes_dbl[4] = {(double)p.ns, (double)(p.ns - 1), (double)p.mpol,
                           1.0};
    FILE* fs = fopen("dump/cuMES/step_precon_sizes_iter_1.bin", "wb");
    if (fs) {
        uint64_t n = 4;
        fwrite(&n, sizeof(uint64_t), 1, fs);
        fwrite(sizes_dbl, sizeof(double), 4, fs);
        fclose(fs);
    }
}

// step_E/F window (post-force): half-grid geometry + force outputs.
template <typename T>
static void dumpStepEF(int iter,
                       int iter2,
                       const DeviceParams<T>& p,
                       const cumes::BaseGeometryHalfViews<T>& base,
                       const cumes::MagneticFieldViews<T>& field,
                       const cumes::RealSpaceStorage<T>& rs) {
    if (!dumpEnabled()) return;
    if (iter == 0 || iter2 == 2) {
        // iter 2 = first pass with lambda != 0 (E3-B blmn blending check)
        size_t n_half = (size_t)(p.ns - 1) * (size_t)p.nZnT;
        if (iter == 0) {
            dumpDeviceArray("dump/cuMES/step_half_r12_iter_1.bin",
                            base.r12.data(), n_half);
            dumpDeviceArray("dump/cuMES/step_half_zu12_iter_1.bin",
                            base.zu12.data(), n_half);
            dumpDeviceArray("dump/cuMES/step_half_tau_iter_1.bin",
                            base.tau.data(), n_half);
            dumpDeviceArray("dump/cuMES/step_half_gsqrt_iter_1.bin",
                            base.gsqrt.data(), n_half);
            dumpDeviceArray("dump/cuMES/step_half_totalP_iter_1.bin",
                            field.total_pressure.data(), n_half);
            dumpDeviceArray("dump/cuMES/step_half_bsupu_iter_1.bin",
                            field.bsupu.data(), n_half);
            dumpDeviceArray("dump/cuMES/step_half_bsupv_iter_1.bin",
                            field.bsupv.data(), n_half);
            dumpDeviceArray("dump/cuMES/step_half_bsubu_iter_1.bin",
                            field.bsubu.data(), n_half);
            dumpDeviceArray("dump/cuMES/step_half_bsubv_iter_1.bin",
                            field.bsubv.data(), n_half);
            dumpDeviceArray("dump/cuMES/step_half_rs_iter_1.bin",
                            base.rs.data(), n_half);
            dumpDeviceArray("dump/cuMES/step_half_zs_iter_1.bin",
                            base.zs.data(), n_half);
        }

        size_t n_real = (size_t)p.ns * (size_t)p.nZnT;
        char fn[128];
        int itag = (iter == 0) ? 1 : iter2;
        snprintf(fn, sizeof fn, "dump/cuMES/step_F_brmn_e_iter_%d.bin", itag);
        dumpDeviceArray(fn, rs.d_brmn_e, n_real);
        snprintf(fn, sizeof fn, "dump/cuMES/step_F_brmn_o_iter_%d.bin", itag);
        dumpDeviceArray(fn, rs.d_brmn_o, n_real);
        snprintf(fn, sizeof fn, "dump/cuMES/step_F_bzmn_e_iter_%d.bin", itag);
        dumpDeviceArray(fn, rs.d_bzmn_e, n_real);
        snprintf(fn, sizeof fn, "dump/cuMES/step_F_bzmn_o_iter_%d.bin", itag);
        dumpDeviceArray(fn, rs.d_bzmn_o, n_real);
        snprintf(fn, sizeof fn, "dump/cuMES/step_F_blmn_e_iter_%d.bin", itag);
        dumpDeviceArray(fn, rs.d_blmn_e, n_real);
        snprintf(fn, sizeof fn, "dump/cuMES/step_F_blmn_o_iter_%d.bin", itag);
        dumpDeviceArray(fn, rs.d_blmn_o, n_real);
        if (iter == 0) {
            dumpDeviceArray("dump/cuMES/step_E_armn_e_iter_1.bin", rs.d_armn_e,
                            n_real);
            dumpDeviceArray("dump/cuMES/step_E_armn_o_iter_1.bin", rs.d_armn_o,
                            n_real);
            dumpDeviceArray("dump/cuMES/step_E_azmn_e_iter_1.bin", rs.d_azmn_e,
                            n_real);
            dumpDeviceArray("dump/cuMES/step_E_azmn_o_iter_1.bin", rs.d_azmn_o,
                            n_real);
            dumpDeviceArray("dump/cuMES/step_E_brmn_e_iter_1.bin", rs.d_brmn_e,
                            n_real);
            dumpDeviceArray("dump/cuMES/step_E_brmn_o_iter_1.bin", rs.d_brmn_o,
                            n_real);
            dumpDeviceArray("dump/cuMES/step_E_bzmn_e_iter_1.bin", rs.d_bzmn_e,
                            n_real);
            dumpDeviceArray("dump/cuMES/step_E_bzmn_o_iter_1.bin", rs.d_bzmn_o,
                            n_real);
            dumpDeviceArray("dump/cuMES/step_E_crmn_e_iter_1.bin", rs.d_crmn_e,
                            n_real);
            dumpDeviceArray("dump/cuMES/step_E_crmn_o_iter_1.bin", rs.d_crmn_o,
                            n_real);
            dumpDeviceArray("dump/cuMES/step_E_czmn_e_iter_1.bin", rs.d_czmn_e,
                            n_real);
            dumpDeviceArray("dump/cuMES/step_E_czmn_o_iter_1.bin", rs.d_czmn_o,
                            n_real);
            dumpDeviceArray("dump/cuMES/step_E_blmn_e_iter_1.bin", rs.d_blmn_e,
                            n_real);
            dumpDeviceArray("dump/cuMES/step_E_blmn_o_iter_1.bin", rs.d_blmn_o,
                            n_real);
            dumpDeviceArray("dump/cuMES/step_E_clmn_e_iter_1.bin", rs.d_clmn_e,
                            n_real);
            dumpDeviceArray("dump/cuMES/step_E_clmn_o_iter_1.bin", rs.d_clmn_o,
                            n_real);
            // NOTE: no combined-force dumps — the force combine buffers
            // were removed (they were allocated/dumped but never
            // produced; the parity-split arrays above are the source of
            // truth).
        }
    }
}

// step_G/GC window (iter 0, post-constraint): constraint-chain intermediates.
template <typename T>
static void dumpStepG(int iter,
                      const DeviceParams<T>& p,
                      const cumes::RealSpaceStorage<T>& rs,
                      cumes::SpectralStorage<T>& storage,
                      cumes::ConstraintOperator<T>& constraint) {
    if (iter != 0 || !dumpEnabled()) return;
    size_t n_real = (size_t)p.ns * (size_t)p.nZnT;
    dumpDeviceArray("dump/cuMES/step_G_brmn_e_iter_1.bin", rs.d_brmn_e, n_real);
    dumpDeviceArray("dump/cuMES/step_G_brmn_o_iter_1.bin", rs.d_brmn_o, n_real);
    dumpDeviceArray("dump/cuMES/step_G_bzmn_e_iter_1.bin", rs.d_bzmn_e, n_real);
    dumpDeviceArray("dump/cuMES/step_G_bzmn_o_iter_1.bin", rs.d_bzmn_o, n_real);
    // Constraint-chain intermediates (stage-by-stage vs vmecpp)
    // State as consumed by the constraint chain at iter 1 (post-descent)
    size_t n_spec2 = (size_t)p.ns * (size_t)p.mnmax;
    dumpDeviceArray("dump/cuMES/step_GC_rmncc_iter_1.bin",
                    storage.family_ptr(cumes::SpectralComponent::Rcc), n_spec2);
    dumpDeviceArray("dump/cuMES/step_GC_rmnss_iter_1.bin",
                    storage.family_ptr(cumes::SpectralComponent::Rss), n_spec2);
    dumpDeviceArray("dump/cuMES/step_GC_zmnsc_iter_1.bin",
                    storage.family_ptr(cumes::SpectralComponent::Zsc), n_spec2);
    dumpDeviceArray("dump/cuMES/step_GC_zmncs_iter_1.bin",
                    storage.family_ptr(cumes::SpectralComponent::Zcs), n_spec2);
    dumpDeviceArray("dump/cuMES/step_GC_rCon_iter_1.bin",
                    constraint.rcon_view(p).data(), n_real);
    dumpDeviceArray("dump/cuMES/step_GC_zCon_iter_1.bin",
                    constraint.zcon_view(p).data(), n_real);
    dumpDeviceArray("dump/cuMES/step_GC_gConEff_iter_1.bin",
                    constraint.gcon_eff(), n_real);
    dumpDeviceArray("dump/cuMES/step_GC_gCon_iter_1.bin", constraint.gcon(),
                    n_real);
    dumpDeviceArray("dump/cuMES/step_GC_frcon_e_iter_1.bin",
                    constraint.constraint_force_views(p).frcon_e.data(),
                    n_real);
    dumpDeviceArray("dump/cuMES/step_GC_frcon_o_iter_1.bin",
                    constraint.constraint_force_views(p).frcon_o.data(),
                    n_real);
    dumpDeviceArray("dump/cuMES/step_GC_fzcon_e_iter_1.bin",
                    constraint.constraint_force_views(p).fzcon_e.data(),
                    n_real);
    dumpDeviceArray("dump/cuMES/step_GC_fzcon_o_iter_1.bin",
                    constraint.constraint_force_views(p).fzcon_o.data(),
                    n_real);
    // tcon/faccon profiles (device arrays; h_tcon is stale -- the
    // kernel writes d_tcon directly)
    dumpDeviceArray("dump/cuMES/step_GC_tcon_iter_1.bin", constraint.tcon(),
                    p.ns);
    dumpDeviceArray("dump/cuMES/step_GC_faccon_iter_1.bin", constraint.faccon(),
                    p.mpol);
}

// step_H window (post-decomposition-scaling): decomposed force slab + the
// E2-start invariant-force window.
template <typename T>
static void dumpStepH(int iter,
                      int iter2,
                      const DeviceParams<T>& p,
                      const T* f_spec,
                      int kDumpIter,
                      int kE2Start) {
    if (!dumpEnabled()) return;
    if (iter == 0 || iter2 == kDumpIter) {
        // Dump AFTER the decomposition scaling, matching vmecpp's dump
        // of m_decomposed_f (post-decomposeInto). Keyed on iter2 so a
        // handoff/plateau comparison can use the same effective counter
        // as vmecpp's dump blocks.
        size_t n_fspec = (size_t)6 * (size_t)p.mnmax * (size_t)p.ns;
        char fn[128];
        snprintf(fn, sizeof fn, "dump/cuMES/step_H_f_spec_iter_%d.bin",
                 iter == 0 ? 1 : iter2);
        dumpDeviceArray(fn, f_spec, n_fspec);
    }
    if (iter2 >= kE2Start && iter2 < kE2Start + 40) {
        char fn[128];
        snprintf(fn, sizeof fn, "dump/cuMES/fspec_invariant_iter_%d.bin",
                 iter2);
        dumpDeviceArray(fn, f_spec, (size_t)6 * p.mnmax * p.ns);
    }
}

// Final-pass force-slab dump (post-m1-gauge, pre-invariant-residual).
template <typename T>
static void dumpStepFinal(int iter,
                          const DeviceParams<T>& p,
                          const T* f_spec,
                          int kMaxIterEff) {
    if (!dumpEnabled()) return;
    if (iter == kMaxIterEff - 1) {
        dumpDeviceArray("dump/cuMES/step_final_f_spec.bin", f_spec,
                        (size_t)6 * p.mnmax * p.ns);
    }
}

// step_I window (post-preconditioner): preconditioned slab + state/velocity
// handoff dumps + the E2-start preconditioned-force window.
template <typename T>
static void dumpStepI(int iter,
                      int iter2,
                      const DeviceParams<T>& p,
                      const T* f_spec,
                      cumes::SpectralStorage<T>& storage,
                      int kDumpIter,
                      int kE2Start) {
    if (!dumpEnabled()) return;
    if (iter == 0 || iter2 == 51 ||
        (iter2 >= kDumpIter && iter2 <= kDumpIter + 2) ||
        (iter2 >= 2 && iter2 <= 4)) {
        size_t n_fspec = (size_t)6 * (size_t)p.mnmax * (size_t)p.ns;
        size_t n_spec = (size_t)p.mnmax * (size_t)p.ns;
        char fn[128];
        snprintf(fn, sizeof fn, "dump/cuMES/step_I_f_spec_iter_%d.bin",
                 iter == 0 ? 1 : iter2);
        dumpDeviceArray(fn, f_spec, n_fspec);
        // State + velocities at the handoff window (pre-descent of the
        // pass, matching vmecpp's dump phase at vmec.cc). Also at the
        // iter-2..4 window (first lambda != 0 passes) for the state check.
        if (iter2 >= kDumpIter || iter2 == 51 || (iter2 >= 2 && iter2 <= 4)) {
            snprintf(fn, sizeof fn, "dump/cuMES/state_rmncc_iter_%d.bin",
                     iter2);
            dumpDeviceArray(
                fn, storage.family_ptr(cumes::SpectralComponent::Rcc), n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/state_zmnsc_iter_%d.bin",
                     iter2);
            dumpDeviceArray(
                fn, storage.family_ptr(cumes::SpectralComponent::Zsc), n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/state_lmnsc_iter_%d.bin",
                     iter2);
            dumpDeviceArray(
                fn, storage.family_ptr(cumes::SpectralComponent::Lsc), n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/state_rmnss_iter_%d.bin",
                     iter2);
            dumpDeviceArray(
                fn, storage.family_ptr(cumes::SpectralComponent::Rss), n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/state_zmncs_iter_%d.bin",
                     iter2);
            dumpDeviceArray(
                fn, storage.family_ptr(cumes::SpectralComponent::Zcs), n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/state_lmncs_iter_%d.bin",
                     iter2);
            dumpDeviceArray(
                fn, storage.family_ptr(cumes::SpectralComponent::Lcs), n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/vel_vrmncc_iter_%d.bin", iter2);
            dumpDeviceArray(
                fn, storage.velocity_family_ptr(cumes::SpectralComponent::Rcc),
                n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/vel_vzmnsc_iter_%d.bin", iter2);
            dumpDeviceArray(
                fn, storage.velocity_family_ptr(cumes::SpectralComponent::Zsc),
                n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/vel_vlmnsc_iter_%d.bin", iter2);
            dumpDeviceArray(
                fn, storage.velocity_family_ptr(cumes::SpectralComponent::Lsc),
                n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/vel_vrmnss_iter_%d.bin", iter2);
            dumpDeviceArray(
                fn, storage.velocity_family_ptr(cumes::SpectralComponent::Rss),
                n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/vel_vzmncs_iter_%d.bin", iter2);
            dumpDeviceArray(
                fn, storage.velocity_family_ptr(cumes::SpectralComponent::Zcs),
                n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/vel_vlmncs_iter_%d.bin", iter2);
            dumpDeviceArray(
                fn, storage.velocity_family_ptr(cumes::SpectralComponent::Lcs),
                n_spec);
        }
    }
    if (iter2 >= kE2Start && iter2 < kE2Start + 40) {
        char fn[128];
        snprintf(fn, sizeof fn, "dump/cuMES/fspec_precon_iter_%d.bin", iter2);
        dumpDeviceArray(fn, f_spec, (size_t)6 * p.mnmax * p.ns);
    }
}
#endif

// ---------------------------------------------------------------------------
// EquilibriumOperator (blueprint §6.11/§7): composes the per-iteration device
// DAG. The constructor builds the operator/workspace/views/buffers the DAG
// enqueues; enqueue() runs one pass from axis extrapolation through the
// preconditioned residual, reducing into the owned control record. The
// host-side controller decisions (descent, checkpoint capture/restore) stay
// with solverRun.
// ---------------------------------------------------------------------------
// One of the solver's own device buffers: carved from the stage DeviceArena
// when one is present (6.4 — one cudaMalloc backs the whole stage), else a
// standalone allocation (the legacy per-array path used by solverRun callers
// that pass no arena). The arena plan includes these spans (see
// stage_solver.hpp's measure_stage_stack).
template <typename T>
static cumes::DeviceBuffer<T> solverArenaBuffer(cumes::DeviceArena* arena,
                                                const char* name,
                                                std::size_t count) {
    if (arena) {
        return cumes::DeviceBuffer<T>(arena->alloc_span<T>(name, count), count);
    }
    return cumes::DeviceBuffer<T>(count);
}

template <typename T>
cumes::EquilibriumOperator<T>::EquilibriumOperator(
    const DeviceParams<T>& p,
    cumes::SpectralStorage<T>& storage,
    const cumes::Profiles<T>& profiles,
    cumes::ToroidalFftOperator<T>& transform,
    cumes::RealSpaceStorage<T>& rs,
    cumes::GeometryOperator<T>& geometry,
    cumes::DeviceArena* arena,
    cumes::SpectralOperator<T>* op,
    cumes::FreeBoundaryOperator<T>* vac)
    : p_(p),
      storage_(storage),
      profiles_(profiles),
      transform_(transform),
      rs_(rs),
      geometry_(geometry),
      precon_(p, arena),
      constraint_(p, arena),
      base_views_(geometry.base_geometry_views(p)),
      field_views_(geometry.magnetic_field_views(p)),
      rpv_(profiles.profile_views()),
      d_f_spec_(solverArenaBuffer<T>(arena,
                                     "solver/f_spec",
                                     6 * (size_t)p.ns * p.mnmax)),
      d_control_(
          solverArenaBuffer<cumes::ControlRecord>(arena, "solver/control", 1)),
      d_psum_(
          solverArenaBuffer<T>(arena, "solver/psum", 4 * (size_t)(p.ns - 1))),
      state_view_(storage.physical()),
      state_view_const_(storage.physical_const()),
      velocity_view_(storage.velocity()),
      residual_view_(d_f_spec_.data(), p.ns, p.mnmax),
      residual_view_const_(d_f_spec_.data(), p.ns, p.mnmax),
      transform_op_((op != nullptr) ? op : &transform),
      vac_(vac),
      d_buco_bvco_(solverArenaBuffer<T>(arena,
                                        "solver/buco_bvco",
                                        2 * (size_t)(p.ns - 1))),
      d_repack_(
          solverArenaBuffer<T>(arena,
                               "solver/lcfs_repack",
                               4 * (size_t)p.mpol * (size_t)(p.ntor + 1))),
      d_axis_(solverArenaBuffer<T>(arena, "solver/axis", 2 * (size_t)p.nzeta)),
      d_rbsq_(solverArenaBuffer<T>(arena, "solver/rbsq", (size_t)p.nZnT)),
      d_delbsq_(solverArenaBuffer<T>(arena, "solver/delbsq", 1)) {
    // View bundles over the stage-owned real-space storage + the constraint's
    // force buffers (built once per stage; cheap pointer aggregates).
    {
        auto geom = [&](T* d) {
            return cumes::RealFieldView<T>(d, p.ns, p.ntheta, p.nzeta);
        };
        geom_views_.r_e = geom(rs.d_r_e);
        geom_views_.z_e = geom(rs.d_z_e);
        geom_views_.l_e = geom(rs.d_l_e);
        geom_views_.ru_e = geom(rs.d_ru_e);
        geom_views_.zu_e = geom(rs.d_zu_e);
        geom_views_.lu_e = geom(rs.d_lu_e);
        geom_views_.r_o = geom(rs.d_r_o);
        geom_views_.z_o = geom(rs.d_z_o);
        geom_views_.l_o = geom(rs.d_l_o);
        geom_views_.ru_o = geom(rs.d_ru_o);
        geom_views_.zu_o = geom(rs.d_zu_o);
        geom_views_.lu_o = geom(rs.d_lu_o);
        geom_views_.rv_e = geom(rs.d_rv_e);
        geom_views_.zv_e = geom(rs.d_zv_e);
        geom_views_.lv_e = geom(rs.d_lv_e);
        geom_views_.rv_o = geom(rs.d_rv_o);
        geom_views_.zv_o = geom(rs.d_zv_o);
        geom_views_.lv_o = geom(rs.d_lv_o);

        auto forc = [&](T* d) {
            return cumes::RealFieldView<const T>(d, p.ns, p.ntheta, p.nzeta);
        };
        force_views_.armn_e = forc(rs.d_armn_e);
        force_views_.armn_o = forc(rs.d_armn_o);
        force_views_.azmn_e = forc(rs.d_azmn_e);
        force_views_.azmn_o = forc(rs.d_azmn_o);
        force_views_.brmn_e = forc(rs.d_brmn_e);
        force_views_.brmn_o = forc(rs.d_brmn_o);
        force_views_.bzmn_e = forc(rs.d_bzmn_e);
        force_views_.bzmn_o = forc(rs.d_bzmn_o);
        force_views_.blmn_e = forc(rs.d_blmn_e);
        force_views_.blmn_o = forc(rs.d_blmn_o);
        force_views_.clmn_e = forc(rs.d_clmn_e);
        force_views_.clmn_o = forc(rs.d_clmn_o);
        force_views_.crmn_e = forc(rs.d_crmn_e);
        force_views_.crmn_o = forc(rs.d_crmn_o);
        force_views_.czmn_e = forc(rs.d_czmn_e);
        force_views_.czmn_o = forc(rs.d_czmn_o);

        conforce_views_ = constraint_.constraint_force_views(p);
    }

    // Transform-timing event pairs (recorded in enqueue, read at the fence).
    cumes::check_cuda(cudaEventCreate(&ev0_inv_), "event create ev0_inv");
    cumes::check_cuda(cudaEventCreate(&ev1_inv_), "event create ev1_inv");
    cumes::check_cuda(cudaEventCreate(&ev0_fwd_), "event create ev0_fwd");
    cumes::check_cuda(cudaEventCreate(&ev1_fwd_), "event create ev1_fwd");

    // Env-gated dump-window knobs (defaults = input values).
    kMaxIterEff_ = p.max_iter;
    if (const char* e = getenv("CUMES_MAX_ITER")) kMaxIterEff_ = atoi(e);
    if (const char* e = getenv("CUMES_DUMP_ITER")) kDumpIter_ = atoi(e);
    if (const char* e = getenv("CUMES_E2_START")) kE2Start_ = atoi(e);
}

template <typename T>
cumes::EquilibriumOperator<T>::~EquilibriumOperator() {
    // Destructor context: report failures to stderr instead of throwing
    // (same policy as DeviceBuffer::release — a free failure must not throw
    // out of a destructor).
    const char* tags[4] = {"ev0_inv", "ev1_inv", "ev0_fwd", "ev1_fwd"};
    cudaEvent_t* evs[4] = {&ev0_inv_, &ev1_inv_, &ev0_fwd_, &ev1_fwd_};
    for (int i = 0; i < 4; ++i) {
        if (*evs[i] == nullptr) continue;
        cudaError_t err = cudaEventDestroy(*evs[i]);
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                         "EquilibriumOperator: cudaEventDestroy(%s) "
                         "failed: %s\n",
                         tags[i], cudaGetErrorString(err));
        }
    }
}

template <typename T>
void cumes::EquilibriumOperator<T>::enqueue(
    int iter,
    int iter2,
    const cumes::EvaluationSchedule& schedule,
    cudaStream_t stream,
    double f_norm_rz,
    double f_norm_l) {
    // Fixed-boundary composition: prefix + suffix back-to-back reproduce the
    // historical single enqueue launch-for-launch (the free-boundary path
    // calls the two halves separately with the host vacuum update between).
    enqueue_prefix(iter, iter2, schedule, stream, f_norm_rz, f_norm_l);
    enqueue_suffix(iter, iter2, schedule, stream, f_norm_rz, f_norm_l);
}

template <typename T>
void cumes::EquilibriumOperator<T>::enqueue_prefix(
    int iter,
    int iter2,
    const cumes::EvaluationSchedule& schedule,
    cudaStream_t stream,
    double f_norm_rz,
    double f_norm_l) {
#ifndef DUMP_CUMES_VERIFY
    // iter/iter2 feed only the dump windows; with the dump machinery
    // compiled out they are unused (completion plan step 4.1 warning-clean).
    (void)iter;
    (void)iter2;
#endif
    (void)f_norm_rz;  // consumed by the suffix's terminal predicate
    (void)f_norm_l;
    // Local aliases mirror the pre-step-12 solverRun variable names so the DAG
    // body below is a verbatim move (same arithmetic, same order).
    const DeviceParams<T>& p = p_;
    cumes::SpectralStorage<T>& storage = storage_;
    const cumes::Profiles<T>& profiles = profiles_;
    cumes::ToroidalFftOperator<T>& transform = transform_;
    cumes::RealSpaceStorage<T>& rs = rs_;
    cumes::GeometryOperator<T>& geometry = geometry_;
    cumes::Preconditioner<T>& precon = precon_;
    cumes::ConstraintOperator<T>& constraint = constraint_;
    cumes::BaseGeometryHalfViews<T>& base = base_views_;
    cumes::MagneticFieldViews<T>& field = field_views_;
    const cumes::RadialProfileViews<T>& rpv = rpv_;
    cumes::DeviceBuffer<T>& d_f_spec = d_f_spec_;
    cumes::DeviceBuffer<cumes::ControlRecord>& d_control = d_control_;
    cumes::DeviceBuffer<T>& d_psum = d_psum_;
    cumes::SpectralOperator<T>* transform_op = transform_op_;
    cumes::GeometryParityViews<T>& geom_views = geom_views_;
    cumes::ForceParityViews<const T>& force_views = force_views_;
    cumes::ConstraintForceViews<const T>& conforce_views = conforce_views_;
    cumes::SpectralView<T, cumes::DecomposedResidualDomain>& residual_view =
        residual_view_;
    cumes::SpectralView<const T, cumes::DecomposedResidualDomain>&
        residual_view_const = residual_view_const_;
    cudaEvent_t& ev0_inv = ev0_inv_;
    cudaEvent_t& ev1_inv = ev1_inv_;
    cudaEvent_t& ev0_fwd = ev0_fwd_;
    cudaEvent_t& ev1_fwd = ev1_fwd_;
    const int kDumpIter = kDumpIter_, kE2Start = kE2Start_,
              kMaxIterEff = kMaxIterEff_;

    // ---- device status reset (completion plan step 1.4) ----
    // Zero the WHOLE control record at pass start: the status bits then cannot
    // leak across passes, and any slot a guarded no-op leaves unwritten reads
    // as the deterministic zero sentinel (stale slots are never read by the
    // host on invalid/terminal passes, but determinism is the point).
    cumes::check_cuda(cudaMemsetAsync(d_control.data(), 0,
                                      sizeof(cumes::ControlRecord), stream),
                      "control reset");

    // Extrapolate m=1 coefficients to the magnetic axis (j=0)
    // before inverse DFT, matching vmecpp's extrapolateTowardsAxis().
    // Must be done each iteration since the descent step updates j=1
    // but skips j=0 for m>0 (axis regularity).
    extrapolateAxisKernel<T><<<(p.mnmax + 31) / 32, 32, 0, stream>>>(
        state_view_, p.ns, p.mnmax, p.ntor + 1);
    cumes::check_cuda(cudaGetLastError(), "extrapAxis");

    cumes::check_cuda(cudaEventRecord(ev0_inv, stream), "event record ev0_inv");
    // Fused inverse (blueprint §8.4/§8.5): geometry + the xmpq-weighted
    // rCon/zCon, dispatched through the unified SpectralOperator interface.
    // The generic backend accumulates rCon/zCon in the same launch as the
    // geometry; the axisymmetric backend runs its direct-poloidal rzCon
    // kernel right after its synthesis — both leave rCon/zCon produced on
    // the stream before this returns.
    transform_op->enqueue_inverse(storage.physical_const(), geom_views,
                                  constraint.rcon_view(p),
                                  constraint.zcon_view(p), stream);
    cumes::check_cuda(cudaEventRecord(ev1_inv, stream), "event record ev1_inv");

#ifdef DUMP_CUMES_VERIFY
    dumpIter0LoopDiag<T>(iter, p, rs);
#endif
#ifdef DUMP_CUMES_VERIFY
    dumpStepA<T>(iter, iter2, p, rs, transform, stream);
#endif

    // Base geometry (blueprint §6.7): staggered interpolation, Jacobian,
    // covariant metric — no 1/√g division.
    geometry.enqueue(rs, p, rpv, stream);

    // ---- Jacobian statistics: reset -> reduce -> finalize (plan step 1.4) --
    // Reduced into the typed record (device-only), ordered after the base
    // geometry but before the magnetic field. The finalize kernel writes
    // status.jacobian_valid with the SAME rule the host controller applies
    // at the fence (kJacobianEps / nZnT); every downstream cache/state
    // mutation and 1/√g consumer reads that bit and no-ops on an invalid
    // pass. The host gate remains authoritative for the restore/delt
    // bookkeeping — the device bit only prevents forbidden mutations.
    geometry.jacobian_stats(p, d_control.data(), stream);
    {
        jacobianFinalizeKernel<<<1, 1, 0, stream>>>(d_control.data(), p.nZnT);
        cumes::check_cuda(cudaGetLastError(), "jacobianFinalize");
    }

    // Magnetic field (1/√g B^θ/B^ζ + covariant B + total pressure + ncurr
    // closure) + the full-grid iota/chip update: every pass for ncurr=1
    // (current closure evolves iotaH/chipH), but for ncurr=0 the half-grid
    // profiles are fixed so the update is idempotent and runs only on the
    // first pass. Status-guarded: on an invalid-Jacobian pass none of the
    // field arrays or the evolved iotaH/chipH cache are written.
    cumes::MagneticFieldOperator<T> field_op;
    field_op.enqueue(rs, p, rpv, base, field, &d_control.data()->status, stream,
                     schedule.update_iota_chi);

#ifdef DUMP_CUMES_VERIFY
    dumpStepD<T>(iter, iter2, p, base, field);
#endif

    // Free-boundary bridge (vmecpp HandOverBoundaryGeometry /
    // HandOverMagneticAxis + the buco/bvco surface averages of
    // radialForceBalance): the vacuum block consumes THIS pass's LCFS state,
    // axis real-space geometry, and half-grid B field.
    if (schedule.run_vacuum_block) {
        vac_->enqueue_surface_averages(field.bsubu.data(), field.bsubv.data(),
                                       d_buco_bvco_.data(), p.ns, p.ntheta,
                                       p.nzeta, stream);
        vac_->enqueue_lcfs_repack(
            storage.family_ptr(cumes::SpectralComponent::Rcc),
            storage.family_ptr(cumes::SpectralComponent::Rss),
            storage.family_ptr(cumes::SpectralComponent::Zsc),
            storage.family_ptr(cumes::SpectralComponent::Zcs), d_repack_.data(),
            p.ns, p.mnmax, p.mpol, p.ntor, stream);
        vac_->enqueue_axis_extract(geom_views.r_e.data(), geom_views.z_e.data(),
                                   d_axis_.data(), p.ntheta, p.nzeta, stream);
    }
}

template <typename T>
void cumes::EquilibriumOperator<T>::enqueue_suffix(
    int iter,
    int iter2,
    const cumes::EvaluationSchedule& schedule,
    cudaStream_t stream,
    double f_norm_rz,
    double f_norm_l) {
    // Local aliases mirror the pre-split enqueue (same arithmetic, same
    // order — the suffix re-binds the aliases the prefix used).
    const DeviceParams<T>& p = p_;
    cumes::SpectralStorage<T>& storage = storage_;
    const cumes::Profiles<T>& profiles = profiles_;
    cumes::ToroidalFftOperator<T>& transform = transform_;
    cumes::RealSpaceStorage<T>& rs = rs_;
    cumes::GeometryOperator<T>& geometry = geometry_;
    cumes::Preconditioner<T>& precon = precon_;
    cumes::ConstraintOperator<T>& constraint = constraint_;
    cumes::BaseGeometryHalfViews<T>& base = base_views_;
    cumes::MagneticFieldViews<T>& field = field_views_;
    const cumes::RadialProfileViews<T>& rpv = rpv_;
    cumes::DeviceBuffer<T>& d_f_spec = d_f_spec_;
    cumes::DeviceBuffer<cumes::ControlRecord>& d_control = d_control_;
    cumes::DeviceBuffer<T>& d_psum = d_psum_;
    cumes::SpectralOperator<T>* transform_op = transform_op_;
    cumes::GeometryParityViews<T>& geom_views = geom_views_;
    cumes::ForceParityViews<const T>& force_views = force_views_;
    cumes::ConstraintForceViews<const T>& conforce_views = conforce_views_;
    cumes::SpectralView<T, cumes::DecomposedResidualDomain>& residual_view =
        residual_view_;
    cumes::SpectralView<const T, cumes::DecomposedResidualDomain>&
        residual_view_const = residual_view_const_;
    cudaEvent_t& ev0_fwd = ev0_fwd_;
    cudaEvent_t& ev1_fwd = ev1_fwd_;
    const int kDumpIter = kDumpIter_, kE2Start = kE2Start_,
              kMaxIterEff = kMaxIterEff_;
    (void)d_f_spec;    // consumed by the dump windows only
    (void)storage;     // consumed by the dump windows only
    (void)geom_views;  // consumed by the prefix's axis extract

    // rCon/zCon were produced by the fused inverse DFT above (blueprint
    // §8.4). Reset the constraint-force reference (rCon0/zCon0) to the
    // LCFS-extrapolated profile on the first iteration and after every
    // restart (iter2 == iter1), matching vmecpp's rzConIntoVolume
    // ("initialization/soft reset"). Status-guarded: the reference cache is
    // not mutated on an invalid pass (the re-anchored next pass re-runs it).
    if (schedule.reset_constraint_reference) {
        constraint.reset_reference(p, rpv.sqrtS_F, &d_control.data()->status,
                                   stream);
    }

    // Update the radial tridiagonal + lambda preconditioners BEFORE the
    // forces, so the constraint-force multiplier (tcon) can use the
    // current iteration's preconditioner elements, matching vmecpp
    // (updateRadialPreconditioner + constraintForceMultiplier at the
    // start of update()). Cadence matches vmecpp's
    // shouldUpdateRadialPreconditioner: (iter2 - iter1) % 25 == 0.
    // Status-guarded: the preconditioner element cache is not rebuilt on an
    // invalid pass (the re-anchor makes the next pass a refresh pass).
    const bool precon_updated = schedule.refresh_preconditioner;
    if (precon_updated) {
        precon.enqueue_compute(rs, transform.xm(), transform.xn(), p, rpv, base,
                               field, &d_control.data()->status, stream,
                               vac_ != nullptr);

        // vmecpp computeForceNorms (same cadence): device-side reduction
        // of the force-norm partial sums into the typed control record
        // (guarded + force_norms_evaluated set on device).
        enqueueForceNorms(storage.physical_const(), transform.xm(),
                          transform.xn(), p, rpv, geometry, d_psum.data(),
                          d_control.data(), stream);

        // Device finalize of the force-norm factors (completion-plan
        // follow-up §2.3): on a refresh pass the normalization is now
        // available BEFORE the device terminal predicate, so the converged
        // classification is no longer structurally disabled there. The
        // kernel reproduces the host's finalizeForceNorms expressions
        // exactly (see device_predicates.cuh); the host consumes the same
        // record fields at the fence instead of recomputing them.
        forceNormFinalizeKernel<<<1, 1, 0, stream>>>(
            d_control.data(), (double)profiles_.delta_s(), (double)p_.lamscale);
        cumes::check_cuda(cudaGetLastError(), "forceNormFinalize");

#ifdef DUMP_CUMES_VERIFY
        dumpStepPrecon<T>(iter, p, precon);
#endif
    }

    // Status-guarded MHD force: no force buffers are written on an invalid
    // pass (the host gate restores before anything consumes them).
    cumes::ForceOperator<T> force_op;
    force_op.enqueue(rs, p, rpv, base, field, &d_control.data()->status,
                     stream);

#ifdef DUMP_CUMES_VERIFY
    dumpStepEF<T>(iter, iter2, p, base, field, rs);
#endif

    // Vacuum edge force (vmecpp assembleTotalForces): rBSq from the vacuum
    // magnetic pressure (reduced-grid mirror + edge pressure) and the
    // LCFS-row force increment, gated on the vacuum state.
    if (schedule.apply_vacuum_edge_force) {
        cumes::check_cuda(
            cudaMemsetAsync(d_delbsq_.data(), 0, sizeof(T), stream),
            "delbsq reset");
        vac_->enqueue_rbsq(rs.d_r_e, rs.d_r_o, field.total_pressure.data(),
                           d_rbsq_.data(), d_delbsq_.data(), p.ns, p.ntheta,
                           p.nzeta, p.nZnT, T(profiles.delta_s()), stream);
        vac_->enqueue_edge_force(rs.d_armn_e, rs.d_armn_o, rs.d_azmn_e,
                                 rs.d_azmn_o, rs.d_zu_e, rs.d_zu_o, rs.d_ru_e,
                                 rs.d_ru_o, d_rbsq_.data(), p.ns, p.ntheta,
                                 p.nzeta, stream);
    }
    // rCon0/zCon0 decay on vacuum-active passes (vmecpp ideal_mhd_model.cc
    // :651-661), before the constraint force consumes the reference.
    if (schedule.decay_rcon0_zcon0) {
        vac_->enqueue_rcon_decay(constraint.rcon0(), constraint.zcon0(), p.ns,
                                 p.ntheta, p.nzeta, stream);
    }

    // Add spectral condensation constraint force to brmn/bzmn.
    // Uses the current-iteration tcon (refreshed above when the
    // preconditioner was updated), matching vmecpp. The de-alias bandpass
    // is dispatched through the unified SpectralOperator interface.
    // Status-guarded: the tcon cache and the constraint-force scratch are
    // not written on an invalid pass.
    constraint.enqueue(p, rs, precon.ard(), precon.azd(), rpv.sqrtS_F,
                       precon_updated, transform_op, &d_control.data()->status,
                       stream);

#ifdef DUMP_CUMES_VERIFY
    dumpStepG<T>(iter, p, rs, storage, constraint);
#endif

    cumes::check_cuda(cudaEventRecord(ev0_fwd, stream), "event record ev0_fwd");
    transform_op->enqueue_forward(force_views, conforce_views, residual_view,
                                  stream, schedule.apply_vacuum_edge_force);
    cumes::check_cuda(cudaEventRecord(ev1_fwd, stream), "event record ev1_fwd");

    // Apply the odd-m decomposition scaling (vmecpp decomposeInto).
    // The forward DFT already zeroed the LCFS R/Z entries and the axis
    // m>0 entries, so the boundary stays rigid and only the lambda
    // force is present at the LCFS (free gauge, evolved by descent).
    {
        dim3 bs(256), gs((p.ns * p.mnmax + 255) / 256);
        scalxcApplyKernel<T><<<gs, bs, 0, stream>>>(
            residual_view, rpv.sqrtS_F, transform.xm(), p.ns, p.mnmax,
            std::sqrt(T(1.0) / T(p.ns - 1)));
        cumes::check_cuda(cudaGetLastError(), "scalxc");
    }

#ifdef DUMP_CUMES_VERIFY
    dumpStepH<T>(iter, iter2, p, d_f_spec.data(), kDumpIter, kE2Start);
#endif

    // ---- m1 gauge constraint (vmecpp FourierCoeffs::m1Constraint) ----
    // Applied to the decomposed forces after the forward DFT (post
    // step_H dump), before the residuals and the preconditioner,
    // matching vmecpp ideal_mhd_model.cc. The fzcs zeroing mirrors
    // vmecpp's `fix_m1_gauge = always_fix_m1_gauge || fsqz < 1e-6 ||
    // iter2 < 2` with always_fix_m1_gauge = false (the standalone
    // default): zeroZ only on the first pass and once the previous
    // pass's invariant Z-residual dropped below 1e-6.
    {
        dim3 b1(256), g1((p.ns + 255) / 256);
        int zeroZ = schedule.zero_z_force_m1;
        m1ConstraintKernel<T><<<g1, b1, 0, stream>>>(residual_view, p.ns,
                                                     p.mnmax, p.ntor, zeroZ);
        cumes::check_cuda(cudaGetLastError(), "m1Constraint");
    }

    // ---- Invariant (unpreconditioned) residuals ----
    // Reduced into rec.invariant_raw. Stream order guarantees the reduction
    // completes before the terminal predicate and the in-place
    // preconditioner, so the three never race on the residual slab.
#ifdef DUMP_CUMES_VERIFY
    dumpStepFinal<T>(iter, p, d_f_spec.data(), kMaxIterEff);
#endif
    cumes::ResidualOperator<T> residual_op;
    residual_op.enqueue(residual_view_const, p.ns, p.mnmax,
                        schedule.include_edge_rz_invariant,
                        d_control.data()->invariant_raw, stream);

    // ---- Device terminal predicate (blueprint §6.9/§7 "Terminal") ----
    // Classify the invariant residual ON DEVICE before in-place
    // preconditioning: nonfinite always; converged on every pass
    // (completion-plan follow-up §2.3). On a refresh pass the factors were
    // finalized ON DEVICE from THIS pass's force norms by
    // forceNormFinalizeKernel above, so the predicate reads them from the
    // record (use_record_factors=1); on other passes it uses the host's
    // cached factors by value. The host consumes the same record fields at
    // the fence, so the device bits and the host decision share bit-identical
    // inputs. The preconditioner and the preconditioned reduction read the
    // bits and no-op on terminal passes, marking their fields not_evaluated.
    {
        invariantPredicateKernel<<<1, 1, 0, stream>>>(
            d_control.data(), f_norm_rz, f_norm_l,
            (double)p.mnmax * (double)p.ns, (double)p.ftol,
            schedule.refresh_preconditioner ? 1 : 0);
        cumes::check_cuda(cudaGetLastError(), "invariantPredicate");
    }

    // vmecpp applyM1Preconditioner: m=1 frss scale, before the RZ solve.
    // Moved into the Preconditioner operator (it reads the odd-parity
    // diagonal elements pw.d_ard/d_brd/azd/bzd). Terminal-guarded.
    precon.enqueue_m1_scale(residual_view, p, &d_control.data()->status,
                            stream);

    // Apply the radial tridiagonal + lambda preconditioners to the
    // (decomposed) spectral forces. Terminal-guarded. The boundary row
    // joins the solve only when the vacuum edge force is active (vmecpp
    // applyRZPreconditioner's jMax gate).
    precon.enqueue_apply(residual_view, p, &d_control.data()->status, stream,
                         schedule.apply_vacuum_edge_force);

#ifdef DUMP_CUMES_VERIFY
    dumpStepI<T>(iter, iter2, p, d_f_spec.data(), storage, kDumpIter, kE2Start);
#endif

    // ---- Preconditioned residuals (vmecpp fsqr1/fsqz1/fsql1) ----
    // Terminal-guarded reduction: zero sentinel + not_evaluated on
    // nonfinite/converged passes, real values + evaluated bit otherwise.
    residual_op.enqueue_preconditioned(residual_view_const, p.ns, p.mnmax,
                                       d_control.data(), stream);
}

template <typename T>
SolverResult<T> solverRun(cumes::SpectralStorage<T>& storage,
                          const DeviceParams<T>& p,
                          const cumes::Profiles<T>& profiles,
                          cumes::ToroidalFftOperator<T>& transform,
                          cumes::RealSpaceStorage<T>& rs,
                          cumes::GeometryOperator<T>& geometry,
                          cumes::DeviceArena* arena,
                          cudaStream_t stream,
                          cumes::SolverBench* bench,
                          cumes::SpectralOperator<T>* op,
                          cumes::FreeBoundaryOperator<T>* vac) {
    const cumes::RadialProfileViews<T> rpv = profiles.profile_views();
#ifndef DUMP_CUMES_VERIFY
    (void)rpv;  // only the dump-window step_0 snapshots consume it
#endif
    SolverResult<T> res{false, 0, T(1.0), T(1.0), T(1.0), p.delt, {}};

    // The per-iteration DAG (blueprint §6.11/§7): owns the operators,
    // workspaces, views, residual/control buffers and the interleaved dump
    // machinery. solverRun below is a thin loop over the controller + this
    // operator (migration step 12).
    cumes::EquilibriumOperator<T> equilibrium(p, storage, profiles, transform,
                                              rs, geometry, arena, op, vac);

    // Stateless descent operator (the DAG's field/force/residual operators live
    // inside the EquilibriumOperator; descent stays with the solver).
    cumes::DescentOperator<T> descent_op;

    // Bind every cuFFT plan to the explicit compute stream so the batched
    // ζ-transforms execute in stream order with the surrounding kernels
    // (blueprint §6.6). Plans are created once per stage; the operator binds
    // its own plans here, before any transform runs.
    transform.bind_stream(stream);

    // ---- env-gated knobs for convergence experiments (defaults = input
    // values; set via CUMES_MAX_ITER, CUMES_DELT0, CUMES_DTAU_FLOOR) ----
    // (CUMES_DUMP_ITER / CUMES_E2_START are consumed by the
    // EquilibriumOperator.)
    int kMaxIterEff = p.max_iter;
    double kDelt0Eff = p.delt;  // double: atof knob, converted to T at use
    double kDtauFloor = 0.0;
    if (const char* e = getenv("CUMES_MAX_ITER")) kMaxIterEff = atoi(e);
    if (const char* e = getenv("CUMES_DELT0")) kDelt0Eff = atof(e);
    if (const char* e = getenv("CUMES_DTAU_FLOOR")) kDtauFloor = atof(e);
    printf("knobs: max_iter=%d delt0=%.3f dtau_floor=%.3e\n", kMaxIterEff,
           kDelt0Eff, kDtauFloor);

    // ---- vmecpp VMEC_8_52 time-step control state ----
    // iter2: effective iteration counter — does NOT advance on restart
    // passes (vmecpp's bad_resets mechanism). iter1: branch-point marker,
    // set to the current iteration on every restart; grace periods and the
    // invTau reinitialization key off (iter2 - iter1). res0: running minimum
    // of the preconditioned residual sum fsq. All of this lives in a pure
    // host state machine (Phase 4): the solver below launches kernels and
    // applies the returned decisions in the exact frozen order.
    // ADR-0001 follow-up: the controller runs in DOUBLE in both builds
    // (double build: identity — Class A bitwise; float build: the double
    // accumulations reach the host decisions unrounded — Class B).
    cumes::IterationController<double> controller(
        {kDelt0Eff, (double)p.ftol, kDtauFloor});

    // vmecpp residual normalization factors (computeForceNorms), refreshed on
    // the same cadence as the preconditioner (every kPreconInterval passes).
    double fNormRZ = 0.0, fNormL = 0.0, fNorm1 = 0.0;

    // The previous pass's invariant residuals: the vacuum state machine's
    // ramp/nvacskip extension reads fsqr+fsqz (vmecpp m_fc_.fsqr/fsqz at the
    // free-boundary block). 1.0 keeps the first pass's gate safely open
    // (the block does not run at iter2==1 anyway).
    double prev_fsqr = 1.0, prev_fsqz = 1.0;
    // Host copies of the buco/bvco surface averages for the vacuum update.
    std::vector<double> h_buco_bvco(2 * (size_t)(p.ns - 1), 0.0);

    // Pinned mirror of the typed control record (one async D2H per pass,
    // delivered by the single control fence — completion plan step 1.3).
    cumes::PinnedBuffer<cumes::ControlRecord> h_control_pin(1);
    // Persistent pinned mirror for the displayed axis/boundary values
    // (completion plan step 3.3): [0..ntor] = the m=0 axis R coefficients,
    // [ntor+1] = the boundary rmncc(0,0) LCFS value. Copied ONCE per pass on
    // the compute stream at the single control fence — the console output
    // reads pure host memory and never adds a device/stream-wide fence or a
    // per-print allocation.
    cumes::PinnedBuffer<T> h_axis_pin(static_cast<std::size_t>(p.ntor) + 2);

    // State rollback: one contiguous state-only checkpoint slab (6*mnmax*ns),
    // replacing the six separate d_bk_* arrays. The slab order matches the six
    // old families, so backup/restore become single copies.
    cumes::DeviceBuffer<T> checkpoint(6 * (size_t)p.ns * p.mnmax);

    // Helper: copy current spectral state to backup (one device-to-device copy)
    auto backupState = [&]() {
        checkpoint.copy_from_async(storage.state_buffer(), stream);
    };

    // Helper: restore spectral state from backup + zero velocities
    auto restoreState = [&]() {
        storage.state_buffer().copy_from_async(checkpoint, stream);
        storage.velocity_buffer().zero_async(stream);
    };

    // Take initial backup
    backupState();

    // Radial location of the magnetic axis at zeta=0, matching vmecpp's r00
    // (Printout: geometric_offset.r_00 = r1_e[0] — the real-space even-m R
    // at the axis at theta=0, zeta=0). With the axis coefficients this is
    // the sum of the m=0 row: sum_n rmncc(0,n)@axis * cos(n*nfp*0) — the
    // plain R_00 coefficient alone misses the axis R wobble with zeta
    // (~+0.36 for W7-X, dominated by rmnc(0,1)@axis = +0.35).
    // Fence-time axis R read from the pinned telemetry mirror (pure host
    // memory — no sync, no allocation). The displayed value lags the post-
    // descent state by one descent (the mirror is copied at the pass's
    // control fence); controller decisions never consume it.
    auto axisRAtZeta0 = [&]() {
        T h = T(0.0);
        for (int n = 0; n <= p.ntor; ++n) h += h_axis_pin.data()[n];
        return h;
    };

#ifdef DUMP_CUMES_VERIFY
    // Per-pass record for convergence analysis (mirrors vmecpp's
    // per_iter_residuals.bin + control scalars). Typed PassRecord; the field
    // order is the frozen 15-column on-disk contract (fsqr_i fsqz_i fsql_i
    // fsqr fsqz fsql delt otav dtau b1 fac iter2 iter1 reason rax). Observers
    // read these scalars and cannot affect the controller's decisions.
    std::vector<cumes::PassRecord> per_iter;
    per_iter.reserve((size_t)kMaxIterEff);
    // The dump record (per_iter_residuals_cumes.bin) must stay byte-identical
    // to the frozen baseline, whose axis_r column was read POST-descent with
    // a synchronized copy. Keep that exact read for the dump-only record
    // (dump mode already performs device-wide dumps; the extra sync costs
    // nothing there and is compiled out of production).
    auto axisRAtZeta0Sync = [&]() {
        cudaStreamSynchronize(stream);
        std::vector<T> h_ax(static_cast<std::size_t>(p.ntor) + 1);
        cumes::check_cuda(
            cudaMemcpy2D(h_ax.data(), sizeof(T),
                         storage.family_ptr(cumes::SpectralComponent::Rcc),
                         (size_t)p.ns * sizeof(T), sizeof(T), p.ntor + 1,
                         cudaMemcpyDeviceToHost),
            "cpy Rax");
        T h = T(0.0);
        for (int n = 0; n <= p.ntor; ++n) h += h_ax[n];
        return h;
    };
    auto recordPass = [&](int reason, double fRi, double fZi, double fLi,
                          double fR, double fZ, double fL, double d, double o,
                          double dt, double b1v, double fcv, int it2, int it1) {
        if (!dumpEnabled()) return;
        if ((int)per_iter.size() < kMaxIterEff) {
            cumes::PassRecord r;
            r.invariant_fsqr = fRi;
            r.invariant_fsqz = fZi;
            r.invariant_fsql = fLi;
            r.preconditioned_fsqr = fR;
            r.preconditioned_fsqz = fZ;
            r.preconditioned_fsql = fL;
            r.delta_t = d;
            r.otav = o;
            r.dtau = dt;
            r.b1 = b1v;
            r.fac = fcv;
            r.iter2 = (double)it2;
            r.iter1 = (double)it1;
            r.reason = (double)reason;
            r.axis_r = (double)axisRAtZeta0Sync();
            per_iter.push_back(r);
        }
    };
#endif

#ifdef DUMP_CUMES_VERIFY
    // Diagnostic: test inverse DFT at specified surface (CUMES_DUMP=1 only).
    // do_combine=false: the diagnostic reads only the parity arrays; the
    // combined *_real buffers are materialized on demand (fourierCombineParity)
    // at the dump site, never read stale.
    if (dumpEnabled()) {
        transform.enqueue_inverse_dump(storage.physical_const(), stream);
        cudaDeviceSynchronize();  // dump-only read of compute-stream data
        auto* h_re = new T[p.nZnT * p.ns];
        auto* h_ro = new T[p.nZnT * p.ns];
        cumes::check_cuda(cudaMemcpy(h_re, rs.d_r_e, p.nZnT * p.ns * sizeof(T),
                                     cudaMemcpyDeviceToHost),
                          "diag re");
        cumes::check_cuda(cudaMemcpy(h_ro, rs.d_r_o, p.nZnT * p.ns * sizeof(T),
                                     cudaMemcpyDeviceToHost),
                          "diag ro");
        // Check surface j=ns-1 (LCFS): r_e should be rbc[0]*cos(0)=3.999, r_o
        // should be sum of odd m
        int jB = p.ns - 1;
        double re_lcfs = h_re[0 + jB * p.nZnT];  // theta=0
        double ro_lcfs = h_ro[0 + jB * p.nZnT];
        printf(
            "  [diag] LCFS theta=0: r_e=%.4f r_o=%.4f r_total=%.4f (expect "
            "~3.93 + ~1.03 = ~4.96)\n",
            re_lcfs, ro_lcfs, re_lcfs + ro_lcfs);
        delete[] h_re;
        delete[] h_ro;
    }
#endif  // DUMP_CUMES_VERIFY

    printf("\n ITER |    FSQR        FSQZ        FSQL    |   DELT\n");
    printf("------+------------------------------------+----------\n");

    // One table row: effective iteration, invariant residuals, delt, and the
    // axis / boundary R_00 (same columns as vmecpp's iteration printout).
    auto printIterRow = [&](int it2, double fsqr_v, double fsqz_v,
                            double fsql_v, double delt_v) {
        printf("%5d | %11.3e %11.3e %11.3e | %8.2e", it2, (double)fsqr_v,
               (double)fsqz_v, (double)fsql_v, (double)delt_v);
        // Both values come from the pinned telemetry mirror (completion
        // plan step 3.3) — no per-print device copy or synchronization.
        T h_rmncc_axis = axisRAtZeta0();
        T h_rmncc_bnd = h_axis_pin.data()[p.ntor + 1];
        printf(" | Rax=%.4f Rbnd=%.4f\n", (double)h_rmncc_axis,
               (double)h_rmncc_bnd);
    };

#ifdef DUMP_CUMES_VERIFY
    {
        dumpEnsureDir();
        size_t n_spec = (size_t)p.ns * (size_t)p.mnmax;
        dumpDeviceArray("dump/cuMES/step_0_rmncc.bin",
                        storage.family_ptr(cumes::SpectralComponent::Rcc),
                        n_spec);
        dumpDeviceArray("dump/cuMES/step_0_zmnsc.bin",
                        storage.family_ptr(cumes::SpectralComponent::Zsc),
                        n_spec);
        dumpDeviceArray("dump/cuMES/step_0_lmnsc.bin",
                        storage.family_ptr(cumes::SpectralComponent::Lsc),
                        n_spec);
        dumpDeviceArray("dump/cuMES/step_0_rmnss.bin",
                        storage.family_ptr(cumes::SpectralComponent::Rss),
                        n_spec);
        dumpDeviceArray("dump/cuMES/step_0_zmncs.bin",
                        storage.family_ptr(cumes::SpectralComponent::Zcs),
                        n_spec);
        dumpDeviceArray("dump/cuMES/step_0_lmncs.bin",
                        storage.family_ptr(cumes::SpectralComponent::Lcs),
                        n_spec);
        dumpDeviceArray("dump/cuMES/step_0_currH.bin", rpv.curr_H, p.ns - 1);
        dumpDeviceArray("dump/cuMES/step_0_chipH.bin", rpv.chip_H, p.ns - 1);
        dumpDeviceArray("dump/cuMES/step_0_iotaH.bin", rpv.iota_H, p.ns - 1);
        dumpDeviceArray("dump/cuMES/step_0_iotaF.bin", rpv.iota_F, p.ns);
    }
#endif

    // Opt-in benchmark observer: wall-clock the hot loop at the single control
    // fence (see solver_bench.hpp). bench_t_prev is reset at each fence so each
    // sample spans one full evaluated pass (host decisions + descent enqueue +
    // the next pass's kernels, synced by that next fence).
    std::chrono::steady_clock::time_point bench_t_prev{};
    if (bench && bench->enabled) {
        bench->pass_wall_us.reserve((size_t)kMaxIterEff);
        bench_t_prev = std::chrono::steady_clock::now();
    }

    for (int iter = 0; iter < kMaxIterEff; ++iter) {
        // Snapshot of the controller's effective iteration for this pass's
        // dump windows (constant until after_descent at the end of the body;
        // the post-descent output block reads the controller directly).
        const int iter2 = controller.effective_iteration();
        // vmecpp: after 25/50 bad Jacobians, restore the state and reset the
        // time step to 0.98/0.96 of the INITIAL delt (vmec.cc, "HAVING A
        // CONVERGENCE PROBLEM: RESETTING DELT"). The controller's
        // next_schedule() performs the ++ijacob / delt reset / re-anchor; the
        // restoreState() below mirrors vmecpp's RestartIteration(BAD_JACOBIAN).
        if (controller.next_schedule()) {
            restoreState();
            printf(
                "  -> CONVERGENCE PROBLEM: RESETTING DELT to %.3e "
                "(ijacob=%d)\n",
                (double)controller.delta_t(), controller.bad_jacobian_count());
            continue;
        }

        // Free-boundary host state machine (vmecpp ideal_mhd_model.cc
        // :605-646): the top-of-block advance computes the block gate,
        // ivacskip, the ramp (full update while not yet ACTIVE), the
        // adaptive nvacskip extension, and the rCon0/zCon0 decay flag from
        // the PREVIOUS pass's invariant residuals.
        if (vac != nullptr) {
            vac->advance(iter2, controller.restart_anchor(), prev_fsqr,
                         prev_fsqz);
        }

        // Extrapolate
        // Build the per-iteration schedule from the controller's pure host
        // decisions, then enqueue the device DAG (blueprint §6.11/§7).
        cumes::EvaluationSchedule schedule;
        schedule.update_iota_chi = (p.ncurr == 1) || (iter == 0);
        schedule.reset_constraint_reference =
            controller.reset_constraint_reference() &&
            // vmecpp rzConIntoVolume gate: the reference reset runs only
            // while the vacuum pressure is OFF or just ramping up.
            (vac == nullptr ||
             (vac->state() != cumes::VacuumState::INITIALIZED &&
              vac->state() != cumes::VacuumState::ACTIVE));
        schedule.refresh_preconditioner = controller.refresh_preconditioner();
        schedule.zero_z_force_m1 = (controller.effective_iteration() < 2) ||
                                   (controller.fsqz_prev() < 1.0e-6);
        schedule.run_vacuum_block = (vac != nullptr) && vac->run_vacuum_block();
        // VMEC normally excludes the free-boundary LCFS R/Z force from the
        // invariant norm. Its compatibility gate includes it only briefly
        // near convergence (or on the first pass of a hot restart).
        if (vac != nullptr) {
            const bool almost_converged = (prev_fsqr + prev_fsqz) < 1.0e-6;
            const bool hot_restart =
                iter2 == 1 && vac->state() == cumes::VacuumState::INITIALIZED;
            schedule.include_edge_rz_invariant =
                (iter2 - controller.restart_anchor()) < 50 &&
                (almost_converged || hot_restart);
        }
        schedule.decay_rcon0_zcon0 =
            (vac != nullptr) && vac->decay_rcon0_zcon0();
        // apply_vacuum_edge_force is set below by the HOST vacuum update
        // (it depends on the post-update state); the fixed path and the
        // block-inactive passes keep it false.
        schedule.apply_vacuum_edge_force = false;

        if (vac == nullptr || !schedule.run_vacuum_block) {
            equilibrium.enqueue(iter, iter2, schedule, stream, fNormRZ, fNormL);
        } else {
            // Free-boundary passes: prefix -> vacuum fence -> HOST vacuum
            // update -> suffix. The vfield solver runs on the legacy
            // default stream with synchronous copies, so the compute stream
            // must be drained first (its kernels produced the LCFS state,
            // the repack, and the buco/bvco averages).
            equilibrium.enqueue_prefix(iter, iter2, schedule, stream, fNormRZ,
                                       fNormL);
            cumes::check_cuda(cudaStreamSynchronize(stream), "vacuum fence");
            cumes::check_cuda(
                cudaMemcpy(h_buco_bvco.data(), equilibrium.buco_bvco_device(),
                           2 * (size_t)(p.ns - 1) * sizeof(T),
                           cudaMemcpyDeviceToHost),
                "cpy buco/bvco");
            // The host update runs the vacuum solve (its own device work on
            // the legacy stream), promotes the state, checks the rBtor/
            // cTor consistency, and arms the soft-restart/edge-force gates.
            vac->run_host_update(
                p.ns, h_buco_bvco.data(), h_buco_bvco.data() + (p.ns - 1),
                equilibrium.repack_device(), equilibrium.axis_device(),
                equilibrium.axis_device() + p.nzeta, stream);
            schedule.apply_vacuum_edge_force = vac->apply_edge_force();
            equilibrium.enqueue_suffix(iter, iter2, schedule, stream, fNormRZ,
                                       fNormL);
        }

        // ---- ONE combined control fence (Phase 6A) ----
        // Jacobian stats + invariant + preconditioned residuals are one device
        // record; transfer it with one async copy and sync once. This replaces
        // the three per-pass host barriers (Jacobian gate, invariant,
        // preconditioned) of the pre-6A loop.
        cumes::check_cuda(
            cudaMemcpyAsync(h_control_pin.data(), equilibrium.control_device(),
                            sizeof(cumes::ControlRecord),
                            cudaMemcpyDeviceToHost, stream),
            "cpy control");
        // Axis/boundary telemetry mirror: copied on the SAME stream before
        // the fence (mode-major [mode*ns + j]: the m=0 modes at the axis row
        // sit at indices n*ns — one 2D strided copy grabs all ntor+1).
        cumes::check_cuda(
            cudaMemcpy2DAsync(h_axis_pin.data(), sizeof(T),
                              storage.family_ptr(cumes::SpectralComponent::Rcc),
                              (size_t)p.ns * sizeof(T), sizeof(T),
                              static_cast<std::size_t>(p.ntor) + 1,
                              cudaMemcpyDeviceToHost, stream),
            "cpy Rax mirror");
        cumes::check_cuda(
            cudaMemcpyAsync(
                h_axis_pin.data() + (p.ntor + 1),
                storage.family_ptr(cumes::SpectralComponent::Rcc) + (p.ns - 1),
                sizeof(T), cudaMemcpyDeviceToHost, stream),
            "cpy Rbnd mirror");
        cumes::check_cuda(cudaStreamSynchronize(stream), "control sync");
        if (vac != nullptr) {
            // delBSq surface-mean diagnostic (log only): the suffix's rBSq
            // kernel reduced it; read it at the same fence.
            T h_delbsq = T(0);
            cumes::check_cuda(cudaMemcpy(&h_delbsq, equilibrium.delbsq_device(),
                                         sizeof(T), cudaMemcpyDeviceToHost),
                              "cpy delbsq");
            vac->set_delbsq(h_delbsq);
        }
        if (bench && bench->enabled) {
            auto bench_now = std::chrono::steady_clock::now();
            bench->pass_wall_us.push_back(
                std::chrono::duration<double, std::micro>(bench_now -
                                                          bench_t_prev)
                    .count());
            bench_t_prev = bench_now;
        }
        const cumes::ControlRecord& rec = *h_control_pin.data();
        // Sample the transform-timing events at this fence (both transforms
        // preceded it on the same stream).
        equilibrium.sample_transform_timing();

        const double plain_per_el = (double)p.mnmax * (double)p.ns;

        // ---- Oriented-Jacobian validity gate (vmecpp bad-jacobian) ----
        // Checked FIRST: an invalid geometry can make the downstream residual
        // (reduced on that geometry) nonfinite or garbage, so the restore
        // decision must precede the residual classification. The delt shrink
        // and re-anchor bookkeeping live in the controller; the solver only
        // restores. The persistent preconditioner/constraint caches the pass
        // may have touched are self-healing: the re-anchor makes the next pass
        // a refresh+reset pass (iter2==iter1), which rebuilds them from the
        // restored geometry.
        cumes::JacobianStatus<double> js;
        js.min_oriented = rec.jacobian_min_oriented;
        js.max_abs = rec.jacobian_max_abs;
        js.nonfinite_count = rec.jacobian_nonfinite_count;
        js.min_index = (int)rec.jacobian_min_index;
        const double delt_before = controller.delta_t();
        const int it2_before = controller.effective_iteration();
        const int it1_before = controller.restart_anchor();
        const bool host_jac_invalid = controller.jacobian_invalid(js, p.nZnT);
#ifdef DUMP_CUMES_VERIFY
        // Device/host rule consistency (dump-only observability): the finalize
        // kernel and the controller must decide identically, or the guards
        // would suppress work the controller later consumes.
        if (dumpEnabled() &&
            ((rec.status.jacobian_valid != 0) == host_jac_invalid)) {
            fprintf(stderr,
                    "cuMES: WARNING: device jacobian status (%s) disagrees "
                    "with the host gate (%s) at pass %d\n",
                    rec.status.jacobian_valid ? "valid" : "invalid",
                    host_jac_invalid ? "invalid" : "valid",
                    controller.effective_iteration());
        }
#endif
        if (host_jac_invalid) {
#ifdef DUMP_CUMES_VERIFY
            recordPass(1, 0, 0, 0, 0, 0, 0, delt_before, 0, 0, 0, 0, it2_before,
                       it1_before);
#endif
            restoreState();
#ifdef DUMP_CUMES_VERIFY
            // Event line compiled out of release (fast) builds along with the
            // rest of the dump machinery; the iteration table still gets a row.
            printf(
                "  -> BAD JACOBIAN (invalid √g: min(signJ·√g)=%.3e "
                "max|√g|=%.3e nonfinite=%.0f at jH=%d) delt=%.3e\n",
                (double)js.min_oriented, (double)js.max_abs,
                (double)js.nonfinite_count, js.min_index / p.nZnT,
                (double)controller.delta_t());
#endif
            printIterRow(controller.effective_iteration(), T(1.0), T(1.0),
                         T(1.0), controller.delta_t());
            continue;
        }

        // On a refresh pass, consume the DEVICE-finalized force-norm factors
        // (completion-plan follow-up §2.3): forceNormFinalizeKernel computed
        // them from THIS pass's force norms before the terminal predicate, so
        // the host decision and the device converged bit share bit-identical
        // inputs — no recomputation, no disagreement window. On non-refresh
        // passes the cached factors are reused.
        if (schedule.refresh_preconditioner) {
            fNormRZ = rec.final_f_norm_rz;
            fNormL = rec.final_f_norm_l;
            fNorm1 = rec.final_f_norm1;
            dumpForceNorms(rec.force_norms, (double)profiles.delta_s(), iter2,
                           fNormRZ, fNormL, fNorm1);
        }

        // Vacuum-activation soft restart (vmecpp UpdateForwardModel ->
        // RestartIteration): vmecpp restores the state to the RUNNING-MINIMUM
        // BACKUP (the physical_x_backup_ refreshed on do_refresh passes — the
        // initial state before the first refresh) and zeroes the velocities;
        // the controller re-anchors and counts the event (see
        // vacuum_soft_restart). NO delt shrink: vmecpp applies the x0.9 to a
        // local delt0 copy, not the maintained one. The pass still classifies,
        // descends, and advances iter2; the re-anchored decide_restart
        // re-initializes the 1/tau history and rebaselines res0 exactly as
        // vmecpp's Evolve does after the restart. The checkpoint slab IS
        // vmecpp's backup (same refresh cadence), so the restore is NOT a
        // no-op — verified against the explore-branch stage-1 trajectory,
        // where the pass-3 state equals the initial state bit-for-bit.
        if (vac != nullptr && vac->soft_restart_requested()) {
            restoreState();  // backup state + zero velocities (vmecpp
                             // RestartIteration BAD_JACOBIAN branch)
            controller.vacuum_soft_restart();
        }

        // ---- Invariant residuals (vmecpp evalFResInvar) ----
        // fsqr = fResInvar[0]·fNormRZ·0.25 (same for fsqz), fsql =
        // fResInvar[2]·fNormL. The kernel returns ΣF²/(mnmax·ns); undo first.
        double fsqr_i = rec.invariant_raw[0] * plain_per_el * fNormRZ * 0.25;
        double fsqz_i = rec.invariant_raw[1] * plain_per_el * fNormRZ * 0.25;
        double fsql_i = rec.invariant_raw[2] * plain_per_el * fNormL;
        const double inv_triple[3] = {fsqr_i, fsqz_i, fsql_i};
        // Hand the residuals to the vacuum state machine for the NEXT pass
        // (vmecpp's free-boundary block reads the current m_fc_.fsqr/fsqz).
        prev_fsqr = fsqr_i;
        prev_fsqz = fsqz_i;

        // ---- Stopping criterion (vmecpp Evolve) ----
        // classify_invariant records fsqz_prev for the next pass's gauge
        // condition, then reports nonfinite (recover) or converged (stop).
        cumes::InvariantVerdict verdict =
            controller.classify_invariant(inv_triple);
#ifdef DUMP_CUMES_VERIFY
        // Device/host terminal-predicate consistency (dump-only): the device
        // bits and the host verdict must now agree on EVERY pass — on refresh
        // passes both consume the record's device-finalized factors
        // (completion-plan follow-up §2.3), on other passes the host's cached
        // factors travel by value into the predicate.
        if (dumpEnabled()) {
            const bool dev_nf = rec.status.invariant_nonfinite != 0;
            const bool dev_cv = rec.status.invariant_converged != 0;
            const bool nf_ok = dev_nf == verdict.nonfinite;
            const bool cv_ok = dev_cv == verdict.converged;
            const bool ev_ok = verdict.nonfinite || verdict.converged ||
                               rec.status.preconditioned_evaluated != 0;
            if (!nf_ok || !cv_ok || !ev_ok) {
                fprintf(stderr,
                        "cuMES: WARNING: device predicates (nf=%d cv=%d "
                        "prec_eval=%d) disagree with host (%d %d) at pass %d\n",
                        (int)rec.status.invariant_nonfinite,
                        (int)rec.status.invariant_converged,
                        (int)rec.status.preconditioned_evaluated,
                        (int)verdict.nonfinite, (int)verdict.converged,
                        controller.effective_iteration());
            }
        }
#endif
        if (verdict.nonfinite) {
            // vmecpp hard-fails on non-finite residuals (status BAD_JACOBIAN);
            // we recover instead: restore the last good state and shrink delt.
#ifdef DUMP_CUMES_VERIFY
            recordPass(1, fsqr_i, fsqz_i, fsql_i, 0, 0, 0, delt_before, 0, 0, 0,
                       0, it2_before, it1_before);
#endif
            restoreState();
#ifdef DUMP_CUMES_VERIFY
            // Event line compiled out of release (fast) builds (see above).
            printf("  -> BAD JACOBIAN (non-finite residuals) delt=%.3e\n",
                   (double)controller.delta_t());
#endif
            printIterRow(controller.effective_iteration(), fsqr_i, fsqz_i,
                         fsql_i, controller.delta_t());
            continue;
        }
        if (verdict.converged) {
#ifdef DUMP_CUMES_VERIFY
            recordPass(0, fsqr_i, fsqz_i, fsql_i, 0, 0, 0, controller.delta_t(),
                       0, 0, 0, 0, controller.effective_iteration(),
                       controller.restart_anchor());
#endif
            res.converged = true;
            res.iterations = controller.effective_iteration();
            res.fsqr = (T)fsqr_i;
            res.fsqz = (T)fsqz_i;
            res.fsql = (T)fsql_i;
            res.delt = (T)controller.delta_t();
            // Report the EFFECTIVE iteration count (iter2): restart passes
            // don't advance it, matching vmecpp's bad_resets counter and the
            // ITER column of the table above (the raw pass count, iter+1,
            // would disagree after any restart).
#ifdef DUMP_CUMES_VERIFY
            // Event line compiled out of release (fast) builds (see above).
            printf("  -> CONVERGED at iter %d\n",
                   controller.effective_iteration());
#endif
            printIterRow(controller.effective_iteration(), fsqr_i, fsqz_i,
                         fsql_i, controller.delta_t());
            break;
        }

        // ---- Preconditioned residuals (vmecpp evalFResPrecd) ----
        // fsqr1 = fResPrecd[0]·fNorm1 (same for fsqz1), fsql1 =
        // fResPrecd[2]·deltaS — NOTE: deltaS, not fNormL.
        double fsqr = rec.preconditioned_raw[0] * plain_per_el * fNorm1;
        double fsqz = rec.preconditioned_raw[1] * plain_per_el * fNorm1;
        double fsql =
            rec.preconditioned_raw[2] * plain_per_el * profiles.delta_s();
        // ---- Damping + time-step control (vmecpp Evolve / VMEC_8_52) ----
        // 1/tau tracks the rate of decrease of fsq (log-ratio), capped at
        // 0.15/delt, averaged over a 10-iteration window; res0 is the running
        // minimum of fsq. All of this (and the refresh/restart predicate) is
        // now the controller's decide_restart(), preserving the exact order.
        const double prec_triple[3] = {fsqr, fsqz, fsql};
        cumes::RestartDecision<double> decision =
            controller.decide_restart(prec_triple, inv_triple);

#ifdef DUMP_CUMES_VERIFY
        recordPass((int)decision.reason, fsqr_i, fsqz_i, fsql_i, fsqr, fsqz,
                   fsql, controller.delta_t(), decision.damping.otav,
                   decision.damping.dtau, decision.damping.b1,
                   decision.damping.fac, controller.effective_iteration(),
                   controller.restart_anchor());
#endif

        // ---- Descent step (Garabedian second-order Richardson) ----------
        // Runs BEFORE the time-step control, matching vmecpp's pass order:
        // Evolve() descends, then SolveEquilibriumLoop's control block
        // refreshes the backup or restores. The backup refresh must capture
        // the POST-descent state (vmecpp's RestartIteration(NO_RESTART) runs
        // after Evolve): a pre-descent backup restores the state one descent
        // step earlier at every restart, which offsets the weakly-determined
        // lambda gauge modes by ~1e-2 (one step of their ~1e-2/pass drift) at
        // the pass-56/57 BAD_PROGRESS restore and splits the trajectory
        // (2791 vs 2953 iters; converged lambda gauge modes 1.4e-2 off).
        cumes::DescentAction dact;
        dact.perform_descent = true;
        dact.move_lcfs = (vac != nullptr);  // free boundary: the LCFS R/Z row
                                            // evolves (vmecpp jMaxRZ = ns)
        dact.delta_t = controller.delta_t();
        dact.damping_b1 = decision.damping.b1;
        dact.damping_fac = decision.damping.fac;
        descent_op.enqueue(equilibrium.state(), equilibrium.velocity(),
                           equilibrium.residual_const(), equilibrium.xm(),
                           equilibrium.xn(), p.ns, p.mnmax, dact, stream);

        if (decision.do_refresh) {
            backupState();  // POST-descent state (vmecpp RestartIteration
                            // NO_RESTART semantics — see comment above)
        }
        if (decision.reason != cumes::RestartReason::kNone) {
            // Restore overwrites the just-descended state and zeroes the
            // velocities (vmecpp does the same: Evolve()'s descent is
            // discarded by the control block's RestartIteration).
            restoreState();
            controller.after_descent(decision);
#ifdef DUMP_CUMES_VERIFY
            // The restart event lines (BAD JACOBIAN / BAD PROGRESS) are
            // compiled out of release (fast) builds with the rest of the
            // dump machinery (see the other event-line gates above).
            printf("  -> %s (iter2=%d) delt=%.3e\n",
                   decision.reason == cumes::RestartReason::kBadJacobian
                       ? "BAD JACOBIAN"
                       : "BAD PROGRESS",
                   controller.effective_iteration(),
                   (double)controller.delta_t());
#endif
        } else {
            controller.after_descent(
                decision);  // advances iter2 on good passes
        }

        // INITIALIZED is the single activation pass, not a stage-long state.
        // vmecpp promotes it at this exact loop-bottom point, after restart
        // control and before the next force iteration.
        if (vac != nullptr) vac->on_iteration_end();

        // ---- Output (every 100 effective iters on the restart-anchored
        // grid, plus the final pass of a max-iteration run) ----
        if ((controller.effective_iteration() - controller.output_anchor()) %
                    100 ==
                0 ||
            iter == kMaxIterEff - 1) {
            printIterRow(controller.effective_iteration(), fsqr_i, fsqz_i,
                         fsql_i, controller.delta_t());
        }

        if (iter == kMaxIterEff - 1) {
            res.iterations = controller.effective_iteration();
            res.fsqr = (T)fsqr_i;
            res.fsqz = (T)fsqz_i;
            res.fsql = (T)fsql_i;
            res.delt = (T)controller.delta_t();
        }
    }

#ifdef DUMP_CUMES_VERIFY
    if (dumpEnabled()) {
        // Per-pass record, column-major: 15 blocks of per_iter.size() doubles
        // (byte-identical to the legacy array-of-doubles layout).
        FILE* fpr = fopen("dump/cuMES/per_iter_residuals_cumes.bin", "wb");
        if (fpr) {
            uint64_t n = (uint64_t)per_iter.size();
            fwrite(&n, sizeof(uint64_t), 1, fpr);
            for (int c = 0; c < cumes::PassRecord::kColumnCount; ++c) {
                for (const cumes::PassRecord& r : per_iter) {
                    const double* base = &r.invariant_fsqr;
                    fwrite(base + c, sizeof(double), 1, fpr);
                }
            }
            fclose(fpr);
        }
    }
#endif

    // Drain the compute stream before destroying the stage's cuFFT plans (the
    // last descent/backup enqueue may still be in flight when the loop exits).
    cumes::check_cuda(cudaStreamSynchronize(stream), "solver end sync");
    // precon/constraint/equilibrium are RAII (their destructors free the
    // arena-backed workspaces and the transform-timing events); nothing else.
    float t_inv_ms = 0.0f, t_fwd_ms = 0.0f;
    equilibrium.transform_timing_ms(t_inv_ms, t_fwd_ms);
    printf(
        "transform timing: inverseDFT total %.1f ms (%.3f ms/iter), "
        "forwardDFT total %.1f ms (%.3f ms/iter)\n",
        t_inv_ms, t_inv_ms / (res.iterations > 0 ? res.iterations : 1),
        t_fwd_ms, t_fwd_ms / (res.iterations > 0 ? res.iterations : 1));
    // Restart history for the stage report (v1 container): every pass that
    // restored state and re-anchored, with the effective iteration.
    res.restarts = controller.restart_events();
    return res;
}
