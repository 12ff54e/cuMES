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
#define DUMP_CUMES_VERIFY
#include "solver.cuh"
#include "precon.cuh"
#include "constraint.cuh"
#include "input.h"
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <vector>

#include "cumes/runtime/cuda_status.hpp"
#include "cumes/runtime/device_buffer.cuh"
#include "cumes/runtime/pinned_buffer.hpp"
#include "cumes/numerics/accumulation.hpp"
#include "cumes/solver/iteration_controller.hpp"
#include "cumes/solver/pass_record.hpp"
#include "cumes/solver/solver_bench.hpp"

#ifdef DUMP_CUMES_VERIFY
static bool dumpEnabled();  // defined below with the dump machinery
template <typename T>
static void dumpDeviceArray(const char* filename, const T* d_data,
                            size_t nelem);  // defined below
#endif

template <typename T>
__global__ void rzNormKernel(  // defined below (before computeResidualsKernel)
    cumes::SpectralView<const T, cumes::PhysicalStateDomain> st,
    const int* __restrict__ xm, const int* __restrict__ xn,
    int ns, int mnmax, T* __restrict__ out);

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
    int nH, T* __restrict__ out)  // [5]
{
    using A = typename cumes::NormAccum<T>::type;  // double for mixed-float
    int tid = threadIdx.x;
    A sRZ = A(0), sL = A(0), sMag = A(0), eTherm = A(0), vol = A(0);
    for (int j = tid; j < nH; j += blockDim.x) {
        sRZ += psum[4 * j + 0];
        sL  += psum[4 * j + 1];
        sMag += psum[4 * j + 2];
        eTherm += presH[j] * dVdsH[j];
        vol += dVdsH[j];
    }
    __shared__ A s_buf[5][256];
    s_buf[0][tid] = sRZ;  s_buf[1][tid] = sL;  s_buf[2][tid] = sMag;
    s_buf[3][tid] = eTherm;  s_buf[4][tid] = vol;
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
        out[0] = s_buf[0][0];  out[1] = s_buf[1][0];  out[2] = s_buf[2][0];
        out[3] = s_buf[3][0];  out[4] = s_buf[4][0];
    }
}

// Device-only force-norm reduction (no host copy or fence): writes the six
// scalars {sRZ, sL, sMag, eTherm, vol, rzNorm} into d_out[0..5], which the
// solver folds into the combined control record and transfers at the single
// control fence. Called on the preconditioner-refresh cadence.
template <typename T>
static void enqueueForceNorms(cumes::SpectralView<const T, cumes::PhysicalStateDomain> st,
                              const FourierPlan<T>& fp, const GridParams<T>& p,
                              const RadialProfiles<T>& rp, const MetricWorkspace<T>& mw,
                              T* d_psum, T* d_out, cudaStream_t stream) {
    computeForceNormPartials(p, mw, rp.d_dVds_H, d_psum, stream);
    { dim3 b1(256), g1(1);
      forceNormReduceKernel<T><<<g1, b1, 0, stream>>>(d_psum, rp.d_dVds_H, rp.d_pres_H,
                                                       p.ns - 1, d_out); }
    { dim3 b2(256), g2(1);
      rzNormKernel<T><<<g2, b2, 0, stream>>>(st, fp.basis.d_xm, fp.basis.d_xn,
                                              p.ns, p.mnmax, d_out + 5); }
    cumes::check_cuda(cudaGetLastError(), "force norms");
}

// Host finalize (called after the single control fence, on refresh passes):
// reduce the six device scalars hc[0..5] = {sRZ, sL, sMag, eTherm, vol, rzNorm}
// into fNormRZ/fNormL/fNorm1, and dump the force-norm record.
template <typename T>
static void finalizeForceNorms(const T* hc, const GridParams<T>& p,
                               const RadialProfiles<T>& rp, int iter2,
                               T& fNormRZ, T& fNormL, T& fNorm1) {
    T sRZ = hc[0], sL = hc[1], sMag = hc[2], eTherm = hc[3], vol = hc[4], h_rz = hc[5];
    T deltaS = rp.delta_s;
    T eMag = fabs(sMag) * deltaS;   // vmecpp: fabs(localMagneticEnergy)*deltaS
    eTherm *= deltaS;
    vol *= deltaS;
    T energyDensity = std::max(eMag, eTherm) / vol;
    // Scale-free-division guards: on degenerate geometry (empty volume, zero
    // surface norms, zero flux) the vmecpp 1/denominator would silently
    // produce inf/NaN normalization factors that poison every residual. A
    // fallback factor of 1.0 keeps the residuals finite (they then fail the
    // ftol/BAD_PROGRESS checks instead of the finiteness check).
    T denomRZ = sRZ * energyDensity * energyDensity;
    fNormRZ = denomRZ > T(0.0) ? (T(1.0) / denomRZ) : T(1.0);
    T denomL = sL * p.lamscale * p.lamscale;
    fNormL = denomL > T(0.0) ? (T(1.0) / denomL) : T(1.0);
    fNorm1 = h_rz > T(0.0) ? (T(1.0) / h_rz) : T(1.0);

#ifdef DUMP_CUMES_VERIFY
    if (dumpEnabled()) {
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
                    (double)eMag, (double)eTherm, (double)vol, (double)energyDensity,
                    (double)sRZ, (double)sL, (double)h_rz,
                    (double)fNormRZ, (double)fNormL, (double)fNorm1);
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
    int ns, int mnmax, int ntorp1) {
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
        st(cumes::SpectralComponent::Lcs, mode, 0) = st(cumes::SpectralComponent::Lcs, mode, 1);
        return;
    }
    if (m != 1) return;     // only m=1 needs extrapolation
    // Copy from j=1 to j=0
    st(cumes::SpectralComponent::Rcc, mode, 0) = st(cumes::SpectralComponent::Rcc, mode, 1);
    st(cumes::SpectralComponent::Zsc, mode, 0) = st(cumes::SpectralComponent::Zsc, mode, 1);
    st(cumes::SpectralComponent::Lsc, mode, 0) = st(cumes::SpectralComponent::Lsc, mode, 1);
    st(cumes::SpectralComponent::Rss, mode, 0) = st(cumes::SpectralComponent::Rss, mode, 1);
    st(cumes::SpectralComponent::Zcs, mode, 0) = st(cumes::SpectralComponent::Zcs, mode, 1);
    st(cumes::SpectralComponent::Lcs, mode, 0) = st(cumes::SpectralComponent::Lcs, mode, 1);
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
    const int* __restrict__ xm, int ns, int mnmax, T sqrtS1)
{
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
__global__ void m1ConstraintKernel(cumes::SpectralView<T, cumes::DecomposedResidualDomain> f_spec,
                                   int ns, int mnmax, int ntor, int zeroZ) {
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
            f_spec(cumes::SpectralComponent::Zcs, mn, j) = T(0.0);  // zeroZForceForM1
        } else {
            f_spec(cumes::SpectralComponent::Zcs, mn, j) = (old_rss - old_zcs) * s;  // mixed zcs
        }
    }
}

// vmecpp's applyM1Preconditioner (FourierForces): scales the m=1 frss by
// (ard+brd)/denom and fzcs by (azd+bzd)/denom using the odd-parity diagonal
// precon elements. The fzcs scale matters only when the mixed fzcs is
// nonzero (fix_m1_gauge = false), i.e. for iter2 >= 2 before convergence.
// Applied right before the RZ preconditioner (after the invariant residuals).
template <typename T>
__global__ void m1PreconScaleKernel(cumes::SpectralView<T, cumes::DecomposedResidualDomain> f_spec,
                                    const T* __restrict__ ard,
                                    const T* __restrict__ brd,
                                    const T* __restrict__ azd,
                                    const T* __restrict__ bzd,
                                    int ns, int mnmax, int ntor) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ns) return;
    int m1base = ntor + 1;
    T denom = ard[j * 2 + 1] + brd[j * 2 + 1] +
              azd[j * 2 + 1] + bzd[j * 2 + 1];
    // Degenerate-denominator guard: all-zero odd-parity precon diagonals
    // (e.g. a zero-√g surface) would make both scales NaN. Leave the forces
    // unscaled instead — the jacobian-stats check fails such surfaces before
    // this kernel normally runs.
    if (fabs(denom) < T(1e-30)) return;
    T scaleR = (ard[j * 2 + 1] + brd[j * 2 + 1]) / denom;
    T scaleZ = (azd[j * 2 + 1] + bzd[j * 2 + 1]) / denom;
    for (int n = 0; n < ntor + 1; ++n) {
        int mn = m1base + n;
        f_spec(cumes::SpectralComponent::Rss, mn, j) *= scaleR;
        f_spec(cumes::SpectralComponent::Zcs, mn, j) *= scaleZ;
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
    const int* __restrict__ xm, const int* __restrict__ xn,
    int ns, int mnmax, T* __restrict__ out)
{
    using A = typename cumes::NormAccum<T>::type;  // double for mixed-float
    A sum = A(0.0);
    int total = mnmax * ns;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int m = i / ns, j = i % ns, mm = xm[m], nn = xn[m];
        if (j == 0 && mm > 0) continue;  // vmecpp keeps the stored axis m>0
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
    if (tid == 0) out[0] = s_sum[0];
}

// Residual groups match vmecpp's FourierForces::residuals (folded basis):
//   fsqr = Σ frcc² + frss²,  fsqz = Σ fzsc² + fzcs²,  fsql = Σ flsc² + flcs²
// (components 0..5 of f_spec: frcc fzsc flsc frss fzcs flcs).
template <typename T>
__global__ void computeResidualsKernel(
    cumes::SpectralView<const T, cumes::DecomposedResidualDomain> f_spec,
    int ns, int mnmax, T* __restrict__ sq_out) {
    using A = typename cumes::NormAccum<T>::type;  // double for mixed-float
    int comp = blockIdx.x; if(comp>=3)return;
    A sum=A(0); int total=mnmax*ns;
    for(int i=threadIdx.x; i<total; i+=blockDim.x){
        int mode = i / ns, j = i % ns;
        T a = f_spec(static_cast<cumes::SpectralComponent>(comp), mode, j);
        T b = f_spec(static_cast<cumes::SpectralComponent>(comp + 3), mode, j);
        sum += a * a + b * b;
    }
    __shared__ A s_sum[256]; int tid=threadIdx.x; s_sum[tid]=sum; __syncthreads();
    for(int s=blockDim.x/2; s>0; s>>=1){if(tid<s)s_sum[tid]+=s_sum[tid+s]; __syncthreads();}
    if(tid==0) sq_out[comp]=s_sum[0]/(mnmax*ns);
}

template <typename T>
__global__ void descentStepKernel(
    cumes::SpectralView<T, cumes::PhysicalStateDomain> x,
    cumes::SpectralView<T, cumes::DecomposedVelocityDomain> v,
    cumes::SpectralView<const T, cumes::DecomposedResidualDomain> f_spec,
    const int* __restrict__ xm, const int* __restrict__ xn,
    int ns, int mnmax, T delt, T b1, T fac)
{
    using cumes::SpectralComponent;
    int i = blockIdx.x*blockDim.x + threadIdx.x, total = mnmax*ns;
    if(i>=total)return;
    int m=i/ns, j=i%ns, mm=xm[m];
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

    // R/Z components (0,1,3,4): LCFS is fixed — the force was zeroed by
    // fixBoundaryKernel and the coefficient must not move. Lambda (comps
    // 2,5) is free on all surfaces including the LCFS, matching vmecpp.
    if (j < ns-1) {
        T vr=v(SpectralComponent::Rcc,m,j); vr=fac*(b1*vr+ delt*f_spec(SpectralComponent::Rcc,m,j)); v(SpectralComponent::Rcc,m,j)=vr; x(SpectralComponent::Rcc,m,j)+=delt*vr*f;
        T vz=v(SpectralComponent::Zsc,m,j); vz=fac*(b1*vz+ delt*f_spec(SpectralComponent::Zsc,m,j)); v(SpectralComponent::Zsc,m,j)=vz; x(SpectralComponent::Zsc,m,j)+=delt*vz*f;
        T vs=v(SpectralComponent::Rss,m,j); vs=fac*(b1*vs+ delt*f_spec(SpectralComponent::Rss,m,j)); v(SpectralComponent::Rss,m,j)=vs;
        T vzc=v(SpectralComponent::Zcs,m,j);vzc=fac*(b1*vzc+ delt*f_spec(SpectralComponent::Zcs,m,j));v(SpectralComponent::Zcs,m,j)=vzc;
        if (mm == 1) {
            // m1 gauge: the state is stored in the UNDONE gauge while the
            // velocities/forces are vmecpp-decomposed (mixed gauge). vmecpp's
            // state evolves in the mixed gauge and is undone each update, so
            // the undone state must increment with the UNDONE velocity:
            //   rmnss += (vrss+vzcs), zmncs += (vrss-vzcs)
            // (FIXED 2026-08-02: without the mixing the iter-2+ m=1 states
            // drifted from vmecpp by ~0.07, corrupting the real-space.)
            x(SpectralComponent::Rss,m,j) += delt * (vs + vzc) * f;
            x(SpectralComponent::Zcs,m,j) += delt * (vs - vzc) * f;
        } else {
            x(SpectralComponent::Rss,m,j) += delt * vs * f;
            x(SpectralComponent::Zcs,m,j) += delt * vzc * f;
        }
    }
    T vl=v(SpectralComponent::Lsc,m,j);vl=fac*(b1*vl+ delt*f_spec(SpectralComponent::Lsc,m,j));v(SpectralComponent::Lsc,m,j)=vl;x(SpectralComponent::Lsc,m,j)+=delt*vl*f;
    T vlc=v(SpectralComponent::Lcs,m,j);vlc=fac*(b1*vlc+ delt*f_spec(SpectralComponent::Lcs,m,j));v(SpectralComponent::Lcs,m,j)=vlc;x(SpectralComponent::Lcs,m,j)+=delt*vlc*f;
}

#ifdef DUMP_CUMES_VERIFY
// Master switch for the dump/debug machinery: off unless CUMES_DUMP=1.
// All dump output routes through dumpEnsureDir/dumpDeviceArray, which no-op
// when disabled, so default runs write nothing to dump/ and print no debug
// noise. (The CUMES_DUMP_ITER knob below still selects WHICH iterations the
// windowed dumps fire on; CUMES_DUMP is the master enable.)
static bool dumpEnabled() {
    const char* e = getenv("CUMES_DUMP");
    return e != nullptr && atoi(e) != 0;
}

static void dumpEnsureDir() {
    if (!dumpEnabled()) return;
    int rc = system("mkdir -p dump/cuMES");
    if (rc != 0) fprintf(stderr, "dumpEnsureDir: mkdir -p failed (rc=%d)\n", rc);
}

// T-native dump: written as sizeof(T) elements; only read back by same-build
// tooling (e.g. tests/test_geometry_iso.cu, which is double-build-only).
template <typename T>
static void dumpDeviceArray(const char* filename, const T* d_data, size_t nelem) {
    if (!dumpEnabled()) return;
    // The dump machinery reads device data on the (synchronous) default stream
    // while the hot loop produces it on the nonblocking compute stream. Sync
    // everything first so a dump never reads a stale/in-flight buffer. This is
    // compile- and runtime-gated observability, so the extra fence is free on
    // the production path.
    cudaDeviceSynchronize();
    T* h_tmp = new T[nelem];
    cudaError_t err = cudaMemcpy(h_tmp, d_data, nelem * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "dumpDeviceArray cudaMemcpy failed for %s: %s\n", filename, cudaGetErrorString(err));
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
#endif

template <typename T>
SolverResult<T> solverRun(cumes::SpectralStorage<T>& storage, const GridParams<T>& p,
                          const RadialProfiles<T>& rp, FourierPlan<T>& fp,
                          MetricWorkspace<T>& mw, cumes::DeviceArena* arena,
                          cudaStream_t stream, cumes::SolverBench* bench) {
    // The legacy 12-pointer view over the contiguous slabs: every kernel and
    // consumer below keeps its unchanged pointer arithmetic and layout.
    SpectralState<T> st = storage.legacy_view();
    SolverResult<T> res{false, 0, T(1.0), T(1.0), T(1.0), p.delt};
    cumes::DeviceBuffer<T> d_f_spec(6 * (size_t)p.ns * p.mnmax);
    // Combined control record (Phase 6A one-fence): [0..3] oriented-Jacobian
    // stats, [4..6] invariant residual, [7..9] preconditioned residual,
    // [10..15] force-norm scalars (Phase 6B device reduction) — one device
    // buffer reduced into and transferred with one async copy per pass.
    cumes::DeviceBuffer<T> d_control(16);
    // Typed spectral views over the contiguous slabs + residual buffer (the
    // migrated kernel inputs). Each indexes bit-for-bit like the legacy
    // pointers it replaces; the const views are the read-only side.
    cumes::SpectralView<T, cumes::PhysicalStateDomain> state_view = storage.physical();
    cumes::SpectralView<T, cumes::DecomposedVelocityDomain> velocity_view = storage.velocity();
    cumes::SpectralView<T, cumes::DecomposedResidualDomain> residual_view(d_f_spec.data(), p.ns, p.mnmax);
    cumes::SpectralView<const T, cumes::DecomposedResidualDomain> residual_view_const(
        d_f_spec.data(), p.ns, p.mnmax);
    PreconWorkspace<T> pw = preconCreate(p, arena);
    ConstraintWorkspace<T> cw = constraintCreate(p, arena);

    // Bind every cuFFT plan to the explicit compute stream so the batched
    // ζ-transforms execute in stream order with the surrounding kernels
    // (blueprint §6.6). Plans are created once per stage; binding once here,
    // before any transform runs, is sufficient.
    cumes::check_cufft(cufftSetStream(fp.plan_z2d, stream), "set stream z2d");
    cumes::check_cufft(cufftSetStream(fp.plan_d2z, stream), "set stream d2z");
    cumes::check_cufft(cufftSetStream(cw.plan_d2z_da, stream), "set stream d2z_da");
    cumes::check_cufft(cufftSetStream(cw.plan_z2d_da, stream), "set stream z2d_da");

    // ---- transform timing (cudaEvent pairs around inverseDFT/forwardDFT) ----
    // The events are RECORDED on the compute stream every iteration but only
    // READ at the already-required invariant-residual fence (Phase 6A removes
    // the per-iteration cudaEventSynchronize that used to follow each transform,
    // turning two host barriers per pass into zero).
    cudaEvent_t ev0_inv, ev1_inv, ev0_fwd, ev1_fwd;
    cudaEventCreate(&ev0_inv); cudaEventCreate(&ev1_inv);
    cudaEventCreate(&ev0_fwd); cudaEventCreate(&ev1_fwd);
    float t_inv_ms = 0.0f, t_fwd_ms = 0.0f;

    // ---- env-gated knobs for convergence experiments (defaults = input
    // values; set via CUMES_MAX_ITER, CUMES_DELT0, CUMES_DTAU_FLOOR,
    // CUMES_DUMP_ITER, CUMES_E2_START) ----
    int kMaxIterEff = p.max_iter;
    double kDelt0Eff = p.delt;      // double: atof knob, converted to T at use
    double kDtauFloor = 0.0;
    int kDumpIter = 150, kE2Start = 560;
    if (const char* e = getenv("CUMES_MAX_ITER"))   kMaxIterEff = atoi(e);
    if (const char* e = getenv("CUMES_DELT0"))      kDelt0Eff = atof(e);
    if (const char* e = getenv("CUMES_DTAU_FLOOR")) kDtauFloor = atof(e);
    if (const char* e = getenv("CUMES_DUMP_ITER"))  kDumpIter = atoi(e);
    if (const char* e = getenv("CUMES_E2_START"))   kE2Start = atoi(e);
    printf("knobs: max_iter=%d delt0=%.3f dtau_floor=%.3e dump_iter=%d e2_start=%d\n",
           kMaxIterEff, kDelt0Eff, kDtauFloor, kDumpIter, kE2Start);

    // ---- vmecpp VMEC_8_52 time-step control state ----
    // iter2: effective iteration counter — does NOT advance on restart
    // passes (vmecpp's bad_resets mechanism). iter1: branch-point marker,
    // set to the current iteration on every restart; grace periods and the
    // invTau reinitialization key off (iter2 - iter1). res0: running minimum
    // of the preconditioned residual sum fsq. All of this lives in a pure
    // host state machine (Phase 4): the solver below launches kernels and
    // applies the returned decisions in the exact frozen order.
    cumes::IterationController<T> controller(
        {T(kDelt0Eff), p.ftol, T(kDtauFloor)});

    // vmecpp residual normalization factors (computeForceNorms), refreshed on
    // the same cadence as the preconditioner (every kPreconInterval passes).
    T fNormRZ = T(0.0), fNormL = T(0.0), fNorm1 = T(0.0);
    cumes::DeviceBuffer<T> d_psum(4 * (size_t)(p.ns - 1));

    // Pinned mirror of the combined control record (one async D2H per pass,
    // delivered by the single control fence).
    cumes::PinnedBuffer<T> h_control_pin(16);

    // State rollback: one contiguous state-only checkpoint slab (6*mnmax*ns),
    // replacing the six separate d_bk_* arrays. The slab order matches the six
    // old families, so backup/restore become single copies.
    cumes::DeviceBuffer<T> checkpoint(6 * (size_t)p.ns * p.mnmax);

    // Helper: copy current spectral state to backup (one device-to-device copy)
    auto backupState = [&]() { checkpoint.copy_from_async(storage.state_buffer(), stream); };

    // Helper: restore spectral state from backup + zero velocities
    auto restoreState = [&]() {
        storage.state_buffer().copy_from_async(checkpoint, stream);
        storage.velocity_buffer().zero_async(stream);
    };

    // Take initial backup
    backupState();

#ifdef DUMP_CUMES_VERIFY
    // Per-pass record for convergence analysis (mirrors vmecpp's
    // per_iter_residuals.bin + control scalars). Typed PassRecord; the field
    // order is the frozen 15-column on-disk contract (fsqr_i fsqz_i fsql_i
    // fsqr fsqz fsql delt otav dtau b1 fac iter2 iter1 reason rax). Observers
    // read these scalars and cannot affect the controller's decisions.
    std::vector<cumes::PassRecord> per_iter;
    per_iter.reserve((size_t)kMaxIterEff);
    // Radial location of the magnetic axis at zeta=0, matching vmecpp's r00
    // (Printout: geometric_offset.r_00 = r1_e[0] — the real-space even-m R
    // at the axis at theta=0, zeta=0). With the axis coefficients this is
    // the sum of the m=0 row: sum_n rmncc(0,n)@axis * cos(n*nfp*0) — the
    // plain R_00 coefficient alone misses the axis R wobble with zeta
    // (~+0.36 for W7-X, dominated by rmnc(0,1)@axis = +0.35).
    auto axisRAtZeta0 = [&]() {
        // mode-major layout [mode*ns + j]: the m=0 modes (mode = n) at the
        // axis row (j=0) sit at indices n*ns — strided, so one cudaMemcpy2D
        // grabs all ntor+1 values instead of ntor+1 individual 1-double
        // copies (each of which synchronized the device). The state is
        // produced on the compute stream, so sync it before the default-stream
        // read below (called from printIterRow's post-descent output path).
        cudaStreamSynchronize(stream);
        T h_ax[64];   // ntor+1 <= 64 for the hardcoded inputs
        cumes::check_cuda(cudaMemcpy2D(h_ax, sizeof(T), st.d_rmncc,
                               (size_t)p.ns * sizeof(T),
                               sizeof(T), p.ntor + 1,
                               cudaMemcpyDeviceToHost), "cpy Rax");
        T h = T(0.0);
        for (int n = 0; n <= p.ntor; ++n) h += h_ax[n];
        return h;
    };
    auto recordPass = [&](int reason, double fRi, double fZi, double fLi,
                          double fR, double fZ, double fL, double d,
                          double o, double dt, double b1v, double fcv,
                          int it2, int it1) {
        if (!dumpEnabled()) return;
        if ((int)per_iter.size() < kMaxIterEff) {
            cumes::PassRecord r;
            r.invariant_fsqr = fRi; r.invariant_fsqz = fZi; r.invariant_fsql = fLi;
            r.preconditioned_fsqr = fR; r.preconditioned_fsqz = fZ;
            r.preconditioned_fsql = fL;
            r.delta_t = d; r.otav = o; r.dtau = dt; r.b1 = b1v; r.fac = fcv;
            r.iter2 = (double)it2; r.iter1 = (double)it1; r.reason = (double)reason;
            r.axis_r = (double)axisRAtZeta0();
            per_iter.push_back(r);
        }
    };
#endif

    // Diagnostic: test inverse DFT at specified surface (CUMES_DUMP=1 only).
    // do_combine=false: the diagnostic reads only the parity arrays; the
    // combined *_real buffers are materialized on demand (fourierCombineParity)
    // at the dump site, never read stale.
    if (dumpEnabled()) {
        inverseDFT(fp, storage.physical_const(), p, false, stream);
        cudaDeviceSynchronize();  // dump-only read of compute-stream data
        auto* h_re = new T[p.nZnT * p.ns];
        auto* h_ro = new T[p.nZnT * p.ns];
        cumes::check_cuda(cudaMemcpy(h_re, fp.d_r_e, p.nZnT*p.ns*sizeof(T), cudaMemcpyDeviceToHost), "diag re");
        cumes::check_cuda(cudaMemcpy(h_ro, fp.d_r_o, p.nZnT*p.ns*sizeof(T), cudaMemcpyDeviceToHost), "diag ro");
        // Check surface j=ns-1 (LCFS): r_e should be rbc[0]*cos(0)=3.999, r_o should be sum of odd m
        int jB = p.ns - 1;
        double re_lcfs = h_re[0 + jB * p.nZnT];  // theta=0
        double ro_lcfs = h_ro[0 + jB * p.nZnT];
        printf("  [diag] LCFS theta=0: r_e=%.4f r_o=%.4f r_total=%.4f (expect ~3.93 + ~1.03 = ~4.96)\n",
               re_lcfs, ro_lcfs, re_lcfs + ro_lcfs);
        delete[] h_re;
        delete[] h_ro;
    }

    printf("\n ITER |    FSQR        FSQZ        FSQL    |   DELT\n");
    printf("------+------------------------------------+----------\n");

    // One table row: effective iteration, invariant residuals, delt, and the
    // axis / boundary R_00 (same columns as vmecpp's iteration printout).
    auto printIterRow = [&](int it2, T fsqr_v, T fsqz_v, T fsql_v, T delt_v) {
        printf("%5d | %11.3e %11.3e %11.3e | %8.2e", it2,
               (double)fsqr_v, (double)fsqz_v, (double)fsql_v, (double)delt_v);
        T h_rmncc_axis = axisRAtZeta0(), h_rmncc_bnd;
        cumes::check_cuda(cudaMemcpy(&h_rmncc_bnd, st.d_rmncc + (p.ns - 1), sizeof(T),
                             cudaMemcpyDeviceToHost), "cpy Rbnd");
        printf(" | Rax=%.4f Rbnd=%.4f\n", (double)h_rmncc_axis, (double)h_rmncc_bnd);
    };

#ifdef DUMP_CUMES_VERIFY
    {
        dumpEnsureDir();
        size_t n_spec = (size_t)p.ns * (size_t)p.mnmax;
        dumpDeviceArray("dump/cuMES/step_0_rmncc.bin",      st.d_rmncc, n_spec);
        dumpDeviceArray("dump/cuMES/step_0_zmnsc.bin",      st.d_zmnsc, n_spec);
        dumpDeviceArray("dump/cuMES/step_0_lmnsc.bin",      st.d_lmnsc, n_spec);
        dumpDeviceArray("dump/cuMES/step_0_rmnss.bin",      st.d_rmnss, n_spec);
        dumpDeviceArray("dump/cuMES/step_0_zmncs.bin",      st.d_zmncs, n_spec);
        dumpDeviceArray("dump/cuMES/step_0_lmncs.bin",      st.d_lmncs, n_spec);
        dumpDeviceArray("dump/cuMES/step_0_currH.bin",      rp.d_curr_H, p.ns - 1);
        dumpDeviceArray("dump/cuMES/step_0_chipH.bin",      rp.d_chip_H, p.ns - 1);
        dumpDeviceArray("dump/cuMES/step_0_iotaH.bin",      rp.d_iota_H, p.ns - 1);
        dumpDeviceArray("dump/cuMES/step_0_iotaF.bin",      rp.d_iota_F, p.ns);
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

    for(int iter=0; iter<kMaxIterEff; ++iter){
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
            printf("  -> CONVERGENCE PROBLEM: RESETTING DELT to %.3e (ijacob=%d)\n",
                   (double)controller.delta_t(), controller.bad_jacobian_count());
            continue;
        }

        // Extrapolate m=1 coefficients to the magnetic axis (j=0)
        // before inverse DFT, matching vmecpp's extrapolateTowardsAxis().
        // Must be done each iteration since the descent step updates j=1
        // but skips j=0 for m>0 (axis regularity).
        extrapolateAxisKernel<T><<<(p.mnmax + 31) / 32, 32, 0, stream>>>(
            state_view, p.ns, p.mnmax, p.ntor + 1);
        cumes::check_cuda(cudaGetLastError(), "extrapAxis");

        cudaEventRecord(ev0_inv, stream);
        // Fused inverse (blueprint §8.4): the xmpq-weighted rCon/zCon are
        // accumulated alongside the geometry (no separate rzCon transform).
        inverseDFTFused(fp, storage.physical_const(), p, false,
                        cw.d_rCon, cw.d_zCon, stream);
        cudaEventRecord(ev1_inv, stream);

        if (iter == 0 && dumpEnabled()) {
            auto* h_test = new T[p.nZnT * p.ns];
            cumes::check_cuda(cudaMemcpy(h_test, fp.d_r_e, p.nZnT*p.ns*sizeof(T), cudaMemcpyDeviceToHost), "loop test");
            int jB = p.ns - 1;
            printf("  [loop diag] LCFS theta=0: r_e=%.4f (expect ~3.93)\n", (double)h_test[0 + jB * p.nZnT]);
            // Also write to file for comparison
            FILE* dbg = fopen("dump/cuMES/debug_r_e.bin", "wb");
            if (dbg) {
                uint64_t n = p.nZnT * p.ns;
                fwrite(&n, sizeof(uint64_t), 1, dbg);
                fwrite(h_test, sizeof(T), n, dbg);
                fclose(dbg);
            }
            delete[] h_test;
        }

#ifdef DUMP_CUMES_VERIFY
        if (iter == 0 || iter2 == 2) {
            // iter 2 = first pass with lambda != 0: dump the real-space
            // lambda derivatives for the basis-convention check.
            size_t n_real = (size_t)p.ns * (size_t)p.nZnT;
            char fn[128];
            snprintf(fn, sizeof fn, "dump/cuMES/step_A_lu_e_iter_%d.bin", iter == 0 ? 1 : 2);
            dumpDeviceArray(fn, fp.d_lu_e, n_real);
            snprintf(fn, sizeof fn, "dump/cuMES/step_A_lu_o_iter_%d.bin", iter == 0 ? 1 : 2);
            dumpDeviceArray(fn, fp.d_lu_o, n_real);
            snprintf(fn, sizeof fn, "dump/cuMES/step_A_l_real_iter_%d.bin", iter == 0 ? 1 : 2);
            dumpDeviceArray(fn, fp.d_l_real, n_real);
        }
        if (iter == 0 || iter2 == 2) {
            size_t n_real = (size_t)p.ns * (size_t)p.nZnT;
            char fn[128];
            snprintf(fn, sizeof fn, "dump/cuMES/step_A_lv_e_iter_%d.bin", iter == 0 ? 1 : 2);
            dumpDeviceArray(fn, fp.d_lv_e, n_real);
            snprintf(fn, sizeof fn, "dump/cuMES/step_A_lv_o_iter_%d.bin", iter == 0 ? 1 : 2);
            dumpDeviceArray(fn, fp.d_lv_o, n_real);
        }
        if (iter == 0) {
            size_t n_real = (size_t)p.ns * (size_t)p.nZnT;
            // The combined *_real arrays are NOT refreshed by the hot loop
            // (inverseDFT runs with do_combine=false); materialize a fresh
            // snapshot from the current parity arrays before dumping them.
            fourierCombineParity(fp, p, stream);
            // Full R, Z, lambda (even+odd)
            dumpDeviceArray("dump/cuMES/step_A_r_real_iter_1.bin", fp.d_r_real, n_real);
            dumpDeviceArray("dump/cuMES/step_A_z_real_iter_1.bin", fp.d_z_real, n_real);
            // Even-m parity
            dumpDeviceArray("dump/cuMES/step_A_r_e_iter_1.bin", fp.d_r_e, n_real);
            dumpDeviceArray("dump/cuMES/step_A_z_e_iter_1.bin", fp.d_z_e, n_real);
            dumpDeviceArray("dump/cuMES/step_A_l_e_iter_1.bin", fp.d_l_e, n_real);
            // Odd-m parity
            dumpDeviceArray("dump/cuMES/step_A_r_o_iter_1.bin", fp.d_r_o, n_real);
            dumpDeviceArray("dump/cuMES/step_A_z_o_iter_1.bin", fp.d_z_o, n_real);
            dumpDeviceArray("dump/cuMES/step_A_l_o_iter_1.bin", fp.d_l_o, n_real);
            // Poloidal derivatives
            dumpDeviceArray("dump/cuMES/step_A_ru_real_iter_1.bin", fp.d_ru_real, n_real);
            dumpDeviceArray("dump/cuMES/step_A_zu_real_iter_1.bin", fp.d_zu_real, n_real);
            dumpDeviceArray("dump/cuMES/step_A_lu_real_iter_1.bin", fp.d_lu_real, n_real);
            // Toroidal derivatives
            dumpDeviceArray("dump/cuMES/step_A_rv_real_iter_1.bin", fp.d_rv_real, n_real);
            dumpDeviceArray("dump/cuMES/step_A_zv_real_iter_1.bin", fp.d_zv_real, n_real);
            dumpDeviceArray("dump/cuMES/step_A_lv_real_iter_1.bin", fp.d_lv_real, n_real);
            // Even-m poloidal derivatives
            dumpDeviceArray("dump/cuMES/step_A_ru_e_iter_1.bin", fp.d_ru_e, n_real);
            dumpDeviceArray("dump/cuMES/step_A_zu_e_iter_1.bin", fp.d_zu_e, n_real);
            dumpDeviceArray("dump/cuMES/step_A_lu_e_iter_1.bin", fp.d_lu_e, n_real);
            // Odd-m poloidal derivatives
            dumpDeviceArray("dump/cuMES/step_A_ru_o_iter_1.bin", fp.d_ru_o, n_real);
            dumpDeviceArray("dump/cuMES/step_A_zu_o_iter_1.bin", fp.d_zu_o, n_real);
            dumpDeviceArray("dump/cuMES/step_A_lu_o_iter_1.bin", fp.d_lu_o, n_real);
        }
#endif

        // Full-grid iota/chip update: every pass for ncurr=1 (current closure
        // evolves iotaH/chipH), but for ncurr=0 the half-grid profiles are
        // fixed so the update is idempotent and runs only on the first pass.
        computeGeometry(fp, p, rp, mw, stream,
                        /*update_iota_chi=*/ (p.ncurr == 1) || (iter == 0));

        // ---- Jacobian statistics (vmecpp's bad-jacobian detection) ----
        // Reduced into d_control[0..3] (device-only). The validity decision is
        // made at the single control fence after the full DAG is enqueued; a
        // collapsed or sign-flipped surface then fails the pass there and the
        // state is restored (see the gate below). The geometry kernels' own
        // inv_gsqrt guards keep their buffers finite in the interim.
        computeJacobianStats(p, mw, d_control.data(), stream);

#ifdef DUMP_CUMES_VERIFY
        if (iter == 0 || iter2 == 2) {
            // iter 2 = first pass with lambda != 0 (E3-D bsupv check)
            size_t n_half = (size_t)(p.ns - 1) * (size_t)p.nZnT;
            char fn[128];
            snprintf(fn, sizeof fn, "dump/cuMES/step_D_bsupu_iter_%d.bin", iter == 0 ? 1 : 2);
            dumpDeviceArray(fn, mw.d_bsupu, n_half);
            snprintf(fn, sizeof fn, "dump/cuMES/step_D_bsupv_iter_%d.bin", iter == 0 ? 1 : 2);
            dumpDeviceArray(fn, mw.d_bsupv, n_half);
        }
        if (iter == 0) {
            size_t n_half = (size_t)(p.ns - 1) * (size_t)p.nZnT;
            dumpDeviceArray("dump/cuMES/step_B_gsqrt_iter_1.bin", mw.d_gsqrt, n_half);
            dumpDeviceArray("dump/cuMES/step_C_guu_iter_1.bin",   mw.d_guu, n_half);
            dumpDeviceArray("dump/cuMES/step_C_guv_iter_1.bin",   mw.d_guv, n_half);
            dumpDeviceArray("dump/cuMES/step_C_gvv_iter_1.bin",   mw.d_gvv, n_half);
        }
#endif

        // rCon/zCon were produced by the fused inverse DFT above (blueprint
        // §8.4). Reset the constraint-force reference (rCon0/zCon0) to the
        // LCFS-extrapolated profile on the first iteration and after every
        // restart (iter2 == iter1), matching vmecpp's rzConIntoVolume
        // ("initialization/soft reset").
        if (controller.reset_constraint_reference()) {
            constraintResetRzCon0(p, cw, rp.d_sqrtS_F, stream);
        }

        // Update the radial tridiagonal + lambda preconditioners BEFORE the
        // forces, so the constraint-force multiplier (tcon) can use the
        // current iteration's preconditioner elements, matching vmecpp
        // (updateRadialPreconditioner + constraintForceMultiplier at the
        // start of update()). Cadence matches vmecpp's
        // shouldUpdateRadialPreconditioner: (iter2 - iter1) % 25 == 0.
        // Diagnostic mode must NOT change the trajectory: the old
        // `|| (dumpEnabled() && iter2 == kDumpIter)` extra refresh made a
        // CUMES_DUMP=1 run a different run than the no-dump production path,
        // so observers could alter the solver arithmetic. The refresh cadence
        // is now a pure function of the iteration counters.
        bool precon_updated = controller.refresh_preconditioner();
        if (precon_updated) {
            preconCompute(fp, p, rp, mw, pw, stream);

            // vmecpp computeForceNorms (same cadence): device-side reduction
            // of the force-norm partial sums into the combined control record
            // (finalized on the host after the single control fence).
            enqueueForceNorms(storage.physical_const(), fp, p, rp, mw,
                              d_psum.data(), d_control.data() + 10, stream);

#ifdef DUMP_CUMES_VERIFY
            if (iter == 0) {
                // Dump tridiagonal preconditioner matrices for comparison
                // with vmecpp. cuMES layout: ar[mode * ns + jF] (mode-major).
                size_t n_tri = (size_t)p.mnmax * (size_t)p.ns;
                size_t n_half_2 = (size_t)2 * (size_t)(p.ns - 1);
                size_t n_full_2 = (size_t)2 * (size_t)p.ns;
                size_t n_full_1 = (size_t)p.ns;

                // Tridiagonal matrix elements (mode-major: [mode, jF])
                dumpDeviceArray("dump/cuMES/step_precon_ar_iter_1.bin", pw.d_ar, n_tri);
                dumpDeviceArray("dump/cuMES/step_precon_dr_iter_1.bin", pw.d_dr, n_tri);
                dumpDeviceArray("dump/cuMES/step_precon_br_iter_1.bin", pw.d_br, n_tri);
                dumpDeviceArray("dump/cuMES/step_precon_az_iter_1.bin", pw.d_az, n_tri);
                dumpDeviceArray("dump/cuMES/step_precon_dz_iter_1.bin", pw.d_dz, n_tri);
                dumpDeviceArray("dump/cuMES/step_precon_bz_iter_1.bin", pw.d_bz, n_tri);

                // jMin per mode (stored as int, convert to double for dump)
                {
                    int* h_jMin = new int[p.mnmax];
                    cudaMemcpy(h_jMin, pw.d_jMin, p.mnmax * sizeof(int), cudaMemcpyDeviceToHost);
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
                dumpDeviceArray("dump/cuMES/step_precon_arm_iter_1.bin", pw.d_arm, n_half_2);
                dumpDeviceArray("dump/cuMES/step_precon_ard_iter_1.bin", pw.d_ard, n_full_2);
                dumpDeviceArray("dump/cuMES/step_precon_brm_iter_1.bin", pw.d_brm, n_half_2);
                dumpDeviceArray("dump/cuMES/step_precon_brd_iter_1.bin", pw.d_brd, n_full_2);
                dumpDeviceArray("dump/cuMES/step_precon_azm_iter_1.bin", pw.d_azm, n_half_2);
                dumpDeviceArray("dump/cuMES/step_precon_azd_iter_1.bin", pw.d_azd, n_full_2);
                dumpDeviceArray("dump/cuMES/step_precon_bzm_iter_1.bin", pw.d_bzm, n_half_2);
                dumpDeviceArray("dump/cuMES/step_precon_bzd_iter_1.bin", pw.d_bzd, n_full_2);
                dumpDeviceArray("dump/cuMES/step_precon_cxd_iter_1.bin", pw.d_cxd, n_full_1);

                // Sizes for comparison script
                double sizes_dbl[4] = {(double)p.ns, (double)(p.ns-1), (double)p.mpol, 1.0};
                FILE* fs = fopen("dump/cuMES/step_precon_sizes_iter_1.bin", "wb");
                if (fs) {
                    uint64_t n = 4;
                    fwrite(&n, sizeof(uint64_t), 1, fs);
                    fwrite(sizes_dbl, sizeof(double), 4, fs);
                    fclose(fs);
                }
            }
#endif  // DUMP_CUMES_VERIFY
        }

        computeForces(fp, p, rp, mw, stream);

#ifdef DUMP_CUMES_VERIFY
        if (iter == 0 || iter2 == 2) {
            // iter 2 = first pass with lambda != 0 (E3-B blmn blending check)
            size_t n_half = (size_t)(p.ns - 1) * (size_t)p.nZnT;
            if (iter == 0) {
                dumpDeviceArray("dump/cuMES/step_half_r12_iter_1.bin", mw.d_r12, n_half);
                dumpDeviceArray("dump/cuMES/step_half_zu12_iter_1.bin", mw.d_zu12, n_half);
                dumpDeviceArray("dump/cuMES/step_half_tau_iter_1.bin", mw.d_tau, n_half);
                dumpDeviceArray("dump/cuMES/step_half_gsqrt_iter_1.bin", mw.d_gsqrt, n_half);
                dumpDeviceArray("dump/cuMES/step_half_totalP_iter_1.bin", mw.d_totalPressure, n_half);
                dumpDeviceArray("dump/cuMES/step_half_bsupu_iter_1.bin", mw.d_bsupu, n_half);
                dumpDeviceArray("dump/cuMES/step_half_bsupv_iter_1.bin", mw.d_bsupv, n_half);
                dumpDeviceArray("dump/cuMES/step_half_bsubu_iter_1.bin", mw.d_bsubu, n_half);
                dumpDeviceArray("dump/cuMES/step_half_bsubv_iter_1.bin", mw.d_bsubv, n_half);
                dumpDeviceArray("dump/cuMES/step_half_rs_iter_1.bin", mw.d_rs, n_half);
                dumpDeviceArray("dump/cuMES/step_half_zs_iter_1.bin", mw.d_zs, n_half);
            }

            size_t n_real = (size_t)p.ns * (size_t)p.nZnT;
            char fn[128];
            int itag = (iter == 0) ? 1 : iter2;
            snprintf(fn, sizeof fn, "dump/cuMES/step_F_brmn_e_iter_%d.bin", itag);
            dumpDeviceArray(fn, fp.d_brmn_e, n_real);
            snprintf(fn, sizeof fn, "dump/cuMES/step_F_brmn_o_iter_%d.bin", itag);
            dumpDeviceArray(fn, fp.d_brmn_o, n_real);
            snprintf(fn, sizeof fn, "dump/cuMES/step_F_bzmn_e_iter_%d.bin", itag);
            dumpDeviceArray(fn, fp.d_bzmn_e, n_real);
            snprintf(fn, sizeof fn, "dump/cuMES/step_F_bzmn_o_iter_%d.bin", itag);
            dumpDeviceArray(fn, fp.d_bzmn_o, n_real);
            snprintf(fn, sizeof fn, "dump/cuMES/step_F_blmn_e_iter_%d.bin", itag);
            dumpDeviceArray(fn, fp.d_blmn_e, n_real);
            snprintf(fn, sizeof fn, "dump/cuMES/step_F_blmn_o_iter_%d.bin", itag);
            dumpDeviceArray(fn, fp.d_blmn_o, n_real);
            if (iter == 0) {
                dumpDeviceArray("dump/cuMES/step_E_armn_e_iter_1.bin", fp.d_armn_e, n_real);
                dumpDeviceArray("dump/cuMES/step_E_armn_o_iter_1.bin", fp.d_armn_o, n_real);
                dumpDeviceArray("dump/cuMES/step_E_azmn_e_iter_1.bin", fp.d_azmn_e, n_real);
                dumpDeviceArray("dump/cuMES/step_E_azmn_o_iter_1.bin", fp.d_azmn_o, n_real);
                dumpDeviceArray("dump/cuMES/step_E_brmn_e_iter_1.bin", fp.d_brmn_e, n_real);
                dumpDeviceArray("dump/cuMES/step_E_brmn_o_iter_1.bin", fp.d_brmn_o, n_real);
                dumpDeviceArray("dump/cuMES/step_E_bzmn_e_iter_1.bin", fp.d_bzmn_e, n_real);
                dumpDeviceArray("dump/cuMES/step_E_bzmn_o_iter_1.bin", fp.d_bzmn_o, n_real);
                dumpDeviceArray("dump/cuMES/step_E_crmn_e_iter_1.bin", fp.d_crmn_e, n_real);
                dumpDeviceArray("dump/cuMES/step_E_crmn_o_iter_1.bin", fp.d_crmn_o, n_real);
                dumpDeviceArray("dump/cuMES/step_E_czmn_e_iter_1.bin", fp.d_czmn_e, n_real);
                dumpDeviceArray("dump/cuMES/step_E_czmn_o_iter_1.bin", fp.d_czmn_o, n_real);
                dumpDeviceArray("dump/cuMES/step_E_blmn_e_iter_1.bin", fp.d_blmn_e, n_real);
                dumpDeviceArray("dump/cuMES/step_E_blmn_o_iter_1.bin", fp.d_blmn_o, n_real);
                dumpDeviceArray("dump/cuMES/step_E_clmn_e_iter_1.bin", fp.d_clmn_e, n_real);
                dumpDeviceArray("dump/cuMES/step_E_clmn_o_iter_1.bin", fp.d_clmn_o, n_real);
                // NOTE: no combined-force dumps — the force combine buffers
                // were removed (they were allocated/dumped but never
                // produced; the parity-split arrays above are the source of
                // truth).
            }
        }
#endif

        // Add spectral condensation constraint force to brmn/bzmn.
        // Uses the current-iteration tcon (refreshed above when the
        // preconditioner was updated), matching vmecpp.
        constraintCompute(p, fp, pw, cw, rp.d_sqrtS_F, precon_updated, stream);

#ifdef DUMP_CUMES_VERIFY
        if (iter == 0) {
            size_t n_real = (size_t)p.ns * (size_t)p.nZnT;
            dumpDeviceArray("dump/cuMES/step_G_brmn_e_iter_1.bin", fp.d_brmn_e, n_real);
            dumpDeviceArray("dump/cuMES/step_G_brmn_o_iter_1.bin", fp.d_brmn_o, n_real);
            dumpDeviceArray("dump/cuMES/step_G_bzmn_e_iter_1.bin", fp.d_bzmn_e, n_real);
            dumpDeviceArray("dump/cuMES/step_G_bzmn_o_iter_1.bin", fp.d_bzmn_o, n_real);
            // Constraint-chain intermediates (stage-by-stage vs vmecpp)
            // State as consumed by the constraint chain at iter 1 (post-descent)
            size_t n_spec2 = (size_t)p.ns * (size_t)p.mnmax;
            dumpDeviceArray("dump/cuMES/step_GC_rmncc_iter_1.bin", st.d_rmncc, n_spec2);
            dumpDeviceArray("dump/cuMES/step_GC_rmnss_iter_1.bin", st.d_rmnss, n_spec2);
            dumpDeviceArray("dump/cuMES/step_GC_zmnsc_iter_1.bin", st.d_zmnsc, n_spec2);
            dumpDeviceArray("dump/cuMES/step_GC_zmncs_iter_1.bin", st.d_zmncs, n_spec2);
            dumpDeviceArray("dump/cuMES/step_GC_rCon_iter_1.bin", cw.d_rCon, n_real);
            dumpDeviceArray("dump/cuMES/step_GC_zCon_iter_1.bin", cw.d_zCon, n_real);
            dumpDeviceArray("dump/cuMES/step_GC_gConEff_iter_1.bin", cw.d_gConEff, n_real);
            dumpDeviceArray("dump/cuMES/step_GC_gCon_iter_1.bin", cw.d_gCon, n_real);
            dumpDeviceArray("dump/cuMES/step_GC_frcon_e_iter_1.bin", cw.d_frcon_e, n_real);
            dumpDeviceArray("dump/cuMES/step_GC_frcon_o_iter_1.bin", cw.d_frcon_o, n_real);
            dumpDeviceArray("dump/cuMES/step_GC_fzcon_e_iter_1.bin", cw.d_fzcon_e, n_real);
            dumpDeviceArray("dump/cuMES/step_GC_fzcon_o_iter_1.bin", cw.d_fzcon_o, n_real);
            // tcon/faccon profiles (device arrays; h_tcon is stale -- the
            // kernel writes d_tcon directly)
            dumpDeviceArray("dump/cuMES/step_GC_tcon_iter_1.bin", cw.d_tcon, p.ns);
            dumpDeviceArray("dump/cuMES/step_GC_faccon_iter_1.bin", cw.d_faccon, p.mpol);
        }
#endif

        cudaEventRecord(ev0_fwd, stream);
        forwardDFT(fp, cumes::SpectralView<T, cumes::DecomposedResidualDomain>(
                          d_f_spec.data(), p.ns, p.mnmax),
                   p, cw, stream);
        cudaEventRecord(ev1_fwd, stream);

        // Apply the odd-m decomposition scaling (vmecpp decomposeInto).
        // The forward DFT already zeroed the LCFS R/Z entries and the axis
        // m>0 entries, so the boundary stays rigid and only the lambda
        // force is present at the LCFS (free gauge, evolved by descent).
        { dim3 bs(256), gs((p.ns*p.mnmax+255)/256);
          scalxcApplyKernel<T><<<gs,bs,0,stream>>>(residual_view, rp.d_sqrtS_F, fp.basis.d_xm,
                                          p.ns, p.mnmax,
                                          std::sqrt(T(1.0) / T(p.ns - 1)));
          cumes::check_cuda(cudaGetLastError(), "scalxc"); }

#ifdef DUMP_CUMES_VERIFY
        if (iter == 0 || iter2 == kDumpIter) {
            // Dump AFTER the decomposition scaling, matching vmecpp's dump
            // of m_decomposed_f (post-decomposeInto). Keyed on iter2 so a
            // handoff/plateau comparison can use the same effective counter
            // as vmecpp's dump blocks.
            size_t n_fspec = (size_t)6 * (size_t)p.mnmax * (size_t)p.ns;
            char fn[128];
            snprintf(fn, sizeof fn, "dump/cuMES/step_H_f_spec_iter_%d.bin",
                     iter == 0 ? 1 : iter2);
            dumpDeviceArray(fn, d_f_spec.data(), n_fspec);
        }
        if (iter2 >= kE2Start && iter2 < kE2Start + 40) {
            char fn[128];
            snprintf(fn, sizeof fn, "dump/cuMES/fspec_invariant_iter_%d.bin", iter2);
            dumpDeviceArray(fn, d_f_spec.data(), (size_t)6 * p.mnmax * p.ns);
        }
#endif

        // ---- m1 gauge constraint (vmecpp FourierCoeffs::m1Constraint) ----
        // Applied to the decomposed forces after the forward DFT (post
        // step_H dump), before the residuals and the preconditioner,
        // matching vmecpp ideal_mhd_model.cc. The fzcs zeroing mirrors
        // vmecpp's `fix_m1_gauge = always_fix_m1_gauge || fsqz < 1e-6 ||
        // iter2 < 2` with always_fix_m1_gauge = false (the standalone
        // default): zeroZ only on the first pass and once the previous
        // pass's invariant Z-residual dropped below 1e-6.
        { dim3 b1(256), g1((p.ns + 255) / 256);
          int zeroZ = (controller.effective_iteration() < 2) ||
                      (controller.fsqz_prev() < T(1.0e-6));
          m1ConstraintKernel<T><<<g1, b1, 0, stream>>>(residual_view, p.ns, p.mnmax, p.ntor,
                                            zeroZ);
          cumes::check_cuda(cudaGetLastError(), "m1Constraint"); }

        // ---- Invariant (unpreconditioned) residuals ----
        // Reduced into d_control[4..6]. The nonfinite/converged decision and
        // the in-place preconditioner are both deferred to the single control
        // fence below (Phase 6A one-fence path); stream order guarantees the
        // invariant reduction completes before the preconditioner runs, so the
        // two never race on the residual slab.
#ifdef DUMP_CUMES_VERIFY
        if (iter == kMaxIterEff - 1) {
            dumpDeviceArray("dump/cuMES/step_final_f_spec.bin", d_f_spec.data(),
                            (size_t)6 * p.mnmax * p.ns);
        }
#endif
        { dim3 b3(256),g3(3); computeResidualsKernel<T><<<g3,b3,0,stream>>>(residual_view_const,p.ns,p.mnmax,d_control.data()+4); }

        // vmecpp applyM1Preconditioner: m=1 frss scale, before the RZ solve
        { dim3 b1(256), g1((p.ns + 255) / 256);
          m1PreconScaleKernel<T><<<g1, b1, 0, stream>>>(residual_view, pw.d_ard, pw.d_brd,
                                             pw.d_azd, pw.d_bzd,
                                             p.ns, p.mnmax, p.ntor);
          cumes::check_cuda(cudaGetLastError(), "m1PreconScale"); }

        // Apply the radial tridiagonal + lambda preconditioners to the
        // (decomposed) spectral forces.
        preconApply(residual_view, p, pw, fp.basis.d_xm, fp.basis.d_xn, stream);

#ifdef DUMP_CUMES_VERIFY
        if (iter == 0 || iter2 == 51 || (iter2 >= kDumpIter && iter2 <= kDumpIter + 2) ||
            (iter2 >= 2 && iter2 <= 4)) {
            size_t n_fspec = (size_t)6 * (size_t)p.mnmax * (size_t)p.ns;
            size_t n_spec = (size_t)p.mnmax * (size_t)p.ns;
            char fn[128];
            snprintf(fn, sizeof fn, "dump/cuMES/step_I_f_spec_iter_%d.bin",
                     iter == 0 ? 1 : iter2);
            dumpDeviceArray(fn, d_f_spec.data(), n_fspec);
            // State + velocities at the handoff window (pre-descent of the
            // pass, matching vmecpp's dump phase at vmec.cc). Also at the
            // iter-2..4 window (first lambda != 0 passes) for the state check.
            if (iter2 >= kDumpIter || iter2 == 51 || (iter2 >= 2 && iter2 <= 4)) {
                snprintf(fn, sizeof fn, "dump/cuMES/state_rmncc_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_rmncc, n_spec);
                snprintf(fn, sizeof fn, "dump/cuMES/state_zmnsc_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_zmnsc, n_spec);
                snprintf(fn, sizeof fn, "dump/cuMES/state_lmnsc_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_lmnsc, n_spec);
                snprintf(fn, sizeof fn, "dump/cuMES/state_rmnss_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_rmnss, n_spec);
                snprintf(fn, sizeof fn, "dump/cuMES/state_zmncs_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_zmncs, n_spec);
                snprintf(fn, sizeof fn, "dump/cuMES/state_lmncs_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_lmncs, n_spec);
                snprintf(fn, sizeof fn, "dump/cuMES/vel_vrmncc_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_v_rmncc, n_spec);
                snprintf(fn, sizeof fn, "dump/cuMES/vel_vzmnsc_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_v_zmnsc, n_spec);
                snprintf(fn, sizeof fn, "dump/cuMES/vel_vlmnsc_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_v_lmnsc, n_spec);
                snprintf(fn, sizeof fn, "dump/cuMES/vel_vrmnss_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_v_rmnss, n_spec);
                snprintf(fn, sizeof fn, "dump/cuMES/vel_vzmncs_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_v_zmncs, n_spec);
                snprintf(fn, sizeof fn, "dump/cuMES/vel_vlmncs_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_v_lmncs, n_spec);
            }
        }
        if (iter2 >= kE2Start && iter2 < kE2Start + 40) {
            char fn[128];
            snprintf(fn, sizeof fn, "dump/cuMES/fspec_precon_iter_%d.bin", iter2);
            dumpDeviceArray(fn, d_f_spec.data(), (size_t)6 * p.mnmax * p.ns);
        }
#endif

        // ---- Preconditioned residuals (vmecpp fsqr1/fsqz1/fsql1) ----
        { dim3 b3(256),g3(3); computeResidualsKernel<T><<<g3,b3,0,stream>>>(residual_view_const,p.ns,p.mnmax,d_control.data()+7); }

        // ---- ONE combined control fence (Phase 6A) ----
        // Jacobian stats + invariant + preconditioned residuals are one device
        // record; transfer it with one async copy and sync once. This replaces
        // the three per-pass host barriers (Jacobian gate, invariant,
        // preconditioned) of the pre-6A loop.
        cumes::check_cuda(cudaMemcpyAsync(h_control_pin.data(), d_control.data(),
                               16 * sizeof(T), cudaMemcpyDeviceToHost, stream), "cpy control");
        cumes::check_cuda(cudaStreamSynchronize(stream), "control sync");
        if (bench && bench->enabled) {
            auto bench_now = std::chrono::steady_clock::now();
            bench->pass_wall_us.push_back(
                std::chrono::duration<double, std::micro>(bench_now - bench_t_prev)
                    .count());
            bench_t_prev = bench_now;
        }
        const T* hc = h_control_pin.data();
        // Sample the transform-timing events at this fence (both transforms
        // preceded it on the same stream).
        { float ms; cudaEventElapsedTime(&ms, ev0_inv, ev1_inv); t_inv_ms += ms; }
        { float ms; cudaEventElapsedTime(&ms, ev0_fwd, ev1_fwd); t_fwd_ms += ms; }

        const T plainPerEl = T(p.mnmax) * T(p.ns);

        // ---- Oriented-Jacobian validity gate (vmecpp bad-jacobian) ----
        // Checked FIRST: an invalid geometry can make the downstream residual
        // (reduced on that geometry) nonfinite or garbage, so the restore
        // decision must precede the residual classification. The delt shrink
        // and re-anchor bookkeeping live in the controller; the solver only
        // restores. The persistent preconditioner/constraint caches the pass
        // may have touched are self-healing: the re-anchor makes the next pass
        // a refresh+reset pass (iter2==iter1), which rebuilds them from the
        // restored geometry.
        cumes::JacobianStatus<T> js;
        js.min_oriented = hc[0];
        js.max_abs = hc[1];
        js.nonfinite_count = hc[2];
        js.min_index = (int)hc[3];
        const T delt_before = controller.delta_t();
        const int it2_before = controller.effective_iteration();
        const int it1_before = controller.restart_anchor();
        if (controller.jacobian_invalid(js, p.nZnT)) {
#ifdef DUMP_CUMES_VERIFY
            recordPass(1, 0, 0, 0, 0, 0, 0, delt_before, 0, 0, 0, 0,
                       it2_before, it1_before);
#endif
            restoreState();
            printf("  -> BAD JACOBIAN (invalid √g: min(signJ·√g)=%.3e "
                   "max|√g|=%.3e nonfinite=%.0f at jH=%d) delt=%.3e\n",
                   (double)js.min_oriented, (double)js.max_abs,
                   (double)js.nonfinite_count,
                   js.min_index / p.nZnT, (double)controller.delta_t());
            printIterRow(controller.effective_iteration(), T(1.0), T(1.0),
                         T(1.0), controller.delta_t());
            continue;
        }

        // On a refresh pass, finalize the force-norm factors from the combined
        // record's force-norm scalars (reduced on device above). On non-refresh
        // passes the cached factors are reused.
        if (precon_updated) {
            finalizeForceNorms(hc + 10, p, rp, iter2, fNormRZ, fNormL, fNorm1);
        }

        // ---- Invariant residuals (vmecpp evalFResInvar) ----
        // fsqr = fResInvar[0]·fNormRZ·0.25 (same for fsqz), fsql =
        // fResInvar[2]·fNormL. The kernel returns ΣF²/(mnmax·ns); undo first.
        T fsqr_i = hc[4] * plainPerEl * fNormRZ * T(0.25);
        T fsqz_i = hc[5] * plainPerEl * fNormRZ * T(0.25);
        T fsql_i = hc[6] * plainPerEl * fNormL;
        const T inv_triple[3] = {fsqr_i, fsqz_i, fsql_i};

        // ---- Stopping criterion (vmecpp Evolve) ----
        // classify_invariant records fsqz_prev for the next pass's gauge
        // condition, then reports nonfinite (recover) or converged (stop).
        cumes::InvariantVerdict<T> verdict = controller.classify_invariant(inv_triple);
        if (verdict.nonfinite) {
            // vmecpp hard-fails on non-finite residuals (status BAD_JACOBIAN);
            // we recover instead: restore the last good state and shrink delt.
#ifdef DUMP_CUMES_VERIFY
            recordPass(1, fsqr_i, fsqz_i, fsql_i, 0,0,0, delt_before,0,0,0,0,
                       it2_before, it1_before);
#endif
            restoreState();
            printf("  -> BAD JACOBIAN (non-finite residuals) delt=%.3e\n",
                   (double)controller.delta_t());
            printIterRow(controller.effective_iteration(), fsqr_i, fsqz_i, fsql_i,
                         controller.delta_t());
            continue;
        }
        if (verdict.converged) {
#ifdef DUMP_CUMES_VERIFY
            recordPass(0, fsqr_i, fsqz_i, fsql_i, 0,0,0, controller.delta_t(),0,0,0,0,
                       controller.effective_iteration(), controller.restart_anchor());
#endif
            res.converged=true; res.iterations=controller.effective_iteration();
            res.fsqr=fsqr_i;res.fsqz=fsqz_i;res.fsql=fsql_i;res.delt=controller.delta_t();
            // Report the EFFECTIVE iteration count (iter2): restart passes
            // don't advance it, matching vmecpp's bad_resets counter and the
            // ITER column of the table above (the raw pass count, iter+1,
            // would disagree after any restart).
            printf("  -> CONVERGED at iter %d\n", controller.effective_iteration());
            printIterRow(controller.effective_iteration(), fsqr_i, fsqz_i, fsql_i,
                         controller.delta_t()); break;
        }

        // ---- Preconditioned residuals (vmecpp evalFResPrecd) ----
        // fsqr1 = fResPrecd[0]·fNorm1 (same for fsqz1), fsql1 =
        // fResPrecd[2]·deltaS — NOTE: deltaS, not fNormL.
        T fsqr = hc[7] * plainPerEl * fNorm1;
        T fsqz = hc[8] * plainPerEl * fNorm1;
        T fsql = hc[9] * plainPerEl * rp.delta_s;
        // ---- Damping + time-step control (vmecpp Evolve / VMEC_8_52) ----
        // 1/tau tracks the rate of decrease of fsq (log-ratio), capped at
        // 0.15/delt, averaged over a 10-iteration window; res0 is the running
        // minimum of fsq. All of this (and the refresh/restart predicate) is
        // now the controller's decide_restart(), preserving the exact order.
        const T prec_triple[3] = {fsqr, fsqz, fsql};
        cumes::RestartDecision<T> decision =
            controller.decide_restart(prec_triple, inv_triple);

#ifdef DUMP_CUMES_VERIFY
        recordPass((int)decision.reason, fsqr_i, fsqz_i, fsql_i,
                   fsqr, fsqz, fsql, controller.delta_t(), decision.damping.otav,
                   decision.damping.dtau, decision.damping.b1, decision.damping.fac,
                   controller.effective_iteration(), controller.restart_anchor());
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
        { dim3 bd(256), gd((p.ns*p.mnmax+255)/256);
          descentStepKernel<T><<<gd,bd,0,stream>>>(
              state_view, velocity_view, residual_view_const,
              fp.basis.d_xm, fp.basis.d_xn,
              p.ns,p.mnmax,controller.delta_t(),decision.damping.b1,decision.damping.fac);
          cumes::check_cuda(cudaGetLastError(),"descent"); }

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
            printf("  -> %s (iter2=%d) delt=%.3e\n",
                   decision.reason == cumes::RestartReason::kBadJacobian
                       ? "BAD JACOBIAN" : "BAD PROGRESS",
                   controller.effective_iteration(), (double)controller.delta_t());
        } else {
            controller.after_descent(decision);  // advances iter2 on good passes
        }

        // ---- Output (every 100 effective iters on the restart-anchored
        // grid, plus the final pass of a max-iteration run) ----
        if ((controller.effective_iteration() - controller.output_anchor()) % 100 == 0 ||
            iter == kMaxIterEff - 1) {
            printIterRow(controller.effective_iteration(), fsqr_i, fsqz_i, fsql_i,
                         controller.delta_t());
        }

        if(iter==kMaxIterEff-1){ res.iterations=controller.effective_iteration();
            res.fsqr=fsqr_i;res.fsqz=fsqz_i;res.fsql=fsql_i;res.delt=controller.delta_t(); }
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
    preconFree(pw); constraintFree(cw);
    printf("transform timing: inverseDFT total %.1f ms (%.3f ms/iter), "
           "forwardDFT total %.1f ms (%.3f ms/iter)\n",
           t_inv_ms, t_inv_ms / (res.iterations > 0 ? res.iterations : 1),
           t_fwd_ms, t_fwd_ms / (res.iterations > 0 ? res.iterations : 1));
    cudaEventDestroy(ev0_inv); cudaEventDestroy(ev1_inv);
    cudaEventDestroy(ev0_fwd); cudaEventDestroy(ev1_fwd);
    return res;
}

