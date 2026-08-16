// precon.cuh — radial tridiagonal preconditioner for MHD force balance.
// Reference: vmecpp's computePreconditioningMatrix / assembleRZPreconditioner
// / TridiagonalSolveSerial (the solver itself is a parallel cyclic reduction
// — see tridiagSolveKernel in precon.cu).
//
// Computes a flux-surface-averaged Hessian approximation twice per update:
//   R preconditioner uses Z-derivatives (zs, zu12, zu_e, zu_o)
//   Z preconditioner uses R-derivatives (rs, ru12, ru_e, ru_o)
//
// Each (m,n) mode gets an independent tridiagonal system solved in parallel
// via PCR (128-thread grid-stride). Even/odd parity components use separate
// matrix elements incorporating sm/sp radial scaling factors.
#pragma once
#include "vmec_types.h"
#include "fourier.cuh"
#include "geometry.cuh"

namespace cumes { class DeviceArena; }

// Per-mode tridiagonal preconditioner workspace.
// All arrays are GPU-resident.
template <typename T>
struct PreconWorkspace {
    // ---- Half-grid temporary ax/bx/cx (one value per half-grid surface) ---
    // ax[4*nH]: even-diag, mixed-offdiag, odd-diag-inside, odd-diag-outside
    T* d_ax_R;   // R preconditioner (from Z geometry)
    T* d_ax_Z;   // Z preconditioner (from R geometry)
    // bx[3*nH]: mixed-offdiag, even-diag-inside, odd-diag-outside
    T* d_bx_R;
    T* d_bx_Z;
    T* d_cx;     // toroidal term (same for R and Z) [nH]

    // ---- Assembled off-diagonal terms on half-grid (even=0, odd=1) ----
    T* d_arm;    // R: d²/ds² off-diagonal  [2*nH]
    T* d_brm;    // R: m² poloidal off-diag   [2*nH]
    T* d_azm;    // Z: d²/ds² off-diagonal  [2*nH]
    T* d_bzm;    // Z: m² poloidal off-diag   [2*nH]

    // ---- Assembled diagonal terms on full-grid (even=0, odd=1) ----
    T* d_ard;    // R: d²/ds² diagonal [2*ns]
    T* d_brd;    // R: m² diagonal    [2*ns]
    T* d_azd;    // Z: d²/ds² diagonal [2*ns]
    T* d_bzd;    // Z: m² diagonal    [2*ns]
    T* d_cxd;    // toroidal diagonal [ns] (same for R and Z)

    // ---- Ratio factors sm/sp per half-grid surface ----
    T* d_sm;     // sqrtS_H / sqrtS_F_inner
    T* d_sp;     // sqrtS_H / sqrtS_F_outer

    // ---- Tridiagonal matrix elements per (m,n) mode per radial position ----
    // ar[j] = sup-diagonal, dr[j] = diagonal, br[j] = sub-diagonal
    // Indexed as [mode * ns + jF]
    T* d_ar; T* d_dr; T* d_br;  // R system
    T* d_az; T* d_dz; T* d_bz;  // Z system

    // ---- jMin per mode (where the tridiagonal solve starts) ----
    int* d_jMin;

    // ---- Lambda diagonal preconditioner (vmecpp updateLambdaPreconditioner) ----
    // lambdaPrec[mode * ns + jF]: per-mode diagonal applied to component 2.
    T* d_lambdaPrec;
    // Scratch [ns+1] each: flux-surface averages of guu/gsqrt, guv/gsqrt
    // (3D) and gvv/gsqrt. vmecpp shifted half-grid layout: half-grid jH ->
    // index jH+1; index ns stays zero (never written), used by the LCFS
    // full-grid average.
    T* d_bLambda;
    T* d_dLambda;
    T* d_cLambda;
    T* d_rmsPhiP;  // scratch [1]: sum of phipH^2 (for lamscale)

    // Phase 8 scale-aware pivot support: per-mode coefficient scale (max
    // |lower|/|diagonal|/|upper| over the assembled R+Z tridiagonal systems)
    // computed once per refresh and read by the PCR/Thomas solve kernels as
    // the pivot floor reference. [mnmax]
    T* d_preconScale;
    // Breakdown accumulator (device int[1]): incremented by the solve kernels
    // for each system whose pivot falls below the scale-aware floor. Reset by
    // preconApply before the solves; not yet folded into the control record.
    int* d_preconStatus;

    // true when the device arrays above are subspans of a shared DeviceArena.
    bool arena_backed = false;
};

template <typename T>
PreconWorkspace<T> preconCreate(const GridParams<T>& p,
                                cumes::DeviceArena* arena = nullptr);
template <typename T>
void preconFree(PreconWorkspace<T>& pw);

// Compute (or update) the preconditioner matrix elements from current
// geometry and metric arrays.  Called every ~25 iterations. `rs` carries the
// parity-split geometry derivatives; `fp` the mode tables.
template <typename T>
void preconCompute(const cumes::RealSpaceStorage<T>& rs, const FourierPlan<T>& fp,
                   const GridParams<T>& p, const RadialProfiles<T>& rp,
                   const MetricWorkspace<T>& mw, PreconWorkspace<T>& pw,
                   cudaStream_t stream = 0);

// Apply the tridiagonal preconditioner to the 5-component spectral forces.
// Solves tridiagonal systems for R (components 0,3) and Z (components 1,4).
// Component 2 (lambda) gets the diagonal lambda preconditioner.
// Modifies forces in-place.
template <typename T>
void preconApply(cumes::SpectralView<T, cumes::DecomposedResidualDomain> f,
                 const GridParams<T>& p,
                 const PreconWorkspace<T>& pw,
                 const int* xm, const int* xn, cudaStream_t stream = 0);
