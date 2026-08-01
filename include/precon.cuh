// precon.cuh — radial tridiagonal preconditioner for MHD force balance.
// Reference: vmecpp's computePreconditioningMatrix / assembleRZPreconditioner /
// TridiagonalSolveSerial.
//
// Computes a flux-surface-averaged Hessian approximation twice per update:
//   R preconditioner uses Z-derivatives (zs, zu12, zu_e, zu_o)
//   Z preconditioner uses R-derivatives (rs, ru12, ru_e, ru_o)
//
// Each (m,n) mode gets an independent tridiagonal system solved via the
// Thomas algorithm. Even/odd parity components use separate matrix elements
// incorporating sm/sp radial scaling factors.
#pragma once
#include "vmec_types.h"
#include "fourier.cuh"
#include "geometry.cuh"

// Per-mode tridiagonal preconditioner workspace.
// All arrays are GPU-resident.
struct PreconWorkspace {
    // ---- Half-grid temporary ax/bx/cx (one value per half-grid surface) ---
    // ax[4*nH]: even-diag, mixed-offdiag, odd-diag-inside, odd-diag-outside
    double* d_ax_R;   // R preconditioner (from Z geometry)
    double* d_ax_Z;   // Z preconditioner (from R geometry)
    // bx[3*nH]: mixed-offdiag, even-diag-inside, odd-diag-outside
    double* d_bx_R;
    double* d_bx_Z;
    double* d_cx;     // toroidal term (same for R and Z) [nH]

    // ---- Assembled off-diagonal terms on half-grid (even=0, odd=1) ----
    double* d_arm;    // R: d²/ds² off-diagonal  [2*nH]
    double* d_brm;    // R: m² poloidal off-diag   [2*nH]
    double* d_azm;    // Z: d²/ds² off-diagonal  [2*nH]
    double* d_bzm;    // Z: m² poloidal off-diag   [2*nH]

    // ---- Assembled diagonal terms on full-grid (even=0, odd=1) ----
    double* d_ard;    // R: d²/ds² diagonal [2*ns]
    double* d_brd;    // R: m² diagonal    [2*ns]
    double* d_azd;    // Z: d²/ds² diagonal [2*ns]
    double* d_bzd;    // Z: m² diagonal    [2*ns]
    double* d_cxd;    // toroidal diagonal [ns] (same for R and Z)

    // ---- Ratio factors sm/sp per half-grid surface ----
    double* d_sm;     // sqrtS_H / sqrtS_F_inner
    double* d_sp;     // sqrtS_H / sqrtS_F_outer

    // ---- Tridiagonal matrix elements per (m,n) mode per radial position ----
    // ar[j] = sup-diagonal, dr[j] = diagonal, br[j] = sub-diagonal
    // Indexed as [mode * ns + jF]
    double* d_ar; double* d_dr; double* d_br;  // R system
    double* d_az; double* d_dz; double* d_bz;  // Z system

    // ---- jMin per mode (where the tridiagonal solve starts) ----
    int* d_jMin;

    // ---- Lambda diagonal preconditioner (vmecpp updateLambdaPreconditioner) ----
    // lambdaPrec[mode * ns + jF]: per-mode diagonal applied to component 2.
    double* d_lambdaPrec;
    // Scratch [ns+1] each: flux-surface averages of guu/gsqrt and gvv/gsqrt.
    // vmecpp shifted half-grid layout: half-grid jH -> index jH+1; index ns
    // stays zero (never written), used by the LCFS full-grid average.
    double* d_bLambda;
    double* d_cLambda;
    double* d_rmsPhiP;  // scratch [1]: sum of phipH^2 (for lamscale)
};

PreconWorkspace preconCreate(const GridParams& p);
void preconFree(PreconWorkspace& pw);

// Compute (or update) the preconditioner matrix elements from current
// geometry and metric arrays.  Called every ~25 iterations.
void preconCompute(const FourierPlan& fp, const GridParams& p,
                   const RadialProfiles& rp, const MetricWorkspace& mw,
                   PreconWorkspace& pw);

// Apply the tridiagonal preconditioner to the 5-component spectral forces.
// Solves tridiagonal systems for R (components 0,3) and Z (components 1,4).
// Component 2 (lambda) gets the diagonal lambda preconditioner.
// Modifies forces in-place.
void preconApply(double* d_f_inout, const GridParams& p,
                 const PreconWorkspace& pw,
                 const int* xm, const int* xn);
