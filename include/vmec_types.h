// vmec_types.h — common data structures.
// Parity convention (matches vmecpp):
//   Even m -> "e" arrays, odd m -> "o" arrays.
//   Each parity array receives the FULL contribution from its modes.
//
// Internal (folded, n>=0) product basis, matching vmecpp's internal "fc"
// representation (FourierToReal3DSymmFastPoloidal):
//   R = rmncc*cos(mθ)cos(nζ) + rmnss*sin(mθ)sin(nζ)
//   Z = zmnsc*sin(mθ)cos(nζ) + zmncs*cos(mθ)sin(nζ)
//   λ = lmnsc*sin(mθ)cos(nζ) + lmncs*cos(mθ)sin(nζ)
// Mode index: mode = m*(ntor+1) + n with m = 0..mpol-1, n = 0..ntor
// (mnmax = mpol*(ntor+1)); the physical cos(mθ-nζ) coefficients fold as
//   rmncc[m,n] = rmnc[n]+rmnc[-n],  rmnss[m,n] = rmnc[n]-rmnc[-n],
//   zmnsc[m,n] = zmns[n]+zmns[-n],  zmncs[m,n] = zmns[-n]-zmns[n].
// The toroidal derivative of λ is stored as -∂λ/∂ζ (vmecpp convention:
// lv = -(lmksc_n*sinmu + lmkcs_n*cosmu) with sinnvn=-n*nfp*sin(nζ),
// cosnvn=+n*nfp*cos(nζ)); bsupu = (lamscale*lv + chip')/√g.
#pragma once

struct GridParams {
    int ns; int mnmax; int ntheta; int nzeta;
    int nfp; int nZnT; int mpol; int ntor;
    // Runtime input knobs (host-side; a JSON parser will fill them later).
    int ncurr;              // 0: prescribed iota, 1: prescribed current
    double delt;            // initial time step
    double ftol;            // convergence tolerance (invariant residuals)
    int max_iter;
    double lamscale;        // sqrt(deltaS * sum phipH^2), vmecpp constants_.
    static constexpr int kSignJacobian = -1;
    static constexpr double kMu0 = 4.0 * M_PI * 1.0e-7;  // exact, = vmecpp MU_0
};

struct FourierBasis {
    double* d_cos_mt_nz; double* d_sin_mt_nz;
    double* d_cc; double* d_ss; double* d_sc; double* d_cs;
    int* d_xm; int* d_xn;
};

struct RadialProfiles {
    double* d_iota_F; double* d_pres_F; double* d_phip_F;
    double* d_chi_F;  double* d_sqrtS_F;
    double* d_iota_H; double* d_pres_H; double* d_phip_H;
    double* d_mass_H; double* d_dVds_H; double* d_sqrtS_H;
    double* d_curr_H;  // prescribed toroidal current profile (ncurr=1), half grid
    double* d_chip_H;  // dχ/ds (poloidal flux derivative), half grid
    double delta_s;
};

// Spectral state with independent coefficient arrays.
// Each is shape (ns, mnmax) column-major.
// Total DOFs: 6 * ns * mnmax  (6 components in 3D stellarator-symmetric).
// Real-space parity (e/o) is determined by m parity (even m -> e, odd m -> o).
struct SpectralState {
    // Coefficients for Fourier reconstruction (all m, folded n>=0)
    double* d_rmncc;   // R: cos(mθ)cos(nζ) component
    double* d_zmnsc;   // Z: sin(mθ)cos(nζ) component
    double* d_lmnsc;   // λ: sin(mθ)cos(nζ) component
    double* d_rmnss;   // R: sin(mθ)sin(nζ) component
    double* d_zmncs;   // Z: cos(mθ)sin(nζ) component
    double* d_lmncs;   // λ: cos(mθ)sin(nζ) component

    // Velocities (6 components)
    double* d_v_rmncc; double* d_v_zmnsc; double* d_v_lmnsc;
    double* d_v_rmnss; double* d_v_zmncs; double* d_v_lmncs;
};
