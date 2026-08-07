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

// The computation scalar type, templated on T throughout; the app-level
// compile-time switch between double and float is `Real` (see below).
template <typename T>
struct GridParams {
    int ns; int mnmax; int ntheta; int nzeta;
    int nfp; int nZnT; int mpol; int ntor;
    // Runtime input knobs (host-side; a JSON parser will fill them later).
    int ncurr;              // 0: prescribed iota, 1: prescribed current
    T delt;                 // initial time step
    T ftol;                 // convergence tolerance (invariant residuals)
    int max_iter;
    T lamscale;             // sqrt(deltaS * sum phipH^2), vmecpp constants_.
    static constexpr int kSignJacobian = -1;
    static constexpr T kMu0 = 4.0 * M_PI * 1.0e-7;  // exact, = vmecpp MU_0
};

struct FourierBasis {
    // Mode tables only: the basis functions are evaluated analytically via
    // the cuFFT machinery and the small per-m poloidal tables in FourierPlan.
    int* d_xm; int* d_xn;
};

template <typename T>
struct RadialProfiles {
    T* d_iota_F; T* d_pres_F; T* d_phip_F;
    T* d_chi_F;  T* d_sqrtS_F;
    T* d_iota_H; T* d_pres_H; T* d_phip_H;
    T* d_mass_H; T* d_dVds_H; T* d_sqrtS_H;
    T* d_curr_H;  // prescribed toroidal current profile (ncurr=1), half grid
    T* d_chip_H;  // dχ/ds (poloidal flux derivative), half grid
    T delta_s;
};

// Spectral state with independent coefficient arrays.
// Each is shape (ns, mnmax) column-major.
// Total DOFs: 6 * ns * mnmax  (6 components in 3D stellarator-symmetric).
// Real-space parity (e/o) is determined by m parity (even m -> e, odd m -> o).
template <typename T>
struct SpectralState {
    // Coefficients for Fourier reconstruction (all m, folded n>=0)
    T* d_rmncc;   // R: cos(mθ)cos(nζ) component
    T* d_zmnsc;   // Z: sin(mθ)cos(nζ) component
    T* d_lmnsc;   // λ: sin(mθ)cos(nζ) component
    T* d_rmnss;   // R: sin(mθ)sin(nζ) component
    T* d_zmncs;   // Z: cos(mθ)sin(nζ) component
    T* d_lmncs;   // λ: cos(mθ)sin(nζ) component

    // Velocities (6 components)
    T* d_v_rmncc; T* d_v_zmnsc; T* d_v_lmnsc;
    T* d_v_rmnss; T* d_v_zmncs; T* d_v_lmncs;
};

// App-level precision switch. All modules are templated on T; this alias is
// what main.cu (and the diagnostics) build with. Configure via
//   cmake -B build-float -DCUMES_USE_FLOAT=ON
// The on-disk state files (cumes_state.bin, vmecpp_init.bin) stay double
// regardless; only the GPU computation uses T.
#ifdef CUMES_USE_FLOAT
using Real = float;
#else
using Real = double;
#endif
