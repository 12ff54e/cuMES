// vmec_types.h — common data structures.
// Parity convention (matches vmecpp):
//   Even m -> "e" arrays, odd m -> "o" arrays.
//   Each parity array receives the FULL contribution from its modes:
//     R = rmncc*cos(mθ)cos(nζ) + rmnss*sin(mθ)sin(nζ)
//     Z = zmnsc*sin(mθ)cos(nζ) + zmncs*(-cos(mθ)sin(nζ))
//     λ = lmnsc*sin(mθ)cos(nζ)  (stellarator-symmetric: even parity only)
#pragma once

struct GridParams {
    int ns; int mnmax; int ntheta; int nzeta;
    int nfp; int nZnT; int mpol; int ntor;
    static constexpr int kSignJacobian = -1;
    static constexpr double kMu0 = 1.25663706212e-6;
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
    double delta_s;
};

// Spectral state with independent coefficient arrays.
// Each is shape (ns, mnmax) column-major.
// Total DOFs: 5 * ns * mnmax
// Real-space parity (e/o) is determined by m parity (even m -> e, odd m -> o).
struct SpectralState {
    // Coefficients for Fourier reconstruction (all m)
    double* d_rmncc;   // R: cos(mθ)cos(nζ) component
    double* d_zmnsc;   // Z: sin(mθ)cos(nζ) component
    double* d_lmnsc;   // λ: sin(mθ)cos(nζ) component

    // Coefficients for Fourier reconstruction (all m)
    double* d_rmnss;   // R: sin(mθ)sin(nζ) component
    double* d_zmncs;   // Z: -cos(mθ)sin(nζ) component [note: minus sign convention]

    // Velocities (5 components)
    double* d_v_rmncc; double* d_v_zmnsc; double* d_v_lmnsc;
    double* d_v_rmnss; double* d_v_zmncs;
};
