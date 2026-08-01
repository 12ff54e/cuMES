// input.h — hardcoded Solovev equilibrium matching vmecpp playground example.
// Boundary values match playground/solovev/solovev.json.
#pragma once
#include "vmec_types.h"

// Parameters aligned with solovev.json for verification against vmecpp.
// vmecpp default for ntheta (when input=0, mpol=6): 2*mpol+6 = 18
// vmecpp default for nzeta (when input=0, ntor=0): 1 (axisymmetric)
constexpr int kNsVal       = 11;   // radial resolution (matches ns_array=[11])
constexpr int kMpol        = 6;    // poloidal modes: m = 0 .. mpol-1
constexpr int kNtor        = 1;    // toroidal modes: 1 for n=0 only (axisymmetric)
constexpr int kNtheta      = 18;   // poloidal real-space grid (vmecpp Nyquist for mpol=6)
constexpr int kNzeta       = 1;    // toroidal real-space grid (axisymmetric)
constexpr int kNfp         = 1;    // field periods (1 = axisymmetric)

constexpr int kMnmax       = kMpol * kNtor;  // total Fourier modes
constexpr int kNZnT        = kNtheta * kNzeta;
constexpr int kMaxIter     = 1000;
constexpr double kFtol     = 1e-16;  // matches ftol_array=[1e-16]
constexpr double kDelt0    = 0.9;     // matches delt=0.9 in solovev.json

// Boundary matching vmecpp playground/solovev/solovev.json:
//   R_00=3.999, R_10=1.026, R_20=-0.068
//   Z_10=1.58, Z_20=0.01
inline void setSolovevBoundary(double* h_rbc, double* h_zbs, int mnmax) {
    for (int i = 0; i < mnmax; ++i) { h_rbc[i] = 0.0; h_zbs[i] = 0.0; }
    h_rbc[0 * kNtor + 0] = 3.999;   // R_00
    h_rbc[1 * kNtor + 0] = 1.026;   // R_10
    h_rbc[2 * kNtor + 0] = -0.068;  // R_20
    h_zbs[1 * kNtor + 0] = 1.58;    // Z_10
    h_zbs[2 * kNtor + 0] = 0.01;    // Z_20
}

// Profile functions matching the Solovev analytical equilibrium.
// Solovev: iota = 1.0 (constant), phip = 1.0 (normalized), mass am = [0.125, -0.125]
inline double iotaProfile(double s) { return 1.0; }
inline double massProfile(double s) { return 0.125 * (1.0 - s); }

// Plasma pressure, matching vmecpp's profile chain (radial_profiles.cc:530-540,
// 1183-1187 and ideal_mhd_model.cc pressureAndEnergies):
//   presH = massH / dVds^gamma,  with gamma = 0 (indata_.gamma = 0)
//   massH = evalMassProfile(torflux) * |vpnorm * r00|^gamma
//         = pressureScalingFactor * am(s) * 1
//   pressureScalingFactor = MU_0 * pres_scale = 4*pi*1e-7 * 1 (pres_scale=1)
//   am(s) = 0.125 * (1 - s)
// => presH = 4*pi*1e-7 * 0.125 * (1 - s) = 1.571e-7 * (1 - s)
// (Verified numerically against a vmecpp iter-1 totalP dump: the pressure
// term is the only difference between the codes' totalPressure.)
inline double presProfile(double s) {
    return 4.0e-7 * 3.14159265358979323846 * 0.125 * (1.0 - s);
}
