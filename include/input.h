// input.h — host-side input parameters.
//
// InputParams is the fixed-size host bundle that feeds the GPU solver
// (boundary, profiles, resolution, multigrid stage sequence). It is
// populated from a vmecpp-style JSON input file by src/input_json.cu
// (API in input_json.h): JSON keys map 1:1 onto the fields below with the
// vmecpp indata schema (mpol, ntor, nfp, ..., ns_array/niter_array/
// ftol_array, am/ac/ai/aphi, raxis_c/zaxis_s, rbc/zbs as {n,m,value}
// objects). Every key is optional; a missing key keeps the member default.
//
// applyResolutionDefaults() applies vmecpp's resolution defaults
// (ntheta=0 -> 2*mpol+6; nzeta=0 -> 1 if ntor=0, else 2*ntor+4), and
// foldBoundary() folds the raw rbc/zbs boundary coefficients (n can be
// negative) into the internal n>=0 product basis rbcc/rbss/zbsc/zbcs,
// matching vmecpp's Boundaries::parseToInternalArrays (boundaries.cc:57-174):
//   rbcc[m,n] = rbc[m,+n] + rbc[m,-n]        (all m)
//   rbss[m,n] = rbc[m,+n] - rbc[m,-n]        (m > 0)
//   zbsc[m,n] = zbs[m,+n] + zbs[m,-n]        (m > 0)
//   zbcs[m,n] = zbs[m,-n] - zbs[m,+n]        (all m)
#pragma once
#include "vmec_types.h"
#include <cmath>

struct BoundaryEntry {
    int m;
    int n;
    double value;
};

// Host-side input parameter bundle (shared by the Solovev and W7-X configs).
struct InputParams {
    int mpol = 6, ntor = 0, nfp = 1;
    int ntheta = 0, nzeta = 0;  // 0 => resolution defaults applied in init
    int ns = 11;
    int ncurr = 0;
    double delt = 0.9, ftol = 1e-16;
    int max_iter = 1000;
    // Multi-radial-grid sequence (vmecpp ns_array/niter_array/ftol_array):
    // the solver runs stage g on ns_array[g] with its own iteration cap and
    // ftol, seeded by the previous stage's converged state (interpolated).
    // ip.ns/max_iter/ftol are the stage-0 values. Default = single stage.
    static constexpr int kMaxGrids = 8;
    int ns_array[kMaxGrids] = {}, niter_array[kMaxGrids] = {};
    double ftol_array[kMaxGrids] = {};
    int n_grids = 1;
    double phiedge = 1.0;
    double pres_scale = 1.0, adiabatic_index = 0.0, spres_ped = 1.0;
    double bloat = 1.0, curtor = 0.0, tcon0 = 1.0;

    static constexpr int kMaxCoeff = 16;
    double am[kMaxCoeff] = {0}, ac[kMaxCoeff] = {0}, ai[kMaxCoeff] = {0};
    double aphi[kMaxCoeff] = {0};
    int am_n = 0, ac_n = 0, ai_n = 0, aphi_n = 0;

    double raxis_c[32] = {0}, zaxis_s[32] = {0};
    int raxis_n = 1;

    BoundaryEntry rbc[256] = {}, zbs[256] = {};
    int rbc_n = 0, zbs_n = 0;

    // Folded boundary coefficients [m][n], m=0..mpol-1, n=0..ntor.
    double rbcc[16][16] = {}, rbss[16][16] = {};
    double zbsc[16][16] = {}, zbcs[16][16] = {};
};

// Apply vmecpp's grid resolution defaults (Sizes::computeDerivedSizes,
// sizes.cc:54-85).
inline void applyResolutionDefaults(InputParams& p) {
    if (p.ntheta < 2 * p.mpol + 6) p.ntheta = 2 * p.mpol + 6;
    if (p.ntor == 0) {
        if (p.nzeta < 1) p.nzeta = 1;
    } else if (p.nzeta < 2 * p.ntor + 4) {
        p.nzeta = 2 * p.ntor + 4;
    }
    p.ntheta = 2 * (p.ntheta / 2);  // nThetaEven
}

// Fold raw rbc/zbs entries (n = -ntor..ntor) into the internal n>=0 product
// basis, matching vmecpp's parseToInternalArrays (see header comment).
inline void foldBoundary(InputParams& p) {
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n)
            p.rbcc[m][n] = p.rbss[m][n] = p.zbsc[m][n] = p.zbcs[m][n] = 0.0;

    for (int i = 0; i < p.rbc_n; ++i) {
        const BoundaryEntry& e = p.rbc[i];
        p.rbcc[e.m][(e.n >= 0) ? e.n : -e.n] += e.value;
        // signum(n): the n=0 fold contributes nothing to the sin-sin part
        double sn = (e.n > 0) ? 1.0 : (e.n < 0) ? -1.0 : 0.0;
        if (e.m > 0) p.rbss[e.m][(e.n >= 0) ? e.n : -e.n] += sn * e.value;
    }
    for (int i = 0; i < p.zbs_n; ++i) {
        const BoundaryEntry& e = p.zbs[i];
        double sn = (e.n > 0) ? 1.0 : (e.n < 0) ? -1.0 : 0.0;
        if (e.m > 0) p.zbsc[e.m][(e.n >= 0) ? e.n : -e.n] += e.value;
        p.zbcs[e.m][(e.n >= 0) ? e.n : -e.n] -= sn * e.value;
    }
}
