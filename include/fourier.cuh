// fourier.cuh — DFT transforms between spectral (parity-split) and real space.
#pragma once
#include "vmec_types.h"
#include <cublas_v2.h>

struct ConstraintWorkspace;  // defined in constraint.cuh

struct FourierPlan {
    FourierBasis basis;
    cublasHandle_t handle;

    // Real-space geometry (parity-split, full grid)
    double* d_r_e;  double* d_z_e;  double* d_l_e;
    double* d_ru_e; double* d_zu_e; double* d_lu_e;
    double* d_r_o;  double* d_z_o;  double* d_l_o;
    double* d_ru_o; double* d_zu_o; double* d_lu_o;
    double* d_r_real; double* d_z_real; double* d_l_real;
    double* d_ru_real; double* d_zu_real; double* d_lu_real;
    double* d_rv_real; double* d_zv_real; double* d_lv_real;
    // Parity-split toroidal derivatives (the 3D metric/forces need them)
    double* d_rv_e; double* d_rv_o;
    double* d_zv_e; double* d_zv_o;
    double* d_lv_e; double* d_lv_o;

    // Force components (parity-split). crmn/czmn/clmn are the toroidal force
    // components (3D only; zero for axisymmetric).
    double* d_armn_e; double* d_armn_o;
    double* d_azmn_e; double* d_azmn_o;
    double* d_brmn_e; double* d_brmn_o;
    double* d_bzmn_e; double* d_bzmn_o;
    double* d_blmn_e; double* d_blmn_o;
    double* d_crmn_e; double* d_crmn_o;
    double* d_czmn_e; double* d_czmn_o;
    double* d_clmn_e; double* d_clmn_o;

    // Combined forces (backward compat)
    double* d_fr_real; double* d_fz_real; double* d_fl_real;
};

FourierPlan fourierCreate(const GridParams& p, cublasHandle_t handle);
void fourierFree(FourierPlan& fp);

void inverseDFT(const FourierPlan& fp, const SpectralState& st,
                const GridParams& p);

// forwardDFT: parity forces → 6-component spectral forces
// Layout: (6*mnmax, ns) col-major
//   [0*mnmax*ns .. 1*mnmax*ns): f_rmncc
//   [1*mnmax*ns .. 2*mnmax*ns): f_zmnsc
//   [2*mnmax*ns .. 3*mnmax*ns): f_lmnsc
//   [3*mnmax*ns .. 4*mnmax*ns): f_rmnss
//   [4*mnmax*ns .. 5*mnmax*ns): f_zmncs
//   [5*mnmax*ns .. 6*mnmax*ns): f_lmncs
void forwardDFT(const FourierPlan& fp, double* d_f_spectral,
                const GridParams& p, const ConstraintWorkspace& cw);

void forwardDFTDirect(const FourierPlan& fp, double* d_f_spectral,
                      const GridParams& p);
