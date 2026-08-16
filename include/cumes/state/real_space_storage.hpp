// real_space_storage.hpp — stage-owned real-space arrays (blueprint §6.5/§6.6).
//
// Splits the parity-split geometry + force + combined buffers out of the legacy
// FourierPlan so a transform operator owns only transform scratch. The storage
// owns 43 device arrays with the SAME layout and names the FourierPlan used
// (bit-for-bit identical pointers/indices):
//   18 parity-split geometry (R/Z/λ × value/θ-deriv/ζ-deriv × even/odd),
//    9 combined (e+o) geometry buffers (fourierCombineParity / do_combine=true),
//   16 parity-split force buffers (computeForces output, forward-DFT input).
// It is a plain owning bundle + typed view accessors; no kernels live here.
// Ownership lives in the stage; the transform/geometry/force/constraint
// operators hold non-owning references.
#pragma once

#include "cumes/state/real_fields.cuh"
#include "vmec_types.h"

namespace cumes {

template <class T>
struct RealSpaceStorage {
    // Parity-split geometry (full grid, inverse-DFT output).
    T* d_r_e = nullptr;  T* d_z_e = nullptr;  T* d_l_e = nullptr;
    T* d_ru_e = nullptr; T* d_zu_e = nullptr; T* d_lu_e = nullptr;
    T* d_r_o = nullptr;  T* d_z_o = nullptr;  T* d_l_o = nullptr;
    T* d_ru_o = nullptr; T* d_zu_o = nullptr; T* d_lu_o = nullptr;
    T* d_rv_e = nullptr; T* d_zv_e = nullptr; T* d_lv_e = nullptr;
    T* d_rv_o = nullptr; T* d_zv_o = nullptr; T* d_lv_o = nullptr;

    // Combined (e+o) geometry buffers (fourierCombineParity / do_combine=true).
    T* d_r_real = nullptr;  T* d_z_real = nullptr;  T* d_l_real = nullptr;
    T* d_ru_real = nullptr; T* d_zu_real = nullptr; T* d_lu_real = nullptr;
    T* d_rv_real = nullptr; T* d_zv_real = nullptr; T* d_lv_real = nullptr;

    // Parity-split force buffers (computeForces output, forward-DFT input).
    T* d_armn_e = nullptr; T* d_armn_o = nullptr;
    T* d_azmn_e = nullptr; T* d_azmn_o = nullptr;
    T* d_brmn_e = nullptr; T* d_brmn_o = nullptr;
    T* d_bzmn_e = nullptr; T* d_bzmn_o = nullptr;
    T* d_blmn_e = nullptr; T* d_blmn_o = nullptr;
    T* d_crmn_e = nullptr; T* d_crmn_o = nullptr;
    T* d_czmn_e = nullptr; T* d_czmn_o = nullptr;
    T* d_clmn_e = nullptr; T* d_clmn_o = nullptr;

    bool arena_backed = false;

    // Typed view bundles over the owned arrays (bit-identical indexing).
    GeometryParityViews<T> geometry_views(const GridParams<T>& p) const {
        auto f = [&](T* d) { return RealFieldView<T>(d, p.ns, p.ntheta, p.nzeta); };
        GeometryParityViews<T> v;
        v.r_e = f(d_r_e); v.z_e = f(d_z_e); v.l_e = f(d_l_e);
        v.ru_e = f(d_ru_e); v.zu_e = f(d_zu_e); v.lu_e = f(d_lu_e);
        v.r_o = f(d_r_o); v.z_o = f(d_z_o); v.l_o = f(d_l_o);
        v.ru_o = f(d_ru_o); v.zu_o = f(d_zu_o); v.lu_o = f(d_lu_o);
        v.rv_e = f(d_rv_e); v.zv_e = f(d_zv_e); v.lv_e = f(d_lv_e);
        v.rv_o = f(d_rv_o); v.zv_o = f(d_zv_o); v.lv_o = f(d_lv_o);
        return v;
    }

    ForceParityViews<T> force_views(const GridParams<T>& p) const {
        auto f = [&](T* d) { return RealFieldView<T>(d, p.ns, p.ntheta, p.nzeta); };
        ForceParityViews<T> v;
        v.armn_e = f(d_armn_e); v.armn_o = f(d_armn_o);
        v.azmn_e = f(d_azmn_e); v.azmn_o = f(d_azmn_o);
        v.brmn_e = f(d_brmn_e); v.brmn_o = f(d_brmn_o);
        v.bzmn_e = f(d_bzmn_e); v.bzmn_o = f(d_bzmn_o);
        v.blmn_e = f(d_blmn_e); v.blmn_o = f(d_blmn_o);
        v.clmn_e = f(d_clmn_e); v.clmn_o = f(d_clmn_o);
        v.crmn_e = f(d_crmn_e); v.crmn_o = f(d_crmn_o);
        v.czmn_e = f(d_czmn_e); v.czmn_o = f(d_czmn_o);
        return v;
    }
};

}  // namespace cumes
