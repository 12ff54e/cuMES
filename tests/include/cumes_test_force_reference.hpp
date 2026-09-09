#ifndef CUMES_TESTS_INCLUDE_CUMES_TEST_FORCE_REFERENCE_HPP_
#define CUMES_TESTS_INCLUDE_CUMES_TEST_FORCE_REFERENCE_HPP_

#include <vector>

namespace cumes::test {

// Independent scalar host reference for the MHD weak-form forces. Geometry
// uses full-grid parity arrays; metric/field arrays use (ns-1, nZnT), and the
// sixteen outputs retain the parity split. Keep this separate from production
// kernels so both frozen-input and converged-state tests check their formulas.
template <typename T>
void cpu_forces(const std::vector<T>& r_e,
                const std::vector<T>& r_o,
                const std::vector<T>& z_o,
                const std::vector<T>& ru_e,
                const std::vector<T>& ru_o,
                const std::vector<T>& zu_e,
                const std::vector<T>& zu_o,
                const std::vector<T>& rv_e,
                const std::vector<T>& rv_o,
                const std::vector<T>& zv_e,
                const std::vector<T>& zv_o,
                const std::vector<T>& lu_e,
                const std::vector<T>& lu_o,
                const std::vector<T>& r12,
                const std::vector<T>& ru12,
                const std::vector<T>& zu12,
                const std::vector<T>& rs,
                const std::vector<T>& zs,
                const std::vector<T>& tau,
                const std::vector<T>& gsqrt,
                const std::vector<T>& guv,
                const std::vector<T>& gvv,
                const std::vector<T>& bsupu,
                const std::vector<T>& bsupv,
                const std::vector<T>& bsubu,
                const std::vector<T>& bsubv,
                const std::vector<T>& totalP,
                const std::vector<T>& sqrtS_F,
                const std::vector<T>& sqrtS_H,
                const std::vector<T>& phip_F,
                int ns,
                int nZnT,
                T lamscale,
                T delta_s,
                std::vector<T>& armn_e,
                std::vector<T>& armn_o,
                std::vector<T>& azmn_e,
                std::vector<T>& azmn_o,
                std::vector<T>& brmn_e,
                std::vector<T>& brmn_o,
                std::vector<T>& bzmn_e,
                std::vector<T>& bzmn_o,
                std::vector<T>& crmn_e,
                std::vector<T>& crmn_o,
                std::vector<T>& czmn_e,
                std::vector<T>& czmn_o,
                std::vector<T>& blmn_e,
                std::vector<T>& blmn_o,
                std::vector<T>& clmn_e,
                std::vector<T>& clmn_o) {
    for (int j = 0; j < ns; ++j) {
        T sF_j = sqrtS_F[j];
        T sFull = sF_j * sF_j;
        for (int k = 0; k < nZnT; ++k) {
            const int idx_f = k + j * nZnT;
            T re_j = r_e[idx_f], ro_j = r_o[idx_f], zo_j = z_o[idx_f];
            T rue_j = ru_e[idx_f], ruo_j = ru_o[idx_f];
            T zue_j = zu_e[idx_f], zuo_j = zu_o[idx_f];
            T rve_j = rv_e[idx_f], rvo_j = rv_o[idx_f];
            T zve_j = zv_e[idx_f], zvo_j = zv_o[idx_f];

            T r12_i = T(0), ru12_i = T(0), zu12_i = T(0), rs_i = T(0),
              zs_i = T(0), tau_i = T(0);
            T gsqrt_i = T(0), bsupu_i = T(0), bsupv_i = T(0);
            T bsubu_i = T(0), bsubv_i = T(0), totalP_i = T(0);
            T sH_i = T(0), gvv_i = T(0), guv_i = T(0);
            T r12_o = T(0), ru12_o = T(0), zu12_o = T(0), rs_o = T(0),
              zs_o = T(0), tau_o = T(0);
            T gsqrt_o = T(0), bsupu_o = T(0), bsupv_o = T(0);
            T bsubu_o = T(0), bsubv_o = T(0), totalP_o = T(0);
            T sH_o = T(0), gvv_o = T(0), guv_o = T(0);

            if (j > 0) {
                int h_i = k + (j - 1) * nZnT;
                r12_i = r12[h_i];
                ru12_i = ru12[h_i];
                zu12_i = zu12[h_i];
                rs_i = rs[h_i];
                zs_i = zs[h_i];
                tau_i = tau[h_i];
                gsqrt_i = gsqrt[h_i];
                gvv_i = gvv[h_i];
                guv_i = guv[h_i];
                bsupu_i = bsupu[h_i];
                bsupv_i = bsupv[h_i];
                bsubu_i = bsubu[h_i];
                bsubv_i = bsubv[h_i];
                totalP_i = totalP[h_i];
                sH_i = sqrtS_H[j - 1];
            }
            if (j < ns - 1) {
                int h_o = k + j * nZnT;
                r12_o = r12[h_o];
                ru12_o = ru12[h_o];
                zu12_o = zu12[h_o];
                rs_o = rs[h_o];
                zs_o = zs[h_o];
                tau_o = tau[h_o];
                gsqrt_o = gsqrt[h_o];
                gvv_o = gvv[h_o];
                guv_o = guv[h_o];
                bsupu_o = bsupu[h_o];
                bsupv_o = bsupv[h_o];
                bsubu_o = bsubu[h_o];
                bsubv_o = bsubv[h_o];
                totalP_o = totalP[h_o];
                sH_o = sqrtS_H[j];
            }

            T P_i = r12_i * totalP_i, P_o = r12_o * totalP_o;
            T zup_i = zu12_i * P_i, zup_o = zu12_o * P_o;
            T rup_i = ru12_i * P_i, rup_o = ru12_o * P_o;
            T rsp_i = rs_i * P_i, rsp_o = rs_o * P_o;
            T zsp_i = zs_i * P_i, zsp_o = zs_o * P_o;
            T taup_i = tau_i * totalP_i, taup_o = tau_o * totalP_o;

            T gbubu_i = gsqrt_i * bsupu_i * bsupu_i;
            T gbubu_o = gsqrt_o * bsupu_o * bsupu_o;
            T gbvbv_i = gsqrt_i * bsupv_i * bsupv_i;
            T gbvbv_o = gsqrt_o * bsupv_o * bsupv_o;
            T gbubv_i = gsqrt_i * bsupu_i * bsupv_i;
            T gbubv_o = gsqrt_o * bsupu_o * bsupv_o;

            T inv_ds = T(1.0) / delta_s;
            T inv_sH_i = (j > 0) ? (T(1.0) / sH_i) : T(0.0);
            T inv_sH_o = (j < ns - 1) ? (T(1.0) / sH_o) : T(0.0);

            T P_avg = T(0.5) * (P_o + P_i);
            T P_wavg = T(0.5) * (P_o * inv_sH_o + P_i * inv_sH_i);
            T gbubu_avg = T(0.5) * (gbubu_o + gbubu_i);
            T gbubu_wavg = T(0.5) * (gbubu_o * sH_o + gbubu_i * sH_i);
            T gbvbv_avg = T(0.5) * (gbvbv_o + gbvbv_i);
            T gbvbv_wavg = T(0.5) * (gbvbv_o * sH_o + gbvbv_i * sH_i);
            T gbubv_avg = T(0.5) * (gbubv_o + gbubv_i);
            T gbubv_wavg = T(0.5) * (gbubv_o * sH_o + gbubv_i * sH_i);

            T armn_e_v = (zup_o - zup_i) * inv_ds + T(0.5) * (taup_o + taup_i) -
                         gbvbv_avg * re_j - gbvbv_wavg * ro_j;
            T armn_o_v = (zup_o * sH_o - zup_i * sH_i) * inv_ds -
                         T(0.5) * P_wavg * zue_j - T(0.5) * P_avg * zuo_j +
                         T(0.5) * (taup_o * sH_o + taup_i * sH_i) -
                         gbvbv_wavg * re_j - gbvbv_avg * ro_j * sFull;
            T azmn_e_v = -(rup_o - rup_i) * inv_ds;
            T azmn_o_v = -(rup_o * sH_o - rup_i * sH_i) * inv_ds +
                         T(0.5) * P_wavg * rue_j + T(0.5) * P_avg * ruo_j;
            T brmn_e_v = T(0.5) * (zsp_o + zsp_i) + T(0.5) * P_wavg * zo_j -
                         gbubu_avg * rue_j - gbubu_wavg * ruo_j -
                         gbubv_avg * rve_j - gbubv_wavg * rvo_j;
            T brmn_o_v = T(0.5) * (zsp_o * sH_o + zsp_i * sH_i) +
                         T(0.5) * P_avg * zo_j - gbubu_wavg * rue_j -
                         gbubu_avg * ruo_j * sFull - gbubv_wavg * rve_j -
                         gbubv_avg * rvo_j * sFull;
            T bzmn_e_v = -T(0.5) * (rsp_o + rsp_i) - T(0.5) * P_wavg * ro_j -
                         gbubu_avg * zue_j - gbubu_wavg * zuo_j -
                         gbubv_avg * zve_j - gbubv_wavg * zvo_j;
            T bzmn_o_v = -T(0.5) * (rsp_o * sH_o + rsp_i * sH_i) -
                         T(0.5) * P_avg * ro_j - gbubu_wavg * zue_j -
                         gbubu_avg * zuo_j * sFull - gbubv_wavg * zve_j -
                         gbubv_avg * zvo_j * sFull;
            T crmn_e_v = gbubv_avg * rue_j + gbubv_wavg * ruo_j +
                         gbvbv_avg * rve_j + gbvbv_wavg * rvo_j;
            T crmn_o_v = gbubv_wavg * rue_j + gbubv_avg * ruo_j * sFull +
                         gbvbv_wavg * rve_j + gbvbv_avg * rvo_j * sFull;
            T czmn_e_v = gbubv_avg * zue_j + gbubv_wavg * zuo_j +
                         gbvbv_avg * zve_j + gbvbv_wavg * zvo_j;
            T czmn_o_v = gbubv_wavg * zue_j + gbubv_avg * zuo_j * sFull +
                         gbvbv_wavg * zve_j + gbvbv_avg * zvo_j * sFull;

            T bsubv_avg = T(0.5) * (bsubv_o + bsubv_i);
            T gvv_gsqrt_i = (j > 0) ? (gvv_i / gsqrt_i) : T(0.0);
            T gvv_gsqrt_o = (j < ns - 1) ? (gvv_o / gsqrt_o) : T(0.0);
            T guv_bsupu_i = (j > 0) ? (guv_i * bsupu_i) : T(0.0);
            T guv_bsupu_o = (j < ns - 1) ? (guv_o * bsupu_o) : T(0.0);
            T lu_e_norm = lamscale * lu_e[idx_f] + phip_F[j];
            T lu_o_norm = lamscale * lu_o[idx_f];
            T bsubv_alt =
                T(0.5) * (gvv_gsqrt_i + gvv_gsqrt_o) * lu_e_norm +
                T(0.5) * (gvv_gsqrt_i * sH_i + gvv_gsqrt_o * sH_o) * lu_o_norm +
                T(0.5) * (guv_bsupu_i + guv_bsupu_o);
            T rb = T(2.0) * T(0.05) * (T(1.0) - sFull);
            T _blmn = bsubv_avg * (T(1.0) - rb) + bsubv_alt * rb;
            if (j > 0) _blmn *= -lamscale;
            T blmn_e_v = _blmn;
            T blmn_o_v = _blmn * sF_j;

            T _clmn = T(0.5) * (bsubu_o + bsubu_i);
            if (j > 0) _clmn *= -lamscale;
            T clmn_e_v = _clmn;
            T clmn_o_v = _clmn * sF_j;

            armn_e[idx_f] = armn_e_v;
            armn_o[idx_f] = armn_o_v;
            azmn_e[idx_f] = azmn_e_v;
            azmn_o[idx_f] = azmn_o_v;
            brmn_e[idx_f] = brmn_e_v;
            brmn_o[idx_f] = brmn_o_v;
            bzmn_e[idx_f] = bzmn_e_v;
            bzmn_o[idx_f] = bzmn_o_v;
            crmn_e[idx_f] = crmn_e_v;
            crmn_o[idx_f] = crmn_o_v;
            czmn_e[idx_f] = czmn_e_v;
            czmn_o[idx_f] = czmn_o_v;
            blmn_e[idx_f] = blmn_e_v;
            blmn_o[idx_f] = blmn_o_v;
            clmn_e[idx_f] = clmn_e_v;
            clmn_o[idx_f] = clmn_o_v;
        }
    }
}

}  // namespace cumes::test

#endif  // CUMES_TESTS_INCLUDE_CUMES_TEST_FORCE_REFERENCE_HPP_
