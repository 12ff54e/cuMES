// seed_state.hpp — cold-start and restart spectral-state construction
// (blueprint §6.11; extracted from main.cu so the CLI and the fixed-iteration
// benchmark share the exact same seeding, not two drifting copies).
//
// Three host-side helpers reproduce the legacy seeding verbatim, now consuming
// the immutable ValidatedProblem directly (migration step 13.2 — the legacy
// InputParams fixed-capacity bridge is gone):
//
//   init_params   — DeviceParams<T> from ValidatedProblem (stage 0)
//   init_state    — vmecpp interpFromBoundaryAndAxis cold start
//   restart_state — upload a host EquilibriumSnapshot + LCFS/axis patch
//
// These are the same functions main.cu inlined before Phase 9; the move is a
// pure code relocation (Class A — no arithmetic change), verified by the
// unchanged Solovev/W7-X trajectories.
#pragma once

#include <cstdio>
#include <cmath>
#include <cstdint>

#include "cumes/config/validated_problem.hpp"
#include "cumes/io/equilibrium_snapshot.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/state/spectral_storage.hpp"
#include "vmec_types.h"

namespace cumes {

// DeviceParams<T> from ValidatedProblem. lamscale is set later by profilesCreate;
// ns/max_iter/ftol carry stage 0 (the stage loop overwrites them per stage).
template <typename T>
DeviceParams<T> init_params(const ValidatedProblem& vp) {
    const ProblemSpec& s = vp.spec();
    DeviceParams<T> p;
    p.ns = static_cast<int>(s.stages.front().radial_surfaces);
    p.mpol = s.mpol; p.ntor = s.ntor;
    p.ntheta = s.angular.ntheta; p.nzeta = s.angular.nzeta; p.nfp = s.nfp;
    p.nZnT = p.ntheta * p.nzeta;
    p.mnmax = p.mpol * (p.ntor + 1);   // folded basis: mode = m*(ntor+1)+n
    p.ncurr = (s.current_model == CurrentModel::kPrescribedCurrent) ? 1 : 0;
    p.delt = T(s.delt);
    p.ftol = T(s.stages.front().tolerance);
    p.max_iter = static_cast<int>(s.stages.front().max_iterations);
    p.tcon0 = T(s.physical.tcon0);     // constraint-force multiplier
    p.lamscale = T(0.0);               // set by profilesCreate
    return p;
}

// Initial state from vmecpp's interpFromBoundaryAndAxis (fourier_geometry.cc):
//   m=0: linear interpolation in s between the magnetic axis (raxis_c /
//        zaxis_s) and the boundary; zmnsc/rmnss have no m=0 content.
//   m>0: s^(m/2) radial envelope so higher modes vanish faster near the axis.
// cuMES stores the plain physical coefficients (vmecpp's internal state
// divides by mscale*nscale, but its mscale'd basis makes the real-space
// reconstruction identical).
template <typename T>
SpectralStorage<T> init_state(const DeviceParams<T>& p, const ValidatedProblem& vp) {
    const ProblemSpec& sp = vp.spec();
    const FoldedBoundary& b = vp.boundary();
    const int ntorp1 = p.ntor + 1;
    size_t nb = (size_t)p.ns * p.mnmax * sizeof(T);
    SpectralStorage<T> storage(p.ns, p.mnmax);

    auto* c=new T[p.ns*p.mnmax](), *s=new T[p.ns*p.mnmax]();
    auto* zsc=new T[p.ns*p.mnmax](), *zcs=new T[p.ns*p.mnmax]();
    auto* lsc=new T[p.ns*p.mnmax](), *lcs=new T[p.ns*p.mnmax]();

    for(int j=0;j<p.ns;++j){
        T sFlux = T(j)/T(p.ns-1);          // normalized flux s
        T sqrtS  = std::sqrt(sFlux);        // sqrt(s)
        for(int m=0;m<p.mpol;++m){
            for(int n=0;n<p.ntor+1;++n){
                int mn = m*(p.ntor+1)+n;
                if(m==0){
                    // m=0: linear in s between axis and boundary
                    c[j+mn*p.ns]   = sFlux*T(b.rbcc[0*ntorp1+n]) + (T(1.0)-sFlux)*T(sp.raxis_c[n]);
                    zcs[j+mn*p.ns] = sFlux*T(b.zbcs[0*ntorp1+n]) - (T(1.0)-sFlux)*T(sp.zaxis_s[n]);
                    // rmnss/zmnsc: no m=0 content; lambda: zero initially
                } else if(m==1){
                    // m=1: s^(1/2) radial envelope (s^(m/2)), matching
                    // vmecpp's physical state (interpFromBoundaryAndAxis).
                    // NOTE: the real-space odd-parity values then carry the
                    // 1/max(sqrt(s),sqrt(1/(ns-1))) decomposition factor
                    // (applied in the inverse DFT), so the real-space m=1
                    // contribution is constant across the interior — matching
                    // vmecpp's decomposed real space (its real-space odd =
                    // physical/max).
                    T w = sqrtS;  // s^(1/2)
                    c[j+mn*p.ns]   = w * T(b.rbcc[m*ntorp1+n]);
                    s[j+mn*p.ns]   = w * T(b.rbss[m*ntorp1+n]);
                    zsc[j+mn*p.ns] = w * T(b.zbsc[m*ntorp1+n]);
                    zcs[j+mn*p.ns] = w * T(b.zbcs[m*ntorp1+n]);
                } else {
                    // m>=2: s^(m/2) radial envelope, vanishing at axis
                    T w = std::pow(sqrtS, m);  // s^(m/2)
                    c[j+mn*p.ns]   = w * T(b.rbcc[m*ntorp1+n]);
                    s[j+mn*p.ns]   = w * T(b.rbss[m*ntorp1+n]);
                    zsc[j+mn*p.ns] = w * T(b.zbsc[m*ntorp1+n]);
                    zcs[j+mn*p.ns] = w * T(b.zbcs[m*ntorp1+n]);
                }
                // lmnsc/lmncs: zero initially (lambda is a free gauge)
            }
        }
    }
    printf("  initState: vmecpp interpFromBoundaryAndAxis (m>0 s^(m/2))\n");

    SpectralState<T> st = storage.legacy_view();
    check_cuda(cudaMemcpy(st.d_rmncc,c,nb,cudaMemcpyHostToDevice),"cpy cc");
    check_cuda(cudaMemcpy(st.d_rmnss,s,nb,cudaMemcpyHostToDevice),"cpy ss");
    check_cuda(cudaMemcpy(st.d_zmnsc,zsc,nb,cudaMemcpyHostToDevice),"cpy zsc");
    check_cuda(cudaMemcpy(st.d_zmncs,zcs,nb,cudaMemcpyHostToDevice),"cpy zcs");
    check_cuda(cudaMemcpy(st.d_lmnsc,lsc,nb,cudaMemcpyHostToDevice),"cpy lsc");
    check_cuda(cudaMemcpy(st.d_lmncs,lcs,nb,cudaMemcpyHostToDevice),"cpy lcs");
    delete[] c; delete[] s; delete[] zsc; delete[] zcs; delete[] lsc; delete[] lcs;
    return storage;
}

// Restart state from a host snapshot (read_checkpoint / convert_legacy_init),
// uploading the six families and applying the same LCFS-boundary + axis-
// regularity patch the legacy CUMES_LOAD_INIT path did. The checkpoint stores
// doubles regardless of T; the conversion mirrors outputSaveBinary's T->double
// in reverse.
template <typename T>
SpectralStorage<T> restart_state(const DeviceParams<T>& p, const ValidatedProblem& vp,
                                 const EquilibriumSnapshot& snap) {
    const FoldedBoundary& b = vp.boundary();
    const int ntorp1 = p.ntor + 1;
    const size_t one = (size_t)p.ns * p.mnmax;
    const size_t nb = one * sizeof(T);
    SpectralStorage<T> storage(p.ns, p.mnmax);

    auto* c = new T[one];   // rmncc
    auto* zsc = new T[one]; // zmnsc
    auto* lsc = new T[one]; // lmnsc
    auto* s = new T[one];   // rmnss
    auto* zcs = new T[one]; // zmncs
    auto* lcs = new T[one]; // lmncs
    for (size_t i = 0; i < one; ++i) {
        c[i]   = T(snap.families[EquilibriumSnapshot::kRmncc][i]);
        zsc[i] = T(snap.families[EquilibriumSnapshot::kZmnsc][i]);
        lsc[i] = T(snap.families[EquilibriumSnapshot::kLmnsc][i]);
        s[i]   = T(snap.families[EquilibriumSnapshot::kRmnss][i]);
        zcs[i] = T(snap.families[EquilibriumSnapshot::kZmncs][i]);
        lcs[i] = T(snap.families[EquilibriumSnapshot::kLmncs][i]);
    }

    // vmecpp stores boundary values separately (not in the spectral state);
    // cuMES embeds the boundary in the spectral coefficients at j=ns-1. Patch
    // the LCFS values to match the folded boundary; also zero m>0 modes at the
    // magnetic axis (j=0) — vmecpp does this via extrapolateTowardsAxis().
    {
        int jB = p.ns - 1;  // LCFS index
        for (int m = 0; m < p.mpol; ++m) {
            for (int n = 0; n < p.ntor + 1; ++n) {
                int mn = m * (p.ntor + 1) + n;
                c[jB + mn * p.ns] = T(b.rbcc[m * ntorp1 + n]);
                s[jB + mn * p.ns] = T(b.rbss[m * ntorp1 + n]);
                zsc[jB + mn * p.ns] = T(b.zbsc[m * ntorp1 + n]);
                zcs[jB + mn * p.ns] = T(b.zbcs[m * ntorp1 + n]);
                if (m > 0) {
                    c[0 + mn * p.ns] = T(0.0);
                    s[0 + mn * p.ns] = T(0.0);
                    zsc[0 + mn * p.ns] = T(0.0);
                    zcs[0 + mn * p.ns] = T(0.0);
                    lsc[0 + mn * p.ns] = T(0.0);
                    lcs[0 + mn * p.ns] = T(0.0);
                }
            }
        }
    }

    SpectralState<T> st = storage.legacy_view();
    check_cuda(cudaMemcpy(st.d_rmncc, c, nb, cudaMemcpyHostToDevice), "restart cc");
    check_cuda(cudaMemcpy(st.d_zmnsc, zsc, nb, cudaMemcpyHostToDevice), "restart zsc");
    check_cuda(cudaMemcpy(st.d_lmnsc, lsc, nb, cudaMemcpyHostToDevice), "restart lsc");
    check_cuda(cudaMemcpy(st.d_rmnss, s, nb, cudaMemcpyHostToDevice), "restart ss");
    check_cuda(cudaMemcpy(st.d_zmncs, zcs, nb, cudaMemcpyHostToDevice), "restart zcs");
    check_cuda(cudaMemcpy(st.d_lmncs, lcs, nb, cudaMemcpyHostToDevice), "restart lcs");
    delete[] c; delete[] s; delete[] zsc; delete[] zcs; delete[] lsc; delete[] lcs;
    printf("  restartState: uploaded checkpoint + LCFS/axis patch\n");
    return storage;
}

}  // namespace cumes
