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

#include "cumes/config/validated_problem.hpp"
#include "cumes/io/equilibrium_snapshot.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/state/spectral_storage.hpp"
#include "vmec_types.h"

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace cumes {

// DeviceParams<T> from ValidatedProblem. lamscale is set later by
// profilesCreate; ns/max_iter/ftol carry stage 0 (the stage loop overwrites
// them per stage).
template <typename T>
DeviceParams<T> init_params(const ValidatedProblem& vp) {
    const ProblemSpec& s = vp.spec();
    DeviceParams<T> p;
    p.ns = static_cast<int>(s.stages.front().radial_surfaces);
    p.mpol = s.mpol;
    p.ntor = s.ntor;
    p.ntheta = s.angular.ntheta;
    p.nzeta = s.angular.nzeta;
    p.nfp = s.nfp;
    p.nZnT = p.ntheta * p.nzeta;
    p.mnmax = p.mpol * (p.ntor + 1);  // folded basis: mode = m*(ntor+1)+n
    p.ncurr = (s.current_model == CurrentModel::kPrescribedCurrent) ? 1 : 0;
    p.delt = T(s.delt);
    p.ftol = T(s.stages.front().tolerance);
    p.max_iter = static_cast<int>(s.stages.front().max_iterations);
    p.tcon0 = T(s.physical.tcon0);  // constraint-force multiplier
    p.lamscale = T(0.0);            // set by profilesCreate
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
SpectralStorage<T> init_state(const DeviceParams<T>& p,
                              const ValidatedProblem& vp) {
    const ProblemSpec& sp = vp.spec();
    const FoldedBoundary& b = vp.boundary();
    const int ntorp1 = p.ntor + 1;
    const size_t one = (size_t)p.ns * p.mnmax;
    const size_t nb = one * sizeof(T);
    SpectralStorage<T> storage(p.ns, p.mnmax);

    // One staging buffer in the exact state_slab() order
    // (Rcc Zsc Lsc Rss Zcs Lcs — spectral_storage.hpp), so the six per-family
    // H2D copies become a single upload. The host staging exists only for the
    // double->T conversion; the layout and values are unchanged
    // (bit-identical).
    auto* h_state = new T[6 * one]();
    auto* h_c = h_state + 0 * one;    // rmncc
    auto* h_zsc = h_state + 1 * one;  // zmnsc
    auto* h_lsc = h_state + 2 * one;  // lmnsc
    auto* h_s = h_state + 3 * one;    // rmnss
    auto* h_zcs = h_state + 4 * one;  // zmncs
    auto* h_lcs = h_state + 5 * one;  // lmncs

    for (int j = 0; j < p.ns; ++j) {
        T sFlux = T(j) / T(p.ns - 1);  // normalized flux s
        T sqrtS = std::sqrt(sFlux);    // sqrt(s)
        for (int m = 0; m < p.mpol; ++m) {
            for (int n = 0; n < p.ntor + 1; ++n) {
                int mn = m * (p.ntor + 1) + n;
                if (m == 0) {
                    // m=0: linear in s between axis and boundary
                    h_c[j + mn * p.ns] = sFlux * T(b.rbcc[0 * ntorp1 + n]) +
                                         (T(1.0) - sFlux) * T(sp.raxis_c[n]);
                    h_zcs[j + mn * p.ns] = sFlux * T(b.zbcs[0 * ntorp1 + n]) -
                                           (T(1.0) - sFlux) * T(sp.zaxis_s[n]);
                    // rmnss/zmnsc: no m=0 content; lambda: zero initially
                } else if (m == 1) {
                    // m=1: s^(1/2) radial envelope (s^(m/2)), matching
                    // vmecpp's physical state (interpFromBoundaryAndAxis).
                    // NOTE: the real-space odd-parity values then carry the
                    // 1/max(sqrt(s),sqrt(1/(ns-1))) decomposition factor
                    // (applied in the inverse DFT), so the real-space m=1
                    // contribution is constant across the interior — matching
                    // vmecpp's decomposed real space (its real-space odd =
                    // physical/max).
                    T w = sqrtS;  // s^(1/2)
                    h_c[j + mn * p.ns] = w * T(b.rbcc[m * ntorp1 + n]);
                    h_s[j + mn * p.ns] = w * T(b.rbss[m * ntorp1 + n]);
                    h_zsc[j + mn * p.ns] = w * T(b.zbsc[m * ntorp1 + n]);
                    h_zcs[j + mn * p.ns] = w * T(b.zbcs[m * ntorp1 + n]);
                } else {
                    // m>=2: s^(m/2) radial envelope, vanishing at axis
                    T w = std::pow(sqrtS, m);  // s^(m/2)
                    h_c[j + mn * p.ns] = w * T(b.rbcc[m * ntorp1 + n]);
                    h_s[j + mn * p.ns] = w * T(b.rbss[m * ntorp1 + n]);
                    h_zsc[j + mn * p.ns] = w * T(b.zbsc[m * ntorp1 + n]);
                    h_zcs[j + mn * p.ns] = w * T(b.zbcs[m * ntorp1 + n]);
                }
                // lmnsc/lmncs: zero initially (lambda is a free gauge)
            }
        }
    }
    printf("  initState: vmecpp interpFromBoundaryAndAxis (m>0 s^(m/2))\n");

    check_cuda(cudaMemcpy(storage.state_slab(), h_state, 6 * nb,
                          cudaMemcpyHostToDevice),
               "init state slab");
    delete[] h_state;
    return storage;
}

// Restart state from a host snapshot (read_checkpoint), uploading the six
// families and applying the same LCFS-boundary + axis-regularity patch the
// legacy CUMES_LOAD_INIT path did. The checkpoint stores doubles regardless
// of T; the upload converts double -> T.
template <typename T>
SpectralStorage<T> restart_state(const DeviceParams<T>& p,
                                 const ValidatedProblem& vp,
                                 const EquilibriumSnapshot& snap) {
    const FoldedBoundary& b = vp.boundary();
    const int ntorp1 = p.ntor + 1;
    const size_t one = (size_t)p.ns * p.mnmax;
    const size_t nb = one * sizeof(T);
    SpectralStorage<T> storage(p.ns, p.mnmax);

    // One staging buffer in the exact state_slab() order
    // (Rcc Zsc Lsc Rss Zcs Lcs — spectral_storage.hpp), so the six per-family
    // H2D copies become a single upload. The host staging exists only for the
    // double->T conversion; the layout and values are unchanged
    // (bit-identical).
    auto* h_state = new T[6 * one]();
    auto* h_c = h_state + 0 * one;    // rmncc
    auto* h_zsc = h_state + 1 * one;  // zmnsc
    auto* h_lsc = h_state + 2 * one;  // lmnsc
    auto* h_s = h_state + 3 * one;    // rmnss
    auto* h_zcs = h_state + 4 * one;  // zmncs
    auto* h_lcs = h_state + 5 * one;  // lmncs
    for (size_t i = 0; i < one; ++i) {
        h_c[i] = T(snap.families[EquilibriumSnapshot::kRmncc][i]);
        h_zsc[i] = T(snap.families[EquilibriumSnapshot::kZmnsc][i]);
        h_lsc[i] = T(snap.families[EquilibriumSnapshot::kLmnsc][i]);
        h_s[i] = T(snap.families[EquilibriumSnapshot::kRmnss][i]);
        h_zcs[i] = T(snap.families[EquilibriumSnapshot::kZmncs][i]);
        h_lcs[i] = T(snap.families[EquilibriumSnapshot::kLmncs][i]);
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
                h_c[jB + mn * p.ns] = T(b.rbcc[m * ntorp1 + n]);
                h_s[jB + mn * p.ns] = T(b.rbss[m * ntorp1 + n]);
                h_zsc[jB + mn * p.ns] = T(b.zbsc[m * ntorp1 + n]);
                h_zcs[jB + mn * p.ns] = T(b.zbcs[m * ntorp1 + n]);
                if (m > 0) {
                    h_c[0 + mn * p.ns] = T(0.0);
                    h_s[0 + mn * p.ns] = T(0.0);
                    h_zsc[0 + mn * p.ns] = T(0.0);
                    h_zcs[0 + mn * p.ns] = T(0.0);
                    h_lsc[0 + mn * p.ns] = T(0.0);
                    h_lcs[0 + mn * p.ns] = T(0.0);
                }
            }
        }
    }

    check_cuda(cudaMemcpy(storage.state_slab(), h_state, 6 * nb,
                          cudaMemcpyHostToDevice),
               "restart state slab");
    delete[] h_state;
    printf("  restartState: uploaded checkpoint + LCFS/axis patch\n");
    return storage;
}

}  // namespace cumes
