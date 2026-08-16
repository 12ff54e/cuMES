// legacy_provenance.hpp — fixed-capacity provenance mirroring the deleted
// legacy InputParams.
//
// The NetCDF/HDF5 v0 writers emit padded arrays (8 stages, 16 coefficients, 32
// axis entries, 16x16 folded boundary) for byte-compatibility with the
// pre-overhaul output. The legacy InputParams struct that used to carry these
// arrays is gone (migration step 13.2); this struct reconstructs them from the
// immutable ValidatedProblem so the v0 writers keep their on-disk layout
// unchanged. A validated problem exceeding a capacity is truncated — the
// shipped configs never do; schema v1 removes the padding entirely.
#pragma once

#include "cumes/config/validated_problem.hpp"

#include <cstddef>

namespace cumes {

struct LegacyInputProvenance {
    static constexpr int kMaxGrids = 8;
    static constexpr int kMaxCoeff = 16;
    static constexpr int kMaxAxis = 32;
    static constexpr int kMaxM = 16;  // folded [m][n]
    static constexpr int kMaxN = 16;

    int n_grids = 1;
    int ns_array[kMaxGrids] = {};
    int niter_array[kMaxGrids] = {};
    double ftol_array[kMaxGrids] = {};
    double am[kMaxCoeff] = {};
    double ac[kMaxCoeff] = {};
    double ai[kMaxCoeff] = {};
    double aphi[kMaxCoeff] = {};
    int am_n = 0, ac_n = 0, ai_n = 0, aphi_n = 0;
    double raxis_c[kMaxAxis] = {};
    double zaxis_s[kMaxAxis] = {};
    int raxis_n = 0;
    int rbc_n = 0, zbs_n = 0;
    double rbcc[kMaxM][kMaxN] = {};
    double rbss[kMaxM][kMaxN] = {};
    double zbsc[kMaxM][kMaxN] = {};
    double zbcs[kMaxM][kMaxN] = {};
    double phiedge = 1.0, pres_scale = 1.0, adiabatic_index = 0.0;
    double spres_ped = 1.0, bloat = 1.0, curtor = 0.0, tcon0 = 1.0;

    // Reconstruct the fixed provenance from the validated model, mirroring the
    // deleted ValidatedProblem::to_input_params() field-for-field.
    static LegacyInputProvenance from_validated(const ValidatedProblem& vp) {
        const ProblemSpec& s = vp.spec();
        const FoldedBoundary& b = vp.boundary();
        LegacyInputProvenance p;

        p.phiedge = s.physical.phiedge;
        p.pres_scale = s.physical.pres_scale;
        p.adiabatic_index = s.physical.adiabatic_index;
        p.spres_ped = s.physical.spres_ped;
        p.bloat = s.physical.bloat;
        p.curtor = s.physical.curtor;
        p.tcon0 = s.physical.tcon0;

        const std::size_t ng = s.stages.size() < kMaxGrids ? s.stages.size() : kMaxGrids;
        p.n_grids = static_cast<int>(s.stages.size());
        for (std::size_t g = 0; g < ng; ++g) {
            p.ns_array[g] = static_cast<int>(s.stages[g].radial_surfaces);
            p.niter_array[g] = static_cast<int>(s.stages[g].max_iterations);
            p.ftol_array[g] = s.stages[g].tolerance;
        }

        auto copy_profile = [](const ProfileSpec& prof, double* dst, int& count) {
            count = static_cast<int>(prof.coefficients.size() < kMaxCoeff
                                         ? prof.coefficients.size() : kMaxCoeff);
            for (int i = 0; i < count; ++i) dst[i] = prof.coefficients[i];
        };
        copy_profile(s.mass, p.am, p.am_n);
        copy_profile(s.iota, p.ai, p.ai_n);
        copy_profile(s.current, p.ac, p.ac_n);
        copy_profile(s.toroidal_flux, p.aphi, p.aphi_n);

        const int na = static_cast<int>(s.raxis_c.size());
        p.raxis_n = na;
        for (int i = 0; i < na && i < kMaxAxis; ++i) p.raxis_c[i] = s.raxis_c[i];
        for (int i = 0; i < na && i < kMaxAxis; ++i) p.zaxis_s[i] = s.zaxis_s[i];

        p.rbc_n = static_cast<int>(s.rbc.size());
        p.zbs_n = static_cast<int>(s.zbs.size());

        const int ntorp1 = s.ntor + 1;
        for (int m = 0; m < s.mpol && m < kMaxM; ++m) {
            for (int n = 0; n < ntorp1 && n < kMaxN; ++n) {
                const std::size_t mode = static_cast<std::size_t>(m) * ntorp1 + n;
                p.rbcc[m][n] = b.rbcc[mode];
                p.rbss[m][n] = b.rbss[mode];
                p.zbsc[m][n] = b.zbsc[mode];
                p.zbcs[m][n] = b.zbcs[mode];
            }
        }
        return p;
    }
};

}  // namespace cumes
