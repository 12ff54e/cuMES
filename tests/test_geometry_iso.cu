// test_geometry_iso.cu — self-contained coverage check of the ncurr=1
// geometry chain. Runs computeGeometry on a manufactured W7-X-shaped state
// (full mpol/ntor, ntheta/nzeta) and verifies the bsupu/bsubu write
// coverage: every angular point of an interior surface must be written (a
// zero on an interior surface is not a physical bsupu/bsubu value, so a
// launch-shape bug that skips points surfaces as exact zeros).
//
// The coverage checks ASSERT (nonzero exit on failure) — a passing exit code
// must mean every angular point of the surface was written.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "vmec_types.h"
#include "fourier.cuh"
#include "cumes/state/mode_table.cuh"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/magnetic_field_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes_test_support.cuh"


// Manufacture a non-degenerate spectral state on the W7-X shape. The content
// is deliberately generic (all modes get a mild radial envelope) so no
// interior surface collapses to zero geometry: R has a strong m=0/n=0 DC
// plus a few m>0 modes, Z and lambda get m=1..3 content with the same s
// envelopes. This replaces the previous external dump/cuMES/step_0_*.bin
// dependency (gitignored, only present after a CUMES_DUMP=1 run) — the test
// is now self-contained and registerable.
template <typename T>
static void fillState(SpectralState<T>& st, const DeviceParams<T>& p) {
    size_t nb = (size_t)p.ns * p.mnmax * sizeof(T);
    std::vector<T> hcc_(p.ns * p.mnmax, T(0)), hss(p.ns * p.mnmax, T(0));
    std::vector<T> hzsc(p.ns * p.mnmax, T(0)), hzcs(p.ns * p.mnmax, T(0));
    std::vector<T> hlsc(p.ns * p.mnmax, T(0)), hlcs(p.ns * p.mnmax, T(0));
    for (int j = 0; j < p.ns; ++j) {
        double s = (double)j / (p.ns - 1.0);
        for (int mode = 0; mode < p.mnmax; ++mode) {
            int m = mode / (p.ntor + 1), n = mode % (p.ntor + 1);
            if (m == 0 && n == 0) { hcc_[j + mode * p.ns] = T(5.6); hzcs[j + mode * p.ns] = T(0.0); }
            else if (m == 0)      { hcc_[j + mode * p.ns] = T(0.02 * s * s); hzcs[j + mode * p.ns] = T(0.01 * s * s); }
            else if (m == 1)      { hcc_[j + mode * p.ns] = T(0.3 * s); hss[j + mode * p.ns] = T(0.1 * s);
                                    hzsc[j + mode * p.ns] = T(0.2 * s); hzcs[j + mode * p.ns] = T(-0.1 * s); }
            else if (m == 2)      { hcc_[j + mode * p.ns] = T(0.04 * s * s); hss[j + mode * p.ns] = T(0.02 * s * s);
                                    hzsc[j + mode * p.ns] = T(0.03 * s * s); hzcs[j + mode * p.ns] = T(0.01 * s * s); }
            else if (m <= 6)      { hcc_[j + mode * p.ns] = T(0.01 * s * s); hzsc[j + mode * p.ns] = T(0.008 * s * s); }
            hlsc[j + mode * p.ns] = T(0.02 * (m + 1) * s * s);
            hlcs[j + mode * p.ns] = T(0.01 * (m + 1) * s * s);
        }
    }
    cc(cudaMemcpy(st.d_rmncc, hcc_.data(), nb, cudaMemcpyHostToDevice), "cc");
    cc(cudaMemcpy(st.d_rmnss, hss.data(), nb, cudaMemcpyHostToDevice), "ss");
    cc(cudaMemcpy(st.d_zmnsc, hzsc.data(), nb, cudaMemcpyHostToDevice), "zsc");
    cc(cudaMemcpy(st.d_zmncs, hzcs.data(), nb, cudaMemcpyHostToDevice), "zcs");
    cc(cudaMemcpy(st.d_lmnsc, hlsc.data(), nb, cudaMemcpyHostToDevice), "lsc");
    cc(cudaMemcpy(st.d_lmncs, hlcs.data(), nb, cudaMemcpyHostToDevice), "lcs");
}

int main() {
    cumes::ValidatedProblem vp = loadValidated("inputs/w7x.json");
    const cumes::ProblemSpec& spec = vp.spec();
    DeviceParams<double> p{};
    // Full W7-X shape (the largest angular grid the solver runs), exercising
    // the same kernel launch shapes the real runs use.
    p.ns = static_cast<int>(spec.stages.back().radial_surfaces);
    p.mpol = spec.mpol; p.ntor = spec.ntor;
    p.ntheta = spec.angular.ntheta; p.nzeta = spec.angular.nzeta; p.nfp = spec.nfp;
    p.nZnT = p.ntheta * p.nzeta;
    p.mnmax = p.mpol * (p.ntor + 1);
    p.ncurr = (spec.current_model == cumes::CurrentModel::kPrescribedCurrent) ? 1 : 0;
    p.delt = spec.delt; p.ftol = spec.stages.back().tolerance;
    p.max_iter = static_cast<int>(spec.stages.back().max_iterations);
    p.lamscale = 0.0;

    cumes::SpectralStorage<double> storage(p.ns, p.mnmax);
    SpectralState<double> st = storage.legacy_view();
    size_t nb = (size_t)p.ns * p.mnmax * sizeof(double);

    fillState(st, p);
    cumes::Profiles<double> profiles(p, vp, nullptr); cumes::RadialProfileViews<double> rp = profiles.profile_views();
    FourierPlan<double> fp = fourierCreate(p);
    cumes::DeviceModeTable mt = cumes::modeTableCreate(p);
    cumes::RealSpaceStorage<double> rs = realSpaceCreate(p);
    cumes::GeometryOperator<double> geometry(p, nullptr);

    // extrapolate m=1 to the axis (as the solver does each iteration)
    {
        auto* hcc = new double[p.ns * p.mnmax];
        cudaMemcpy(hcc, st.d_rmncc, nb, cudaMemcpyDeviceToHost);
        for (int n = 0; n < p.ntor + 1; ++n) {
            int mn = 1 * (p.ntor + 1) + n;
            hcc[0 + mn * p.ns] = hcc[1 + mn * p.ns];
        }
        cudaMemcpy(st.d_rmncc, hcc, nb, cudaMemcpyHostToDevice);
        delete[] hcc;
    }

    inverseDFT(fp, rs, storage.physical_const(), p, mt.d_xm, mt.d_xn);
    geometry.enqueue(rs, p, rp, 0); cumes::MagneticFieldOperator<double>{}.enqueue(rs, p, rp, geometry.base_geometry_views(p), geometry.magnetic_field_views(p), 0, true);

    // check bsupu coverage on a mid-volume surface. All indices are computed
    // from the actual grid (the old hardcoded ks/1080 were W7-X-specific).
    int nZnT = p.nZnT;
    int jMid = (p.ns - 1) / 2;  // a surface in the middle of the volume
    int nks = 8;
    int ks[8];
    for (int i = 0; i < nks; ++i) ks[i] = (i * nZnT) / nks;  // spread over the plane
    double hb[8];
    for (int i = 0; i < nks; ++i)
        cudaMemcpy(&hb[i], geometry.magnetic_field_views(p).bsupu.data() + jMid * nZnT + ks[i], sizeof(double), cudaMemcpyDeviceToHost);
    int nz = 0;
    double* h_all = new double[(p.ns - 1) * nZnT];
    cudaMemcpy(h_all, geometry.magnetic_field_views(p).bsupu.data(), (p.ns - 1) * nZnT * sizeof(double), cudaMemcpyDeviceToHost);
    for (int k = 0; k < nZnT; ++k) if (h_all[jMid * nZnT + k] == 0.0) ++nz;
    printf("bsupu[jMid=%d] zeros: %d/%d\n", jMid, nz, nZnT);
    for (int i = 0; i < nks; ++i)
        printf("  k=%d: %.6f\n", ks[i], hb[i]);
    // also check the bsubu coverage
    cudaMemcpy(h_all, geometry.magnetic_field_views(p).bsubu.data(), (p.ns - 1) * nZnT * sizeof(double), cudaMemcpyDeviceToHost);
    int nz2 = 0;
    for (int k = 0; k < nZnT; ++k) if (h_all[jMid * nZnT + k] == 0.0) ++nz2;
    printf("bsubu[jMid=%d] zeros: %d/%d\n", jMid, nz2, nZnT);
    delete[] h_all;

    realSpaceFree(rs);
    fourierFree(fp); cumes::modeTableFree(mt);

    // Assertions: a full-coverage kernel must leave no unwritten point on an
    // interior surface (zero is not a physical bsupu/bsubu value there).
    int bad = 0;
    if (nz != 0) { fprintf(stderr, "FAIL: bsupu coverage incomplete (%d zeros)\n", nz); bad = 1; }
    if (nz2 != 0) { fprintf(stderr, "FAIL: bsubu coverage incomplete (%d zeros)\n", nz2); bad = 1; }
    if (bad == 0) printf("test_geometry_iso: coverage OK\n");
    return bad;
}
