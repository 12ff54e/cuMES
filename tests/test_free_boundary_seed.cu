// test_free_boundary_seed.cu — free-boundary trajectory diagnostic: seed the
// cuMES solver with vmecpp's exact pre-pass-2 stage-2 state (decoded from the
// evolve dump; the evolve dump is written post-pass, so the vacuum input of
// pass 2 is evolve_000001's xc_after — verified bit-identical to the
// vac1n_vacuum_000002 boundary) and run the ns=32 stage. vmecpp converges
// from this state (multigrid_result_00032_000774); if cuMES diverges from it,
// the divergence is in the free-boundary coupling. DEBUG harness — the dump
// path is pinned to the vmecpp checkout; a vendored seed replaces it when
// this becomes a permanent test.
#include "JsonParser.h"
#include "cumes/config/profile_functions.hpp"
#include "cumes/physics/free_boundary_operator.hpp"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes/runtime/device_buffer.cuh"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/transforms/toroidal_fft_operator.hpp"
#include "cumes_test_cuda_helper.cuh"
#include "solver.cuh"
#include "vmec_types.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using cumes::test::cc;

int main(int argc, char** argv) {
    const std::string dump =
        "/lustre/qzhong/magnetic-equilibrium-solver/vmecpp/src/vmecpp/cpp/"
        "vmecpp_large_cpp_tests/test_data/solovev_free_bdy/evolve/"
        "evolve_00032_000001_01.solovev_free_bdy.json";
    const char* mgrid =
        "/lustre/qzhong/magnetic-equilibrium-solver/vacuum-field/tests/data/"
        "mgrid_solovev.nc";

    // ---- decode vmecpp's xc_after: [3 families][1][ns][1][mpol] ----
    json::Value root = json::parse_file(dump);
    const json::Value& xc = root.at("xc_after");
    const int ns = 32, mpol = 12, ntor = 0, ntheta = 30, nzeta = 1;
    const int mnmax = mpol * (ntor + 1);
    std::vector<double> h_cc((size_t)ns * mnmax, 0.0);
    std::vector<double> h_zsc((size_t)ns * mnmax, 0.0);
    std::vector<double> h_lsc((size_t)ns * mnmax, 0.0);
    std::vector<double> h_ss((size_t)ns * mnmax, 0.0);
    std::vector<double> h_zcs((size_t)ns * mnmax, 0.0);
    std::vector<double> h_lcs((size_t)ns * mnmax, 0.0);
    if (argc > 1) {
        // Binary seed override: 6 families x [mode][surface] doubles
        // (cuMES layout), as produced from the explore-branch vmecpp state
        // dumps. Bypasses the JSON decode entirely.
        FILE* f = fopen(argv[1], "rb");
        if (f == nullptr) {
            printf("cannot open seed %s\n", argv[1]);
            return 2;
        }
        uint64_t n = 0;
        size_t got = fread(&n, sizeof(n), 1, f);
        (void)got;
        const size_t per_fam = (size_t)ns * mnmax;
        if (n != 6 * per_fam) {
            printf("seed size mismatch: %llu != %zu\n", (unsigned long long)n,
                   6 * per_fam);
            return 2;
        }
        std::vector<double> buf(6 * per_fam);
        if (fread(buf.data(), sizeof(double), buf.size(), f) != buf.size()) {
            printf("seed read failed\n");
            return 2;
        }
        fclose(f);
        std::memcpy(h_cc.data(), buf.data(), per_fam * sizeof(double));
        std::memcpy(h_zsc.data(), buf.data() + per_fam,
                    per_fam * sizeof(double));
        std::memcpy(h_lsc.data(), buf.data() + 2 * per_fam,
                    per_fam * sizeof(double));
    } else {
        // cuMES_state = vmecpp_decomposed * ms*ns at every mode and surface
        // (verified empirically against the explore-branch state dumps);
        // the scalxc odd-m factor is ALREADY inside the decomposed
        // coefficients and must NOT be divided back out.
        for (int j = 0; j < ns; ++j) {
            for (int mode = 0; mode < mnmax; ++mode) {
                const int m = mode / (ntor + 1), n = mode % (ntor + 1);
                const double mfac = (m == 0) ? 1.0 : std::sqrt(2.0);
                const double nfac = (n == 0) ? 1.0 : std::sqrt(2.0);
                const double scale = mfac * nfac;
                h_cc[j + (size_t)mode * ns] =
                    scale * static_cast<double>(xc[0][0][j][0][mode]);
                h_zsc[j + (size_t)mode * ns] =
                    scale * static_cast<double>(xc[1][0][j][0][mode]);
                h_lsc[j + (size_t)mode * ns] =
                    scale * static_cast<double>(xc[2][0][j][0][mode]);
            }
        }
    }

    DeviceParams<double> p;
    p.ns = ns;
    p.mnmax = mnmax;
    p.ntheta = ntheta;
    p.nzeta = nzeta;
    p.nfp = 1;
    p.nZnT = ntheta * nzeta;
    p.mpol = mpol;
    p.ntor = ntor;
    p.ncurr = 0;
    p.delt = 0.9;
    p.ftol = 1e-14;
    p.max_iter = 800;
    p.tcon0 = 1.0;
    p.lamscale = 0.0;

    cumes::ValidatedProblem vp =
        cumes::test::load_validated("inputs/free_bdy/solovev_free_bdy.json");

    cumes::SpectralStorage<double> storage(p.ns, p.mnmax);
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Rcc),
                  h_cc.data(), h_cc.size() * sizeof(double),
                  cudaMemcpyHostToDevice),
       "seed rcc");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Zsc),
                  h_zsc.data(), h_zsc.size() * sizeof(double),
                  cudaMemcpyHostToDevice),
       "seed zsc");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Lsc),
                  h_lsc.data(), h_lsc.size() * sizeof(double),
                  cudaMemcpyHostToDevice),
       "seed lsc");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Rss),
                  h_ss.data(), h_ss.size() * sizeof(double),
                  cudaMemcpyHostToDevice),
       "seed rss");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Zcs),
                  h_zcs.data(), h_zcs.size() * sizeof(double),
                  cudaMemcpyHostToDevice),
       "seed zcs");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Lcs),
                  h_lcs.data(), h_lcs.size() * sizeof(double),
                  cudaMemcpyHostToDevice),
       "seed lcs");

    // The free-boundary operator (hot start: the vacuum state starts
    // INITIALIZED, like vmecpp's stage transition).
    cumes::FreeBoundaryOperator<double>::HostParams hp;
    hp.mgrid_file = mgrid;
    hp.extcur = vp.spec().free_boundary.extcur;
    hp.nvacskip = vp.spec().free_boundary.nvacskip;
    hp.hot_start = true;
    cumes::FreeBoundaryOperator<double> vac(hp, p);

    cumes::Profiles<double> profiles(p, vp, nullptr);
    cumes::RealSpaceStorage<double> rs = realSpaceCreate(p);
    cumes::DeviceModeTable mt = cumes::modeTableCreate<double>(p);
    cumes::ToroidalFftOperator<double> transform(p, rs, mt, nullptr);
    cumes::GeometryOperator<double> geometry(p, nullptr);

    cudaStream_t stream = 0;
    // Per-stage edge pressure (mirrors StageSolver::run).
    {
        const cumes::RadialProfileViews<double> rpv = profiles.profile_views();
        double pres_last = 0.0;
        cc(cudaMemcpy(&pres_last, rpv.pres_H + (p.ns - 2), sizeof(double),
                      cudaMemcpyDeviceToHost),
           "presH");
        const double mass_edge = cumes::evalMassProfile<double>(vp.spec(), 1.0);
        double edge_pressure = cumes::evalMassProfile<double>(
            vp.spec(), (p.ns - 1.5) / (p.ns - 1.0));
        if (edge_pressure != 0.0) {
            edge_pressure = mass_edge / edge_pressure * pres_last;
        }
        vac.set_edge_pressure(edge_pressure);
        printf("edge_pressure = %.6e\n", edge_pressure);
    }

    SolverResult<double> result{false, 0, 1.0, 1.0, 1.0, 0.9, {}};
    try {
        result =
            solverRun<double>(storage, p, profiles, transform, rs, geometry,
                              nullptr, stream, nullptr, nullptr, &vac);
    } catch (const std::exception& e) {
        printf("seeded run: THREW: %s\n", e.what());
        return 2;
    }
    printf(
        "seeded run: converged=%d iterations=%d fsqr=%.3e fsqz=%.3e "
        "fsql=%.3e\n",
        (int)result.converged, result.iterations, (double)result.fsqr,
        (double)result.fsqz, (double)result.fsql);
    return result.converged ? 0 : 1;
}
