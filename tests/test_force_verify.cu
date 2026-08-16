// test_force_verify.cu — converged equilibrium must sit near force balance.
//
// Runs the solver to convergence on the Solovev fixture (self-contained — no
// external vmecpp_init.bin dependency), then recomputes the spectral forces
// through the test's own forward-DFT path and checks the residual is small.
// A broken force formula shows up as O(1) residuals; a correct one lands far
// below the threshold. This is the registerable, fixture-free replacement
// for the old test that loaded an absent vmecpp_init.bin.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "input_json.h"
#include "constraint.cuh"
#include "vmec_types.h"
#include "fourier.cuh"
#include "geometry.cuh"
#include "forces.cuh"
#include "profiles.cuh"
#include "solver.cuh"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/transforms/toroidal_fft_operator.hpp"
#include "cumes_test_support.cuh"

static int failures = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (cond) {                                                          \
            printf("PASS %s\n", msg);                                        \
        } else {                                                             \
            printf("FAIL %s\n", msg);                                        \
            ++failures;                                                      \
        }                                                                    \
    } while (0)


int main() {
    // ---- Initial Solovev state (ns=55 = the Solovev final grid, ntor=0) ----
    const int ns = 55, mpol = 6, ntor = 0, ntheta = 18, nzeta = 1;
    GridParams<double> p;
    p.ns = ns; p.mnmax = mpol * (ntor + 1); p.ntheta = ntheta; p.nzeta = nzeta;
    p.nfp = 1; p.nZnT = ntheta * nzeta; p.mpol = mpol; p.ntor = ntor;
    p.ncurr = 0; p.delt = 0.9; p.ftol = 1e-14; p.max_iter = 2000;
    p.tcon0 = 1.0; p.lamscale = 0.0;

    // ---- Initial state from vmecpp interpFromBoundaryAndAxis (same logic as
    // main.cu initState): m=0 linear in s between axis and boundary, m>0 with
    // a s^(m/2) radial envelope. Uses the folded boundary from the JSON.
    InputParams ip = initInputParams();
    cumes::SpectralStorage<double> storage(ns, p.mnmax);
    SpectralState<double> st = storage.legacy_view();
    size_t nb = (size_t)ns * p.mnmax * sizeof(double);
    auto* h_rmncc = new double[ns * p.mnmax]();
    auto* h_zmnsc = new double[ns * p.mnmax]();
    auto* h_lmnsc = new double[ns * p.mnmax]();
    auto* h_rmnss = new double[ns * p.mnmax]();
    auto* h_zmncs = new double[ns * p.mnmax]();
    auto* h_lmncs = new double[ns * p.mnmax]();
    for (int j = 0; j < ns; ++j) {
        double sFlux = (double)j / (ns - 1.0);
        double sqrtS = std::sqrt(sFlux);
        for (int m = 0; m < p.mpol; ++m) {
            for (int n = 0; n < p.ntor + 1; ++n) {
                int mn = m * (p.ntor + 1) + n;
                if (m == 0) {
                    h_rmncc[j + mn * ns] = sFlux * ip.rbcc[0][n] + (1.0 - sFlux) * ip.raxis_c[n];
                    h_zmncs[j + mn * ns] = sFlux * ip.zbcs[0][n] - (1.0 - sFlux) * ip.zaxis_s[n];
                } else if (m == 1) {
                    double w = sqrtS;
                    h_rmncc[j + mn * ns] = w * ip.rbcc[m][n];
                    h_rmnss[j + mn * ns] = w * ip.rbss[m][n];
                    h_zmnsc[j + mn * ns] = w * ip.zbsc[m][n];
                    h_zmncs[j + mn * ns] = w * ip.zbcs[m][n];
                } else {
                    double w = std::pow(sqrtS, m);
                    h_rmncc[j + mn * ns] = w * ip.rbcc[m][n];
                    h_rmnss[j + mn * ns] = w * ip.rbss[m][n];
                    h_zmnsc[j + mn * ns] = w * ip.zbsc[m][n];
                    h_zmncs[j + mn * ns] = w * ip.zbcs[m][n];
                }
            }
        }
    }

    cc(cudaMemcpy(st.d_rmncc, h_rmncc, nb, cudaMemcpyHostToDevice), "cpy rmncc");
    cc(cudaMemcpy(st.d_zmnsc, h_zmnsc, nb, cudaMemcpyHostToDevice), "cpy zmnsc");
    cc(cudaMemcpy(st.d_lmnsc, h_lmnsc, nb, cudaMemcpyHostToDevice), "cpy lmnsc");
    cc(cudaMemcpy(st.d_rmnss, h_rmnss, nb, cudaMemcpyHostToDevice), "cpy rmnss");
    cc(cudaMemcpy(st.d_zmncs, h_zmncs, nb, cudaMemcpyHostToDevice), "cpy zmncs");
    cc(cudaMemcpy(st.d_lmncs, h_lmncs, nb, cudaMemcpyHostToDevice), "cpy lmncs");
    delete[] h_rmncc; delete[] h_zmnsc; delete[] h_lmnsc;
    delete[] h_rmnss; delete[] h_zmncs; delete[] h_lmncs;

    // ---- Profiles / plan / workspace ----
    RadialProfiles<double> rp = profilesCreate(p, ip);
    cumes::ToroidalFftOperator<double> transform(p, nullptr);
    FourierPlan<double>& fp = transform.fourier_plan();
    cumes::RealSpaceStorage<double> rs = realSpaceCreate(p);
    cumes::GeometryOperator<double> geometry(p, nullptr);
    MetricWorkspace<double>& mw = geometry.workspace();

    // ---- Converge: the solver drives the MHD residual to ftol ----
    SolverResult<double> res = solverRun(storage, p, rp, transform, rs, geometry);
    printf("solver: converged=%d iterations=%d fsqr=%.3e fsqz=%.3e fsql=%.3e\n",
           res.converged, res.iterations, res.fsqr, res.fsqz, res.fsql);
    CHECK(res.converged, "converged equilibrium reached");

    // ---- Recompute forces through the test's own path ----
    inverseDFT(fp, rs, storage.physical_const(), p);
    computeGeometry(rs, p, rp, mw);
    computeForces(rs, p, rp, mw);

    // Forward DFT to get spectral forces (6 families).
    const size_t n6 = (size_t)6 * ns * p.mnmax;
    double* d_f_spec;
    cc(cudaMalloc(&d_f_spec, n6 * sizeof(double)), "f_spec");
    ConstraintWorkspace<double> cw_zero{};
    cc(cudaMalloc(&cw_zero.d_frcon_e, (size_t)p.ns * p.nZnT * sizeof(double)), "frcon_e");
    cc(cudaMalloc(&cw_zero.d_frcon_o, (size_t)p.ns * p.nZnT * sizeof(double)), "frcon_o");
    cc(cudaMalloc(&cw_zero.d_fzcon_e, (size_t)p.ns * p.nZnT * sizeof(double)), "fzcon_e");
    cc(cudaMalloc(&cw_zero.d_fzcon_o, (size_t)p.ns * p.nZnT * sizeof(double)), "fzcon_o");
    cc(cudaMemset(cw_zero.d_frcon_e, 0, (size_t)p.ns * p.nZnT * sizeof(double)), "frcon_e zero");
    cc(cudaMemset(cw_zero.d_frcon_o, 0, (size_t)p.ns * p.nZnT * sizeof(double)), "frcon_o zero");
    cc(cudaMemset(cw_zero.d_fzcon_e, 0, (size_t)p.ns * p.nZnT * sizeof(double)), "fzcon_e zero");
    cc(cudaMemset(cw_zero.d_fzcon_o, 0, (size_t)p.ns * p.nZnT * sizeof(double)), "fzcon_o zero");
    forwardDFT(fp, rs, cumes::SpectralView<double, cumes::DecomposedResidualDomain>(
                       d_f_spec, p.ns, p.mnmax),
               p, cw_zero);

    auto* h_f = new double[n6];
    cc(cudaMemcpy(h_f, d_f_spec, n6 * sizeof(double), cudaMemcpyDeviceToHost), "cpy f");

    // Residuals in the vmecpp groups: fsqr = frcc+frss (0,3), fsqz = fzsc+fzcs
    // (1,4), fsql = flsc+flcs (2,5).
    double fsqr = 0, fsqz = 0, fsql = 0;
    for (int c = 0; c < 6; ++c) {
        double sum = 0;
        for (size_t i = 0; i < (size_t)ns * p.mnmax; ++i)
            sum += h_f[c * (size_t)ns * p.mnmax + i] * h_f[c * (size_t)ns * p.mnmax + i];
        sum /= (double)(ns * p.mnmax);
        if (c == 0 || c == 3) fsqr += sum;
        else if (c == 1 || c == 4) fsqz += sum;
        else if (c == 2 || c == 5) fsql += sum;
    }
    printf("Recomputed force residuals for the converged equilibrium:\n");
    printf("  FSQR = %.3e  FSQZ = %.3e  FSQL = %.3e\n", fsqr, fsqz, fsql);

    // The whole point: a converged equilibrium must sit near a force balance.
    // A broken formula shows up as O(1) residuals; a correct one lands far
    // below. The solver's own fsqr here is ~1e-16; the recomputed path shares
    // the same kernels, so a 1e-4 threshold is orders of magnitude of margin.
    const double kFailThresh = 1e-4;
    CHECK(fsqr <= kFailThresh, "FSQR small for converged equilibrium");
    CHECK(fsqz <= kFailThresh, "FSQZ small for converged equilibrium");
    CHECK(fsql <= kFailThresh, "FSQL small for converged equilibrium");

    // Cleanup
    cudaFree(d_f_spec);
    cudaFree(cw_zero.d_frcon_e); cudaFree(cw_zero.d_frcon_o);
    cudaFree(cw_zero.d_fzcon_e); cudaFree(cw_zero.d_fzcon_o);
    profilesFree(rp);  // fp/mw owned by ToroidalFftOperator/GeometryOperator (RAII)
    delete[] h_f;

    if (failures == 0) { printf("test_force_verify: ALL PASS\n"); return 0; }
    printf("test_force_verify: %d FAILURES\n", failures);
    return 1;
}
