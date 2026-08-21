// test_regression_kernels.cu — containment-fix regression gates.
//
// Two fixes from the 2026-08-12 issue pass are regression-tested here against
// CPU scalar references:
//
//  1. De-alias bandpass theta coverage (constraint.cu deAliasAnalyzeKernel).
//     The analyze kernel sums gConEff over the FULL theta grid in 32-point
//     groups (for (int it0 = 4*t; it0 < ntheta; it0 += 32)), so grids with
//     ntheta > 32 are fully summed instead of silently dropping the tail.
//     The regression runs ntheta = 30, 32, 34, 40 and compares the GPU
//     bandpass output against a CPU projection that sums ALL theta samples:
//     30 bounds the 32-point-group loop from below, 32 is the shipped
//     boundary, 34/40 have ntheta > 32 (the case the fix targets) and
//     exercise the second group iteration (the old kernel hard-capped at
//     32 and silently dropped the tail).
//     NOTE: odd ntheta is deliberately NOT exercised here.  The de-alias
//     analyze loop itself handles odd ntheta, but the surrounding pipeline
//     does not: inverseDFT's poloidal accumulation (and the geometry it
//     feeds) uses a 2*(ntheta/2) thread split that leaves the last theta
//     point unwritten, and its stale value then poisons the preconCompute
//     surface sums with NaN (tau = 0 -> pTau = 0/0).  applyResolutionDefaults
//     likewise forces ntheta even.  Odd ntheta is not a supported solver
//     configuration, so the coverage regression uses the even sizes above.
//
//  2. PCR solve row coverage (precon.cu tridiagSolveKernel).  The kernel uses
//     a 128-thread grid-stride loop so solved rows above 128 (ns > 129) are
//     no longer left at their staged values.  The regression runs
//     ns = 3, 17, 65, 127, 129, 130, 257 and compares the preconditioned
//     force buffer against a serial Thomas solve on the SAME assembled matrix
//     coefficients (precon.ar()/dr/br/az/dz/bz, precon.jmin(), precon.lambdaPrec() copied
//     to host after the real preconCompute).
//
// Conventions match the other tests: everything is templated on T and both
// double and float are instantiated; the CPU references always compute in
// double from the (possibly float) T inputs; the float leg compares at a
// relaxed tolerance.
//
// The bandpass test drives the PUBLIC operator: it manufactures the
// effective-constraint-force inputs (rCon/zCon/rCon0/zCon0 + ruFull/zuFull)
// so that constraintCompute's effectiveConstraintKernel produces the desired
// deterministic gConEff = pattern exactly, then the built-in analyze->D2Z->
// coeff-pack->Z2D->synthesize chain bandpasses it.  tcon comes from the real
// preconditioner (preconCompute on the manufactured geometry) through the
// public constraintCompute(precon_updated=true) path.
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include "vmec_types.h"
#include "cumes/state/real_space_storage.hpp"
#include "cumes/state/mode_table.cuh"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/magnetic_field_operator.hpp"
#include "cumes/physics/constraint_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes/transforms/toroidal_fft_operator.hpp"
#include "cumes/numerics/preconditioner.hpp"
#include "cumes/runtime/device_buffer.cuh"
#include "cumes_test_cuda_helper.cuh"
using namespace cumes::test;



// Absolute comparison against a precomputed tolerance (see the callers: the
// tolerances are scale-based — rel * the grid/component reference scale + an
// absolute floor — because the GPU runs in float and the CPU reference in
// double, and the float result carries rounding amplified by the coefficient
// scale; a per-point relative test would spuriously fail at genuine zero
// crossings of the output).
static void checkNear(double gpu, double ref, double tol,
                      const char* s, long long a, long long b, long long c) {
    if (!(fabs(gpu - ref) <= tol)) {
        fprintf(stderr, "FAIL [%s] a=%lld b=%lld c=%lld gpu=%.15e ref=%.15e\n",
                s, a, b, c, gpu, ref);
        ++failures();
    }
}

// ---------------------------------------------------------------------------
// DeviceParams / ValidatedProblem construction (self-contained, no JSON/files).
// ---------------------------------------------------------------------------
template <typename T>
static DeviceParams<T> makeParams(int ns, int mpol, int ntor, int ntheta, int nzeta) {
    DeviceParams<T> p;
    p.ns = ns; p.mnmax = mpol * (ntor + 1);
    p.ntheta = ntheta; p.nzeta = nzeta; p.nfp = 1;
    p.nZnT = ntheta * nzeta; p.mpol = mpol; p.ntor = ntor;
    p.ncurr = 0; p.delt = T(1.0); p.ftol = T(1e-14); p.max_iter = 10;
    p.tcon0 = T(1.0); p.lamscale = T(1.0);
    return p;
}

// Solovev-like profiles (matches inputs/solovev.json: am=[0.125,-0.125],
// ai=[1.0]; aphi=[1.0] so the toroidal flux / phip profile is non-degenerate).
static cumes::ValidatedProblem solovevInput() {
    cumes::ProblemSpec spec;
    spec.mpol = 6; spec.ntor = 0; spec.nfp = 1;
    spec.current_model = cumes::CurrentModel::kFixedIota;
    spec.delt = 0.9;
    spec.mass.coefficients = {0.125, -0.125};
    spec.iota.coefficients = {1.0};
    spec.toroidal_flux.coefficients = {1.0};
    spec.rbc = {{1, 0, 1.0}};
    spec.zbs = {{1, 0, 0.5}};
    spec.stages = {{11, 1000, 1e-16}};
    return validate_spec(std::move(spec));
}

// ---------------------------------------------------------------------------
// Manufactured spectral state (Solovev-like: R_00 ~ 4, some m=1/2/3 content).
// ---------------------------------------------------------------------------
template <typename T>
static void fillState(std::vector<T>& cc, std::vector<T>& ss,
                      std::vector<T>& zsc, std::vector<T>& zcs,
                      std::vector<T>& lsc, std::vector<T>& lcs,
                      int ns, int mnmax, int ntor) {
    for (int j = 0; j < ns; ++j) {
        double s = (double)j / (ns - 1.0);
        for (int mode = 0; mode < mnmax; ++mode) {
            int m = mode / (ntor + 1), n = mode % (ntor + 1);
            if (m == 0 && n == 0)        cc[j + mode * ns] = T(4.0);
            else if (m == 1 && n == 0)   cc[j + mode * ns] = T(0.3 * s);
            else if (m == 2 && n == 0)   cc[j + mode * ns] = T(0.2 * s);
            else if (m == 3 && n == 0)   cc[j + mode * ns] = T(0.1 * s);
            ss[j + mode * ns] = cc[j + mode * ns];
            if (m == 1 && n == 0)        { zsc[j + mode * ns] = T(-0.5 * s); zcs[j + mode * ns] = T(-0.5 * s); }
            else if (m == 2 && n == 0)   { zsc[j + mode * ns] = T(0.15 * s); zcs[j + mode * ns] = T(0.15 * s); }
            else if (m == 3 && n == 0)   { zsc[j + mode * ns] = T(-0.05 * s); zcs[j + mode * ns] = T(-0.05 * s); }
            lsc[j + mode * ns] = T(0.01 * (m + 1) * s);
            lcs[j + mode * ns] = T(0.01 * (m + 1) * s);
        }
    }
}

// (upload_state is the shared six-family upload in cumes_test_cuda_helper.cuh.)

// ---------------------------------------------------------------------------
// CPU scalar reference for the de-alias bandpass.
//
// Replicates constraint.cu deAliasAnalyzeKernel -> D2Z -> deAliasCoeffPackKernel
// -> Z2D -> deAliasSynthesizeKernel exactly:
//   * analysis: UNIFORM sums of gConEff*sin(mθ) / gConEff*cos(mθ) over the
//     FULL theta grid (per zeta plane) — the kernel does not use the reduced
//     [0,pi] trapezoid here, it sums all ntheta points in 32-point groups.
//   * D2Z (unnormalized 1D real FFT over ζ): Re F_sc(n) = Σ_k S_sc cos(2πnk/N),
//     Im F_cs(n) = -Σ_k S_cs sin(2πnk/N).
//   * pack: norm = 2/nZnT (n=0), 4/nZnT (n>0); scale = tcon[jF]*faccon[m];
//     the sc channel keeps Re F_sc, the cs channel keeps -Im F_cs, and the
//     Z2D synthesis expands the real spectrum as X[0]+2ΣRe cos − 2ΣIm sin
//     (cuFFT is unnormalized), which is why the packed bins carry half the
//     coefficient (the 0.5 in deAliasCoeffPackKernel cancels the 2×).
//   * synthesis: gCon[jF][k][l] = Σ_m slot0*sin(mθ_l) + slot1*cos(mθ_l).
//
// gCon at jF == 0 is not written (the kernel skips the axis); callers compare
// jF >= 1 only.
static void cpuDealiasBandpass(const double* gConEff, int ns, int mpol, int ntor,
                               int ntheta, int nzeta, int nZnT,
                               const double* tcon, const double* faccon,
                               double* gCon) {
    int nz2 = nzeta / 2 + 1;
    std::vector<double> coeffSc((size_t)(mpol - 2) * (ns - 1) * nz2, 0.0);
    std::vector<double> coeffCs((size_t)(mpol - 2) * (ns - 1) * nz2, 0.0);
    for (int jF = 1; jF < ns; ++jF) {
        int jF1 = jF - 1;
        for (int m = 1; m <= mpol - 2; ++m) {
            int m1 = m - 1;
            // Uniform full-theta projection per zeta plane.
            std::vector<double> Ssc(nzeta, 0.0), Scs(nzeta, 0.0);
            for (int k = 0; k < nzeta; ++k) {
                const double* g = gConEff + (size_t)jF * nZnT + (size_t)k * ntheta;
                double ssc = 0.0, scs = 0.0;
                for (int it = 0; it < ntheta; ++it) {
                    double th = 2.0 * M_PI * it / ntheta;
                    ssc += g[it] * sin(m * th);
                    scs += g[it] * cos(m * th);
                }
                Ssc[k] = ssc; Scs[k] = scs;
            }
            double scale = tcon[jF] * faccon[m];
            for (int n = 0; n <= ntor; ++n) {
                double reFsc = 0.0, imFcs = 0.0;
                for (int k = 0; k < nzeta; ++k) {
                    double ang = 2.0 * M_PI * n * k / nzeta;
                    reFsc += Ssc[k] * cos(ang);
                    imFcs += -Scs[k] * sin(ang);
                }
                double norm = (n > 0) ? 4.0 / nZnT : 2.0 / nZnT;
                coeffSc[(size_t)(m1 * (ns - 1) + jF1) * nz2 + n] = norm * scale * reFsc;
                coeffCs[(size_t)(m1 * (ns - 1) + jF1) * nz2 + n] = norm * scale * (-imFcs);
            }
            // Unnormalized Z2D + poloidal synthesis.
            for (int k = 0; k < nzeta; ++k) {
                size_t base = (size_t)(m1 * (ns - 1) + jF1) * nz2;
                double slot0 = coeffSc[base];   // n=0 (cs n=0 is dropped: shalf=0)
                double slot1 = 0.0;
                for (int n = 1; n <= ntor; ++n) {
                    double ang = 2.0 * M_PI * n * k / nzeta;
                    slot0 += coeffSc[base + n] * cos(ang);
                    slot1 += coeffCs[base + n] * sin(ang);
                }
                for (int l = 0; l < ntheta; ++l) {
                    double th = 2.0 * M_PI * l / ntheta;
                    gCon[(size_t)jF * nZnT + (size_t)k * ntheta + l] +=
                        slot0 * sin(m * th) + slot1 * cos(m * th);
                }
            }
        }
    }
}

// Deterministic gConEff pattern: DC + sin(m=1..8) + cos(m=2,4) with a mild
// surface factor.  The bandpass keeps sin(m=1..4) (scaled by tcon*faccon),
// drops m=0, m=5..8, and (for n=0) drops the entire cos channel — so the
// expected output is analytically known and every mode is exercised.
static double daPatternValue(int jF, int it, int ntheta, int ns) {
    double sF = (ns > 1) ? (double)jF / (ns - 1.0) : 0.0;
    double th = 2.0 * M_PI * it / ntheta;
    double v = 0.5
        + 0.7 * sin(1.0 * th) + 0.6 * sin(2.0 * th) + 0.5 * sin(3.0 * th)
        + 0.4 * sin(4.0 * th) + 0.3 * sin(5.0 * th) + 0.2 * sin(6.0 * th)
        + 0.15 * sin(7.0 * th) + 0.1 * sin(8.0 * th)
        + 0.3 * cos(2.0 * th) + 0.25 * cos(4.0 * th);
    return (1.0 + 0.1 * sF) * v;
}

// ---------------------------------------------------------------------------
// CPU serial Thomas solve.
//
// Solves rows jMin..jMax-1 of
//   b[j]*x[j-1] + d[j]*x[j] + a[j]*x[j+1] = rhs[j]
// with the boundary conditions x[jMin-1] = 0 and x[jMax] = 0 (the LCFS row is
// OUTSIDE the solve — the PCR kernel's hasL/hasR guards impose exactly these,
// see the boundary analysis in the test header).  Only x[j] for jMin..jMax-1
// are written; x[jMax] and x[j<jMin] are left untouched.
static void thomasSolve(const double* a, const double* d, const double* b,
                        int jMin, int jMax, const double* rhs, double* x) {
    int n = jMax - jMin;
    if (n <= 0) return;
    std::vector<double> cp(n), dp(n);
    double denom = d[jMin];
    cp[0] = a[jMin] / denom;
    dp[0] = rhs[jMin] / denom;
    for (int i = 1; i < n; ++i) {
        int j = jMin + i;
        denom = d[j] - b[j] * cp[i - 1];
        cp[i] = a[j] / denom;
        dp[i] = (rhs[j] - b[j] * dp[i - 1]) / denom;
    }
    x[jMax - 1] = dp[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        int j = jMin + i;
        x[j] = dp[i] - cp[i] * x[j + 1];
    }
}

// CPU reference for preconApply (precon.cu tridiagSolveKernel):
//   comps 0,3 -> R tridiagonal (ar/dr/br), comps 1,4 -> Z tridiagonal (az/dz/bz)
//   comps 0..4 zeroed for j < jMin, comps 2,5 multiplied by lambdaPrec
//   (all surfaces 0..ns-1, including the LCFS; comp 5 is NOT zeroed below
//   jMin — matching the kernel).
// f is the manufactured 6-family buffer [comp*mnmax*ns + mode*ns + jF],
// modified in place (in double).
template <typename T>
static void cpuPreconApplyRef(const std::vector<T>& ar, const std::vector<T>& dr,
                              const std::vector<T>& br, const std::vector<T>& az,
                              const std::vector<T>& dz, const std::vector<T>& bz,
                              const std::vector<int>& jMin,
                              const std::vector<T>& lambdaPrec,
                              int ns, int mnmax, std::vector<double>& f) {
    int stride = mnmax * ns;
    for (int mode = 0; mode < mnmax; ++mode) {
        int jm = jMin[mode];
        int jMax = ns - 1;
        std::vector<double> arD(ns), drD(ns), brD(ns), azD(ns), dzD(ns), bzD(ns);
        for (int j = 0; j < ns; ++j) {
            arD[j] = (double)ar[mode * ns + j]; drD[j] = (double)dr[mode * ns + j];
            brD[j] = (double)br[mode * ns + j]; azD[j] = (double)az[mode * ns + j];
            dzD[j] = (double)dz[mode * ns + j]; bzD[j] = (double)bz[mode * ns + j];
        }
        // R system (comps 0, 3); Z system (comps 1, 4).
        for (int c = 0; c < 6; ++c) {
            if (c == 2 || c == 5) continue;
            const std::vector<double>* aP = (c == 0 || c == 3) ? &arD : &azD;
            const std::vector<double>* dP = (c == 0 || c == 3) ? &drD : &dzD;
            const std::vector<double>* bP = (c == 0 || c == 3) ? &brD : &bzD;
            std::vector<double> rhs(ns), x(ns, 0.0);
            for (int j = 0; j < ns; ++j) rhs[j] = f[c * stride + mode * ns + j];
            thomasSolve(aP->data(), dP->data(), bP->data(), jm, jMax, rhs.data(), x.data());
            for (int j = jm; j < jMax; ++j) f[c * stride + mode * ns + j] = x[j];
        }
        // Zero comps 0..4 below jMin (comp 5 keeps its value).
        for (int j = 0; j < jm; ++j)
            for (int c = 0; c < 5; ++c)
                f[c * stride + mode * ns + j] = 0.0;
        // Lambda diagonal preconditioner on comps 2, 5 (all surfaces).
        for (int j = 0; j < ns; ++j) {
            double lp = (double)lambdaPrec[mode * ns + j];
            f[2 * stride + mode * ns + j] *= lp;
            f[5 * stride + mode * ns + j] *= lp;
        }
    }
}

// ---------------------------------------------------------------------------
// De-alias bandpass theta-coverage scenario (one ntheta value).
// ---------------------------------------------------------------------------
template <typename T>
static int testDealias(int ntheta) {
    int lf = failures();
    printf("  de-alias bandpass theta coverage: ns=11 mpol=6 ntor=0 nzeta=1 ntheta=%d ... ", ntheta);
    const int ns = 11, mpol = 6, ntor = 0, nzeta = 1;
    DeviceParams<T> p = makeParams<T>(ns, mpol, ntor, ntheta, nzeta);
    cumes::ValidatedProblem vp = solovevInput();
    cumes::Profiles<T> profiles(p, vp, nullptr); cumes::RadialProfileViews<T> rp = profiles.profile_views();
    cumes::DeviceModeTable mt = cumes::modeTableCreate(p);
    cumes::RealSpaceStorage<T> rs = realSpaceCreate(p);
    cumes::GeometryOperator<T> geometry(p, nullptr);
    cumes::ToroidalFftOperator<T> transform(p, rs, mt, nullptr);
    cumes::Preconditioner<T> precon(p, nullptr);
    cumes::ConstraintOperator<T> constraint(p, nullptr);
    cumes::SpectralStorage<T> storage(ns, p.mnmax);
    std::vector<T> cc(ns * p.mnmax), ss(ns * p.mnmax), zsc(ns * p.mnmax);
    std::vector<T> zcs(ns * p.mnmax), lsc(ns * p.mnmax), lcs(ns * p.mnmax);
    fillState(cc, ss, zsc, zcs, lsc, lcs, ns, p.mnmax, ntor);
    upload_state(storage, cc, ss, zsc, zcs, lsc, lcs, ns, p.mnmax);
    transform.inverse(storage.physical_const(), /*do_combine=*/true);
    geometry.enqueue(rs, p, rp, 0); cumes::MagneticFieldOperator<T>{}.enqueue(rs, p, rp, geometry.base_geometry_views(p), geometry.magnetic_field_views(p), nullptr, 0, true);
    precon.enqueue_compute(rs, mt.d_xm, mt.d_xn, p, rp, geometry.base_geometry_views(p), geometry.magnetic_field_views(p), nullptr, 0);

    // Manufacture the bandpass INPUT through the public operator.  With
    // ruFull = zuFull = 1 (ru_e = zu_e = 1, ru_o = zu_o = 0), rCon = pattern,
    // zCon = rCon0 = zCon0 = 0, the effectiveConstraintKernel computes
    //   gConEff = (rCon - rCon0)*ruFull + (zCon - zCon0)*zuFull = pattern
    // exactly, and the built-in bandpass chain filters it.
    size_t nF = (size_t)ns * p.nZnT * sizeof(T);
    std::vector<T> pat(ns * p.nZnT, T(0));
    for (int jF = 0; jF < ns; ++jF)
        for (int k = 0; k < nzeta; ++k)
            for (int it = 0; it < p.ntheta; ++it)
                pat[jF * p.nZnT + k * p.ntheta + it] = T(daPatternValue(jF, it, p.ntheta, ns));
    check_cuda(cudaMemcpy(constraint.rcon_view(p).data(), pat.data(), nF, cudaMemcpyHostToDevice), "rCon up");
    check_cuda(cudaMemset(constraint.zcon_view(p).data(), 0, nF), "zCon zero");
    check_cuda(cudaMemset(constraint.rcon0(), 0, nF), "rCon0 zero");
    check_cuda(cudaMemset(constraint.zcon0(), 0, nF), "zCon0 zero");
    std::vector<T> ones(ns * p.nZnT, T(1.0));
    check_cuda(cudaMemcpy(rs.d_ru_e, ones.data(), nF, cudaMemcpyHostToDevice), "ru_e up");
    check_cuda(cudaMemcpy(rs.d_zu_e, ones.data(), nF, cudaMemcpyHostToDevice), "zu_e up");
    check_cuda(cudaMemset(rs.d_ru_o, 0, nF), "ru_o zero");
    check_cuda(cudaMemset(rs.d_zu_o, 0, nF), "zu_o zero");
    // Deterministic addConstraint outputs (not compared, but keeps the pass clean).
    check_cuda(cudaMemset(rs.d_brmn_e, 0, nF), "brmn_e zero");
    check_cuda(cudaMemset(rs.d_brmn_o, 0, nF), "brmn_o zero");
    check_cuda(cudaMemset(rs.d_bzmn_e, 0, nF), "bzmn_e zero");
    check_cuda(cudaMemset(rs.d_bzmn_o, 0, nF), "bzmn_o zero");

    // tcon is refreshed from the (real) preconditioner + manufactured ru/zu;
    // the bandpass runs on the manufactured gConEff.
    constraint.enqueue(p, rs, precon.ard(), precon.azd(), rp.sqrtS_F, true, &transform, nullptr, 0);

    std::vector<T> h_tcon(p.ns);
    check_cuda(cudaMemcpy(h_tcon.data(), constraint.tcon(), p.ns * sizeof(T), cudaMemcpyDeviceToHost), "tcon get");
    std::vector<T> h_gCon(ns * p.nZnT);
    check_cuda(cudaMemcpy(h_gCon.data(), constraint.gcon(), nF, cudaMemcpyDeviceToHost), "gCon get");
    std::vector<double> tcon(p.ns), faccon(p.mnmax);
    for (int jF = 0; jF < p.ns; ++jF) tcon[jF] = (double)h_tcon[jF];
    for (int m = 0; m < p.mnmax; ++m) faccon[m] = (double)constraint.h_faccon()[m];

    // CPU reference bandpass of the same (float-rounded) pattern, in double.
    std::vector<double> gEff(ns * p.nZnT, 0.0), gRef(ns * p.nZnT, 0.0);
    for (int i = 0; i < ns * p.nZnT; ++i) gEff[i] = (double)pat[i];
    cpuDealiasBandpass(gEff.data(), ns, mpol, ntor, p.ntheta, nzeta, p.nZnT,
                       tcon.data(), faccon.data(), gRef.data());

    // The synthesize kernel's theta thread split is 2*(ntheta/2): for odd
    // ntheta the last theta point is not written by the kernel.  All the
    // ntheta values exercised here are even, so this equals ntheta; the
    // defensive 2*(ntheta/2) keeps the comparison honest if an odd size is
    // ever added (the analyze coverage is still validated on every written
    // point, since each depends on the FULL-theta analysis sum).
    int nCompTheta = 2 * (p.ntheta / 2);
    // Scale-based tolerance: the GPU float arithmetic multiplies large
    // coefficients by small (but non-zero in float) sin/cos values at the
    // output's genuine zero crossings, so the per-point error is bounded by
    // ~rel * the OUTPUT SCALE (not rel * the local value).  A dropped theta
    // sample in the analyze (the regression this guards) changes every
    // coefficient by O(1/ntheta) of its value — far above rel*scale.
    double rel   = (sizeof(T) == sizeof(float)) ? 1e-4 : 1e-9;
    double absF  = (sizeof(T) == sizeof(float)) ? 1e-7 : 1e-12;
    double scale = 0.0;
    for (int jF = 1; jF < ns; ++jF)
        for (int k = 0; k < nzeta; ++k)
            for (int l = 0; l < nCompTheta; ++l) {
                int idx = jF * p.nZnT + k * p.ntheta + l;
                scale = fmax(scale, fabs(gRef[idx]));
            }
    double tol = rel * scale + absF;
    for (int jF = 1; jF < ns; ++jF)
        for (int k = 0; k < nzeta; ++k)
            for (int l = 0; l < nCompTheta; ++l) {
                int idx = jF * p.nZnT + k * p.ntheta + l;
                checkNear((double)h_gCon[idx], gRef[idx], tol, "dealias", jF, k, l);
            }

    
    realSpaceFree(rs);
    cumes::modeTableFree(mt);
    printf(failures() == lf ? "PASS\n" : "FAIL\n");
    return failures() - lf;
}

// ---------------------------------------------------------------------------
// PCR solve row-coverage scenario (one ns value).
// ---------------------------------------------------------------------------
template <typename T>
static int testPcr(int ns) {
    int lf = failures();
    printf("  PCR solve row coverage: mpol=4 ntor=0 ntheta=18 nzeta=1 ns=%d ... ", ns);
    const int mpol = 4, ntor = 0, ntheta = 18, nzeta = 1;
    DeviceParams<T> p = makeParams<T>(ns, mpol, ntor, ntheta, nzeta);
    cumes::ValidatedProblem vp = solovevInput();
    cumes::Profiles<T> profiles(p, vp, nullptr); cumes::RadialProfileViews<T> rp = profiles.profile_views();
    cumes::DeviceModeTable mt = cumes::modeTableCreate(p);
    cumes::RealSpaceStorage<T> rs = realSpaceCreate(p);
    cumes::GeometryOperator<T> geometry(p, nullptr);
    cumes::ToroidalFftOperator<T> transform(p, rs, mt, nullptr);
    cumes::Preconditioner<T> precon(p, nullptr);
    cumes::SpectralStorage<T> storage(ns, p.mnmax);
    std::vector<T> cc(ns * p.mnmax), ss(ns * p.mnmax), zsc(ns * p.mnmax);
    std::vector<T> zcs(ns * p.mnmax), lsc(ns * p.mnmax), lcs(ns * p.mnmax);
    fillState(cc, ss, zsc, zcs, lsc, lcs, ns, p.mnmax, ntor);
    upload_state(storage, cc, ss, zsc, zcs, lsc, lcs, ns, p.mnmax);
    transform.inverse(storage.physical_const(), /*do_combine=*/true);
    geometry.enqueue(rs, p, rp, 0); cumes::MagneticFieldOperator<T>{}.enqueue(rs, p, rp, geometry.base_geometry_views(p), geometry.magnetic_field_views(p), nullptr, 0, true);
    precon.enqueue_compute(rs, mt.d_xm, mt.d_xn, p, rp, geometry.base_geometry_views(p), geometry.magnetic_field_views(p), nullptr, 0);

    // Copy the assembled matrix coefficients to host: the CPU Thomas reference
    // must use the EXACT same ar/dr/br/az/dz/bz, jMin and lambdaPrec the GPU
    // PCR kernel solves with.
    std::vector<T> ar(p.mnmax * ns), dr(p.mnmax * ns), br(p.mnmax * ns);
    std::vector<T> az(p.mnmax * ns), dz(p.mnmax * ns), bz(p.mnmax * ns);
    std::vector<T> lam(p.mnmax * ns);
    std::vector<int> jMin(p.mnmax);
    size_t szMN = (size_t)p.mnmax * ns * sizeof(T);
    check_cuda(cudaMemcpy(ar.data(), precon.ar(), szMN, cudaMemcpyDeviceToHost), "ar get");
    check_cuda(cudaMemcpy(dr.data(), precon.dr(), szMN, cudaMemcpyDeviceToHost), "dr get");
    check_cuda(cudaMemcpy(br.data(), precon.br(), szMN, cudaMemcpyDeviceToHost), "br get");
    check_cuda(cudaMemcpy(az.data(), precon.az(), szMN, cudaMemcpyDeviceToHost), "az get");
    check_cuda(cudaMemcpy(dz.data(), precon.dz(), szMN, cudaMemcpyDeviceToHost), "dz get");
    check_cuda(cudaMemcpy(bz.data(), precon.bz(), szMN, cudaMemcpyDeviceToHost), "bz get");
    check_cuda(cudaMemcpy(lam.data(), precon.lambdaPrec(), szMN, cudaMemcpyDeviceToHost), "lambdaPrec get");
    check_cuda(cudaMemcpy(jMin.data(), precon.jmin(), p.mnmax * sizeof(int), cudaMemcpyDeviceToHost), "jMin get");

    // Manufacture a smooth, non-trivial 6-family spectral force buffer.
    int stride = p.mnmax * ns;
    std::vector<T> h_f(6 * stride);
    for (int c = 0; c < 6; ++c)
        for (int mode = 0; mode < p.mnmax; ++mode)
            for (int j = 0; j < ns; ++j)
                h_f[c * stride + mode * ns + j] =
                    T(sin(0.7 * c + 1.3 * mode + 0.11 * j) * (0.4 + 0.05 * c));

    // (check_cuda, not cc: the local spectral vectors named cc/ss/... shadow
    // the cc() helper in this TU.)
    cumes::DeviceBuffer<T> d_f(6 * stride);
    check_cuda(cudaMemcpy(d_f.data(), h_f.data(), 6 * stride * sizeof(T), cudaMemcpyHostToDevice), "f up");
    precon.enqueue_apply(cumes::SpectralView<T, cumes::DecomposedResidualDomain>(
                    d_f.data(), p.ns, p.mnmax),
                p, nullptr, 0);
    std::vector<T> h_g(6 * stride);
    check_cuda(cudaMemcpy(h_g.data(), d_f.data(), 6 * stride * sizeof(T), cudaMemcpyDeviceToHost), "f down");

    // CPU reference: serial Thomas on the SAME coefficients.
    std::vector<double> fRef(6 * stride);
    for (int i = 0; i < 6 * stride; ++i) fRef[i] = (double)h_f[i];
    cpuPreconApplyRef(ar, dr, br, az, dz, bz, jMin, lam, ns, p.mnmax, fRef);

    // Per-component scale tolerance.  The PCR and the serial Thomas are
    // different algorithms; the preconditioner conditioning amplifies float
    // rounding to ~0.5% relative (double stays at ~1e-13), so the float
    // tolerance is 1e-2 of the component's solution scale.  A row-coverage
    // regression (rows above 128 left at their manufactured RHS values) is a
    // ~100% error per affected row — far above this tolerance.
    double rel   = (sizeof(T) == sizeof(float)) ? 1e-2 : 1e-7;
    double absF  = (sizeof(T) == sizeof(float)) ? 1e-8 : 1e-12;
    std::vector<double> scaleC(6, 0.0);
    for (int c = 0; c < 6; ++c)
        for (int mode = 0; mode < p.mnmax; ++mode)
            for (int j = 0; j < ns; ++j)
                scaleC[c] = fmax(scaleC[c], fabs(fRef[c * stride + mode * ns + j]));
    for (int i = 0; i < 6 * stride; ++i) {
        int c = i / stride, rem = i % stride;
        int mode = rem / ns, j = rem % ns;
        double tol = rel * scaleC[c] + absF;
        checkNear((double)h_g[i], fRef[i], tol, "pcr", c, mode, j);
    }

    realSpaceFree(rs);
    cumes::modeTableFree(mt);
    printf(failures() == lf ? "PASS\n" : "FAIL\n");
    return failures() - lf;
}

template <typename T>
static int runDealiasTests() {
    int nf = 0;
    const int nthetaList[] = {30, 32, 34, 40};
    for (int nt : nthetaList) nf += testDealias<T>(nt);
    return nf;
}

template <typename T>
static int runPcrTests() {
    int nf = 0;
    const int nsList[] = {3, 17, 65, 127, 129, 130, 257};
    for (int ns : nsList) nf += testPcr<T>(ns);
    return nf;
}

int main() {
    printf("=== Regression: de-alias theta coverage + PCR row coverage ===\n");
    int nf = 0;
    nf += runDealiasTests<double>();
    nf += runDealiasTests<float>();
    nf += runPcrTests<double>();
    nf += runPcrTests<float>();
    failures() = nf;
    printf(failures() == 0 ? "ALL PASS\n" : "%d FAILURES\n", failures());
    return summary();
}
