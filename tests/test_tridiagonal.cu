// test_tridiagonal.cu — backend-neutral batched tridiagonal solve gates
// (blueprint §8.9).
//
// Phase 8 introduces the TridiagonalBackend interface with two concrete
// backends — PcrBackend (the production grid-stride PCR, extracted from the
// legacy tridiagSolveKernel) and ThomasBackend (the serial scalar reference).
// This test drives the PUBLIC interface directly (StridedBatchTridiagonalView +
// enqueue_solve) on manufactured systems, complementing
// test_regression_kernels.cu which exercises the production preconApply path on
// real geometry.
//
// Coverage:
//   * CPU serial Thomas reference vs GPU Thomas vs GPU PCR across awkward row
//     counts (3, 17, 65, 99, 130, 257, 512) and mixed jMin (m-parity),
//     rhs_count = 2 (the production shared-elimination layout).
//   * the scale-aware pivot/breakdown contract: a healthy system reports status
//     == 0 and matches the CPU reference; a system with a zero diagonal reports
//     status > 0 (a breakdown is detected, not silently clamped to +1e-30).
//
// Conventions match the other tests: everything is templated on T and both
// double and float are instantiated; the CPU reference computes in double from
// the (possibly float) T inputs; the float leg compares at a relaxed tolerance.
#include "cumes/numerics/tridiagonal_backend.hpp"
#include "cumes_test_cuda_helper.cuh"

#include <cmath>
#include <cstdlib>
#include <vector>
using namespace cumes::test;

// Scale-based tolerance: the GPU runs in float and the CPU reference in double,
// and the PCR elimination order differs from Thomas, so a per-point relative
// test spuriously fails at genuine zero crossings. rel * solution-scale +
// floor.
static void checkNear(double gpu,
                      double ref,
                      double tol,
                      const char* s,
                      int a,
                      int b,
                      int c) {
    if (!(fabs(gpu - ref) <= tol)) {
        std::cerr << format(
            "FAIL [{}] a={} b={} c={} gpu={:.15e} ref={:.15e}\n", s, a, b, c,
            gpu, ref);
        ++failures();
    }
}

// CPU serial Thomas solve (matches precon_impl.cuh thomasSolveKernel and
// test_regression_kernels.cu thomasSolve): rows [jMin, jMax) with x[jMin-1]=0
// and x[jMax]=0.
static void cpuThomas(const double* lower,
                      const double* diagonal,
                      const double* upper,
                      int jMin,
                      int jMax,
                      const double* rhs,
                      double* x) {
    int n = jMax - jMin;
    if (n <= 0) return;
    std::vector<double> cp(n), dp(n);
    double denom = diagonal[jMin];
    cp[0] = upper[jMin] / denom;
    dp[0] = rhs[jMin] / denom;
    for (int i = 1; i < n; ++i) {
        int j = jMin + i;
        denom = diagonal[j] - lower[j] * cp[i - 1];
        cp[i] = upper[j] / denom;
        dp[i] = (rhs[j] - lower[j] * dp[i - 1]) / denom;
    }
    x[jMax - 1] = dp[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        int j = jMin + i;
        x[j] = dp[i] - cp[i] * x[j + 1];
    }
}

// ---------------------------------------------------------------------------
// Healthy-solve scenario (one ns value): CPU Thomas vs GPU Thomas vs GPU PCR.
// ---------------------------------------------------------------------------
template <typename T>
static int testSolve(int ns) {
    int lf = failures();
    std::cout << format("  tridiagonal backend solve: ns={} rhs_count=2 ... ",
                        ns);
    const int modes = 8;
    const int rhs_count = 2;
    const int jMax = ns - 1;

    // Manufacture diagonally-dominant systems with mixed jMin (m-parity).
    std::vector<T> lower(modes * ns), diag(modes * ns), upper(modes * ns);
    std::vector<T> rhs(rhs_count * modes * ns);
    std::vector<int> jMin(modes);
    std::vector<T> scale(modes, T(0));
    for (int mode = 0; mode < modes; ++mode) {
        jMin[mode] = (mode % 2 == 0) ? 0 : 1;  // m=0 -> 0, m>0 -> 1
        for (int j = 0; j < ns; ++j) {
            double d = 4.0 + 0.5 * sin(0.3 * mode + 0.7 * j);
            double l = (j > 0) ? -1.0 - 0.1 * sin(0.5 * mode + j) : 0.0;
            double u = (j < ns - 1) ? -1.0 + 0.2 * cos(0.4 * mode + j) : 0.0;
            lower[mode * ns + j] = T(l);
            diag[mode * ns + j] = T(d);
            upper[mode * ns + j] = T(u);
            scale[mode] =
                fmax(scale[mode], T(fmax(fabs(d), fmax(fabs(l), fabs(u)))));
            for (int c = 0; c < rhs_count; ++c)
                rhs[c * modes * ns + mode * ns + j] =
                    T(sin(0.7 * c + 1.3 * mode + 0.11 * j) * (0.4 + 0.05 * c));
        }
    }

    // Upload.
    T *d_lower, *d_diag, *d_upper, *d_rhs, *d_scale;
    int *d_jMin, *d_status;
    check_cuda(cudaMalloc(&d_lower, modes * ns * sizeof(T)), "lower");
    check_cuda(cudaMalloc(&d_diag, modes * ns * sizeof(T)), "diag");
    check_cuda(cudaMalloc(&d_upper, modes * ns * sizeof(T)), "upper");
    check_cuda(cudaMalloc(&d_rhs, rhs_count * modes * ns * sizeof(T)), "rhs");
    check_cuda(cudaMalloc(&d_scale, modes * sizeof(T)), "scale");
    check_cuda(cudaMalloc(&d_jMin, modes * sizeof(int)), "jMin");
    check_cuda(cudaMalloc(&d_status, sizeof(int)), "status");
    check_cuda(cudaMemcpy(d_lower, lower.data(), modes * ns * sizeof(T),
                          cudaMemcpyHostToDevice),
               "lower up");
    check_cuda(cudaMemcpy(d_diag, diag.data(), modes * ns * sizeof(T),
                          cudaMemcpyHostToDevice),
               "diag up");
    check_cuda(cudaMemcpy(d_upper, upper.data(), modes * ns * sizeof(T),
                          cudaMemcpyHostToDevice),
               "upper up");
    check_cuda(cudaMemcpy(d_rhs, rhs.data(), rhs_count * modes * ns * sizeof(T),
                          cudaMemcpyHostToDevice),
               "rhs up");
    check_cuda(cudaMemcpy(d_scale, scale.data(), modes * sizeof(T),
                          cudaMemcpyHostToDevice),
               "scale up");
    check_cuda(cudaMemcpy(d_jMin, jMin.data(), modes * sizeof(int),
                          cudaMemcpyHostToDevice),
               "jMin up");

    cumes::StridedBatchTridiagonalView<T> v;
    v.lower = d_lower;
    v.diagonal = d_diag;
    v.upper = d_upper;
    v.rhs = d_rhs;
    v.first_surface = d_jMin;
    v.scale = d_scale;
    v.rhs_count = rhs_count;
    v.rhs_stride = modes * ns;  // [rhs_count][mode][surface]
    v.modes = modes;
    v.surfaces = ns;
    v.last_surface = jMax;

    // GPU Thomas backend.
    check_cuda(cudaMemset(d_status, 0, sizeof(int)), "status zero");
    {
        cumes::ThomasBackend<T> th;
        th.enqueue_solve(v, d_status, 0);
    }
    check_cuda(cudaDeviceSynchronize(), "thomas sync");
    int th_status = 0;
    check_cuda(
        cudaMemcpy(&th_status, d_status, sizeof(int), cudaMemcpyDeviceToHost),
        "status get th");
    std::vector<T> th_out(rhs_count * modes * ns);
    check_cuda(
        cudaMemcpy(th_out.data(), d_rhs, rhs_count * modes * ns * sizeof(T),
                   cudaMemcpyDeviceToHost),
        "th out");

    // GPU PCR backend (fresh rhs copy).
    check_cuda(cudaMemcpy(d_rhs, rhs.data(), rhs_count * modes * ns * sizeof(T),
                          cudaMemcpyHostToDevice),
               "rhs reset");
    check_cuda(cudaMemset(d_status, 0, sizeof(int)), "status zero 2");
    {
        cumes::PcrBackend<T> pc;
        pc.enqueue_solve(v, d_status, 0);
    }
    check_cuda(cudaDeviceSynchronize(), "pcr sync");
    int pc_status = 0;
    check_cuda(
        cudaMemcpy(&pc_status, d_status, sizeof(int), cudaMemcpyDeviceToHost),
        "status get pcr");
    std::vector<T> pc_out(rhs_count * modes * ns);
    check_cuda(
        cudaMemcpy(pc_out.data(), d_rhs, rhs_count * modes * ns * sizeof(T),
                   cudaMemcpyDeviceToHost),
        "pc out");

    // Healthy systems must report no breakdown.
    if (th_status != 0) {
        std::cerr << format(
            "FAIL [status] Thomas reported {} breakdowns on a "
            "healthy system\n",
            th_status);
        ++failures();
    }
    if (pc_status != 0) {
        std::cerr << format(
            "FAIL [status] PCR reported {} breakdowns on a "
            "healthy system\n",
            pc_status);
        ++failures();
    }

    // CPU reference: serial Thomas in double on the (possibly float) inputs.
    double rel = (sizeof(T) == sizeof(float)) ? 1e-3 : 1e-8;
    double absF = (sizeof(T) == sizeof(float)) ? 1e-8 : 1e-12;
    for (int mode = 0; mode < modes; ++mode) {
        std::vector<double> lo(ns), dg(ns), up(ns);
        for (int j = 0; j < ns; ++j) {
            lo[j] = (double)lower[mode * ns + j];
            dg[j] = (double)diag[mode * ns + j];
            up[j] = (double)upper[mode * ns + j];
        }
        for (int c = 0; c < rhs_count; ++c) {
            std::vector<double> rh(ns), x(ns, 0.0);
            for (int j = 0; j < ns; ++j)
                rh[j] = (double)rhs[c * modes * ns + mode * ns + j];
            cpuThomas(lo.data(), dg.data(), up.data(), jMin[mode], jMax,
                      rh.data(), x.data());
            double scale_c = 0.0;
            for (int j = jMin[mode]; j < jMax; ++j)
                scale_c = fmax(scale_c, fabs(x[j]));
            double tol = rel * scale_c + absF;
            for (int j = jMin[mode]; j < jMax; ++j) {
                checkNear((double)th_out[c * modes * ns + mode * ns + j], x[j],
                          tol, "thomas", c, mode, j);
                checkNear((double)pc_out[c * modes * ns + mode * ns + j], x[j],
                          tol, "pcr", c, mode, j);
            }
        }
    }

    cudaFree(d_lower);
    cudaFree(d_diag);
    cudaFree(d_upper);
    cudaFree(d_rhs);
    cudaFree(d_scale);
    cudaFree(d_jMin);
    cudaFree(d_status);
    std::cout << (failures() == lf ? "PASS\n" : "FAIL\n");
    return failures() - lf;
}

// ---------------------------------------------------------------------------
// Breakdown scenario: a system with a zero diagonal must report status > 0.
// ---------------------------------------------------------------------------
template <typename T>
static int testBreakdown() {
    int lf = failures();
    std::cout << "  tridiagonal pivot breakdown (zero diagonal) ... ";
    const int modes = 1, ns = 16, rhs_count = 1;
    const int jMax = ns - 1;

    std::vector<T> lower(modes * ns, T(0)), diag(modes * ns, T(1));
    std::vector<T> upper(modes * ns, T(0)), rhs(rhs_count * modes * ns, T(1));
    std::vector<int> jMin(modes, 0);
    std::vector<T> scale(modes, T(1));
    for (int j = 1; j < ns; ++j) {
        lower[j] = T(-1);
        upper[j - 1] = T(-1);
    }
    diag[5] = T(0);  // a genuinely singular pivot

    T *d_lower, *d_diag, *d_upper, *d_rhs, *d_scale;
    int *d_jMin, *d_status;
    check_cuda(cudaMalloc(&d_lower, modes * ns * sizeof(T)), "lower");
    check_cuda(cudaMalloc(&d_diag, modes * ns * sizeof(T)), "diag");
    check_cuda(cudaMalloc(&d_upper, modes * ns * sizeof(T)), "upper");
    check_cuda(cudaMalloc(&d_rhs, rhs_count * modes * ns * sizeof(T)), "rhs");
    check_cuda(cudaMalloc(&d_scale, modes * sizeof(T)), "scale");
    check_cuda(cudaMalloc(&d_jMin, modes * sizeof(int)), "jMin");
    check_cuda(cudaMalloc(&d_status, sizeof(int)), "status");
    check_cuda(cudaMemcpy(d_lower, lower.data(), modes * ns * sizeof(T),
                          cudaMemcpyHostToDevice),
               "lower up");
    check_cuda(cudaMemcpy(d_diag, diag.data(), modes * ns * sizeof(T),
                          cudaMemcpyHostToDevice),
               "diag up");
    check_cuda(cudaMemcpy(d_upper, upper.data(), modes * ns * sizeof(T),
                          cudaMemcpyHostToDevice),
               "upper up");
    check_cuda(cudaMemcpy(d_rhs, rhs.data(), rhs_count * modes * ns * sizeof(T),
                          cudaMemcpyHostToDevice),
               "rhs up");
    check_cuda(cudaMemcpy(d_scale, scale.data(), modes * sizeof(T),
                          cudaMemcpyHostToDevice),
               "scale up");
    check_cuda(cudaMemcpy(d_jMin, jMin.data(), modes * sizeof(int),
                          cudaMemcpyHostToDevice),
               "jMin up");

    cumes::StridedBatchTridiagonalView<T> v;
    v.lower = d_lower;
    v.diagonal = d_diag;
    v.upper = d_upper;
    v.rhs = d_rhs;
    v.first_surface = d_jMin;
    v.scale = d_scale;
    v.rhs_count = rhs_count;
    v.rhs_stride = modes * ns;
    v.modes = modes;
    v.surfaces = ns;
    v.last_surface = jMax;

    // Both backends must report at least one breakdown.
    for (int pass = 0; pass < 2; ++pass) {
        check_cuda(
            cudaMemcpy(d_rhs, rhs.data(), rhs_count * modes * ns * sizeof(T),
                       cudaMemcpyHostToDevice),
            "rhs reset");
        check_cuda(cudaMemset(d_status, 0, sizeof(int)), "status zero");
        if (pass == 0) {
            cumes::ThomasBackend<T> th;
            th.enqueue_solve(v, d_status, 0);
        } else {
            cumes::PcrBackend<T> pc;
            pc.enqueue_solve(v, d_status, 0);
        }
        check_cuda(cudaDeviceSynchronize(), "sync");
        int st = 0;
        check_cuda(
            cudaMemcpy(&st, d_status, sizeof(int), cudaMemcpyDeviceToHost),
            "status get");
        if (st == 0) {
            std::cerr << format(
                "FAIL [status] {} did not report the zero-diagonal "
                "breakdown\n",
                pass == 0 ? "Thomas" : "PCR");
            ++failures();
        }
    }

    cudaFree(d_lower);
    cudaFree(d_diag);
    cudaFree(d_upper);
    cudaFree(d_rhs);
    cudaFree(d_scale);
    cudaFree(d_jMin);
    cudaFree(d_status);
    std::cout << (failures() == lf ? "PASS\n" : "FAIL\n");
    return failures() - lf;
}

int main() {
    std::cout
        << "=== Tridiagonal backend: CPU agreement + pivot breakdown ===\n";
    int nf = 0;
    const int nsList[] = {3, 17, 65, 99, 130, 257, 512};
    for (int ns : nsList) {
        nf += testSolve<double>(ns);
        nf += testSolve<float>(ns);
    }
    nf += testBreakdown<double>();
    nf += testBreakdown<float>();
    failures() = nf;
    std::cout << (failures() == 0 ? "ALL PASS\n"
                                  : format("{} FAILURES\n", failures()));
    return summary();
}
