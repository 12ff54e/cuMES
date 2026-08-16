// test_geometry_ncurr.cu — ncurr=0/ncurr=1 geometry-path regression.
//
// The containment series fixed a half-grid OOB: updateIotaChipFKernel read
// one element beyond the (ns-1)-sized half-grid allocation at the LCFS row
// for ns=33 (the W7-X first stage). This test drives the same kernels at
// ns=33 in BOTH current models:
//
//   ncurr=0 (fixed iota):  geometryKernel -> updateIotaChipFKernel
//   ncurr=1 (fixed curr):  geometryKernel -> ncurr1FinalizeKernel ->
//                          updateIotaChipFKernel
//
// and asserts the half-grid outputs are finite and the Jacobian stats are
// sensible. It is a kernel-driving test (launches CUDA kernels), so it also
// gets a Compute Sanitizer memcheck variant via CUMES_ENABLE_SANITIZER_TESTS
// — the registered sanitizer pass that catches OOB reads like the one this
// regression pins.
#include <cstdio>
#include <cmath>
#include <vector>

#include "vmec_types.h"
#include "cumes/transforms/toroidal_fft_operator.hpp"
#include "cumes/state/mode_table.cuh"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/magnetic_field_operator.hpp"
#include "cumes/physics/profiles.hpp"
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


// Build a Solovev-like state with a few modes and run one geometry pass.
template <typename T>
static void runGeometry(int ns, int ncurr, const char* label) {
    DeviceParams<T> p;
    p.ns = ns; p.mnmax = 4; p.ntheta = 18; p.nzeta = 1;
    p.nfp = 1; p.nZnT = 18; p.mpol = 4; p.ntor = 0;
    p.ncurr = ncurr; p.delt = T(0.9); p.ftol = T(1e-14); p.max_iter = 10;
    p.tcon0 = T(1.0); p.lamscale = T(0.0);

    cumes::SpectralStorage<T> storage(p.ns, p.mnmax);
    size_t nb = (size_t)p.ns * p.mnmax * sizeof(T);
    auto* h_cc = new T[p.ns * p.mnmax]();
    auto* h_ss = new T[p.ns * p.mnmax]();
    auto* h_zsc = new T[p.ns * p.mnmax]();
    auto* h_zcs = new T[p.ns * p.mnmax]();
    auto* h_lsc = new T[p.ns * p.mnmax]();
    auto* h_lcs = new T[p.ns * p.mnmax]();
    for (int j = 0; j < p.ns; ++j) {
        T s = T(j) / T(p.ns - 1);
        for (int m = 0; m < p.mnmax; ++m) {
            if (m == 0) h_cc[j + m * p.ns] = T(4.0);        // R_00
            else if (m == 1) h_cc[j + m * p.ns] = T(0.3) * s; // R_10
            else if (m == 2) h_cc[j + m * p.ns] = T(0.2) * s; // R_20
            h_ss[j + m * p.ns] = h_cc[j + m * p.ns];
            if (m == 1) { h_zsc[j + m * p.ns] = T(-0.5) * s; h_zcs[j + m * p.ns] = T(-0.5) * s; }
        }
    }
    checkCuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Rcc), h_cc, nb, cudaMemcpyHostToDevice), "cc");
    checkCuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Rss), h_ss, nb, cudaMemcpyHostToDevice), "ss");
    checkCuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Zsc), h_zsc, nb, cudaMemcpyHostToDevice), "zsc");
    checkCuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Zcs), h_zcs, nb, cudaMemcpyHostToDevice), "zcs");
    checkCuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Lsc), h_lsc, nb, cudaMemcpyHostToDevice), "lsc");
    checkCuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Lcs), h_lcs, nb, cudaMemcpyHostToDevice), "lcs");
    delete[] h_cc; delete[] h_ss; delete[] h_zsc; delete[] h_zcs;
    delete[] h_lsc; delete[] h_lcs;

    // ncurr=1 needs prescribed-current profiles (curtor/ac); ncurr=0 uses
    // the fixed-iota profiles from inputs/solovev.json.
    cumes::ValidatedProblem vp = [&]() {
        if (ncurr == 1) {
            cumes::ProblemSpec spec;
            spec.mpol = p.mpol; spec.ntor = p.ntor; spec.nfp = 1;
            spec.angular.ntheta = p.ntheta; spec.angular.nzeta = p.nzeta;
            spec.current_model = cumes::CurrentModel::kPrescribedCurrent;
            spec.physical.curtor = 1.0;
            spec.mass.coefficients = {0.1};
            spec.current.coefficients = {1.0};
            spec.rbc = {{1, 0, 1.0}};
            spec.zbs = {{1, 0, 0.5}};
            spec.stages = {{static_cast<std::size_t>(ns), 10, 1e-14}};
            return validateSpec(std::move(spec));
        }
        return loadValidated("inputs/solovev.json");
    }();

    cumes::Profiles<T> profiles(p, vp, nullptr); cumes::RadialProfileViews<T> rp = profiles.profile_views();
    cumes::DeviceModeTable mt = cumes::modeTableCreate(p);
    cumes::RealSpaceStorage<T> rs = realSpaceCreate(p);
    cumes::ToroidalFftOperator<T> op(p, rs, mt);
    cumes::GeometryOperator<T> geometry(p, nullptr);

    op.inverse(storage.physical_const(), /*do_combine=*/true);
    geometry.enqueue(rs, p, rp, 0); cumes::MagneticFieldOperator<T>{}.enqueue(rs, p, rp, geometry.base_geometry_views(p), geometry.magnetic_field_views(p), 0, true);

    // Assert the half-grid outputs that updateIotaChipFKernel /
    // ncurr1FinalizeKernel produce are finite (an OOB read would surface as
    // garbage or a memcheck error, not a finite check failure).
    size_t nH = (size_t)(p.ns - 1) * p.nZnT;
    auto* h_chip = new T[p.ns - 1];
    auto* h_iota = new T[p.ns - 1];
    checkCuda(cudaMemcpy(h_chip, rp.chip_H, (p.ns - 1) * sizeof(T), cudaMemcpyDeviceToHost), "chipH");
    checkCuda(cudaMemcpy(h_iota, rp.iota_H, (p.ns - 1) * sizeof(T), cudaMemcpyDeviceToHost), "iotaH");
    bool all_finite = true;
    for (int j = 0; j < p.ns - 1; ++j) {
        if (!std::isfinite((double)h_chip[j]) || !std::isfinite((double)h_iota[j])) all_finite = false;
    }
    CHECK(all_finite, label);
    // Jacobian stats must be finite and the max nonzero. computeJacobianStats
    // is device-only (Phase 6A one-fence path); copy the 4 values out here.
    T* d_stats; T h_stats[4];
    checkCuda(cudaMalloc(&d_stats, 4 * sizeof(T)), "stats");
    geometry.jacobian_stats(p, d_stats, 0);
    checkCuda(cudaMemcpy(h_stats, d_stats, 4 * sizeof(T), cudaMemcpyDeviceToHost), "stats cpy");
    CHECK(std::isfinite((double)h_stats[0]) && std::isfinite((double)h_stats[1]) &&
          h_stats[2] == T(0.0) && h_stats[1] > T(0.0),
          "jacobian stats finite, nonzero max");
    checkCuda(cudaFree(d_stats), "free stats");

    delete[] h_chip; delete[] h_iota;
    realSpaceFree(rs);
    cumes::modeTableFree(mt);
}

int main() {
    printf("=== Geometry ncurr path regression (ns=33, the OOB size) ===\n");
    // Both current models at the exact failing resolution.
    runGeometry<double>(33, 0, "ncurr=0 fixed-iota geometry finite");
    runGeometry<double>(33, 1, "ncurr=1 prescribed-current geometry finite");
    // Also the smallest valid grid and a mid size, both current models.
    runGeometry<double>(5,  0, "ncurr=0 ns=5 finite");
    runGeometry<double>(5,  1, "ncurr=1 ns=5 finite");
    runGeometry<double>(99, 0, "ncurr=0 ns=99 finite");
    runGeometry<double>(99, 1, "ncurr=1 ns=99 finite");

    if (failures == 0) {
        printf("test_geometry_ncurr: ALL PASS\n");
        return 0;
    }
    printf("test_geometry_ncurr: %d FAILURES\n", failures);
    return 1;
}
