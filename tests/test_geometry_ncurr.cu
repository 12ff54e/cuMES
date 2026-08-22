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
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/magnetic_field_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes/runtime/device_buffer.cuh"
#include "cumes/state/mode_table.cuh"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/transforms/toroidal_fft_operator.hpp"
#include "cumes_test_cuda_helper.cuh"
#include "vmec_types.h"

#include <cmath>
#include <vector>
using namespace cumes::test;

// Build a Solovev-like state with a few modes and run one geometry pass.
template <typename T>
static void runGeometry(int ns, int ncurr, const char* label) {
    DeviceParams<T> p;
    p.ns = ns;
    p.mnmax = 4;
    p.ntheta = 18;
    p.nzeta = 1;
    p.nfp = 1;
    p.nZnT = 18;
    p.mpol = 4;
    p.ntor = 0;
    p.ncurr = ncurr;
    p.delt = T(0.9);
    p.ftol = T(1e-14);
    p.max_iter = 10;
    p.tcon0 = T(1.0);
    p.lamscale = T(0.0);

    cumes::SpectralStorage<T> storage(p.ns, p.mnmax);
    // Shared manufactured state (kSolovevLinear in cumes_test_cuda_helper.cuh):
    // R_00=4.0, R_10=0.3s, R_20=0.2s, Rss=Rcc, Z_10 (sc+cs)=-0.5s, lambda=0.
    std::vector<T> h_cc, h_ss, h_zsc, h_zcs, h_lsc, h_lcs;
    manufactured_state<T>(ManufacturedShape::kSolovevLinear, p.ns, p.mnmax,
                          p.ntor, h_cc, h_ss, h_zsc, h_zcs, h_lsc, h_lcs);
    upload_state(storage, h_cc, h_ss, h_zsc, h_zcs, h_lsc, h_lcs, p.ns,
                 p.mnmax);

    // ncurr=1 needs prescribed-current profiles (curtor/ac); ncurr=0 uses
    // the fixed-iota profiles from inputs/solovev.json.
    cumes::ValidatedProblem vp = [&]() {
        if (ncurr == 1) {
            cumes::ProblemSpec spec;
            spec.mpol = p.mpol;
            spec.ntor = p.ntor;
            spec.nfp = 1;
            spec.angular.ntheta = p.ntheta;
            spec.angular.nzeta = p.nzeta;
            spec.current_model = cumes::CurrentModel::kPrescribedCurrent;
            spec.physical.curtor = 1.0;
            spec.physical.phiedge = 1.0;
            // A well-scaled toroidal-flux profile: the edge normalization
            // T(1) is validated now (completion plan step 1.1) and must be
            // nonzero and finite — the old fixture left it empty (T(1)=0) and
            // relied on the silent no-divide fallback.
            spec.toroidal_flux.coefficients = {1.0};
            spec.mass.coefficients = {0.1};
            spec.current.coefficients = {1.0};
            spec.rbc = {{1, 0, 1.0}};
            spec.zbs = {{1, 0, 0.5}};
            spec.stages = {{static_cast<std::size_t>(ns), 10, 1e-14}};
            return validate_spec(std::move(spec));
        }
        return load_validated("inputs/solovev.json");
    }();

    cumes::Profiles<T> profiles(p, vp, nullptr);
    cumes::RadialProfileViews<T> rp = profiles.profile_views();
    cumes::DeviceModeTable mt = cumes::modeTableCreate(p);
    cumes::RealSpaceStorage<T> rs = realSpaceCreate(p);
    cumes::ToroidalFftOperator<T> op(p, rs, mt);
    cumes::GeometryOperator<T> geometry(p, nullptr);

    op.inverse(storage.physical_const(), /*do_combine=*/true);
    geometry.enqueue(rs, p, rp, 0);
    cumes::MagneticFieldOperator<T>{}.enqueue(
        rs, p, rp, geometry.base_geometry_views(p),
        geometry.magnetic_field_views(p), nullptr, 0, true);

    // Assert the half-grid outputs that updateIotaChipFKernel /
    // ncurr1FinalizeKernel produce are finite (an OOB read would surface as
    // garbage or a memcheck error, not a finite check failure).
    size_t nH = (size_t)(p.ns - 1) * p.nZnT;
    auto* h_chip = new T[p.ns - 1];
    auto* h_iota = new T[p.ns - 1];
    check_cuda(cudaMemcpy(h_chip, rp.chip_H, (p.ns - 1) * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "chipH");
    check_cuda(cudaMemcpy(h_iota, rp.iota_H, (p.ns - 1) * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "iotaH");
    bool all_finite = true;
    for (int j = 0; j < p.ns - 1; ++j) {
        if (!std::isfinite((double)h_chip[j]) ||
            !std::isfinite((double)h_iota[j]))
            all_finite = false;
    }
    check(all_finite, label);
    // Jacobian stats must be finite and the max nonzero. computeJacobianStats
    // is device-only (Phase 6A one-fence path) and writes DOUBLE stats in
    // both builds (ADR-0001); the stats now land in the typed ControlRecord
    // (completion plan step 1.3), copied out of a RAII buffer here.
    cumes::DeviceBuffer<cumes::ControlRecord> rec(1);
    // Zero the whole record first: jacobian_stats writes only its four slots,
    // and initcheck flags a D2H copy of any byte no kernel produced.
    check_cuda(cudaMemset(rec.data(), 0, sizeof(cumes::ControlRecord)),
               "rec zero");
    geometry.jacobian_stats(p, rec.data(), 0);
    cumes::ControlRecord h_rec;
    check_cuda(
        cudaMemcpy(&h_rec, rec.data(), sizeof(h_rec), cudaMemcpyDeviceToHost),
        "rec cpy");
    check(std::isfinite(h_rec.jacobian_min_oriented) &&
              std::isfinite(h_rec.jacobian_max_abs) &&
              h_rec.jacobian_nonfinite_count == 0.0 &&
              h_rec.jacobian_max_abs > 0.0,
          "jacobian stats finite, nonzero max");

    delete[] h_chip;
    delete[] h_iota;
    realSpaceFree(rs);
    cumes::modeTableFree(mt);
}

int main() {
    std::cout
        << "=== Geometry ncurr path regression (ns=33, the OOB size) ===\n";
    // Both current models at the exact failing resolution.
    runGeometry<double>(33, 0, "ncurr=0 fixed-iota geometry finite");
    runGeometry<double>(33, 1, "ncurr=1 prescribed-current geometry finite");
    // Also the smallest valid grid and a mid size, both current models.
    runGeometry<double>(5, 0, "ncurr=0 ns=5 finite");
    runGeometry<double>(5, 1, "ncurr=1 ns=5 finite");
    runGeometry<double>(99, 0, "ncurr=0 ns=99 finite");
    runGeometry<double>(99, 1, "ncurr=1 ns=99 finite");

    return summary();
}
