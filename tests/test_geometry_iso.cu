// test_geometry_iso.cu — self-contained coverage check of the ncurr=1
// geometry chain. Runs computeGeometry on a manufactured W7-X-shaped state
// (full mpol/ntor, ntheta/nzeta) and verifies the bsupu/bsubu write
// coverage: every angular point of an interior surface must be written (a
// zero on an interior surface is not a physical bsupu/bsubu value, so a
// launch-shape bug that skips points surfaces as exact zeros).
//
// The coverage checks ASSERT (nonzero exit on failure) — a passing exit code
// must mean every angular point of the surface was written.
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/magnetic_field_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes/state/mode_table.cuh"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/transforms/toroidal_fft_operator.hpp"
#include "cumes_test_cuda_helper.cuh"
#include "vmec_types.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>
using namespace cumes::test;

// Manufacture a non-degenerate spectral state on the W7-X shape (the shared
// W7X_GENERIC fixture in cumes_test_cuda_helper.cuh). The content is
// deliberately generic (all modes get a mild radial envelope) so no interior
// surface collapses to zero geometry: R has a strong m=0/n=0 DC plus a few m>0
// modes, Z and lambda get m=1..3 content with the same s envelopes. This
// replaces the previous external dump/cuMES/init_*.bin dependency
// (gitignored, only present after a CUMES_DUMP=1 run) — the test is now
// self-contained and registerable.
template <typename T>
static void fill_state(cumes::SpectralStorage<T>& storage,
                       const DeviceParams<T>& p) {
    std::vector<T> hcc, hss, hzsc, hzcs, hlsc, hlcs;
    manufactured_state<T>(ManufacturedShape::W7X_GENERIC, p.ns, p.mnmax, p.ntor,
                          hcc, hss, hzsc, hzcs, hlsc, hlcs);
    upload_state(storage, hcc, hss, hzsc, hzcs, hlsc, hlcs, p.ns, p.mnmax);
}

int main() {
    cumes::ValidatedProblem vp = load_validated("inputs/w7x.json");
    const cumes::ProblemSpec& spec = vp.spec();
    DeviceParams<double> p{};
    // Full W7-X shape (the largest angular grid the solver runs), exercising
    // the same kernel launch shapes the real runs use.
    p.ns = static_cast<int>(spec.stages.back().radial_surfaces);
    p.mpol = spec.mpol;
    p.ntor = spec.ntor;
    p.ntheta = spec.angular.ntheta;
    p.nzeta = spec.angular.nzeta;
    p.nfp = spec.nfp;
    p.nZnT = p.ntheta * p.nzeta;
    p.mnmax = p.mpol * (p.ntor + 1);
    p.ncurr =
        (spec.current_model == cumes::CurrentModel::PRESCRIBED_CURRENT) ? 1 : 0;
    p.delt = spec.delt;
    p.ftol = spec.stages.back().tolerance;
    p.max_iter = static_cast<int>(spec.stages.back().max_iterations);
    p.lamscale = 0.0;

    cumes::SpectralStorage<double> storage(p.ns, p.mnmax);
    size_t nb = (size_t)p.ns * p.mnmax * sizeof(double);

    fill_state(storage, p);
    cumes::Profiles<double> profiles(p, vp, std::nullopt);
    cumes::RadialProfileViews<double> rp = profiles.profile_views();
    cumes::DeviceModeTable mt = cumes::mode_table_create(p);
    cumes::RealSpaceStorage<double> rs = real_space_create(p);
    cumes::ToroidalFftOperator<double> op(p, rs, mt);
    cumes::GeometryOperator<double> geometry(p, std::nullopt);

    // Zero the bsupu/bsubu half-grid buffers BEFORE the geometry pass: the
    // coverage gate below counts exact 0.0 entries as unwritten points, and
    // those buffers are never zero-initialized by the operators (a fresh
    // allocation reads 0 only while the driver's pages stay zeroed; allocator
    // reuse returns stale nonzero bytes), which would make the gate's
    // fire/no-fire depend on allocator state. Deterministic zeroing means a
    // launch-shape bug that skips points always surfaces as exact zeros.
    const size_t nHBytes = (size_t)(p.ns - 1) * p.nZnT * sizeof(double);
    cc(cudaMemset(geometry.magnetic_field_views(p).bsupu.data(), 0, nHBytes),
       "zero bsupu");
    cc(cudaMemset(geometry.magnetic_field_views(p).bsubu.data(), 0, nHBytes),
       "zero bsubu");

    // extrapolate m=1 to the axis (as the solver does each iteration)
    {
        std::vector<double> hcc(p.ns * p.mnmax);
        cc(cudaMemcpy(hcc.data(),
                      storage.family_ptr(cumes::SpectralComponent::Rcc), nb,
                      cudaMemcpyDeviceToHost),
           "get rmncc");
        for (int n = 0; n < p.ntor + 1; ++n) {
            int mn = 1 * (p.ntor + 1) + n;
            hcc[0 + mn * p.ns] = hcc[1 + mn * p.ns];
        }
        cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Rcc),
                      hcc.data(), nb, cudaMemcpyHostToDevice),
           "put rmncc");
    }

    op.inverse(storage.physical_const(), /*do_combine=*/true);
    geometry.enqueue(rs, p, rp, 0);
    cumes::MagneticFieldOperator<double>{}.enqueue(
        rs, p, rp, geometry.base_geometry_views(p),
        geometry.magnetic_field_views(p), nullptr, 0, true);

    // check bsupu coverage on a mid-volume surface. All indices are computed
    // from the actual grid (the old hardcoded ks/1080 were W7-X-specific).
    int nZnT = p.nZnT;
    int jMid = (p.ns - 1) / 2;  // a surface in the middle of the volume
    int nks = 8;
    int ks[8];
    for (int i = 0; i < nks; ++i)
        ks[i] = (i * nZnT) / nks;  // spread over the plane
    double hb[8];
    for (int i = 0; i < nks; ++i)
        cc(cudaMemcpy(&hb[i],
                      geometry.magnetic_field_views(p).bsupu.data() +
                          jMid * nZnT + ks[i],
                      sizeof(double), cudaMemcpyDeviceToHost),
           "get bsupu probe");
    int nz = 0;
    std::vector<double> h_all((p.ns - 1) * nZnT);
    cc(cudaMemcpy(h_all.data(), geometry.magnetic_field_views(p).bsupu.data(),
                  (p.ns - 1) * nZnT * sizeof(double), cudaMemcpyDeviceToHost),
       "get bsupu");
    for (int k = 0; k < nZnT; ++k)
        if (h_all[jMid * nZnT + k] == 0.0) ++nz;
    std::cout << format("bsupu[jMid={}] zeros: {}/{}\n", jMid, nz, nZnT);
    for (int i = 0; i < nks; ++i)
        std::cout << format("  k={}: {:.6f}\n", ks[i], hb[i]);
    // also check the bsubu coverage
    cc(cudaMemcpy(h_all.data(), geometry.magnetic_field_views(p).bsubu.data(),
                  (p.ns - 1) * nZnT * sizeof(double), cudaMemcpyDeviceToHost),
       "get bsubu");
    int nz2 = 0;
    for (int k = 0; k < nZnT; ++k)
        if (h_all[jMid * nZnT + k] == 0.0) ++nz2;
    std::cout << format("bsubu[jMid={}] zeros: {}/{}\n", jMid, nz2, nZnT);

    real_space_free(rs);
    cumes::mode_table_free(mt);

    // Assertions: a full-coverage kernel must leave no unwritten point on an
    // interior surface (zero is not a physical bsupu/bsubu value there).
    int bad = 0;
    if (nz != 0) {
        std::cerr << format("FAIL: bsupu coverage incomplete ({} zeros)\n", nz);
        bad = 1;
    }
    if (nz2 != 0) {
        std::cerr << format("FAIL: bsubu coverage incomplete ({} zeros)\n",
                            nz2);
        bad = 1;
    }
    if (bad == 0) std::cout << "test_geometry_iso: coverage OK\n";
    return bad;
}
