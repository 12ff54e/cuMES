// test_constraint_tcon.cu — non-unit tcon0 constraint-force scaling.
//
// The constraint force is linear in the tcon profile, and tcon is linear in
// the parsed tcon0 multiplier (computeTconKernel: tcon[jF] = tcon_base *
// tcon_multiplier * ..., with tcon_multiplier = tcon0 * (constant); the
// de-alias coeff pack scales by tcon[jF]*faccon[m]; the add kernel adds
// dr*gc to the forces). So running the operator at tcon0 = 0, 1, 2 and
// taking the difference of the post-constraint forces must yield a constant
// increment (linearity), with tcon(2) == 2*tcon(1).
//
// This pins the tcon0 propagation fix (cuMES-issues.md: "tcon0 was parsed
// but ignored"): before the fix a non-unit tcon0 left the constraint force
// unchanged (increment == 0); the regression requires exact linear scaling.
#include "cumes/numerics/preconditioner.hpp"
#include "cumes/physics/constraint_operator.hpp"
#include "cumes/physics/force_operator.hpp"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/magnetic_field_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes/state/mode_table.cuh"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/transforms/toroidal_fft_operator.hpp"
#include "cumes_test_cuda_helper.cuh"
#include "vmec_types.h"

#include <cmath>
#include <vector>
using namespace cumes::test;

// One constraint-compute pass at the given tcon0; returns the post-constraint
// even force families (brmn_e, bzmn_e) and the tcon profile (host).
template <typename T>
static void runConstraint(T tcon0,
                          double* out_brmn_e,
                          double* out_bzmn_e,
                          double* out_tcon,
                          int ns,
                          int mnmax,
                          int nZnT) {
    DeviceParams<T> p;
    p.ns = ns;
    p.mnmax = mnmax;
    p.ntheta = 18;
    p.nzeta = 1;
    p.nfp = 1;
    p.nZnT = nZnT;
    p.mpol = mnmax;
    p.ntor = 0;
    p.ncurr = 0;
    p.delt = T(0.9);
    p.ftol = T(1e-14);
    p.max_iter = 10;
    p.tcon0 = tcon0;
    p.lamscale = T(0.0);

    cumes::SpectralStorage<T> storage(p.ns, p.mnmax);
    // Shared manufactured state — kSolovevQuadM2, whose m>=2 R content uses a
    // QUADRATIC radial envelope (not linear s): with a linear profile
    // rCon(s) == s*rCon_LCFS exactly, so rCon - rCon0 is identically zero and
    // the constraint force vanishes no matter what tcon0 is — the test would
    // pass vacuously. The quadratic envelope puts the interior off the
    // LCFS-extrapolated reference, making the constraint-force contribution
    // genuinely nonzero. (Deliberate — see the ManufacturedShape comment.)
    std::vector<T> h_cc, h_ss, h_zsc, h_zcs, h_lsc, h_lcs;
    manufactured_state<T>(ManufacturedShape::kSolovevQuadM2, p.ns, p.mnmax,
                          p.ntor, h_cc, h_ss, h_zsc, h_zcs, h_lsc, h_lcs);
    upload_state(storage, h_cc, h_ss, h_zsc, h_zcs, h_lsc, h_lcs, p.ns,
                 p.mnmax);

    cumes::ValidatedProblem vp = load_validated("inputs/solovev.json");
    cumes::Profiles<T> profiles(p, vp, nullptr);
    cumes::RadialProfileViews<T> rp = profiles.profile_views();
    cumes::DeviceModeTable mt = cumes::modeTableCreate(p);
    cumes::RealSpaceStorage<T> rs = realSpaceCreate(p);
    cumes::GeometryOperator<T> geometry(p, nullptr);
    cumes::ToroidalFftOperator<T> transform(p, rs, mt, nullptr);
    cumes::Preconditioner<T> precon(p, nullptr);
    cumes::ConstraintOperator<T> constraint(p, nullptr);

    transform.inverse_fused(storage.physical_const(), /*do_combine=*/false,
                            constraint.rcon_view(p).data(),
                            constraint.zcon_view(p).data());
    geometry.enqueue(rs, p, rp, 0);
    cumes::MagneticFieldOperator<T>{}.enqueue(
        rs, p, rp, geometry.base_geometry_views(p),
        geometry.magnetic_field_views(p), nullptr, 0, true);
    constraint.reset_reference(p, rp.sqrtS_F, nullptr, 0);
    precon.enqueue_compute(rs, mt.d_xm, mt.d_xn, p, rp,
                           geometry.base_geometry_views(p),
                           geometry.magnetic_field_views(p), nullptr, 0);
    cumes::ForceOperator<T>{}.enqueue(
        rs, p, rp, geometry.base_geometry_views(p),
        geometry.magnetic_field_views(p), nullptr, 0);
    // precon_updated=true recomputes tcon from the current tcon0.
    constraint.enqueue(p, rs, precon.ard(), precon.azd(), rp.sqrtS_F, true,
                       &transform, nullptr, 0);

    size_t nF = (size_t)p.ns * p.nZnT;
    auto* h = new T[nF];
    check_cuda(
        cudaMemcpy(h, rs.d_brmn_e, nF * sizeof(T), cudaMemcpyDeviceToHost),
        "brmn_e");
    for (size_t i = 0; i < nF; ++i) out_brmn_e[i] = (double)h[i];
    check_cuda(
        cudaMemcpy(h, rs.d_bzmn_e, nF * sizeof(T), cudaMemcpyDeviceToHost),
        "bzmn_e");
    for (size_t i = 0; i < nF; ++i) out_bzmn_e[i] = (double)h[i];
    delete[] h;
    auto* ht = new T[p.ns];
    check_cuda(cudaMemcpy(ht, constraint.tcon(), p.ns * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "tcon");
    for (int j = 0; j < p.ns; ++j) out_tcon[j] = (double)ht[j];
    delete[] ht;

    realSpaceFree(rs);
    cumes::modeTableFree(mt);
}

template <typename T>
static void testScaling() {
    const int ns = 11, mnmax = 4, nZnT = 18;
    const size_t nF = (size_t)ns * nZnT;
    std::vector<double> f0e(nF), f1e(nF), f2e(nF);
    std::vector<double> f0z(nF), f1z(nF), f2z(nF);
    std::vector<double> t0(ns), t1(ns), t2(ns);

    runConstraint<T>(T(0.0), f0e.data(), f0z.data(), t0.data(), ns, mnmax,
                     nZnT);
    runConstraint<T>(T(1.0), f1e.data(), f1z.data(), t1.data(), ns, mnmax,
                     nZnT);
    runConstraint<T>(T(2.0), f2e.data(), f2z.data(), t2.data(), ns, mnmax,
                     nZnT);

    // tcon is linear in tcon0: tcon(2) == 2*tcon(1), and nonzero somewhere.
    double tcon_max = 0.0, tcon_lin = 0.0;
    for (int j = 0; j < ns; ++j) {
        tcon_max = std::max(tcon_max, std::abs(t2[j]));
        tcon_lin = std::max(tcon_lin, std::abs(t2[j] - 2.0 * t1[j]));
    }
    check(tcon_max > 0.0,
          "tcon0: constraint multiplier active (nonzero tcon at tcon0=2)");
    check(tcon_lin / std::max(tcon_max, 1.0) < 1e-8,
          "tcon0: tcon scales linearly with tcon0");

    // Force increments must be equal: F(2)-F(1) == F(1)-F(0) (linearity), and
    // the increment is nonzero (the constraint force actually moves).
    double inc_max = 0.0, inc_nonlin = 0.0;
    for (size_t i = 0; i < nF; ++i) {
        double inc1 = f1e[i] - f0e[i];
        double inc2 = f2e[i] - f1e[i];
        inc_max = std::max(inc_max, std::abs(inc1));
        inc_nonlin = std::max(inc_nonlin, std::abs(inc2 - inc1));
        double inc1z = f1z[i] - f0z[i];
        double inc2z = f2z[i] - f1z[i];
        inc_max = std::max(inc_max, std::abs(inc1z));
        inc_nonlin = std::max(inc_nonlin, std::abs(inc2z - inc1z));
    }
    check(inc_max > 0.0, "tcon0: constraint force contribution is nonzero");
    check(inc_nonlin / std::max(inc_max, 1.0) < 1e-8,
          "tcon0: force increments equal (constraint force linear in tcon0)");
}

int main() {
    std::cout << "=== Constraint tcon0 scaling ===\n";
    testScaling<double>();
    return summary();
}
