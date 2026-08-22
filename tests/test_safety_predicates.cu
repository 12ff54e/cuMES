// test_safety_predicates.cu — manufactured cases for the device safety
// predicates (completion plan step 1.5; blueprint §6.9/§7).
//
// Asserts, on the shipped kernels (public include/cumes/numerics/
// device_predicates.cuh) and the guarded operators:
//
//   1. jacobianFinalizeKernel decides validity with the IDENTICAL rule as
//      IterationController::jacobian_invalid — manufactured healthy,
//      collapsed, inverted, nonfinite, empty-grid, axis-row and interior
//      relative cases;
//   2. invariantPredicateKernel classifies nonfinite / converged /
//      continue exactly like the host's normalized-triple expressions,
//      including the refresh-pass structural disable;
//   3. the guarded operators (magnetic field, constraint reference + tcon,
//      preconditioner element cache, MHD force) write NOTHING when
//      status->jacobian_valid is clear, and run normally when it is set;
//   4. computeResidualsPreconditionedKernel stores the zero sentinel +
//      not_evaluated on terminal passes and real values + evaluated otherwise;
//   5. an end-to-end collapsed-state EquilibriumOperator::enqueue reports
//      jacobian_valid == 0 and performs no forbidden cache mutation.
//
// Both double and float instantiate every case (the predicate math itself is
// double in both builds — ADR-0001).
#include "cumes/numerics/device_predicates.cuh"
#include "cumes/numerics/preconditioner.hpp"
#include "cumes/physics/constraint_operator.hpp"
#include "cumes/physics/force_operator.hpp"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/magnetic_field_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes/solver/control_record.hpp"
#include "cumes/solver/equilibrium_operator.hpp"
#include "cumes/solver/iteration_controller.hpp"
#include "cumes/state/mode_table.cuh"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/transforms/toroidal_fft_operator.hpp"
#include "cumes_test_cuda_helper.cuh"
#include "vmec_types.h"

#include <cmath>
#include <cstring>
#include <vector>
using namespace cumes::test;

// ---------------------------------------------------------------------------
// 1. Jacobian finalize rule == host controller rule
// ---------------------------------------------------------------------------
static void testJacobianFinalizeRules() {
    const int nZnT = 18;
    // {min_oriented, max_abs, nonfinite, min_index} + expected host decision.
    struct Case {
        double min_o, max_a, nf, idx;
        const char* label;
    };
    const Case cases[] = {
        {0.5, 1.0, 0.0, 5.0, "healthy interior"},
        {0.0, 1.0, 0.0, 5.0, "collapsed (zero sqrt(g))"},
        {-0.2, 1.0, 0.0, 5.0, "inverted (sign flip)"},
        {0.5, 1.0, 1.0, 5.0, "nonfinite entry"},
        {0.5, 0.0, 0.0, 5.0, "zero max (empty grid identity)"},
        {1e-14, 1.0, 0.0, (double)(nZnT + 2), "relative at axis row"},
        {1e-14, 1.0, 0.0, 5.0, "relative at interior (valid)"},
        {0.9, 1.0, 0.0, 0.0, "min at first point (valid)"},
    };

    cumes::DeviceBuffer<cumes::ControlRecord> d_rec(1);
    for (const Case& c : cases) {
        cumes::ControlRecord h;
        h.jacobian_min_oriented = c.min_o;
        h.jacobian_max_abs = c.max_a;
        h.jacobian_nonfinite_count = c.nf;
        h.jacobian_min_index = c.idx;
        check_cuda(
            cudaMemcpy(d_rec.data(), &h, sizeof(h), cudaMemcpyHostToDevice),
            "rec up (jac)");

        // Host decision: the controller's own gate (fresh controller per case).
        cumes::IterationController<double> ctl(
            cumes::IterationController<double>::Options{});
        cumes::JacobianStatus<double> js;
        js.min_oriented = c.min_o;
        js.max_abs = c.max_a;
        js.nonfinite_count = c.nf;
        js.min_index = (int)c.idx;
        const bool host_invalid = ctl.jacobian_invalid(js, nZnT);

        jacobianFinalizeKernel<<<1, 1>>>(d_rec.data(), nZnT);
        cc(cudaDeviceSynchronize(), "jacobianFinalize sync");
        check_cuda(
            cudaMemcpy(&h, d_rec.data(), sizeof(h), cudaMemcpyDeviceToHost),
            "rec down (jac)");

        const bool dev_valid = h.status.jacobian_valid != 0;
        check(dev_valid != host_invalid,
              format("jacobian finalize rule: {} (dev {}, host {})", c.label,
                     dev_valid ? "valid" : "invalid",
                     host_invalid ? "invalid" : "valid"));
    }
}

// ---------------------------------------------------------------------------
// 2. Terminal predicate == host normalized-triple classification
// ---------------------------------------------------------------------------
// use_record_factors selects the factor source (completion-plan follow-up
// §2.3): 0 = the cached factors passed by value (non-refresh passes),
// 1 = the record's device-finalized final_f_norm_* fields (refresh passes).
// The host consumes the same record fields on refresh passes, so the device
// bits and the host verdict must agree bit-for-bit on EVERY pass.
static void testInvariantPredicateRules() {
    struct Case {
        double raw[3];
        double f_rz, f_l, plain, ftol;
        double rec_f_rz, rec_f_l;
        int evaluated;
        int use_record_factors;
        bool want_nf, want_cv;
        const char* label;
    };
    const Case cases[] = {
        {{1e-10, 2e-10, 3e-10},
         1.0,
         1.0,
         100.0,
         1e-4,
         0.0,
         0.0,
         1,
         0,
         false,
         true,
         "converged (cached factors, all below ftol)"},
        {{1e-10, 2e-10, 3e-10},
         99.0,
         99.0,
         100.0,
         1e-4,
         1.0,
         1.0,
         1,
         1,
         false,
         true,
         "refresh pass: converged from record factors"},
        {{1e-10, 2e-10, 3e-10},
         99.0,
         99.0,
         100.0,
         1e-14,
         1.0,
         1.0,
         1,
         1,
         false,
         false,
         "refresh pass: record factors above ftol: continue"},
        {{1e-10, 2e-10, 3e-10},
         99.0,
         99.0,
         100.0,
         1e-4,
         0.0,
         0.0,
         0,
         1,
         false,
         false,
         "refresh pass with unevaluated factors: convergence skipped"},
        {{1.0, 2.0, 3.0},
         1.0,
         1.0,
         100.0,
         1e-4,
         0.0,
         0.0,
         1,
         0,
         false,
         false,
         "large residuals: continue"},
        {{1.0, NAN, 3.0},
         1.0,
         1.0,
         100.0,
         1e-4,
         0.0,
         0.0,
         1,
         0,
         true,
         false,
         "nonfinite invariant: recover"},
        {{1e-10, 2e-10, 3e-10},
         0.0,
         1.0,
         100.0,
         0.0,
         0.0,
         0.0,
         1,
         0,
         false,
         true,
         "zero fNormRZ collapses fsqr/fsqz (converged)"},
    };

    cumes::DeviceBuffer<cumes::ControlRecord> d_rec(1);
    for (const Case& c : cases) {
        cumes::ControlRecord h = {};
        h.invariant_raw[0] = c.raw[0];
        h.invariant_raw[1] = c.raw[1];
        h.invariant_raw[2] = c.raw[2];
        h.final_f_norm_rz = c.rec_f_rz;
        h.final_f_norm_l = c.rec_f_l;
        h.status.force_norms_evaluated = (unsigned)c.evaluated;
        check_cuda(
            cudaMemcpy(d_rec.data(), &h, sizeof(h), cudaMemcpyHostToDevice),
            "rec up (pred)");
        invariantPredicateKernel<<<1, 1>>>(d_rec.data(), c.f_rz, c.f_l, c.plain,
                                           c.ftol, c.use_record_factors);
        cc(cudaDeviceSynchronize(), "predicate sync");
        check_cuda(
            cudaMemcpy(&h, d_rec.data(), sizeof(h), cudaMemcpyDeviceToHost),
            "rec down (pred)");

        // Host classification with the identical expressions and factor source.
        const double f_rz = c.use_record_factors ? c.rec_f_rz : c.f_rz;
        const double f_l = c.use_record_factors ? c.rec_f_l : c.f_l;
        const double fsqr_i = c.raw[0] * c.plain * f_rz * 0.25;
        const double fsqz_i = c.raw[1] * c.plain * f_rz * 0.25;
        const double fsql_i = c.raw[2] * c.plain * f_l;
        const bool host_nf = !(std::isfinite(fsqr_i) && std::isfinite(fsqz_i) &&
                               std::isfinite(fsql_i));
        const bool host_cv =
            !host_nf && (!c.use_record_factors || c.evaluated != 0) &&
            fsqr_i <= c.ftol && fsqz_i <= c.ftol && fsql_i <= c.ftol;

        check(
            (h.status.invariant_nonfinite != 0) == host_nf &&
                (h.status.invariant_converged != 0) == host_cv,
            format("terminal predicate: {} (dev nf={} cv={}, host nf={} cv={})",
                   c.label, (int)h.status.invariant_nonfinite,
                   (int)h.status.invariant_converged, (int)host_nf,
                   (int)host_cv));
    }
}

// ---------------------------------------------------------------------------
// 2b. Device force-norm finalize == the host's finalizeForceNorms expressions
// ---------------------------------------------------------------------------
static void testForceNormFinalizeRules() {
    struct Case {
        double norms[6];
        double delta_s, lamscale;
        const char* label;
    };
    const Case cases[] = {
        {{2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, 0.5, 2.0, "healthy factors"},
        {{2.0, 3.0, -4.0, 5.0, 6.0, 7.0}, 0.5, 2.0, "negative magnetic energy"},
        {{0.0, 3.0, 4.0, 5.0, 6.0, 7.0}, 0.5, 2.0, "zero sRZ -> fallback 1"},
        {{2.0, 0.0, 4.0, 5.0, 6.0, 7.0}, 0.5, 0.0, "zero sL -> fallback 1"},
        {{2.0, 3.0, 4.0, 5.0, 6.0, 0.0}, 0.5, 2.0, "zero rzNorm -> fallback 1"},
        {{2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
         0.0,
         2.0,
         "zero deltaS -> NaN density -> fallback 1"},
    };

    cumes::DeviceBuffer<cumes::ControlRecord> d_rec(1);
    for (const Case& c : cases) {
        cumes::ControlRecord h = {};
        for (int i = 0; i < 6; ++i) h.force_norms[i] = c.norms[i];
        h.status.force_norms_evaluated = 1;
        check_cuda(
            cudaMemcpy(d_rec.data(), &h, sizeof(h), cudaMemcpyHostToDevice),
            "rec up (fnfinalize)");
        forceNormFinalizeKernel<<<1, 1>>>(d_rec.data(), c.delta_s, c.lamscale);
        cc(cudaDeviceSynchronize(), "fnfinalize sync");
        check_cuda(
            cudaMemcpy(&h, d_rec.data(), sizeof(h), cudaMemcpyDeviceToHost),
            "rec down (fnfinalize)");

        // Host reference: the exact finalizeForceNorms expressions. Every op
        // is correctly-rounded IEEE double, so the device result must match
        // BIT-FOR-BIT (the solver relies on this for host/device agreement).
        const double sRZ = c.norms[0], sL = c.norms[1], sMag = c.norms[2];
        const double eTherm = c.norms[3] * c.delta_s;
        const double vol = c.norms[4] * c.delta_s;
        const double eMag = fabs(sMag) * c.delta_s;
        const double energyDensity = std::max(eMag, eTherm) / vol;
        const double denomRZ = sRZ * energyDensity * energyDensity;
        const double fRZ = denomRZ > 0.0 ? (1.0 / denomRZ) : 1.0;
        const double denomL = sL * c.lamscale * c.lamscale;
        const double fL = denomL > 0.0 ? (1.0 / denomL) : 1.0;
        const double f1 = c.norms[5] > 0.0 ? (1.0 / c.norms[5]) : 1.0;

        check(h.final_f_norm_rz == fRZ && h.final_f_norm_l == fL &&
                  h.final_f_norm1 == f1,
              format("force-norm finalize: {}", c.label));
    }

    // Not evaluated (invalid-Jacobian refresh pass): the fields keep the
    // deterministic zero sentinel the pass-start control reset wrote.
    cumes::ControlRecord h = {};
    for (int i = 0; i < 6; ++i) h.force_norms[i] = 1.0;
    check_cuda(cudaMemcpy(d_rec.data(), &h, sizeof(h), cudaMemcpyHostToDevice),
               "rec up (fnfinalize skip)");
    forceNormFinalizeKernel<<<1, 1>>>(d_rec.data(), 0.5, 2.0);
    cc(cudaDeviceSynchronize(), "fnfinalize skip sync");
    check_cuda(cudaMemcpy(&h, d_rec.data(), sizeof(h), cudaMemcpyDeviceToHost),
               "rec down (fnfinalize skip)");
    check(h.final_f_norm_rz == 0.0 && h.final_f_norm_l == 0.0 &&
              h.final_f_norm1 == 0.0,
          "force-norm finalize: not evaluated -> zero sentinel");
}

// ---------------------------------------------------------------------------
// 3. Guarded-operator no-op on invalid Jacobian (and normal run when valid)
// ---------------------------------------------------------------------------

// The shared guarded-consumer battery (magnetic field + iotaF/chipF cache,
// constraint reference, preconditioner element cache, MHD force, constraint
// force). Every consumer reads `status` and must write NOTHING when
// jacobian_valid is clear. Shared by runGuardNoop and runSignFlipStats so the
// two invalid-Jacobian scenarios test the identical operator set.
template <typename T>
static void enqueueGuardedConsumers(const DeviceParams<T>& p,
                                    const cumes::RadialProfileViews<T>& rp,
                                    cumes::RealSpaceStorage<T>& rs,
                                    cumes::ToroidalFftOperator<T>& transform,
                                    cumes::GeometryOperator<T>& geometry,
                                    cumes::Preconditioner<T>& precon,
                                    cumes::ConstraintOperator<T>& constraint,
                                    const cumes::DeviceModeTable& mt,
                                    cumes::ControlStatus* status) {
    cumes::MagneticFieldOperator<T>{}.enqueue(
        rs, p, rp, geometry.base_geometry_views(p),
        geometry.magnetic_field_views(p), status, 0, true);
    constraint.reset_reference(p, rp.sqrtS_F, status, 0);
    precon.enqueue_compute(rs, mt.d_xm, mt.d_xn, p, rp,
                           geometry.base_geometry_views(p),
                           geometry.magnetic_field_views(p), status, 0);
    cumes::ForceOperator<T>{}.enqueue(
        rs, p, rp, geometry.base_geometry_views(p),
        geometry.magnetic_field_views(p), status, 0);
    constraint.enqueue(p, rs, precon.ard(), precon.azd(), rp.sqrtS_F, true,
                       &transform, status, 0);
}

template <typename T>
static void runGuardNoop(T label) {
    (void)label;
    DeviceParams<T> p;
    p.ns = 9;
    p.mnmax = 4;
    p.ntheta = 18;
    p.nzeta = 2;
    p.nfp = 1;
    p.nZnT = 36;
    p.mpol = 4;
    p.ntor = 0;
    p.ncurr = 0;
    p.delt = T(0.9);
    p.ftol = T(1e-14);
    p.max_iter = 10;
    p.tcon0 = T(1.0);
    p.lamscale = T(0.0);

    cumes::ValidatedProblem vp = load_validated("inputs/solovev.json");

    cumes::SpectralStorage<T> storage(p.ns, p.mnmax);
    std::vector<T> h_cc, h_ss, h_zsc, h_zcs, h_lsc, h_lcs;
    manufactured_state<T>(ManufacturedShape::kSolovevLinear, p.ns, p.mnmax,
                          p.ntor, h_cc, h_ss, h_zsc, h_zcs, h_lsc, h_lcs);
    upload_state(storage, h_cc, h_ss, h_zsc, h_zcs, h_lsc, h_lcs, p.ns,
                 p.mnmax);

    cumes::Profiles<T> profiles(p, vp, nullptr);
    cumes::RadialProfileViews<T> rp = profiles.profile_views();
    cumes::DeviceModeTable mt = cumes::modeTableCreate(p);
    cumes::RealSpaceStorage<T> rs = realSpaceCreate(p);
    cumes::ToroidalFftOperator<T> transform(p, rs, mt);
    cumes::GeometryOperator<T> geometry(p, nullptr);
    cumes::Preconditioner<T> precon(p, nullptr);
    cumes::ConstraintOperator<T> constraint(p, nullptr);

    cumes::DeviceBuffer<cumes::ControlRecord> d_rec(1);
    cumes::ControlRecord h_rec = {};
    h_rec.status.jacobian_valid = 1;

    const size_t nField = (size_t)(p.ns - 1) * p.nZnT;  // half-grid fields
    const size_t nFull = (size_t)p.ns * p.nZnT;         // full-grid buffers

    // One guarded pass with valid status, snapshot the written caches.
    // The fused inverse produces rCon/zCon into the constraint views (the
    // same production path — a plain inverse would leave them uninitialized).
    auto runPass = [&]() {
        transform.inverse_fused(storage.physical_const(), /*do_combine=*/false,
                                constraint.rcon_view(p).data(),
                                constraint.zcon_view(p).data());
        geometry.enqueue(rs, p, rp, 0);
        enqueueGuardedConsumers<T>(p, rp, rs, transform, geometry, precon,
                                   constraint, mt, &d_rec.data()->status);
        cc(cudaDeviceSynchronize(), "pass sync");
    };
    check_cuda(
        cudaMemcpy(d_rec.data(), &h_rec, sizeof(h_rec), cudaMemcpyHostToDevice),
        "rec up (guard)");
    runPass();

    std::vector<T> snap_bsupu(nField), snap_iotaF(p.ns), snap_ard(2 * p.ns);
    std::vector<T> snap_rcon0(nFull), snap_brmn(nFull);
    const auto* bsupu = geometry.magnetic_field_views(p).bsupu.data();
    check_cuda(cudaMemcpy(snap_bsupu.data(), bsupu, nField * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "snap bsupu");
    check_cuda(cudaMemcpy(snap_iotaF.data(), rp.iota_F, p.ns * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "snap iotaF");
    check_cuda(cudaMemcpy(snap_ard.data(), precon.ard(), 2 * p.ns * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "snap ard");
    check_cuda(cudaMemcpy(snap_rcon0.data(), constraint.rcon0(),
                          nFull * sizeof(T), cudaMemcpyDeviceToHost),
               "snap rcon0");
    check_cuda(cudaMemcpy(snap_brmn.data(), rs.d_brmn_e, nFull * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "snap brmn");

    // Invalid pass: same enqueues, guarded kernels must write NOTHING.
    h_rec.status.jacobian_valid = 0;
    check_cuda(
        cudaMemcpy(d_rec.data(), &h_rec, sizeof(h_rec), cudaMemcpyHostToDevice),
        "rec up (invalid)");
    runPass();

    std::vector<T> now_bsupu(nField), now_iotaF(p.ns), now_ard(2 * p.ns);
    std::vector<T> now_rcon0(nFull), now_brmn(nFull);
    check_cuda(cudaMemcpy(now_bsupu.data(), bsupu, nField * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "now bsupu");
    check_cuda(cudaMemcpy(now_iotaF.data(), rp.iota_F, p.ns * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "now iotaF");
    check_cuda(cudaMemcpy(now_ard.data(), precon.ard(), 2 * p.ns * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "now ard");
    check_cuda(cudaMemcpy(now_rcon0.data(), constraint.rcon0(),
                          nFull * sizeof(T), cudaMemcpyDeviceToHost),
               "now rcon0");
    check_cuda(cudaMemcpy(now_brmn.data(), rs.d_brmn_e, nFull * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "now brmn");

    check(std::memcmp(snap_bsupu.data(), now_bsupu.data(),
                      nField * sizeof(T)) == 0,
          "invalid pass: magnetic-field buffers untouched");
    check(
        std::memcmp(snap_iotaF.data(), now_iotaF.data(), p.ns * sizeof(T)) == 0,
        "invalid pass: iotaF/chipF profile cache untouched");
    check(
        std::memcmp(snap_ard.data(), now_ard.data(), 2 * p.ns * sizeof(T)) == 0,
        "invalid pass: preconditioner element cache untouched");
    check(std::memcmp(snap_rcon0.data(), now_rcon0.data(), nFull * sizeof(T)) ==
              0,
          "invalid pass: constraint reference cache untouched");
    check(
        std::memcmp(snap_brmn.data(), now_brmn.data(), nFull * sizeof(T)) == 0,
        "invalid pass: MHD force buffers untouched");

    // Valid pass again: the guards must re-enable the real work (outputs move).
    h_rec.status.jacobian_valid = 1;
    check_cuda(
        cudaMemcpy(d_rec.data(), &h_rec, sizeof(h_rec), cudaMemcpyHostToDevice),
        "rec up (valid again)");
    runPass();
    check_cuda(cudaMemcpy(now_bsupu.data(), bsupu, nField * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "again bsupu");
    check(std::memcmp(snap_bsupu.data(), now_bsupu.data(),
                      nField * sizeof(T)) == 0,
          "valid pass again: guarded work re-enabled");
}

// ---------------------------------------------------------------------------
// 4. Preconditioned-reduction terminal gate
// ---------------------------------------------------------------------------
template <typename T>
static void runPreconditionedGate(T label) {
    (void)label;
    const int ns = 8, mnmax = 4;
    std::vector<T> h_f(6 * mnmax * ns);
    for (size_t i = 0; i < h_f.size(); ++i) {
        h_f[i] = T(std::sin(0.31 * (double)i + 1.7) + 0.5);
    }
    cumes::DeviceBuffer<T> d_f(h_f.size());
    check_cuda(cudaMemcpy(d_f.data(), h_f.data(), h_f.size() * sizeof(T),
                          cudaMemcpyHostToDevice),
               "f up");
    cumes::SpectralView<const T, cumes::DecomposedResidualDomain> view(
        d_f.data(), ns, mnmax);

    cumes::DeviceBuffer<cumes::ControlRecord> d_rec(1);
    cumes::ControlRecord h = {};

    // Terminal (converged): zero sentinel + not_evaluated.
    h.status.invariant_converged = 1;
    check_cuda(cudaMemcpy(d_rec.data(), &h, sizeof(h), cudaMemcpyHostToDevice),
               "rec up (terminal)");
    computeResidualsPreconditionedKernel<T>
        <<<3, 256>>>(view, ns, mnmax, d_rec.data());
    cc(cudaDeviceSynchronize(), "gate sync terminal");
    check_cuda(cudaMemcpy(&h, d_rec.data(), sizeof(h), cudaMemcpyDeviceToHost),
               "rec down (terminal)");
    check(h.preconditioned_raw[0] == 0.0 && h.preconditioned_raw[1] == 0.0 &&
              h.preconditioned_raw[2] == 0.0 &&
              h.status.preconditioned_evaluated == 0,
          "terminal pass: preconditioned residuals zero sentinel + not "
          "evaluated");

    // Continuing: real reduction + evaluated bit.
    h = cumes::ControlRecord{};
    check_cuda(cudaMemcpy(d_rec.data(), &h, sizeof(h), cudaMemcpyHostToDevice),
               "rec up (continue)");
    computeResidualsPreconditionedKernel<T>
        <<<3, 256>>>(view, ns, mnmax, d_rec.data());
    cc(cudaDeviceSynchronize(), "gate sync continue");
    check_cuda(cudaMemcpy(&h, d_rec.data(), sizeof(h), cudaMemcpyDeviceToHost),
               "rec down (continue)");
    check(h.status.preconditioned_evaluated == 1,
          "continuing pass: preconditioned residuals marked evaluated");
    // Host reference sum for group 0 (fsqr = sum frcc^2 + frss^2 over all).
    double ref0 = 0.0;
    for (int mode = 0; mode < mnmax; ++mode) {
        for (int j = 0; j < ns; ++j) {
            ref0 += (double)h_f[0 * mnmax * ns + mode * ns + j] *
                        (double)h_f[0 * mnmax * ns + mode * ns + j] +
                    (double)h_f[3 * mnmax * ns + mode * ns + j] *
                        (double)h_f[3 * mnmax * ns + mode * ns + j];
        }
    }
    ref0 /= (double)(mnmax * ns);
    const double got0 = h.preconditioned_raw[0];
    // Tree-vs-serial summation order differs at rounding level: 1e-9 relative
    // for double, 1e-4 for the float leg's rounding floor.
    const double tol = (sizeof(T) == sizeof(double)) ? 1e-9 : 1e-4;
    check(std::fabs(got0 - ref0) <= tol * std::max(1.0, std::fabs(ref0)),
          "continuing pass: fsqr group matches host reference sum");
}

// ---------------------------------------------------------------------------
// 5. End-to-end collapsed state through EquilibriumOperator::enqueue
// ---------------------------------------------------------------------------
template <typename T>
static void runCollapsedDag(T label) {
    (void)label;
    DeviceParams<T> p;
    p.ns = 9;
    p.mnmax = 4;
    p.ntheta = 18;
    p.nzeta = 2;
    p.nfp = 1;
    p.nZnT = 36;
    p.mpol = 4;
    p.ntor = 0;
    p.ncurr = 0;
    p.delt = T(0.9);
    p.ftol = T(1e-14);
    p.max_iter = 10;
    p.tcon0 = T(1.0);
    p.lamscale = T(0.0);

    cumes::ValidatedProblem vp = load_validated("inputs/solovev.json");

    cumes::SpectralStorage<T> storage(p.ns, p.mnmax);
    // Seed with the manufactured geometry so the caches are REAL first. The
    // Z families are negated: signJ = -1, and the kSolovevLinear Z sign gives
    // a Jacobian of the opposite orientation (signJ*sqrt(g) < 0 -> invalid).
    // Mirroring Z flips tau and makes the manufactured state a VALID pass.
    {
        std::vector<T> h_cc, h_ss, h_zsc, h_zcs, h_lsc, h_lcs;
        manufactured_state<T>(ManufacturedShape::kSolovevLinear, p.ns, p.mnmax,
                              p.ntor, h_cc, h_ss, h_zsc, h_zcs, h_lsc, h_lcs);
        for (auto& v : h_zsc) v = -v;
        for (auto& v : h_zcs) v = -v;
        upload_state(storage, h_cc, h_ss, h_zsc, h_zcs, h_lsc, h_lcs, p.ns,
                     p.mnmax);
    }

    cumes::Profiles<T> profiles(p, vp, nullptr);
    cumes::RadialProfileViews<T> rp = profiles.profile_views();
    cumes::DeviceModeTable mt = cumes::modeTableCreate(p);
    cumes::RealSpaceStorage<T> rs = realSpaceCreate(p);
    cumes::ToroidalFftOperator<T> transform(p, rs, mt);
    cumes::GeometryOperator<T> geometry(p, nullptr);

    cumes::EquilibriumOperator<T> equilibrium(p, storage, profiles, transform,
                                              rs, geometry, nullptr, nullptr);

    cumes::EvaluationSchedule schedule;
    schedule.update_iota_chi = true;
    schedule.reset_constraint_reference = true;
    schedule.refresh_preconditioner = true;
    schedule.zero_z_force_m1 = true;

    // A valid pass first: real caches, real jacobian validity.
    equilibrium.enqueue(0, 1, schedule, 0, 1.0, 1.0);
    cc(cudaDeviceSynchronize(), "valid dag sync");
    cumes::ControlRecord h_rec;
    check_cuda(cudaMemcpy(&h_rec, equilibrium.control_device(), sizeof(h_rec),
                          cudaMemcpyDeviceToHost),
               "rec down (valid dag)");
    check(h_rec.status.jacobian_valid != 0, "valid state: jacobian_valid set");
    const size_t nField = (size_t)(p.ns - 1) * p.nZnT;
    std::vector<T> snap_iotaF(p.ns), snap_bsupu(nField);
    check_cuda(cudaMemcpy(snap_iotaF.data(), rp.iota_F, p.ns * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "snap iotaF (dag)");
    check_cuda(cudaMemcpy(snap_bsupu.data(),
                          geometry.magnetic_field_views(p).bsupu.data(),
                          nField * sizeof(T), cudaMemcpyDeviceToHost),
               "snap bsupu (dag)");

    // Collapse the state: all-zero spectral coefficients -> sqrt(g) = 0.
    {
        std::vector<T> z((size_t)p.ns * p.mnmax, T(0.0));
        upload_state(storage, z, z, z, z, z, z, p.ns, p.mnmax);
        cc(cudaDeviceSynchronize(), "collapse sync");
    }

    equilibrium.enqueue(0, 2, schedule, 0, 1.0, 1.0);
    cc(cudaDeviceSynchronize(), "collapsed dag sync");
    check_cuda(cudaMemcpy(&h_rec, equilibrium.control_device(), sizeof(h_rec),
                          cudaMemcpyDeviceToHost),
               "rec down (collapsed dag)");
    check(h_rec.status.jacobian_valid == 0,
          "collapsed state: jacobian_valid clear (device finalize)");

    std::vector<T> now_iotaF(p.ns), now_bsupu(nField);
    check_cuda(cudaMemcpy(now_iotaF.data(), rp.iota_F, p.ns * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "now iotaF (dag)");
    check_cuda(cudaMemcpy(now_bsupu.data(),
                          geometry.magnetic_field_views(p).bsupu.data(),
                          nField * sizeof(T), cudaMemcpyDeviceToHost),
               "now bsupu (dag)");
    check(
        std::memcmp(snap_iotaF.data(), now_iotaF.data(), p.ns * sizeof(T)) == 0,
        "collapsed pass: profile cache not mutated");
    check(std::memcmp(snap_bsupu.data(), now_bsupu.data(),
                      nField * sizeof(T)) == 0,
          "collapsed pass: field buffers not mutated");
}

// ---------------------------------------------------------------------------
// 6. Production-path first-sample sign reversal (completion-plan follow-up
//    §2.1): jacobianStatsKernel + jacobianFinalizeKernel over a manufactured
//    half-grid whose ONLY sign reversal is the FIRST sample of reduction
//    lane 20. Before the vmin-init fix that lane seeded the minimum from
//    fabs(g), hiding the flip behind the healthy samples; the oriented
//    minimum must now be the flipped value and the finalize kernel must
//    report invalid, with every guarded consumer writing nothing.
// ---------------------------------------------------------------------------
template <typename T>
static void runSignFlipStats(T label) {
    (void)label;
    DeviceParams<T> p;
    p.ns = 9;
    p.mnmax = 4;
    p.ntheta = 18;
    p.nzeta = 2;
    p.nfp = 1;
    p.nZnT = 36;
    p.mpol = 4;
    p.ntor = 0;
    p.ncurr = 0;
    p.delt = T(0.9);
    p.ftol = T(1e-14);
    p.max_iter = 10;
    p.tcon0 = T(1.0);
    p.lamscale = T(0.0);

    cumes::ValidatedProblem vp = load_validated("inputs/solovev.json");

    cumes::SpectralStorage<T> storage(p.ns, p.mnmax);
    std::vector<T> h_cc, h_ss, h_zsc, h_zcs, h_lsc, h_lcs;
    manufactured_state<T>(ManufacturedShape::kSolovevLinear, p.ns, p.mnmax,
                          p.ntor, h_cc, h_ss, h_zsc, h_zcs, h_lsc, h_lcs);
    upload_state(storage, h_cc, h_ss, h_zsc, h_zcs, h_lsc, h_lcs, p.ns,
                 p.mnmax);

    cumes::Profiles<T> profiles(p, vp, nullptr);
    cumes::RadialProfileViews<T> rp = profiles.profile_views();
    cumes::DeviceModeTable mt = cumes::modeTableCreate(p);
    cumes::RealSpaceStorage<T> rs = realSpaceCreate(p);
    cumes::ToroidalFftOperator<T> transform(p, rs, mt);
    cumes::GeometryOperator<T> geometry(p, nullptr);
    cumes::Preconditioner<T> precon(p, nullptr);
    cumes::ConstraintOperator<T> constraint(p, nullptr);

    const size_t nHalf = (size_t)(p.ns - 1) * p.nZnT;  // 288 half-grid entries
    const size_t nFull = (size_t)p.ns * p.nZnT;

    cumes::DeviceBuffer<cumes::ControlRecord> d_rec(1);
    cumes::ControlRecord h_rec = {};
    h_rec.status.jacobian_valid = 1;

    // Seed a valid pass (the fused inverse + base geometry), then OVERWRITE
    // the gsqrt view with the manufactured buffer. jacobian_stats launches
    // ONE 256-thread block, so lane t's first sample is flat index t: the
    // flip sits at index 20 (< 256), with lane 20's second sample at 276
    // (< nHalf = 288) — exactly the bug shape from the review.
    transform.inverse_fused(storage.physical_const(), /*do_combine=*/false,
                            constraint.rcon_view(p).data(),
                            constraint.zcon_view(p).data());
    geometry.enqueue(rs, p, rp, 0);
    cc(cudaDeviceSynchronize(), "seed sync");

    // Seed the guarded caches with a VALID pass (jacobian_valid = 1), so the
    // later no-op comparison runs against real, initialized data.
    check_cuda(
        cudaMemcpy(d_rec.data(), &h_rec, sizeof(h_rec), cudaMemcpyHostToDevice),
        "rec up (seed)");
    enqueueGuardedConsumers<T>(p, rp, rs, transform, geometry, precon,
                               constraint, mt, &d_rec.data()->status);
    cc(cudaDeviceSynchronize(), "seed guarded sync");

    // signJ = -1 and a valid run has sqrt(g) < 0 everywhere, so the
    // oriented value signJ·sqrt(g) is positive on the healthy grid. The
    // manufactured buffer is negative everywhere EXCEPT one POSITIVE element
    // at lane 20's first sample — the only sign reversal of sqrt(g).
    const double flip = 0.5, maxval = 1.75;
    std::vector<T> h_gsqrt(nHalf);
    for (size_t i = 0; i < nHalf; ++i) {
        h_gsqrt[i] = -T(0.4 + 0.01 * (double)(i % 50));
    }
    h_gsqrt[20] = T(flip);  // lane 20's FIRST sample: the only sign reversal
    h_gsqrt[276] =
        -T(maxval);  // lane 20's SECOND sample (20+256 < 288), above |flip|
    check_cuda(
        cudaMemcpy(geometry.base_geometry_views(p).gsqrt.data(), h_gsqrt.data(),
                   nHalf * sizeof(T), cudaMemcpyHostToDevice),
        "gsqrt up");

    // The PRODUCTION chain: GeometryOperator::jacobian_stats (the exact
    // function the DAG enqueues) followed by jacobianFinalizeKernel.
    geometry.jacobian_stats(p, d_rec.data(), 0);
    jacobianFinalizeKernel<<<1, 1>>>(d_rec.data(), p.nZnT);
    cc(cudaDeviceSynchronize(), "stats sync");
    check_cuda(
        cudaMemcpy(&h_rec, d_rec.data(), sizeof(h_rec), cudaMemcpyDeviceToHost),
        "rec down (stats)");

    check(h_rec.jacobian_min_oriented == (double)T(-flip),
          "sign flip: oriented minimum is signJ·(flipped sample)");
    check(h_rec.jacobian_max_abs == (double)T(maxval),
          "sign flip: max |sqrt(g)| keeps every magnitude");
    check(h_rec.jacobian_nonfinite_count == 0.0,
          "sign flip: no nonfinite entries");
    check(h_rec.jacobian_min_index == 20.0,
          "sign flip: argmin index is the flip");
    check(h_rec.status.jacobian_valid == 0,
          "sign flip: jacobianFinalizeKernel reports invalid");

    // The guarded consumers with the just-finalized (invalid) status: every
    // output/cache sentinel must stay byte-unchanged.
    std::vector<T> snap_bsupu(nHalf), snap_iotaF(p.ns), snap_ard(2 * p.ns);
    std::vector<T> snap_rcon0(nFull), snap_brmn(nFull);
    check_cuda(cudaMemcpy(snap_bsupu.data(),
                          geometry.magnetic_field_views(p).bsupu.data(),
                          nHalf * sizeof(T), cudaMemcpyDeviceToHost),
               "snap bsupu");
    check_cuda(cudaMemcpy(snap_iotaF.data(), rp.iota_F, p.ns * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "snap iotaF");
    check_cuda(cudaMemcpy(snap_ard.data(), precon.ard(), 2 * p.ns * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "snap ard");
    check_cuda(cudaMemcpy(snap_rcon0.data(), constraint.rcon0(),
                          nFull * sizeof(T), cudaMemcpyDeviceToHost),
               "snap rcon0");
    check_cuda(cudaMemcpy(snap_brmn.data(), rs.d_brmn_e, nFull * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "snap brmn");

    enqueueGuardedConsumers<T>(p, rp, rs, transform, geometry, precon,
                               constraint, mt, &d_rec.data()->status);
    cc(cudaDeviceSynchronize(), "guarded sync");

    std::vector<T> now_bsupu(nHalf), now_iotaF(p.ns), now_ard(2 * p.ns);
    std::vector<T> now_rcon0(nFull), now_brmn(nFull);
    check_cuda(cudaMemcpy(now_bsupu.data(),
                          geometry.magnetic_field_views(p).bsupu.data(),
                          nHalf * sizeof(T), cudaMemcpyDeviceToHost),
               "now bsupu");
    check_cuda(cudaMemcpy(now_iotaF.data(), rp.iota_F, p.ns * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "now iotaF");
    check_cuda(cudaMemcpy(now_ard.data(), precon.ard(), 2 * p.ns * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "now ard");
    check_cuda(cudaMemcpy(now_rcon0.data(), constraint.rcon0(),
                          nFull * sizeof(T), cudaMemcpyDeviceToHost),
               "now rcon0");
    check_cuda(cudaMemcpy(now_brmn.data(), rs.d_brmn_e, nFull * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "now brmn");

    check(std::memcmp(snap_bsupu.data(), now_bsupu.data(), nHalf * sizeof(T)) ==
              0,
          "sign flip: magnetic-field buffers untouched");
    check(
        std::memcmp(snap_iotaF.data(), now_iotaF.data(), p.ns * sizeof(T)) == 0,
        "sign flip: iotaF/chipF profile cache untouched");
    check(
        std::memcmp(snap_ard.data(), now_ard.data(), 2 * p.ns * sizeof(T)) == 0,
        "sign flip: preconditioner element cache untouched");
    check(std::memcmp(snap_rcon0.data(), now_rcon0.data(), nFull * sizeof(T)) ==
              0,
          "sign flip: constraint reference cache untouched");
    check(
        std::memcmp(snap_brmn.data(), now_brmn.data(), nFull * sizeof(T)) == 0,
        "sign flip: MHD force buffers untouched");
}

int main() {
    testJacobianFinalizeRules();
    testInvariantPredicateRules();
    testForceNormFinalizeRules();
    runGuardNoop(double(0));
    runGuardNoop(float(0));
    runSignFlipStats(double(0));
    runSignFlipStats(float(0));
    runPreconditionedGate(double(0));
    runPreconditionedGate(float(0));
    runCollapsedDag(double(0));
    runCollapsedDag(float(0));

    return summary();
}
