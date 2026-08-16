// test_controller.cpp — pure-host unit test for IterationController.
//
// The blueprint's Phase 4 exit gate is "recorded residual histories reproduce
// exact restart/damping decisions". This test drives the controller through
// scripted residual sequences and requires the exact historical control
// semantics: convergence, nonfinite recovery, the 1/tau damping history, the
// refresh/restart predicates, the ijacob==25/50 maintenance reset, and the
// effective-iteration / restart-anchor bookkeeping. The end-to-end
// byte-for-byte gate (scripts/compare_bitwise.py) separately proves the
// refactored solver replays the frozen trajectory identically.
#include <cmath>
#include <cstdio>

#include "cumes/solver/iteration_controller.hpp"

static int failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("  FAIL: %s\n", what);
    }
}

static bool near(double a, double b, double eps) {
    return std::abs(a - b) <= eps * std::max(1.0, std::abs(b));
}

static void check_near(double a, double b, double eps, const char* what) {
    if (!near(a, b, eps)) {
        ++failures;
        std::printf("  FAIL: %s (got %.17g, want %.17g)\n", what, a, b);
    }
}

int main() {
    using cumes::IterationController;
    using cumes::RestartReason;

    // ---- initial state ----
    {
        IterationController<double> ctl({0.9, 1e-14, 0.0});
        check(ctl.effective_iteration() == 1, "initial iter2 == 1");
        check(ctl.restart_anchor() == 1, "initial iter1 == 1");
        check(ctl.output_anchor() == 0, "initial log_anchor == 0");
        check_near(ctl.delta_t(), 0.9, 1e-15, "initial delt == 0.9");
        check_near(ctl.fsqz_prev(), 0.0, 0.0, "initial fsqz_prev == 0");
        check(ctl.reset_constraint_reference(), "iter2==iter1 => reset ref");
        check(ctl.refresh_preconditioner(), "(iter2-iter1)%25==0 => refresh");
    }

    // ---- convergence ----
    {
        IterationController<double> ctl({0.9, 1e-14, 0.0});
        const double inv[3] = {1e-16, 1e-16, 1e-16};
        auto v = ctl.classify_invariant(inv);
        check(v.converged, "invariant <= ftol => converged");
        check(!v.nonfinite, "converged pass is finite");
    }

    // ---- nonfinite recovery ----
    {
        IterationController<double> ctl({0.9, 1e-14, 0.0});
        const double inv[3] = {std::nan(""), 0.0, 0.0};
        auto v = ctl.classify_invariant(inv);
        check(v.nonfinite, "NaN invariant => nonfinite");
        check(!v.converged, "nonfinite is not converged");
        check_near(ctl.delta_t(), 0.81, 1e-15, "nonfinite shrinks delt by 0.9");
        check(ctl.restart_anchor() == 1, "nonfinite re-anchors iter1=iter2");
        check(ctl.restart_events().size() == 1 &&
                  ctl.restart_events().back().iteration == 1,
              "nonfinite recovery records one event");
    }

    // ---- damping on the first (anchor) pass ----
    {
        IterationController<double> ctl({0.9, 1e-14, 0.0});
        const double inv[3] = {0.1, 0.1, 0.1};
        const double prec[3] = {0.01, 0.01, 0.01};  // fsq = 0.03
        check(!ctl.classify_invariant(inv).converged, "0.1 > ftol => not converged");
        auto d = ctl.decide_restart(prec, inv);
        check(d.reason == RestartReason::kNone, "anchor pass never restarts");
        check(!d.do_refresh, "anchor pass (age 0) never refreshes");
        // First pass: history all 0.15/0.9, so otav = 0.15/0.9 and
        // dtau = 0.9 * otav / 2 = 0.075, b1 = 0.925, fac = 1/1.075.
        check_near(d.damping.otav, 0.15 / 0.9, 1e-15, "anchor otav");
        check_near(d.damping.dtau, 0.075, 1e-15, "anchor dtau");
        check_near(d.damping.b1, 0.925, 1e-15, "anchor b1");
        check_near(d.damping.fac, 1.0 / 1.075, 1e-15, "anchor fac");
        ctl.after_descent(d);
        check(ctl.effective_iteration() == 2, "good pass advances iter2");
        check(ctl.restart_anchor() == 1, "good pass keeps iter1");
    }

    // ---- bad-Jacobian restart ----
    {
        IterationController<double> ctl({0.9, 1e-14, 0.0});
        const double inv[3] = {0.1, 0.1, 0.1};
        // Pass 1: small fsq establishes res0 = 0.03.
        const double small[3] = {0.01, 0.01, 0.01};
        ctl.classify_invariant(inv);
        ctl.after_descent(ctl.decide_restart(small, inv));
        check(ctl.effective_iteration() == 2, "setup pass advances iter2");
        // Pass 2: fsq = 30 blows up past 100*res0.
        const double blowup[3] = {10.0, 10.0, 10.0};
        ctl.classify_invariant(inv);
        auto d = ctl.decide_restart(blowup, inv);
        check(d.reason == RestartReason::kBadJacobian, "fsq>100*res0 => bad jacobian");
        ctl.after_descent(d);
        check_near(ctl.delta_t(), 0.81, 1e-15, "bad jacobian shrinks delt by 0.9");
        check(ctl.effective_iteration() == 2, "restart does not advance iter2");
        check(ctl.restart_anchor() == 2, "restart re-anchors iter1=iter2");
        check(ctl.restart_events().size() == 1,
              "bad-jacobian restart records one event");
        check(ctl.restart_events().back().iteration == 2,
              "bad-jacobian event carries the effective iteration");
    }

    // ---- Jacobian-gate restart (jacobian_invalid) ----
    {
        IterationController<double> ctl({0.9, 1e-14, 0.0});
        // Good pass first so the event's effective iteration is 2.
        const double inv[3] = {0.1, 0.1, 0.1};
        const double prec[3] = {0.01, 0.01, 0.01};
        ctl.classify_invariant(inv);
        ctl.after_descent(ctl.decide_restart(prec, inv));
        cumes::JacobianStatus<double> js;
        js.min_oriented = -1.0;  // sign-flipped sqrt(g)
        js.max_abs = 1.0;
        js.nonfinite_count = 0.0;
        js.min_index = 4 * 5;   // interior surface
        check(ctl.jacobian_invalid(js, 5), "sign flip => invalid jacobian");
        check_near(ctl.delta_t(), 0.81, 1e-15, "jacobian gate shrinks delt");
        check(ctl.restart_anchor() == 2, "jacobian gate re-anchors");
        check(ctl.restart_events().size() == 1 &&
                  ctl.restart_events().back().iteration == 2,
              "jacobian-gate restart records one event at iter2");
    }

    // ---- bad-progress restart (after >50 effective passes) ----
    {
        IterationController<double> ctl({0.9, 1e-14, 0.0});
        bool saw_refresh = false;
        for (int i = 0; i < 55; ++i) {
            const double inv[3] = {1e-3, 1e-3, 1e-3};
            const double prec[3] = {1e-4, 1e-4, 1e-4};  // fsq = 3e-4 == res0
            ctl.classify_invariant(inv);
            auto d = ctl.decide_restart(prec, inv);
            check(d.reason == RestartReason::kNone, "steady good pass does not restart");
            if (ctl.effective_iteration() - ctl.restart_anchor() > 10) {
                saw_refresh = saw_refresh || d.do_refresh;
            }
            ctl.after_descent(d);
        }
        check(ctl.effective_iteration() == 56, "55 good passes => iter2 == 56");
        check(saw_refresh, "steady progress requests a checkpoint refresh");
        // Stalled pass: fsq = 9e-3 (above res0 but below 100*res0) and a large
        // invariant R+Z residual with age > 12 and iter2 > 50.
        const double inv[3] = {0.1, 0.1, 0.1};
        const double prec[3] = {3e-3, 3e-3, 3e-3};
        ctl.classify_invariant(inv);
        auto d = ctl.decide_restart(prec, inv);
        check(d.reason == RestartReason::kBadProgress, "stalled pass => bad progress");
        check(!d.do_refresh, "bad progress is not a refresh");
        ctl.after_descent(d);
        check_near(ctl.delta_t(), 0.9 / 1.03, 1e-15, "bad progress divides delt by 1.03");
        check(ctl.effective_iteration() == 56, "restart does not advance iter2");
        check(ctl.restart_anchor() == 56, "restart re-anchors iter1=iter2");
        check(ctl.restart_events().size() == 1 &&
                  ctl.restart_events().back().iteration == 56,
              "bad-progress restart records one event at the stalled iteration");
    }

    // ---- ijacob==25/50 maintenance reset ----
    {
        IterationController<double> ctl({0.9, 1e-14, 0.0});
        const double inv[3] = {0.1, 0.1, 0.1};
        const double small[3] = {0.01, 0.01, 0.01};
        const double big[3] = {10.0, 10.0, 10.0};
        for (int k = 0; k < 25; ++k) {
            ctl.classify_invariant(inv);
            ctl.after_descent(ctl.decide_restart(small, inv));  // good
            ctl.classify_invariant(inv);
            ctl.after_descent(ctl.decide_restart(big, inv));     // bad
        }
        check(ctl.next_schedule(), "25 bad jacobians => maintenance reset");
        check_near(ctl.delta_t(), 0.98 * 0.9, 1e-15, "maintenance delt = 0.98*delt0");
        check(ctl.restart_events().size() == 26,
              "25 bad-jacobian events + 1 maintenance reset");
        check(ctl.restart_events().front().iteration == 2 &&
                  ctl.restart_events().back().iteration == 26,
              "events carry the effective iteration at each restart");
    }

    // ---- preconditioner refresh cadence over a fresh anchor ----
    {
        IterationController<double> ctl({0.9, 1e-14, 0.0});
        const double inv[3] = {1e-3, 1e-3, 1e-3};
        const double prec[3] = {1e-4, 1e-4, 1e-4};
        // Advance one pass, then check the cadence predicate at iter2 == 26.
        ctl.classify_invariant(inv);
        ctl.after_descent(ctl.decide_restart(prec, inv));
        for (int i = 0; i < 24; ++i) {
            ctl.classify_invariant(inv);
            ctl.after_descent(ctl.decide_restart(prec, inv));
        }
        check(ctl.effective_iteration() == 26, "cadence: iter2 == 26");
        check(ctl.refresh_preconditioner(), "cadence: (26-1)%25==0 => refresh");
    }

    if (failures) {
        std::printf("test_controller: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_controller: all checks passed\n");
    return 0;
}
