// iteration_controller.hpp — the pure host fixed-point control state machine
// (blueprint §6.10, §4.10).
//
// This class owns every host-side scalar decision the legacy solverRun() made
// inline: the effective-iteration / restart-anchor counters, the residual
// log-ratio damping history, the running-minimum reset logic, the
// bad-Jacobian / bad-progress / maintenance-reset branches, and convergence.
// It is deliberately CUDA-free and deterministic: the same sequence of
// scalar records always produces the same sequence of decisions. Unit tests
// replay recorded residual histories through it and require the exact
// historical restart/damping decisions.
//
// The solver drives it in the exact per-pass order the frozen trajectory
// requires (blueprint §4.10):
//
//   1. next_schedule()                 — top-of-pass ijacob==25/50 maintenance
//   2. jacobian_invalid(stats, nZnT)   — after computeGeometry's stats
//   3. classify_invariant(invariant)   — after the forward transform
//   4. decide_restart(prec, invariant) — after preconditioning
//   5. after_descent(decision)         — after the descent kernel
//
// All arithmetic uses the scalar type T exactly as the legacy loop did, so the
// refactor is Class A bitwise-equivalent (including float builds).
#pragma once

#include <cmath>
#include <cstdint>

#include "cumes/solver/control_record.hpp"

namespace cumes {

// vmecpp's shouldUpdateRadialPreconditioner cadence (blueprint §4.10): the
// radial preconditioner and residual normalization refresh every 25 passes
// measured from the latest restart anchor.
inline constexpr int kPreconInterval = 25;

template <typename T>
class IterationController {
  public:
    struct Options {
        T delta_t0 = T(0.9);     // initial time step (vmecpp delt)
        T ftol = T(1e-16);       // stage convergence tolerance
        T dtau_floor = T(0);     // optional 1/tau floor (0 = disabled)
    };

    explicit IterationController(Options o)
        : delt0_(o.delta_t0),
          ftol_(o.ftol),
          dtau_floor_(o.dtau_floor),
          delt_(o.delta_t0) {
        // vmecpp initializes the ten-sample 1/tau history to 0.15/delt at
        // stage start (NOT 0.15/current-delt; the two coincide only until the
        // first restart or maintenance reset changes delt).
        for (int ii = 0; ii < 10; ++ii) inv_tau_hist_[ii] = T(0.15) / delt0_;
    }

    // ---- pass-invariant accessors (the solver builds the device schedule) ----
    int effective_iteration() const noexcept { return iter2_; }
    int restart_anchor() const noexcept { return iter1_; }
    int output_anchor() const noexcept { return log_anchor_; }
    T delta_t() const noexcept { return delt_; }
    // Previous pass's invariant Z-residual, feeding the m=1 fix_m1_gauge
    // (zeroZForceForM1) condition this pass.
    T fsqz_prev() const noexcept { return fsqz_prev_; }
    // Accumulated bad-Jacobian counter (observability only; the 25/50
    // maintenance reset keys off it internally).
    int bad_jacobian_count() const noexcept { return ijacob_; }
    // rzConIntoVolume soft-reset: true on the first pass and after a restart.
    bool reset_constraint_reference() const noexcept { return iter2_ == iter1_; }
    bool refresh_preconditioner() const noexcept {
        return (iter2_ - iter1_) % kPreconInterval == 0;
    }

    // ---- per-fence decisions ----

    // Top-of-pass maintenance branch (vmec.cc "HAVING A CONVERGENCE PROBLEM:
    // RESETTING DELT"). Returns true when a reset occurred: the caller restores
    // the checkpoint and continues without geometry, descent, or an
    // effective-iteration increment.
    bool next_schedule() {
        if (ijacob_ == 25 || ijacob_ == 50) {
            ++ijacob_;
            delt_ = (ijacob_ < 50 ? T(0.98) : T(0.96)) * delt0_;
            iter1_ = iter2_;
            log_anchor_ = iter2_;
            return true;
        }
        return false;
    }

    // Oriented-Jacobian validity gate. Returns true when the geometry is
    // degenerate (the caller restores and continues); the delt shrink and
    // restart-anchor reset happen here, matching the legacy inline check.
    bool jacobian_invalid(const JacobianStatus<T>& s, int nZnT) {
        if (s.nonfinite_count > T(0) || s.max_abs <= T(0) ||
            s.min_oriented <= T(0) ||
            (s.min_oriented < T(1e-12) * s.max_abs && s.min_index >= nZnT)) {
            delt_ *= T(0.9);
            iter1_ = iter2_;
            log_anchor_ = iter2_;
            return true;
        }
        return false;
    }

    // Classify the invariant (unpreconditioned, normalized) residual triple:
    // updates fsqz_prev for the next pass's gauge condition, then reports
    // nonfinite (recover) or converged (stop).
    InvariantVerdict<T> classify_invariant(const T invariant[3]) {
        fsqz_prev_ = invariant[1];
        const bool nonfinite = !(std::isfinite(invariant[0]) &&
                                 std::isfinite(invariant[1]) &&
                                 std::isfinite(invariant[2]));
        if (nonfinite) {
            delt_ *= T(0.9);
            iter1_ = iter2_;
            log_anchor_ = iter2_;
            return InvariantVerdict<T>{true, false};
        }
        InvariantVerdict<T> v;
        v.converged = invariant[0] <= ftol_ && invariant[1] <= ftol_ &&
                      invariant[2] <= ftol_;
        return v;
    }

    // Update the 1/tau damping history, compute the acceleration coefficients,
    // and choose the refresh/restart action from the preconditioned (and
    // invariant, for the bad-progress threshold) residuals.
    RestartDecision<T> decide_restart(const T preconditioned[3],
                                      const T invariant[3]) {
        const T fsq = preconditioned[0] + preconditioned[1] + preconditioned[2];

        // vmecpp Evolve: on the restart-anchor pass the history is
        // reinitialized, then shifted once per pass, and a new sample is
        // inserted only when iter2 > iter1 (so the anchor pass leaves the last
        // entry at the initialized 0.15/delt value).
        if (iter2_ == iter1_) {
            for (int ii = 0; ii < 10; ++ii) inv_tau_hist_[ii] = T(0.15) / delt_;
        }
        for (int ii = 0; ii < 9; ++ii) inv_tau_hist_[ii] = inv_tau_hist_[ii + 1];
        if (iter2_ > iter1_) {
            T invtau_num = T(0);
            if (fsq != T(0)) {
                invtau_num =
                    std::min(std::abs(std::log(fsq / fsq_prev_)), T(0.15));
            }
            inv_tau_hist_[9] = invtau_num / delt_;
        }
        fsq_prev_ = fsq;

        T otav = T(0);
        for (int ii = 0; ii < 10; ++ii) otav += inv_tau_hist_[ii];
        otav /= T(10);
        T dtau = delt_ * otav / T(2);
        if (dtau_floor_ > T(0)) dtau = std::fmax(dtau, dtau_floor_);
        const T b1 = T(1) - dtau;
        const T fac = T(1) / (T(1) + dtau);

        // Running minimum of the preconditioned sum; reset on the restart
        // anchor (and the sentinel -1 first pass), exactly as vmecpp.
        if (iter2_ == iter1_ || res0_ == T(-1.0)) res0_ = fsq;
        res0_ = std::min(res0_, fsq);

        RestartReason reason = RestartReason::kNone;
        bool do_refresh = false;
        if (fsq <= res0_ && (iter2_ - iter1_) > 10) {
            do_refresh = true;
        } else if (fsq > T(100.0) * res0_ && iter2_ > iter1_) {
            reason = RestartReason::kBadJacobian;
        } else if ((iter2_ - iter1_) > 12 && iter2_ > 50 &&
                   (invariant[0] + invariant[1]) > T(1.0e-2)) {
            reason = RestartReason::kBadProgress;
        }

        RestartDecision<T> d;
        d.reason = reason;
        d.do_refresh = do_refresh;
        d.damping.b1 = b1;
        d.damping.fac = fac;
        d.damping.otav = otav;
        d.damping.dtau = dtau;
        return d;
    }

    // Post-descent bookkeeping: apply the restart's time-step adjustment and
    // re-anchor, or advance the effective-iteration counter on a good pass.
    void after_descent(const RestartDecision<T>& d) {
        if (d.reason != RestartReason::kNone) {
            if (d.reason == RestartReason::kBadJacobian) {
                delt_ *= T(0.9);
                ++ijacob_;
            } else {
                delt_ /= T(1.03);
            }
            iter1_ = iter2_;
            log_anchor_ = iter2_;
        } else {
            ++iter2_;
        }
    }

  private:
    const T delt0_;
    const T ftol_;
    const T dtau_floor_;
    T delt_;
    int iter2_ = 1;        // effective iteration (does not advance on restart)
    int iter1_ = 1;        // latest restart anchor
    int log_anchor_ = 0;   // output-grid anchor (reset on restart)
    int ijacob_ = 0;       // accumulated bad-Jacobian counter
    T res0_ = T(-1.0);     // running minimum of the preconditioned sum
    T fsq_prev_ = T(1.0);  // previous preconditioned sum
    T fsqz_prev_ = T(0.0); // previous invariant Z-residual
    T inv_tau_hist_[10];   // ten-sample 1/tau history
};

}  // namespace cumes
