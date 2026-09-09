#pragma once

#include "cumes/solver/control_policy.hpp"
#include "cumes/webgpu/iteration.hpp"

#include <functional>
#include <memory>

namespace cumes::webgpu {

struct NewtonOptions {
    int iterations = control_policy::NEWTON_KRYLOV_STEPS;
    float tolerance =
        static_cast<float>(control_policy::NEWTON_INNER_TOLERANCE);
    float difference_step =
        static_cast<float>(control_policy::NEWTON_DIFFERENCE_STEP);
    bool central_difference = false;
};

struct NewtonControl {
    float rhs_norm = 0, residual_norm = 0, target_norm = 0;
    int steps = 0, cycles = 0, evaluations = 0;
    bool converged = false;
    // 1: singular projected system; 2: nonfinite arithmetic/invalid probe.
    int breakdown = 0;
};

using NewtonCallback = std::function<void(std::string, NewtonControl)>;

// GPU-resident f32 GMRES. Apply must synchronously enqueue its map and return
// the output view, whose contents are copied before the next Apply. There is
// one asynchronous readback after all maps; no map may itself fence the host.
class DeviceGmres {
   public:
    using Apply = std::function<DeviceFields(const DeviceFields&)>;

    DeviceGmres(const wgpu::Device& device, int size, int max_basis = 32);
    void enqueue(const DeviceFields& rhs,
                 Apply apply,
                 int iterations,
                 float tolerance,
                 NewtonCallback callback);
    DeviceFields correction() const;

   private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

// Opt-in fixed-boundary axisymmetric experiment. State/trials remain paired;
// the frozen production preconditioner and Krylov vectors remain scalar f32.
// The caller owns eligibility, trial acceptance, rollback re-evaluation,
// velocity reset and all outer-controller state.
class NewtonCorrection {
   public:
    NewtonCorrection(const wgpu::Device& device, int ns, int mpol);
    void prepare(IterationCase frozen,
                 const DeviceFields& base,
                 const DeviceFields& preconditioned);
    void solve(const NewtonOptions& options, NewtonCallback callback);
    DeviceFields enqueue_trial(float scale);
    DeviceFields enqueue_trial(const DeviceFields& direction, float scale);
    DeviceFields base_state() const;
    DeviceFields rhs() const;
    DeviceFields correction() const;

    // Diagnostic finite differences use the same frozen production oracle.
    // Consume the returned transient view before the next JVP is submitted.
    DeviceFields enqueue_jvp(const DeviceFields& direction,
                             float step,
                             bool central = false);

   private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace cumes::webgpu
