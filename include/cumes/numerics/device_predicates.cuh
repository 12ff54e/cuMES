// device_predicates.cuh — the per-pass device status/predicate kernels
// (blueprint §6.9/§7; completion plan step 1.4).
//
// These kernels implement the device-side safety decisions of the DAG:
//
//   jacobianFinalizeKernel             reset->reduce->FINALIZE the oriented
//                                      Jacobian validity (the rule is shared
//                                      with IterationController::jacobian_
//                                      invalid via cumes::kJacobianEps);
//   invariantPredicateKernel           classify the invariant residual ON
//                                      DEVICE before in-place preconditioning
//                                      (nonfinite always; converged only when
//                                      the caller's force-norm factors are
//                                      final — refresh passes disable it);
//   computeResidualsPreconditionedKernel  the terminal-guarded preconditioned
//                                      reduction (zero sentinel +
//                                      not_evaluated on terminal passes).
//
// They live in a PUBLIC header so tests can drive them directly with
// manufactured ControlRecords (tests/test_safety_predicates.cu); the
// production DAG (src/solver_impl.cuh) includes the same definitions, so the
// tested kernels are bit-for-bit the shipped ones. All readers/writers are
// ordered on the single compute stream — no atomics.
#pragma once

#include "cumes/core/tensor_view.cuh"
#include "cumes/numerics/accumulation.hpp"
#include "cumes/solver/control_record.hpp"

// Finalize the global Jacobian status (blueprint §7 "JStat"): read the just-
// reduced oriented stats and decide validity with the IDENTICAL rule the host
// controller applies (IterationController::jacobian_invalid + kJacobianEps).
// The bit gates every downstream 1/√g consumer, cache mutation, and force.
static __global__ void jacobianFinalizeKernel(cumes::ControlRecord* __restrict__ rec,
                                            int nZnT) {
    const double eps = cumes::kJacobianEps;
    const bool invalid =
        rec->jacobian_nonfinite_count > 0.0 ||
        rec->jacobian_max_abs <= 0.0 ||
        rec->jacobian_min_oriented <= 0.0 ||
        (rec->jacobian_min_oriented < eps * rec->jacobian_max_abs &&
         rec->jacobian_min_index >= (double)nZnT);
    rec->status.jacobian_valid = invalid ? 0u : 1u;
}

// Classify the invariant (unpreconditioned) residual ON DEVICE before the
// in-place preconditioner (blueprint §6.9/§7 "Terminal"). The normalized
// triples are formed with the host's cached force-norm factors and the exact
// host expressions, so the bits agree with IterationController::classify_
// invariant bit-for-bit. On a preconditioner-refresh pass the host's factors
// are not yet final (they are finalized from THIS pass's force norms at the
// fence), so convergence classification is structurally disabled there
// (classify_converged=0) — a stale-factor false "converged" can then never
// suppress preconditioning on a pass the host later continues. Nonfinite
// classification is factor-independent (finite factors preserve non-finite
// sums) and always runs.
static __global__ void invariantPredicateKernel(cumes::ControlRecord* __restrict__ rec,
                                              double f_norm_rz, double f_norm_l,
                                              double plain_per_el, double ftol,
                                              int classify_converged) {
    const double fsqr_i =
        rec->invariant_raw[0] * plain_per_el * f_norm_rz * 0.25;
    const double fsqz_i =
        rec->invariant_raw[1] * plain_per_el * f_norm_rz * 0.25;
    const double fsql_i = rec->invariant_raw[2] * plain_per_el * f_norm_l;
    const bool nonfinite = !(isfinite(fsqr_i) && isfinite(fsqz_i) &&
                             isfinite(fsql_i));
    rec->status.invariant_nonfinite = nonfinite ? 1u : 0u;
    rec->status.invariant_converged =
        (!nonfinite && classify_converged != 0 && fsqr_i <= ftol &&
         fsqz_i <= ftol && fsql_i <= ftol)
            ? 1u
            : 0u;
}

// Preconditioned residual reduction with the terminal gate: on a nonfinite or
// converged pass the in-place preconditioner no-op'd, so this reduction also
// no-ops — it stores the deterministic zero sentinel and leaves
// preconditioned_evaluated clear (blueprint §6.9 "not_evaluated"). The frozen
// telemetry already records zero preconditioned residuals on terminal passes
// (recordPass), so the on-disk contract is unchanged.
template <typename T>
__global__ void computeResidualsPreconditionedKernel(
    cumes::SpectralView<const T, cumes::DecomposedResidualDomain> f_spec,
    int ns, int mnmax, cumes::ControlRecord* __restrict__ rec) {
    using A = typename cumes::NormAccum<T>::type;  // double for mixed-float
    int comp = blockIdx.x; if (comp >= 3) return;
    const bool terminal = rec->status.invariant_nonfinite != 0 ||
                          rec->status.invariant_converged != 0;
    if (terminal) {
        if (threadIdx.x == 0) rec->preconditioned_raw[comp] = 0.0;
        return;
    }
    A sum = A(0); int total = mnmax * ns;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int mode = i / ns, j = i % ns;
        T a = f_spec(static_cast<cumes::SpectralComponent>(comp), mode, j);
        T b = f_spec(static_cast<cumes::SpectralComponent>(comp + 3), mode, j);
        sum += a * a + b * b;
    }
    __shared__ A s_sum[256]; int tid = threadIdx.x; s_sum[tid] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) s_sum[tid] += s_sum[tid + s]; __syncthreads(); }
    if (tid == 0) {
        rec->preconditioned_raw[comp] = s_sum[0] / (mnmax * ns);
        rec->status.preconditioned_evaluated = 1;  // idempotent (3 blocks)
    }
}
