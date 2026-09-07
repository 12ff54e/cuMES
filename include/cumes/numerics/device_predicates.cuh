// device_predicates.cuh — the per-pass device status/predicate kernels
// (blueprint §6.9/§7; completion plan step 1.4).
//
// These kernels implement the device-side safety decisions of the DAG:
//
//   jacobian_finalize_kernel             reset->reduce->FINALIZE the oriented
//                                      Jacobian validity (the rule is shared
//                                      with IterationController::jacobian_
//                                      invalid via the shared control policy);
//   force_norm_finalize_kernel             finalize the force-norm factors ON
//                                      DEVICE from the refresh-pass force
//                                      norms, before the terminal predicate
//                                      (completion-plan follow-up §2.3);
//   invariant_predicate_kernel           classify the invariant residual ON
//                                      DEVICE before in-place preconditioning
//                                      (nonfinite always; converged on every
//                                      pass — refresh passes use the record's
//                                      device-finalized factors);
//   compute_residuals_preconditioned_kernel  the terminal-guarded
//   preconditioned
//                                      reduction (zero sentinel +
//                                      not_evaluated on terminal passes).
//
// They live in a PUBLIC header so tests can drive them directly with
// manufactured ControlRecords (tests/test_safety_predicates.cu); the
// production DAG (src/kernels/solver_impl.cuh) includes the same definitions,
// so the tested kernels are bit-for-bit the shipped ones. All readers/writers
// are ordered on the single compute stream — no atomics.
#ifndef CUMES_INCLUDE_CUMES_NUMERICS_DEVICE_PREDICATES_CUH_
#define CUMES_INCLUDE_CUMES_NUMERICS_DEVICE_PREDICATES_CUH_

#include "cumes/core/tensor_view.cuh"
#include "cumes/numerics/accumulation.hpp"
#include "cumes/solver/control_record.hpp"

// Finalize the global Jacobian status (blueprint §7 "JStat"): read the just-
// reduced oriented stats and decide validity with the IDENTICAL rule the host
// controller applies (IterationController::jacobian_invalid + the shared
// Jacobian threshold).
// The bit gates every downstream 1/√g consumer, cache mutation, and force.
template <class T>
static __global__ void jacobian_finalize_kernel(
    cumes::DeviceControlRecord<T>* __restrict__ rec,
    int nZnT) {
    const T eps = T(cumes::control_policy::JACOBIAN_RELATIVE_THRESHOLD);
    const bool invalid =
        rec->jacobian_nonfinite_count > T(0.0) ||
        rec->jacobian_max_abs <= T(0.0) ||
        rec->jacobian_min_oriented <= T(0.0) ||
        (rec->jacobian_min_oriented < eps * rec->jacobian_max_abs &&
         rec->jacobian_min_index >= nZnT);
    rec->status.jacobian_valid = invalid ? 0u : 1u;
}

// Finalize force-norm factors in the state scalar on refresh passes. The
// host consumes these values after the fence; it does not recompute them in
// a different precision. Keep std::max comparison semantics for NaN inputs.
// Unevaluated fields retain the zero sentinel from the pass-start reset.
template <class T>
static __global__ void force_norm_finalize_kernel(
    cumes::DeviceControlRecord<T>* __restrict__ rec,
    T delta_s,
    T lamscale) {
    if (!rec->status.force_norms_evaluated) return;
    const T sRZ = rec->force_norms[0];
    const T sL = rec->force_norms[1];
    const T sMag = rec->force_norms[2];
    T eTherm = rec->force_norms[3];
    T vol = rec->force_norms[4];
    const T h_rz = rec->force_norms[5];
    const T eMag = fabs(sMag) * delta_s;
    eTherm *= delta_s;
    vol *= delta_s;
    const T energyDensity = ((eMag < eTherm) ? eTherm : eMag) / vol;
    // Scale-free-division guards (identical to the host): degenerate
    // denominators produce the unit fallback instead of inf/NaN factors.
    const T denomRZ = sRZ * energyDensity * energyDensity;
    rec->final_f_norm_rz = denomRZ > T(0.0) ? (T(1.0) / denomRZ) : T(1.0);
    const T denomL = sL * lamscale * lamscale;
    rec->final_f_norm_l = denomL > T(0.0) ? (T(1.0) / denomL) : T(1.0);
    rec->final_f_norm1 = h_rz > T(0.0) ? (T(1.0) / h_rz) : T(1.0);
}

// Normalize and classify before in-place preconditioning. Publish the same
// normalized T values to the host so both comparisons use identical inputs.
// Refresh passes use this record's new factors; other passes use the cached
// factors. The host Jacobian gate skips unevaluated refresh records.
template <class T>
static __global__ void invariant_predicate_kernel(
    cumes::DeviceControlRecord<T>* __restrict__ rec,
    T f_norm_rz,
    T f_norm_l,
    T plain_per_el,
    T ftol,
    int use_record_factors) {
    const T f_rz = use_record_factors ? rec->final_f_norm_rz : f_norm_rz;
    const T f_l = use_record_factors ? rec->final_f_norm_l : f_norm_l;
    const T fsqr_i = rec->invariant_raw[0] * plain_per_el * f_rz * T(0.25);
    const T fsqz_i = rec->invariant_raw[1] * plain_per_el * f_rz * T(0.25);
    const T fsql_i = rec->invariant_raw[2] * plain_per_el * f_l;
    rec->invariant_scaled[0] = fsqr_i;
    rec->invariant_scaled[1] = fsqz_i;
    rec->invariant_scaled[2] = fsql_i;
    const bool nonfinite =
        !(isfinite(fsqr_i) && isfinite(fsqz_i) && isfinite(fsql_i));
    const bool can_classify =
        !use_record_factors || rec->status.force_norms_evaluated != 0;
    rec->status.invariant_nonfinite = nonfinite ? 1u : 0u;
    rec->status.invariant_converged =
        (!nonfinite && can_classify && fsqr_i <= ftol && fsqz_i <= ftol &&
         fsql_i <= ftol)
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
__global__ void compute_residuals_preconditioned_kernel(
    cumes::SpectralView<const T, cumes::DecomposedResidualDomain> f_spec,
    int ns,
    int mnmax,
    cumes::DeviceControlRecord<T>* __restrict__ rec) {
    using A = typename cumes::NormAccum<T>::type;  // float-float for float
    int comp = blockIdx.x;
    if (comp >= 3) return;
    const bool terminal = rec->status.invariant_nonfinite != 0 ||
                          rec->status.invariant_converged != 0;
    if (terminal) {
        if (threadIdx.x == 0) rec->preconditioned_raw[comp] = T(0.0);
        return;
    }
    A sum = A(0);
    int total = mnmax * ns;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int mode = i / ns, j = i % ns;
        T a = f_spec(static_cast<cumes::SpectralComponent>(comp), mode, j);
        T b = f_spec(static_cast<cumes::SpectralComponent>(comp + 3), mode, j);
        sum += A(a * a + b * b);
    }
    __shared__ A s_sum[256];
    int tid = threadIdx.x;
    s_sum[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        rec->preconditioned_raw[comp] = T(s_sum[0]) / T(mnmax * ns);
        rec->status.preconditioned_evaluated = 1;  // idempotent (3 blocks)
    }
}

#endif  // CUMES_INCLUDE_CUMES_NUMERICS_DEVICE_PREDICATES_CUH_
