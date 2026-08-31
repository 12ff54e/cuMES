// control_policy.hpp — centralized fixed-point iteration policy parameters.
//
// These values govern controller decisions and solver scheduling. Keeping them
// here makes tuning auditable and prevents semantically unrelated thresholds
// that happen to share a value from becoming accidentally coupled.
#ifndef CUMES_INCLUDE_CUMES_SOLVER_CONTROL_POLICY_HPP_
#define CUMES_INCLUDE_CUMES_SOLVER_CONTROL_POLICY_HPP_

namespace cumes::control_policy {

// Controller defaults and damping history.
inline constexpr double DEFAULT_INITIAL_STEP = 0.9;
inline constexpr double DEFAULT_STAGE_TOLERANCE = 1.0e-16;
inline constexpr double DEFAULT_DTAU_FLOOR = 0.0;
inline constexpr int DAMPING_HISTORY_LENGTH = 10;
inline constexpr double DAMPING_LOG_RATIO_LIMIT = 0.15;
inline constexpr double DAMPING_TIME_SCALE_DIVISOR = 2.0;

// Preconditioner/checkpoint refresh policy.
inline constexpr int PRECONDITIONER_REFRESH_INTERVAL = 25;
inline constexpr int CHECKPOINT_REFRESH_MIN_AGE = 10;

// Jacobian validation and restart policy.
inline constexpr double JACOBIAN_RELATIVE_THRESHOLD = 1.0e-12;
inline constexpr double RESTART_STEP_FACTOR = 0.9;
inline constexpr double BAD_JACOBIAN_RESIDUAL_GROWTH_FACTOR = 100.0;
inline constexpr int BAD_PROGRESS_MIN_AGE = 12;
inline constexpr int BAD_PROGRESS_MIN_ITERATION = 50;
inline constexpr double BAD_PROGRESS_RZ_RESIDUAL_THRESHOLD = 1.0e-2;
inline constexpr double BAD_PROGRESS_STEP_DIVISOR = 1.03;

// Repeated-bad-Jacobian maintenance resets.
inline constexpr int FIRST_MAINTENANCE_BAD_JACOBIAN_COUNT = 25;
inline constexpr int SECOND_MAINTENANCE_BAD_JACOBIAN_COUNT = 50;
inline constexpr double FIRST_MAINTENANCE_STEP_FACTOR = 0.98;
inline constexpr double SECOND_MAINTENANCE_STEP_FACTOR = 0.96;

// One-shot recovery after a stable fixed-boundary window.
inline constexpr int STEP_RECOVERY_AGE = 250;
inline constexpr double STEP_RECOVERY_FACTOR = 1.1;

// Per-pass gauge and free-boundary scheduling.
inline constexpr int M1_GAUGE_BOOTSTRAP_ITERATIONS = 2;
inline constexpr double M1_GAUGE_RESIDUAL_THRESHOLD = 1.0e-6;
inline constexpr double VACUUM_ALMOST_CONVERGED_RESIDUAL = 1.0e-6;
inline constexpr int VACUUM_EDGE_INVARIANT_WINDOW = 50;
inline constexpr double VACUUM_ACTIVATION_RESIDUAL = 3.0e-2;
inline constexpr double VACUUM_CONSTRAINT_DECAY_FACTOR = 0.9;

// Stage-start step scaling. Numerators and denominators remain separate so
// applying them in T preserves the established floating-point operation order.
inline constexpr int FREE_BOUNDARY_COARSE_MAX_SURFACES = 25;
inline constexpr int FREE_BOUNDARY_COARSE_STEP_NUMERATOR = 17;
inline constexpr int FREE_BOUNDARY_COARSE_STEP_DENOMINATOR = 14;
inline constexpr int AXISYMMETRIC_SINGLE_GRID_STEP_NUMERATOR = 7;
inline constexpr int AXISYMMETRIC_SINGLE_GRID_STEP_DENOMINATOR = 6;
inline constexpr int AXISYMMETRIC_COARSE_STEP_NUMERATOR = 17;
inline constexpr int AXISYMMETRIC_COARSE_STEP_DENOMINATOR = 15;
inline constexpr int AXISYMMETRIC_REFINED_STEP_NUMERATOR = 6;
inline constexpr int AXISYMMETRIC_REFINED_STEP_DENOMINATOR = 5;

// Progress reporting on the restart-anchored iteration grid.
inline constexpr int ITERATION_OUTPUT_INTERVAL = 100;

}  // namespace cumes::control_policy

#endif  // CUMES_INCLUDE_CUMES_SOLVER_CONTROL_POLICY_HPP_
