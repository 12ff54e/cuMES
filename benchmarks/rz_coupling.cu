// Diagnostic only: probe a frozen production residual at a saved state.
// Finite differences, least squares and trial updates stay on the device.
// The model uses P^-1 F because EquilibriumOperator exposes its residual after
// preconditioning; acceptance uses all three actual invariant residuals.
#include "bench_common.cuh"
#include "cumes/io/checkpoint.hpp"
#include "cumes/runtime/device_buffer.cuh"
#include "cumes/runtime/stream.hpp"
#include "cumes/solver/equilibrium_operator.hpp"
#include "cumes/solver/start_policy.hpp"
#include "cumes/state/seed_state.hpp"

#include <array>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace {

constexpr int THREADS = 256;
constexpr int MAX_BASIS = 64;

struct Basis {
    int component;
    int m;
    int n;
    int radial;
};

__global__ void basis_kernel(double* d_basis,
                             const Basis* d_descriptors,
                             int count,
                             int ns,
                             int mnmax,
                             int ntor) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int column = blockIdx.y;
    const int size = 6 * ns * mnmax;
    if (i >= size || column >= count) return;
    const Basis b = d_descriptors[column];
    const int component = i / (ns * mnmax);
    const int mode = (i / ns) % mnmax;
    const int j = i % ns;
    double value = 0.0;
    if (component == b.component && mode == b.m * (ntor + 1) + b.n &&
        j < ns - 1) {
        // m>0 dependent axis entries follow the same j=1 extrapolation as
        // production. Only Rcc/Zsc are varied: the mixed m=1 gauge is fixed.
        const double s = double(j == 0 ? 1 : j) / double(ns - 1);
        value = pow(s, 0.5 * b.m) * (1.0 - s);
        if (b.radial == 1) value *= 2.0 * s - 1.0;
    }
    d_basis[column * size + i] = value;
}

__global__ void perturb_kernel(double* d_state,
                               const double* d_base,
                               const double* d_direction,
                               int size,
                               double scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        d_state[i] = d_direction[i] == 0.0 ? d_base[i]
                                           : d_base[i] + scale * d_direction[i];
}

__global__ void difference_kernel(double* d_column,
                                  const double* d_plus,
                                  const double* d_minus,
                                  int size,
                                  double epsilon) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) d_column[i] = (d_plus[i] - d_minus[i]) / (2.0 * epsilon);
}

// Fixed tree reduction; each block owns one scalar, with no floating atomics.
__device__ double block_sum(double value) {
    __shared__ double partial[THREADS];
    partial[threadIdx.x] = value;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    return partial[0];
}

__global__ void gram_kernel(const double* d_columns,
                            const double* d_residual,
                            double* d_gram,
                            int count,
                            int size) {
    const int row = blockIdx.x;
    const int col = blockIdx.y;
    double sum = 0.0;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        const double a = d_columns[row * size + i];
        const double b =
            col == count ? -d_residual[i] : d_columns[col * size + i];
        sum += a * b;
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0) d_gram[row * (count + 1) + col] = sum;
}

__global__ void coupling_kernel(const double* d_columns,
                                const Basis* d_descriptors,
                                double* d_stats,
                                int ns,
                                int mnmax,
                                int ntor) {
    const int col = blockIdx.x;
    const int group = blockIdx.y;
    const int size = 6 * ns * mnmax;
    const Basis b = d_descriptors[col];
    const int source_mode = b.m * (ntor + 1) + b.n;
    double sum = 0.0;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        const int component = i / (ns * mnmax);
        const int mode = (i / ns) % mnmax;
        const bool selected = group < 2
                                  ? mode == source_mode && component == group
                                  : mode != source_mode;
        if (selected)
            sum += d_columns[col * size + i] * d_columns[col * size + i];
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0) d_stats[col * 3 + group] = sum;
}

// Tiny diagnostic system, solved on one GPU thread. Column equilibration and
// ridge regularization keep the two radial shapes from amplifying cancellation.
__global__ void solve_kernel(double* d_gram,
                             double* d_coefficients,
                             double* d_scales,
                             int count,
                             double ridge,
                             double trust) {
    if (threadIdx.x != 0) return;
    const int width = count + 1;
    for (int i = 0; i < count; ++i)
        d_scales[i] = sqrt(fmax(d_gram[i * width + i], 1e-300));
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < count; ++j)
            d_gram[i * width + j] /= d_scales[i] * d_scales[j];
        d_gram[i * width + i] += ridge;
        d_gram[i * width + count] /= d_scales[i];
    }
    for (int k = 0; k < count; ++k) {
        int pivot = k;
        for (int i = k + 1; i < count; ++i)
            if (fabs(d_gram[i * width + k]) > fabs(d_gram[pivot * width + k]))
                pivot = i;
        if (!(fabs(d_gram[pivot * width + k]) > 1e-14)) {
            for (int i = 0; i < count; ++i) d_coefficients[i] = 0.0;
            return;
        }
        for (int j = k; j <= count; ++j) {
            const double swap = d_gram[k * width + j];
            d_gram[k * width + j] = d_gram[pivot * width + j];
            d_gram[pivot * width + j] = swap;
        }
        for (int i = k + 1; i < count; ++i) {
            const double factor = d_gram[i * width + k] / d_gram[k * width + k];
            for (int j = k + 1; j <= count; ++j)
                d_gram[i * width + j] -= factor * d_gram[k * width + j];
        }
    }
    for (int i = count - 1; i >= 0; --i) {
        double rhs = d_gram[i * width + count];
        for (int j = i + 1; j < count; ++j)
            rhs -= d_gram[i * width + j] * d_coefficients[j];
        d_coefficients[i] = rhs / d_gram[i * width + i];
    }
    double max_coefficient = 0.0;
    for (int i = 0; i < count; ++i) {
        d_coefficients[i] /= d_scales[i];
        max_coefficient = fmax(max_coefficient, fabs(d_coefficients[i]));
    }
    const double scale =
        max_coefficient > trust ? trust / max_coefficient : 1.0;
    for (int i = 0; i < count; ++i) d_coefficients[i] *= scale;
}

__global__ void correction_kernel(double* d_state,
                                  const double* d_base,
                                  const double* d_basis,
                                  const double* d_coefficients,
                                  int count,
                                  int size,
                                  double alpha) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    double delta = 0.0;
    for (int col = 0; col < count; ++col)
        delta += d_coefficients[col] * d_basis[col * size + i];
    d_state[i] = delta == 0.0 ? d_base[i] : d_base[i] + alpha * delta;
}

double merit(const cumes::ControlRecord& rec) {
    if (!rec.status.jacobian_valid || rec.status.invariant_nonfinite)
        return std::numeric_limits<double>::infinity();
    return rec.invariant_scaled[0] + rec.invariant_scaled[1] +
           rec.invariant_scaled[2];
}

void emit_record(std::ostream& out, const cumes::ControlRecord& rec) {
    out << "{\"invariant\":[" << rec.invariant_scaled[0] << ','
        << rec.invariant_scaled[1] << ',' << rec.invariant_scaled[2]
        << "],\"jacobian_valid\":" << rec.status.jacobian_valid
        << ",\"jacobian_min\":" << rec.jacobian_min_oriented << '}';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string input = "inputs/solovev.json", restart, output;
        int max_m = 3, max_n = 1, radial = 2, ordinary_passes = 0;
        double epsilon = 1e-5, ridge = 1e-8, trust = 1e-2;
        bench_common::ArgParser args(argc, argv, "rz-coupling");
        for (int i = 1; i < argc; ++i) {
            if (const char* value = args.need(i, "input"))
                input = value;
            else if (const char* value = args.need(i, "restart"))
                restart = value;
            else if (const char* value = args.need(i, "out"))
                output = value;
            else if (const char* value = args.need(i, "max-m"))
                max_m = atoi(value);
            else if (const char* value = args.need(i, "max-n"))
                max_n = atoi(value);
            else if (const char* value = args.need(i, "radial"))
                radial = atoi(value);
            else if (const char* value = args.need(i, "ordinary-passes"))
                ordinary_passes = atoi(value);
            else if (const char* value = args.need(i, "epsilon"))
                epsilon = atof(value);
            else if (const char* value = args.need(i, "ridge"))
                ridge = atof(value);
            else if (const char* value = args.need(i, "trust"))
                trust = atof(value);
            else
                throw cumes::CumesError("rz-coupling: unknown option");
        }
        if (restart.empty() || max_m < 1 || max_n < 0 || radial < 1 ||
            radial > 2 || ordinary_passes < 0 || !(epsilon > 0.0) ||
            !std::isfinite(epsilon) || !(ridge > 0.0) ||
            !std::isfinite(ridge) || !(trust > 0.0) || !std::isfinite(trust))
            throw cumes::CumesError(
                "rz-coupling: invalid options / missing --restart");
        auto validated = bench_common::load_validated(input, "rz-coupling");
        if (!validated.has_value()) return 2;
        const auto& vp = validated.value();
        if (vp.spec().free_boundary.lfreeb)
            throw cumes::CumesError("rz-coupling: fixed boundary only");
        auto checkpoint = cumes::read_checkpoint(restart);
        if (!checkpoint.has_value())
            throw cumes::CumesError(checkpoint.error());
        auto p = cumes::init_params<double>(vp, false);
        p.ns = checkpoint.value().ns;
        if (p.mnmax != checkpoint.value().mnmax)
            throw cumes::CumesError("rz-coupling: checkpoint modes mismatch");
        int stage = -1;
        for (std::size_t g = 0; g < vp.spec().stages.size(); ++g)
            if (vp.spec().stages[g].radial_surfaces == std::size_t(p.ns))
                stage = int(g);
        if (stage < 0)
            throw cumes::CumesError(
                "rz-coupling: checkpoint grid not configured");
        p.delt =
            cumes::initial_step_for_stage(p.delt, p.ntor, p.nzeta, false, p.ns,
                                          int(vp.spec().stages.size()), stage);
        // Diagnostic evaluations always expose P^-1 F, including states below
        // the configured tolerance. This harness never declares convergence.
        p.ftol = 0.0;
        auto storage = cumes::restart_state<double>(p, vp, checkpoint.value());
        auto ordinary = cumes::restart_state<double>(p, vp, checkpoint.value());
        cumes::Stream stream;
        std::vector<Basis> descriptors;
        for (int m = 1; m <= std::min(max_m, p.mpol - 1); ++m)
            for (int n = 0; n <= std::min(max_n, p.ntor); ++n)
                for (int component = 0; component < 2; ++component)
                    for (int shape = 0; shape < radial; ++shape)
                        descriptors.push_back({component, m, n, shape});
        const int count = int(descriptors.size());
        if (count > MAX_BASIS)
            throw cumes::CumesError("rz-coupling: too many basis vectors");
        const int size = 6 * p.ns * p.mnmax;
        const std::size_t matrix_size = std::size_t(size) * count;
        cumes::DeviceBuffer<Basis> d_descriptors(count);
        cumes::DeviceBuffer<double> d_base(size), d_best(size), d_f0(size),
            d_plus(size);
        cumes::DeviceBuffer<double> d_basis(matrix_size),
            d_columns(matrix_size);
        cumes::DeviceBuffer<double> d_gram(count * (count + 1)),
            d_coefficients(count);
        cumes::DeviceBuffer<double> d_scales(count), d_stats(3 * count);
        cumes::check_cuda(cudaMemcpy(d_descriptors.data(), descriptors.data(),
                                     descriptors.size() * sizeof(Basis),
                                     cudaMemcpyHostToDevice),
                          "basis descriptors upload");
        basis_kernel<<<dim3((size + THREADS - 1) / THREADS, count), THREADS, 0,
                       stream.get()>>>(d_basis.data(), d_descriptors.data(),
                                       count, p.ns, p.mnmax, p.ntor);
        cumes::check_cuda(cudaGetLastError(), "basis launch");
        std::ostringstream json;
        json << std::setprecision(17) << "{\"ns\":" << p.ns
             << ",\"basis_count\":" << count << ",\"epsilon\":" << epsilon
             << ",\"ridge\":" << ridge << ",\"trust\":" << trust;
        cumes::run_in_stage_arena<double>(p, [&](cumes::DeviceArena& arena) {
            bench_common::OperatorStack<double> stack(p, vp, arena);
            stack.transform.bind_stream(stream.get());
            const auto transform = stack.axisym
                                       ? std::optional<std::reference_wrapper<
                                             cumes::SpectralOperator<double>>>(
                                             std::ref(*stack.axisym))
                                       : std::nullopt;
            cumes::EquilibriumOperator<double> equilibrium(
                p, storage, stack.profiles, stack.transform, stack.rs,
                stack.geometry, std::ref(arena), transform);
            cumes::EvaluationSchedule schedule;
            schedule.update_iota_chi = true;
            schedule.zero_z_force_m1 = true;
            schedule.refresh_preconditioner = true;
            schedule.reset_constraint_reference = true;
            double norm_rz = 1.0, norm_l = 1.0;
            int evaluations = 0;
            auto evaluate = [&]() {
                equilibrium.enqueue(1000 + evaluations, 1000 + evaluations,
                                    schedule, stream.get(), norm_rz, norm_l);
                cumes::ControlRecord rec;
                cumes::check_cuda(
                    cudaMemcpyAsync(&rec, equilibrium.control_device(),
                                    sizeof(rec), cudaMemcpyDeviceToHost,
                                    stream.get()),
                    "record download");
                stream.synchronize();
                ++evaluations;
                return rec;
            };
            auto base_record = evaluate();
            if (!std::isfinite(merit(base_record)))
                throw cumes::CumesError("rz-coupling: invalid base state");
            if (base_record.invariant_scaled[1] >= 1e-6)
                throw cumes::CumesError(
                    "rz-coupling: requires the stable late-stage m1 gauge");
            norm_rz = base_record.final_f_norm_rz;
            norm_l = base_record.final_f_norm_l;
            schedule.refresh_preconditioner = false;
            schedule.reset_constraint_reference = false;
            auto copy = [&](double* destination, const double* source) {
                cumes::check_cuda(
                    cudaMemcpyAsync(destination, source,
                                    std::size_t(size) * sizeof(double),
                                    cudaMemcpyDeviceToDevice, stream.get()),
                    "diagnostic state copy");
            };
            copy(d_base.data(), storage.state_slab());
            copy(d_best.data(), storage.state_slab());
            copy(d_f0.data(), equilibrium.residual().data());
            cumes::stage_detail::ScopedDeviceTimer timer;
            timer.start(stream.get());
            for (int col = 0; col < count; ++col) {
                const double* direction =
                    d_basis.data() + std::size_t(col) * size;
                perturb_kernel<<<(size + THREADS - 1) / THREADS, THREADS, 0,
                                 stream.get()>>>(storage.state_slab(),
                                                 d_base.data(), direction, size,
                                                 epsilon);
                const auto positive = evaluate();
                if (!std::isfinite(merit(positive)))
                    throw cumes::CumesError("invalid positive probe");
                copy(d_plus.data(), equilibrium.residual().data());
                perturb_kernel<<<(size + THREADS - 1) / THREADS, THREADS, 0,
                                 stream.get()>>>(storage.state_slab(),
                                                 d_base.data(), direction, size,
                                                 -epsilon);
                const auto negative = evaluate();
                if (!std::isfinite(merit(negative)))
                    throw cumes::CumesError("invalid negative probe");
                difference_kernel<<<(size + THREADS - 1) / THREADS, THREADS, 0,
                                    stream.get()>>>(
                    d_columns.data() + std::size_t(col) * size, d_plus.data(),
                    equilibrium.residual().data(), size, epsilon);
            }
            const double probe_ms = timer.stop(stream.get());
            timer.start(stream.get());
            coupling_kernel<<<dim3(count, 3), THREADS, 0, stream.get()>>>(
                d_columns.data(), d_descriptors.data(), d_stats.data(), p.ns,
                p.mnmax, p.ntor);
            gram_kernel<<<dim3(count, count + 1), THREADS, 0, stream.get()>>>(
                d_columns.data(), d_f0.data(), d_gram.data(), count, size);
            solve_kernel<<<1, 1, 0, stream.get()>>>(
                d_gram.data(), d_coefficients.data(), d_scales.data(), count,
                ridge, trust);
            cumes::check_cuda(cudaGetLastError(), "coupled solve launch");
            double best_merit = merit(base_record), best_alpha = 0.0;
            auto best_record = base_record;
            json << ",\"base\":";
            emit_record(json, base_record);
            json << ",\"trials\":[";
            int trial = 0;
            for (const double alpha : {1.0, 0.5, 0.25, 0.125}) {
                correction_kernel<<<(size + THREADS - 1) / THREADS, THREADS, 0,
                                    stream.get()>>>(
                    storage.state_slab(), d_base.data(), d_basis.data(),
                    d_coefficients.data(), count, size, alpha);
                const auto rec = evaluate();
                if (trial++) json << ',';
                json << "{\"alpha\":" << alpha << ",\"record\":";
                emit_record(json, rec);
                json << '}';
                if (merit(rec) < best_merit) {
                    best_merit = merit(rec);
                    best_record = rec;
                    best_alpha = alpha;
                    copy(d_best.data(), storage.state_slab());
                }
            }
            const double trial_ms = timer.stop(stream.get());
            // Re-evaluate the base after all probes/trials. Identical residuals
            // prove the reference, preconditioner and norm epoch stayed fixed.
            copy(storage.state_slab(), d_base.data());
            const auto replay_record = evaluate();
            for (int c = 0; c < 3; ++c)
                if (replay_record.invariant_scaled[c] !=
                    base_record.invariant_scaled[c])
                    throw cumes::CumesError(
                        "rz-coupling: frozen base replay changed");
            json << "],\"probe_ms\":" << probe_ms
                 << ",\"solve_trial_ms\":" << trial_ms
                 << ",\"best_alpha\":" << best_alpha << ",\"best\":";
            emit_record(json, best_record);
            std::vector<double> stats(3 * count), coefficients(count);
            cumes::check_cuda(cudaMemcpy(stats.data(), d_stats.data(),
                                         stats.size() * sizeof(double),
                                         cudaMemcpyDeviceToHost),
                              "coupling stats download");
            cumes::check_cuda(
                cudaMemcpy(coefficients.data(), d_coefficients.data(),
                           coefficients.size() * sizeof(double),
                           cudaMemcpyDeviceToHost),
                "coefficients download");
            json << ",\"columns\":[";
            for (int col = 0; col < count; ++col) {
                const Basis b = descriptors[col];
                if (col) json << ',';
                json << "{\"component\":" << b.component << ",\"m\":" << b.m
                     << ",\"n\":" << b.n << ",\"radial\":" << b.radial
                     << ",\"same_mode_r_norm\":" << sqrt(stats[col * 3])
                     << ",\"same_mode_z_norm\":" << sqrt(stats[col * 3 + 1])
                     << ",\"other_mode_norm\":" << sqrt(stats[col * 3 + 2])
                     << ",\"coefficient\":" << coefficients[col] << '}';
            }
            // Comparator uses the ordinary production controller, refreshes
            // and descent from the identical checkpoint. The default nominal
            // budget equals the model/trial evaluation count; actual evaluated
            // passes and elapsed time are reported, including any restarts.
            auto ordinary_p = p;
            ordinary_p.max_iter =
                ordinary_passes ? ordinary_passes : 2 * count + 4;
            cumes::SolverBench bench;
            bench.enabled = true;
            double ordinary_ms = 0.0;
            const auto result = cumes::StageSolver<double>::run(
                ordinary_p, vp, ordinary, stream.get(), std::ref(bench),
                std::nullopt, nullptr, nullptr, true, std::ref(ordinary_ms),
                false, false);
            copy(storage.state_slab(), ordinary.state_slab());
            const auto ordinary_record = evaluate();
            json << "],\"ordinary_evaluations\":" << bench.pass_wall_us.size()
                 << ",\"ordinary_effective_iterations\":" << result.iterations
                 << ",\"ordinary_ms\":" << ordinary_ms
                 << ",\"ordinary_frozen_merit\":";
            emit_record(json, ordinary_record);
            json << ",\"frozen_base_replay_exact\":true}";
            return 0;
        });
        if (output.empty())
            std::printf("%s\n", json.str().c_str());
        else {
            std::ofstream out(output);
            if (!out) throw cumes::CumesError("rz-coupling: cannot open --out");
            out << json.str() << '\n';
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "rz-coupling: %s\n", error.what());
        return 2;
    }
    return 0;
}
