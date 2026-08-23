// bench_common.cuh — shared harness pieces for the three benchmark executables
// (cumes_benchmark_fixed_iteration, cumes_benchmark_graph_overhead,
// cumes_benchmark_graph_realpass).
//
// Header-only: the wall-clock helper, sample statistics, CLI-option scanner,
// config loader, and operator-stack builder that the three harnesses used to
// copy-paste per file (review finding 4.6). Each harness keeps its own stderr
// prefix, CLI flags/defaults, and output format; everything here is
// behavior-neutral for the existing call sites.
//
// The operator classes' moves are deleted (review finding 3.2), so
// OperatorStack constructs its members in place in the caller's frame — the
// caller owns the arena separately because fixed_iteration times
// arena.allocate as part of setup.
#ifndef CUMES_BENCHMARKS_BENCH_COMMON_CUH_
#define CUMES_BENCHMARKS_BENCH_COMMON_CUH_

#include "cumes/config/json_reader.hpp"
#include "cumes/config/solver_options.hpp"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes/runtime/device_arena.cuh"
#include "cumes/solver/stage_solver.hpp"
#include "cumes/state/mode_table.cuh"
#include "cumes/state/real_space_storage.hpp"
#include "cumes/transforms/axisymmetric_operator.hpp"
#include "cumes/transforms/toroidal_fft_operator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace bench_common {

using Clock = std::chrono::steady_clock;

// Wall microseconds since the steady-clock epoch (identical in all harnesses).
inline double now_us() {
    return std::chrono::duration<double, std::micro>(
               Clock::now().time_since_epoch())
        .count();
}

// Median of a by-value copy of the samples (the caller's vector is untouched).
inline double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    if (n % 2) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// 95th percentile: v[ceil(0.95*n)-1] (>= 95% of samples are <= this value).
inline double p95(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    std::size_t k = static_cast<std::size_t>(std::ceil(0.95 * n));
    if (k == 0) k = 1;
    if (k > n) k = n;
    return v[k - 1];
}

// argv scan for the harnesses' "--name value" / "--name=value" options.
// Diagnostics go to stderr with the caller's program prefix.
class ArgParser {
   public:
    ArgParser(int argc, char** argv, std::string_view prog)
        : argc_(argc), argv_(argv), prog_(prog) {}

    // Value of option `name` at index `i` (advancing i past a space-separated
    // value), or nullptr when argv[i] names a different option.
    const char* need(int& i, std::string_view name) const {
        const char* a = argv_[i];
        const std::string opt = std::string("--") + std::string(name);
        if (std::strcmp(a, opt.c_str()) == 0) {
            if (i + 1 >= argc_) {
                std::fprintf(stderr, "%s: --%s needs a value\n", prog_.data(),
                             name.data());
                std::exit(2);
            }
            return argv_[++i];
        }
        const std::string pfx = opt + "=";
        if (std::strncmp(a, pfx.c_str(), pfx.size()) == 0)
            return a + pfx.size();
        return nullptr;
    }

   private:
    int argc_;
    char** argv_;
    std::string_view prog_;
};

// Parse + validate the input JSON (the CLI host model). Prints failures to
// stderr prefixed by `prog` and returns the null (error) result; the caller
// maps that to its own exit path.
inline cumes::ValidationResult load_validated(const std::string& input_path,
                                              std::string_view prog) {
    cumes::SolverOptions opts;  // default precision policy (verify-double)
    try {
        cumes::ValidationResult vr = cumes::read_and_validate(input_path, opts);
        if (!vr.has_value()) {
            std::fprintf(stderr, "%s: input validation failed\n", prog.data());
            for (const auto& issue : vr.error().issues())
                std::fprintf(stderr, "  [%d] %s: %s\n",
                             static_cast<int>(issue.severity),
                             issue.key.c_str(), issue.message.c_str());
        }
        return vr;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: %s\n", prog.data(), e.what());
        return cumes::ValidationResult(cumes::ValidationReport{});
    }
}

// The production operator stack the fixed-iteration and graph-realpass
// harnesses build (mirrors StageSolver::run): the profiles / real-space /
// mode-table / transform / geometry operators over one stage arena, plus the
// axisymmetric direct-poloidal backend for ntor=0/nzeta=1 shapes unless
// CUMES_FORCE_GENERIC=1 (blueprint §8.5). The caller owns the arena (the
// fixed-iteration harness times arena.allocate as part of setup); the
// operators are members so the free-on-destruct operator classes are never
// moved.
template <class T>
class OperatorStack {
   public:
    using val_type = T;

    OperatorStack(DeviceParams<T>& p,
                  const cumes::ValidatedProblem& vp,
                  cumes::DeviceArena& arena)
        : profiles(p, vp, arena),
          rs(::real_space_create<T>(p, arena)),
          mt(cumes::mode_table_create<T>(p, arena)),
          transform(p, rs, mt, arena),
          geometry(p, arena),
          use_axisym(p.ntor == 0 && p.nzeta == 1) {
        if (const char* e = std::getenv("CUMES_FORCE_GENERIC"))
            if (std::atoi(e) != 0) use_axisym = false;
        if (use_axisym)
            axisym = std::make_unique<cumes::AxisymmetricOperator<T>>(p);
    }

    OperatorStack(const OperatorStack&) = delete;
    OperatorStack& operator=(const OperatorStack&) = delete;
    OperatorStack(OperatorStack&&) = delete;
    OperatorStack& operator=(OperatorStack&&) = delete;

    cumes::Profiles<T> profiles;
    cumes::RealSpaceStorage<T> rs;
    cumes::DeviceModeTable mt;
    cumes::ToroidalFftOperator<T> transform;
    cumes::GeometryOperator<T> geometry;
    bool use_axisym;
    std::unique_ptr<cumes::AxisymmetricOperator<T>> axisym;
};

}  // namespace bench_common

#endif  // CUMES_BENCHMARKS_BENCH_COMMON_CUH_
