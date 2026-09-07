// Reference storage must preserve sub-ULP radial structure through the
// transform, differentiation, descent, refinement and physical snapshot API.
#include "cumes/io/snapshot_bridge.cuh"
#include "cumes/numerics/descent_operator.hpp"
#include "cumes/numerics/prolongation.hpp"
#include "cumes/runtime/stream.hpp"
#include "cumes/solver/equilibrium_solver.hpp"
#include "cumes/solver/stage_solver.hpp"
#include "cumes/state/seed_state.hpp"
#include "cumes_test_cuda_helper.cuh"

#include <array>
#include <cmath>
#include <vector>

using namespace cumes;
using namespace cumes::test;

static void test_small_radial_variation() {
    DeviceParams<float> p{};
    p.ns = 99;
    p.mpol = 2;
    p.ntor = 1;
    p.mnmax = p.mpol * (p.ntor + 1);
    p.ntheta = 8;
    p.nzeta = 8;
    p.nfp = 3;
    p.nZnT = p.ntheta * p.nzeta;
    p.radius_reference = 5.5586;
    const std::array<double, 2> reference{p.radius_reference, 0.26447};
    SpectralStorage<float> state(p.ns, p.mnmax, reference);
    const std::size_t size = SPECTRAL_COMPONENT_COUNT * p.ns * p.mnmax;
    std::vector<float> h_state(size, 0.0F);
    constexpr double slope = 1e-6;
    for (int j = 0; j < p.ns; ++j)
        h_state[j] = float(slope * (1.0 - double(j) / (p.ns - 1)));
    cumes::check_cuda(cudaMemcpy(state.state_slab(), h_state.data(),
                                 size * sizeof(float), cudaMemcpyHostToDevice),
                      "upload displacement state");

    stage_detail::ScopedRealSpace<float> rs(p, std::nullopt);
    stage_detail::ScopedModeTable<float> mt(p, std::nullopt);
    ToroidalFftOperator<float> transform(p, *rs, mt.get());
    // Isolate even R radial differentiation without other mode roundoff.
    // This fixture has no Z: it tests geometry arithmetic, not a field solve
    // or a valid-Jacobian equilibrium.
    std::vector<float> h_sqrt_f(p.ns), h_sqrt_h(p.ns - 1);
    for (int j = 0; j < p.ns; ++j)
        h_sqrt_f[j] = std::sqrt(float(j) / (p.ns - 1));
    for (int j = 0; j < p.ns - 1; ++j)
        h_sqrt_h[j] = std::sqrt((float(j) + 0.5F) / (p.ns - 1));
    DeviceBuffer<float> d_sqrt_f(p.ns), d_sqrt_h(p.ns - 1);
    cumes::check_cuda(cudaMemcpy(d_sqrt_f.data(), h_sqrt_f.data(),
                                 p.ns * sizeof(float), cudaMemcpyHostToDevice),
                      "upload sqrt(s) full");
    cumes::check_cuda(
        cudaMemcpy(d_sqrt_h.data(), h_sqrt_h.data(), (p.ns - 1) * sizeof(float),
                   cudaMemcpyHostToDevice),
        "upload sqrt(s) half");
    RadialProfileViews<float> radial{};
    radial.sqrtS_F = d_sqrt_f.data();
    radial.sqrtS_H = d_sqrt_h.data();
    GeometryOperator<float> geometry(p, std::nullopt);
    transform.inverse(state.physical_const(), true);
    geometry.enqueue(*rs, p, radial, 0);
    std::vector<float> h_rs((p.ns - 1) * p.nZnT), h_r(p.ns * p.nZnT),
        h_rv(h_r.size());
    cumes::check_cuda(
        cudaMemcpy(h_rs.data(), geometry.base_geometry_views(p).rs.data(),
                   h_rs.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "read radial derivative");
    cumes::check_cuda(
        cudaMemcpy(h_r.data(), rs->d_r_real, h_r.size() * sizeof(float),
                   cudaMemcpyDeviceToHost),
        "read physical R");
    cumes::check_cuda(
        cudaMemcpy(h_rv.data(), rs->d_rv_real, h_rv.size() * sizeof(float),
                   cudaMemcpyDeviceToHost),
        "read physical R_zeta");
    double max_derivative_error = 0.0, max_radius_error = 0.0,
           max_toroidal_error = 0.0;
    for (float value : h_rs)
        max_derivative_error =
            std::max(max_derivative_error, std::abs(double(value) + slope));
    for (int j = 0; j < p.ns; ++j) {
        for (int k = 0; k < p.nZnT; ++k) {
            double angle = 2.0 * M_PI * (k / p.ntheta) / p.nzeta;
            double r = reference[0] + reference[1] * std::cos(angle) +
                       slope * (1.0 - double(j) / (p.ns - 1));
            double rv = -p.nfp * reference[1] * std::sin(angle);
            max_radius_error =
                std::max(max_radius_error, std::abs(h_r[j * p.nZnT + k] - r));
            max_toroidal_error = std::max(max_toroidal_error,
                                          std::abs(h_rv[j * p.nZnT + k] - rv));
        }
    }
    check(max_derivative_error < 1e-10,
          "reference: radial differences retain small displacements");
    check(max_radius_error < 1e-6,
          "reference: combined R restores the full reference");
    check(max_toroidal_error < 1e-6,
          "reference: toroidal derivative includes the fixed reference");

    DeviceBuffer<float> d_force(size);
    std::vector<float> h_force(size, 0.0F);
    constexpr int surface = 17;
    constexpr float increment = 1e-8F;
    h_force[surface] = increment;
    cumes::check_cuda(cudaMemcpy(d_force.data(), h_force.data(),
                                 size * sizeof(float), cudaMemcpyHostToDevice),
                      "upload small descent increment");
    auto before = snapshot_from_device(state);
    DescentAction action;
    action.perform_descent = true;
    action.delta_t = action.damping_fac = 1.0;
    DescentOperator<float>{}.enqueue(
        state.physical(), state.velocity(),
        SpectralView<const float, DecomposedResidualDomain>(d_force.data(),
                                                            p.ns, p.mnmax),
        mt.get().d_xm, mt.get().d_xn, p.ns, p.mnmax, action, 0);
    auto after = snapshot_from_device(state);
    double actual_increment =
        after.families[0][surface] - before.families[0][surface];
    check(std::abs(actual_increment - increment) < 1e-12,
          "reference: descent retains an increment below one absolute-radius "
          "ULP");
    check(after.families[0][p.ns - 1] == reference[0],
          "reference: LCFS remains physical and fixed");

    DeviceParams<float> fine = p;
    fine.ns = 129;
    auto refined = Prolongation<float>{}.enqueue(
        fine, state, p, 0, RadialInterpolation::LINEAR, {}, false);
    auto snapshot = snapshot_from_device(refined);
    check(refined.radius_references().size() == reference.size(),
          "reference: refinement carries reference modes");
    check(snapshot.families[0][fine.ns - 1] == reference[0],
          "reference: refined LCFS is physical");
    check(snapshot.families[0][2 * fine.ns - 1] == reference[1],
          "reference: refined toroidal mode is physical");
}

static void test_seed_restart() {
    auto vp = load_validated("inputs/w7x.json");
    auto p = init_params<float>(vp);
    check(SolveRequest{}.use_radius_reference,
          "reference: public solver request enables reference by default");
    check(p.radius_reference == float(vp.boundary().rbcc[0]),
          "reference: fixed 3D float enables reference by default");
    check(init_params<float>(vp, false).radius_reference == 0.0,
          "reference: explicit opt-out restores absolute coefficients");
    check(init_params<double>(vp).radius_reference == 0.0,
          "reference: double policy unchanged");
    auto axisymmetric = load_validated("inputs/solovev.json");
    check(init_params<float>(axisymmetric).radius_reference == 0.0,
          "reference: axisymmetric policy unchanged");
    auto free_boundary =
        load_validated("inputs/free_bdy/solovev_free_bdy_embedded.json");
    auto free_spec = free_boundary.spec();
    free_spec.ntor = 1;
    free_spec.raxis_c.resize(2, 0.0);
    free_spec.zaxis_s.resize(2, 0.0);
    free_spec.angular.nzeta = 6;
    auto free_3d = validate(std::move(free_spec), free_boundary.options());
    check(free_3d.has_value(), "reference: 3D free-boundary fixture validates");
    if (free_3d.has_value())
        check(init_params<float>(free_3d.value()).radius_reference == 0.0,
              "reference: free-boundary default remains absolute coefficients");
    auto seeded = init_state(p, vp, false, false);
    auto snapshot = snapshot_from_device(seeded);
    for (int invalid_case = 0; invalid_case < 3; ++invalid_case) {
        auto invalid = snapshot;
        if (invalid_case == 0)
            --invalid.ns;
        else if (invalid_case == 1)
            --invalid.mnmax;
        else
            invalid.families[EquilibriumSnapshot::RMNCC].pop_back();
        bool rejected = false;
        try {
            auto unused = restart_state(p, vp, invalid, false);
        } catch (const CumesError&) { rejected = true; }
        check(
            rejected,
            "reference: reject mismatched or incomplete restart before upload");
    }
    auto replay = restart_state(p, vp, snapshot, false);
    auto restored = snapshot_from_device(replay);
    check(seeded.radius_references().size() == std::size_t(p.ntor + 1),
          "reference: float fixed 3D uses m=0 reference");
    for (int n = 0; n <= p.ntor; ++n)
        for (int j = 0; j < p.ns; ++j)
            check(snapshot.families[0][n * p.ns + j] ==
                      restored.families[0][n * p.ns + j],
                  "reference: physical checkpoint recentering preserves m=0 "
                  "state");
}

static void test_reference_cache() {
    auto vp = load_validated("inputs/w7x.json");
    auto p = init_params<float>(vp);
    p.ns = 9;
    auto state = init_state(p, vp, false, false);
    stage_detail::ScopedRealSpace<float> rs(p, std::nullopt);
    stage_detail::ScopedModeTable<float> mt(p, std::nullopt);
    ToroidalFftOperator<float> transform(p, *rs, mt.get());
    Stream stream;
    transform.bind_stream(stream.get());
    auto read_radius = [&]() {
        stream.synchronize();
        std::vector<float> values(p.ns * p.nZnT);
        cumes::check_cuda(
            cudaMemcpy(values.data(), rs->d_r_real,
                       values.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "read cached R");
        return values;
    };
    auto capture = [&]() {
        cudaGraph_t graph{};
        cumes::check_cuda(
            cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeGlobal),
            "capture reference inverse");
        transform.inverse(state.physical_const(), true, stream.get());
        cumes::check_cuda(cudaStreamEndCapture(stream.get(), &graph),
                          "end reference inverse capture");
        return graph;
    };
    transform.inverse(state.physical_const(), true, stream.get());
    auto expected = read_radius();
    cudaGraph_t uncached = capture();
    transform.prepare_radius_reference(state.physical_const(), stream.get());
    cudaGraph_t cached = capture();
    std::size_t uncached_nodes = 0, cached_nodes = 0;
    cumes::check_cuda(cudaGraphGetNodes(uncached, nullptr, &uncached_nodes),
                      "count uncached nodes");
    cumes::check_cuda(cudaGraphGetNodes(cached, nullptr, &cached_nodes),
                      "count cached nodes");
    check(cached_nodes + 1 == uncached_nodes,
          "reference cache removes reconstruction from the iteration graph");
    cudaGraphExec_t executable{};
    cumes::check_cuda(
        cudaGraphInstantiate(&executable, cached, nullptr, nullptr, 0),
        "instantiate cached inverse");
    cumes::check_cuda(cudaGraphLaunch(executable, stream.get()),
                      "replay cached inverse");
    check(read_radius() == expected, "reference cache preserves inverse bits");
    cumes::check_cuda(cudaGraphExecDestroy(executable),
                      "destroy cached executable");
    cumes::check_cuda(cudaGraphDestroy(cached), "destroy cached graph");
    cumes::check_cuda(cudaGraphDestroy(uncached), "destroy uncached graph");

    // A different immutable reference must not reuse the prepared one, even
    // if its displacement coefficients are identical. Switching back must
    // reconstruct the original reference after that fallback overwrites it.
    std::vector<double> reference(state.radius_references().begin(),
                                  state.radius_references().end());
    reference[1] += 0.125;
    SpectralStorage<float> other(p.ns, p.mnmax, reference);
    cumes::check_cuda(cudaMemcpy(other.state_slab(), state.state_slab(),
                                 6 * p.ns * p.mnmax * sizeof(float),
                                 cudaMemcpyDeviceToDevice),
                      "copy displacement state");
    transform.inverse(other.physical_const(), true, stream.get());
    check(read_radius() != expected, "different reference bypasses cache");
    transform.inverse(state.physical_const(), true, stream.get());
    check(read_radius() == expected, "fallback invalidates overwritten cache");

    // Caller-owned reference output must still be populated by the view API.
    transform.prepare_radius_reference(state.physical_const(), stream.get());
    DeviceBuffer<float> d_reference(p.nZnT);
    auto geom = geometry_parity_views(*rs, p);
    geom.r_reference =
        RealFieldView<float>(d_reference.data(), 1, p.ntheta, p.nzeta);
    transform.enqueue_inverse(state.physical_const(), geom, {}, {},
                              stream.get());
    stream.synchronize();
    std::vector<float> own(p.nZnT), alternate(p.nZnT);
    cumes::check_cuda(
        cudaMemcpy(own.data(), rs->d_r_reference, own.size() * sizeof(float),
                   cudaMemcpyDeviceToHost),
        "read owned reference");
    cumes::check_cuda(
        cudaMemcpy(alternate.data(), d_reference.data(),
                   alternate.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "read alternate reference");
    check(own == alternate, "reference cache honors non-aliasing output views");
}

int main() {
    test_small_radial_variation();
    test_seed_restart();
    test_reference_cache();
    return summary();
}
