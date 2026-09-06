#include "cumes/io/checkpoint.hpp"
#include "cumes/io/snapshot_bridge.cuh"
#include "cumes/runtime/stream.hpp"
#include "cumes/solver/stage_solver.hpp"
#include "cumes/state/seed_state.hpp"
#include "cumes_test_cuda_helper.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string_view>
#include <vector>

using namespace cumes;
using namespace cumes::test;

template <class T>
std::vector<T> download(const T* d_data, std::size_t size) {
    std::vector<T> result(size);
    cumes::check_cuda(cudaMemcpy(result.data(), d_data, size * sizeof(T),
                                 cudaMemcpyDeviceToHost),
                      "odd geometry download");
    return result;
}

// Independent host direct product-basis sum, retaining extra precision through
// both angular directions and the odd normalization. Input is exactly float.
std::array<std::vector<double>, 2> reference(
    const DeviceParams<float>& p,
    const EquilibriumSnapshot& snapshot) {
    std::array<std::vector<double>, 2> result;
    for (auto& field : result) field.resize(p.ns * p.nZnT);
    const long double pi = std::acos(-1.0L);
    for (int k = 0; k < p.nzeta; ++k)
        for (int l = 0; l < p.ntheta; ++l) {
            std::vector<long double> rb(p.mnmax), rsb(p.mnmax), zb(p.mnmax),
                zsb(p.mnmax);
            for (int m = 1; m < p.mpol; m += 2)
                for (int n = 0; n <= p.ntor; ++n) {
                    int mode = m * (p.ntor + 1) + n;
                    long double u = 2 * pi * m * l / p.ntheta;
                    long double v = 2 * pi * n * k / p.nzeta;
                    rb[mode] = std::cos(u) * std::cos(v);
                    rsb[mode] = std::sin(u) * std::sin(v);
                    zb[mode] = std::sin(u) * std::cos(v);
                    zsb[mode] = std::cos(u) * std::sin(v);
                }
            for (int j = 0; j < p.ns; ++j) {
                long double r = 0, z = 0;
                for (int m = 1; m < p.mpol; m += 2)
                    for (int n = 0; n <= p.ntor; ++n) {
                        int mode = m * (p.ntor + 1) + n;
                        int q = mode * p.ns + j;
                        r += snapshot.families[0][q] * rb[mode] +
                             snapshot.families[3][q] * rsb[mode];
                        z += snapshot.families[1][q] * zb[mode] +
                             snapshot.families[4][q] * zsb[mode];
                    }
                long double scale = std::sqrt(
                    static_cast<long double>(p.ns - 1) / std::max(j, 1));
                int i = j * p.nZnT + k * p.ntheta + l;
                result[0][i] = double(r * scale);
                result[1][i] = double(z * scale);
            }
        }
    return result;
}

int main(int argc, char** argv) {
    bool benchmark = argc > 1 && std::string_view(argv[1]) == "--benchmark";
    auto vp = load_validated("inputs/w7x.json");
    auto p = init_params<float>(vp, true);
    p.ns = benchmark ? 99 : 9;
    auto state = init_state(p, vp, false, false);
    if (argc > 2) {
        auto checkpoint = read_checkpoint(argv[2]);
        check(checkpoint.has_value(), "odd geometry checkpoint readable");
        if (!checkpoint.has_value()) return 1;
        state = restart_state(p, vp, checkpoint.value(), false);
    }
    auto expected = reference(p, snapshot_from_device(state));
    constexpr std::array policies{OddGeometryPrecision::NATIVE,
                                  OddGeometryPrecision::FLOAT_ORDER,
                                  OddGeometryPrecision::SUM,
                                  OddGeometryPrecision::POLOIDAL,
                                  OddGeometryPrecision::POLOIDAL_SCALE,
                                  OddGeometryPrecision::FLOAT_FLOAT,
                                  OddGeometryPrecision::DOUBLE};
    constexpr std::array names{"native",   "float-order",    "sum",
                               "poloidal", "poloidal-scale", "float-float",
                               "double"};
    std::vector<float> baseline_ru;
    std::array<double, 2> baseline_radial{};
    Stream stream;
    for (std::size_t v = 0; v < policies.size(); ++v) {
        p.odd_geometry = policies[v];
        stage_detail::ScopedRealSpace<float> rs(p, std::nullopt);
        stage_detail::ScopedModeTable<float> mt(p, std::nullopt);
        ToroidalFftOperator<float> transform(p, *rs, mt.get());
        transform.bind_stream(stream.get());
        transform.inverse(state.physical_const(), true, stream.get());
        cumes::check_cuda(cudaStreamSynchronize(stream.get()),
                          "odd geometry initial inverse");
        auto ru = download(rs->d_ru_o, p.ns * p.nZnT);
        if (v == 0) baseline_ru = ru;
        check(ru == baseline_ru,
              "odd geometry leaves angular derivatives unchanged");
        std::array fields{download(rs->d_r_o, p.ns * p.nZnT),
                          download(rs->d_z_o, p.ns * p.nZnT)};
        std::printf("%s", names[v]);
        for (int c = 0; c < 2; ++c) {
            double error2 = 0, radial2 = 0, worst = 0;
            int differing = 0;
            for (int i = 0; i < p.ns * p.nZnT; ++i) {
                double error = fields[c][i] - expected[c][i];
                error2 += error * error;
                worst = std::max(worst, std::abs(error));
                differing += std::abs(expected[c][i]) > 1e-12 &&
                             fields[c][i] != float(expected[c][i]);
                if (i >= p.nZnT) {
                    double radial = (error - (fields[c][i - p.nZnT] -
                                              expected[c][i - p.nZnT])) *
                                    (p.ns - 1);
                    radial2 += radial * radial;
                }
            }
            if (v == 0) baseline_radial[c] = radial2;
            if (policies[v] == OddGeometryPrecision::POLOIDAL ||
                policies[v] == OddGeometryPrecision::POLOIDAL_SCALE)
                check(radial2 < 0.95 * baseline_radial[c],
                      "poloidal compensation reduces radial difference error");
            std::printf(
                " %c_rms=%.9g %c_radial_rms=%.9g %c_max=%.9g "
                "rounded_mismatch=%d",
                c == 0 ? 'r' : 'z', std::sqrt(error2 / fields[c].size()),
                c == 0 ? 'r' : 'z', std::sqrt(radial2 / ((p.ns - 1) * p.nZnT)),
                c == 0 ? 'r' : 'z', worst, differing);
            if (policies[v] == OddGeometryPrecision::FLOAT_FLOAT ||
                policies[v] == OddGeometryPrecision::DOUBLE) {
                check(
                    worst < 1e-7,
                    "accurate odd positions agree with host direct reference");
                // Values arbitrarily close to a rounding midpoint may differ
                // by one ULP; mathematical zeros are also trig-table sensitive.
                check(
                    differing < int(fields[c].size() / 500),
                    "accurate odd positions are almost all correctly rounded");
            }
        }
        // Explicit nondefault-stream graph capture also verifies no hot-loop
        // allocation and replay of the caller-supplied geometry views.
        cudaGraph_t graph{};
        cudaGraphExec_t executable{};
        cumes::check_cuda(
            cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeGlobal),
            "odd capture");
        transform.enqueue_inverse(state.physical_const(),
                                  geometry_parity_views(*rs, p), {}, {},
                                  stream.get());
        cumes::check_cuda(cudaStreamEndCapture(stream.get(), &graph),
                          "odd end capture");
        cumes::check_cuda(
            cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
            "odd instantiate");
        for (int i = 0; i < (benchmark ? 3000 : 30); ++i)
            cumes::check_cuda(cudaGraphLaunch(executable, stream.get()),
                              "odd warmup");
        cudaEvent_t start{}, stop{};
        cumes::check_cuda(cudaEventCreate(&start), "odd event start");
        cumes::check_cuda(cudaEventCreate(&stop), "odd event stop");
        std::vector<float> timings;
        for (int sample = 0; sample < (benchmark ? 15 : 1); ++sample) {
            cumes::check_cuda(cudaEventRecord(start, stream.get()),
                              "odd start");
            for (int i = 0; i < 100; ++i)
                cumes::check_cuda(cudaGraphLaunch(executable, stream.get()),
                                  "odd replay");
            cumes::check_cuda(cudaEventRecord(stop, stream.get()), "odd stop");
            cumes::check_cuda(cudaEventSynchronize(stop), "odd sync");
            float ms = 0;
            cumes::check_cuda(cudaEventElapsedTime(&ms, start, stop),
                              "odd elapsed");
            timings.push_back(ms * 10.0F);
        }
        std::sort(timings.begin(), timings.end());
        std::printf(" inverse_graph_us=%.6f\n", timings[timings.size() / 2]);
        check(download(rs->d_r_o, p.ns * p.nZnT) == fields[0],
              "odd graph matches direct inverse");
        cumes::check_cuda(cudaEventDestroy(start), "odd destroy start");
        cumes::check_cuda(cudaEventDestroy(stop), "odd destroy stop");
        cumes::check_cuda(cudaGraphExecDestroy(executable),
                          "odd destroy executable");
        cumes::check_cuda(cudaGraphDestroy(graph), "odd destroy graph");
    }
    return summary();
}
