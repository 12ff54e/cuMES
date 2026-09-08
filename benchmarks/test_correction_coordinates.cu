#include "correction_coordinates.cuh"
#include "cumes/numerics/descent_operator.hpp"
#include "cumes/runtime/device_buffer.cuh"
#include "cumes/runtime/stream.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

template <class T>
void check_map(bool fix_m1, T delta_t) {
    DeviceParams<T> p{};
    p.ns = 5;
    p.mpol = 4;
    p.ntor = 2;
    p.mnmax = p.mpol * (p.ntor + 1);
    const int count = p.ns * p.mnmax, size = 6 * count;
    std::vector<T> base(size), q(size), packed(size), got(size), expected(size);
    std::vector<int> xm(p.mnmax), xn(p.mnmax);
    for (int mode = 0; mode < p.mnmax; ++mode) {
        xm[mode] = mode / (p.ntor + 1);
        xn[mode] = mode % (p.ntor + 1);
    }
    for (int i = 0; i < size; ++i) {
        base[i] = i % 11 == 0 ? -T(0) : T(0.0625 * (i % 31 - 15));
        q[i] = T(0.125 * (1 + i % 13));
    }
    cumes::DeviceBuffer<T> d_base(size), d_q(size), d_got(size),
        d_expected(size);
    cumes::DeviceBuffer<T> d_velocity(size);
    cumes::DeviceBuffer<T> d_scale(1), d_physical(size);
    cumes::DeviceBuffer<int> d_xm(p.mnmax), d_xn(p.mnmax);
    cumes::Stream stream;
    auto upload = [&](auto& buffer, const auto& values) {
        cumes::check_cuda(
            cudaMemcpyAsync(buffer.data(), values.data(), buffer.byte_size(),
                            cudaMemcpyHostToDevice, stream.get()),
            "coordinate test upload");
    };
    upload(d_base, base);
    upload(d_q, q);
    upload(d_xm, xm);
    upload(d_xn, xn);
    const std::array<T, 1> scale{delta_t * delta_t};
    upload(d_scale, scale);
    d_expected.copy_from_async(d_base, stream.get());
    d_got.copy_from_async(d_base, stream.get());
    d_velocity.zero_async(stream.get());
    block_bench::CorrectionCoordinates<T> coordinates(p, !fix_m1);
    coordinates.set_fix_m1(fix_m1);
    coordinates.enqueue_pack(d_q.data(), d_q.data(), stream.get());
    cumes::check_cuda(
        cudaMemcpyAsync(packed.data(), d_q.data(), d_q.byte_size(),
                        cudaMemcpyDeviceToHost, stream.get()),
        "packed coordinates");
    coordinates.enqueue_trial(d_got.data(), d_q.data(), d_got.data(), T(1),
                              stream.get(), d_scale.data());
    cumes::DescentAction action;
    action.perform_descent = true;
    action.delta_t = delta_t;
    action.damping_b1 = 0;
    action.damping_fac = 1;
    cumes::DescentOperator<T>{}.enqueue(
        {d_expected.data(), p.ns, p.mnmax}, {d_velocity.data(), p.ns, p.mnmax},
        {d_q.data(), p.ns, p.mnmax}, d_xm.data(), d_xn.data(), p.ns, p.mnmax,
        action, stream.get());
    cumes::check_cuda(
        cudaMemcpyAsync(got.data(), d_got.data(), d_got.byte_size(),
                        cudaMemcpyDeviceToHost, stream.get()),
        "coordinate result");
    cumes::check_cuda(cudaMemcpyAsync(expected.data(), d_expected.data(),
                                      d_expected.byte_size(),
                                      cudaMemcpyDeviceToHost, stream.get()),
                      "descent result");
    stream.synchronize();
    std::array<int, 6> active_counts{};
    for (int i = 0; i < size; ++i)
        active_counts[i / count] += packed[i] != T(0);
    // Analytic mode/surface counts for ns=5, m=0..3, n=0..2. Fixing the
    // m=1 mixed gauge removes 2 toroidal modes * 3 interior surfaces.
    const std::array<int, 6> counts{39, 27, 36, 18, fix_m1 ? 20 : 26, 32};
    if (active_counts != counts)
        throw std::runtime_error("incorrect active coordinate partition");
    T error = T(0);
    for (int i = 0; i < size; ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(expected[i]))
            throw std::runtime_error("nonfinite coordinate result");
        error = std::max(error, std::abs(got[i] - expected[i]));
        const int c = i / count, mode = (i % count) / p.ns, j = i % p.ns;
        if ((j == p.ns - 1 && c != 2 && c != 5) ||
            (j == 0 && (xm[mode] > 0 || c == 2 || c == 5)) ||
            (xm[mode] == 0 && (c == 1 || c == 2 || c == 3)) ||
            (xn[mode] == 0 && c >= 3)) {
            if (std::memcmp(&got[i], &base[i], sizeof(T)))
                throw std::runtime_error("protected coordinate changed bits");
        }
    }
    if (error > (std::is_same_v<T, float> ? T(1e-6) : T(2e-15)))
        throw std::runtime_error(
            "coordinate map differs from production descent");
    coordinates.enqueue_physical(d_q.data(), d_physical.data(), stream.get());
    coordinates.enqueue_physical(d_q.data(), d_q.data(), stream.get());
    cumes::check_cuda(cudaMemcpyAsync(got.data(), d_q.data(), d_q.byte_size(),
                                      cudaMemcpyDeviceToHost, stream.get()),
                      "in-place direction");
    cumes::check_cuda(cudaMemcpyAsync(expected.data(), d_physical.data(),
                                      d_physical.byte_size(),
                                      cudaMemcpyDeviceToHost, stream.get()),
                      "physical direction");
    stream.synchronize();
    if (std::memcmp(got.data(), expected.data(), d_q.byte_size()))
        throw std::runtime_error(
            "physical coordinate mapping cannot run in place");
    std::printf("%s fix_m1=%d delta_t=%.2f production descent error %.3e\n",
                std::is_same_v<T, float> ? "float" : "double", fix_m1,
                double(delta_t), double(error));
}

int main() {
    try {
        for (double delta_t : {1.0, 0.5}) {
            check_map<double>(true, delta_t);
            check_map<double>(false, delta_t);
            check_map<float>(true, float(delta_t));
            check_map<float>(false, float(delta_t));
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::puts("correction coordinates: PASS");
}
