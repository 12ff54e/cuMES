#ifndef CUMES_SRC_KERNELS_ODD_GEOMETRY_IMPL_CUH_
#define CUMES_SRC_KERNELS_ODD_GEOMETRY_IMPL_CUH_

#include "cumes/transforms/odd_geometry_operator.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

namespace cumes {
namespace odd_geometry_detail {
template <class A>
A split_constant(double value) {
    if constexpr (std::is_same_v<A, FloatFloat>)
        return FloatFloat::from_double(value);
    else
        return A(value);
}

template <class A>
void upload(DeviceBuffer<A>& buffer, const std::vector<A>& values) {
    buffer.allocate(values.size());
    check_cuda(cudaMemcpy(buffer.data(), values.data(),
                          values.size() * sizeof(A), cudaMemcpyHostToDevice),
               "odd geometry table");
}

// Four channels, only odd m. Surface-contiguous scratch/coefficient loads.
template <class A>
__global__ void toroidal_kernel(
    SpectralView<const float, PhysicalStateDomain> coeff,
    const A* d_cos,
    const A* d_sin,
    A* d_scratch,
    int ns,
    int ntor,
    int nzeta,
    int odd_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int count = odd_count * nzeta * ns;
    if (i >= count) return;
    int j = i % ns, k = (i / ns) % nzeta;
    int m = 2 * (i / (ns * nzeta)) + 1;
    A rcc{}, rss{}, zsc{}, zcs{};
    for (int n = 0; n <= ntor; ++n) {
        int mode = m * (ntor + 1) + n;
        A c = d_cos[n * nzeta + k], s = d_sin[n * nzeta + k];
        rcc = rcc + A(coeff(SpectralComponent::Rcc, mode, j)) * c;
        rss = rss + A(coeff(SpectralComponent::Rss, mode, j)) * s;
        zsc = zsc + A(coeff(SpectralComponent::Zsc, mode, j)) * c;
        zcs = zcs + A(coeff(SpectralComponent::Zcs, mode, j)) * s;
    }
    d_scratch[i] = rcc;
    d_scratch[count + i] = rss;
    d_scratch[2 * count + i] = zsc;
    d_scratch[3 * count + i] = zcs;
}

template <class A>
__global__ void poloidal_kernel(float* d_r_o,
                                float* d_z_o,
                                const A* d_scratch,
                                const A* d_cos,
                                const A* d_sin,
                                const A* d_scale,
                                int ns,
                                int mpol,
                                int ntheta,
                                int nzeta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nZnT = ntheta * nzeta;
    if (i >= ns * nZnT) return;
    int j = i / nZnT, k = (i % nZnT) / ntheta, l = i % ntheta;
    A r{}, z{};
    for (int m = 1; m < mpol; m += 2) {
        int count = (mpol / 2) * nzeta * ns;
        int q = ((m / 2) * nzeta + k) * ns + j;
        A rcc = d_scratch[q], rss = d_scratch[count + q];
        A zsc = d_scratch[2 * count + q], zcs = d_scratch[3 * count + q];
        A c = d_cos[(m / 2) * ntheta + l];
        A s = d_sin[(m / 2) * ntheta + l];
        r = r + (rcc * c + rss * s);
        z = z + (zsc * s + zcs * c);
    }
    d_r_o[i] = float(r * d_scale[j]);
    d_z_o[i] = float(z * d_scale[j]);
}
}  // namespace odd_geometry_detail

template <class A>
OddGeometryOperator<A>::OddGeometryOperator(const DeviceParams<float>& p)
    : p_(p) {
    using namespace odd_geometry_detail;
    bool full = p.odd_geometry == OddGeometryPrecision::FLOAT_FLOAT;
    std::vector<A> scale(p.ns);
    for (int j = 0; j < p.ns; ++j)
        scale[j] =
            split_constant<A>(std::sqrt(double(p.ns - 1) / std::max(j, 1)));
    upload(d_scale_, scale);
    if (!full) return;
    std::vector<A> cos_zeta((p.ntor + 1) * p.nzeta), sin_zeta(cos_zeta.size());
    for (int n = 0; n <= p.ntor; ++n)
        for (int k = 0; k < p.nzeta; ++k) {
            double angle = 2.0 * M_PI * n * k / p.nzeta;
            cos_zeta[n * p.nzeta + k] = split_constant<A>(std::cos(angle));
            sin_zeta[n * p.nzeta + k] = split_constant<A>(std::sin(angle));
        }
    upload(d_cos_zeta_, cos_zeta);
    upload(d_sin_zeta_, sin_zeta);
    std::vector<A> cos_theta((p.mpol / 2) * p.ntheta),
        sin_theta(cos_theta.size());
    for (int m = 1; m < p.mpol; m += 2)
        for (int l = 0; l < p.ntheta; ++l) {
            double angle = 2.0 * M_PI * m * l / p.ntheta;
            cos_theta[(m / 2) * p.ntheta + l] =
                split_constant<A>(std::cos(angle));
            sin_theta[(m / 2) * p.ntheta + l] =
                split_constant<A>(std::sin(angle));
        }
    upload(d_cos_theta_, cos_theta);
    upload(d_sin_theta_, sin_theta);
    d_scratch_.allocate(4 * (p.mpol / 2) * p.nzeta * p.ns);
}

template <class A>
void OddGeometryOperator<A>::enqueue(
    SpectralView<const float, PhysicalStateDomain> coeff,
    GeometryParityViews<float> geometry,
    cudaStream_t stream) {
    using namespace odd_geometry_detail;
    const auto& p = p_;
    int count = (p.mpol / 2) * p.nzeta * p.ns;
    if (count > 0)
        toroidal_kernel<A><<<(count + 255) / 256, 256, 0, stream>>>(
            coeff, d_cos_zeta_.data(), d_sin_zeta_.data(), d_scratch_.data(),
            p.ns, p.ntor, p.nzeta, p.mpol / 2);
    count = p.ns * p.nZnT;
    poloidal_kernel<A><<<(count + 255) / 256, 256, 0, stream>>>(
        geometry.r_o.data(), geometry.z_o.data(), d_scratch_.data(),
        d_cos_theta_.data(), d_sin_theta_.data(), d_scale_.data(), p.ns, p.mpol,
        p.ntheta, p.nzeta);
    check_cuda(cudaGetLastError(), "odd geometry reconstruction");
}
}  // namespace cumes
#endif
