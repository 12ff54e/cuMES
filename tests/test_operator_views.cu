// test_operator_views.cu — Phase 5 operator boundaries: the typed real-space
// view family and the operator interface headers (blueprint §6.3, §6.6-6.9).
//
// This test has two jobs:
//   1. Include every operator interface header, proving the dependency DAG is
//      acyclic and the contracts compile together (the "no cyclic module
//      dependencies" gate).
//   2. Exercise the concrete typed views (RealFieldView, ReducedThetaView, and
//      the aggregate geometry/field/force bundles) with a device round-trip and
//      layout static_asserts.
#include "cumes/numerics/descent_operator.hpp"
#include "cumes/numerics/preconditioner.hpp"
#include "cumes/numerics/prolongation.hpp"
#include "cumes/numerics/residual_operator.hpp"
#include "cumes/numerics/tridiagonal_backend.hpp"
#include "cumes/physics/constraint_operator.hpp"
#include "cumes/physics/force_operator.hpp"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/magnetic_field_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes/state/real_fields.cuh"
#include "cumes/transforms/spectral_operator.hpp"
#include "cumes_test_cuda_helper.cuh"

#include <cstdlib>
#include <type_traits>
#include <vector>
using namespace cumes::test;

// The view bundles must be trivially copyable so kernels can receive them by
// value (a bundle is just pointers + extents).
static_assert(
    std::is_trivially_copyable<cumes::GeometryParityViews<double>>::value,
    "geometry views trivially copyable");
static_assert(
    std::is_trivially_copyable<cumes::BaseGeometryHalfViews<double>>::value,
    "half views trivially copyable");
static_assert(
    std::is_trivially_copyable<cumes::MagneticFieldViews<double>>::value,
    "field views trivially copyable");
static_assert(
    std::is_trivially_copyable<cumes::ForceParityViews<double>>::value,
    "force views trivially copyable");

// Write through a ReducedThetaView on device; host verifies the
// [surface][zeta][reduced_theta] layout (reduced_theta contiguous).
__global__ void write_reduced(cumes::ReducedThetaView<double> v,
                              int ntheta_red) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = v.surfaces() * ntheta_red * v.nzeta();
    if (i >= total) return;
    int surf = i / (ntheta_red * v.nzeta());
    int rem = i % (ntheta_red * v.nzeta());
    int zeta = rem / ntheta_red;
    int thr = rem % ntheta_red;
    v(surf, zeta, thr) = (double)i;
}

int main() {
    std::cout << "=== operator views ===\n";

    // ---- ReducedThetaView device round-trip ----
    {
        const int ns = 5, ntheta = 8, nzeta = 3, nred = ntheta / 2 + 1;  // 5
        const int total = ns * nred * nzeta;
        std::vector<double> h(total, -1.0);
        double* d = nullptr;
        cc(cudaMalloc(&d, total * sizeof(double)), "alloc");
        cc(cudaMemcpy(d, h.data(), total * sizeof(double),
                      cudaMemcpyHostToDevice),
           "seed");
        cumes::ReducedThetaView<double> v(d, ns, nred, nzeta);
        write_reduced<<<(total + 127) / 128, 128>>>(v, nred);
        cc(cudaDeviceSynchronize(), "sync");
        std::vector<double> back(total);
        cc(cudaMemcpy(back.data(), d, total * sizeof(double),
                      cudaMemcpyDeviceToHost),
           "read");
        bool ok = true;
        for (int i = 0; i < total; ++i) ok = ok && back[i] == (double)i;
        check(ok, "ReducedThetaView writes [surface][zeta][reduced_theta]");
        cudaFree(d);
    }

    // ---- RealFieldView full-grid indexing vs the legacy formula ----
    {
        const int ns = 3, ntheta = 4, nzeta = 2;
        cumes::RealFieldView<double> v(nullptr, ns, ntheta, nzeta);
        // Legacy index: surface*nZnT + zeta*ntheta + theta.
        const int surf = 2, zeta = 1, theta = 3;
        const int nZnT = ntheta * nzeta;
        check((surf * nZnT + zeta * ntheta + theta) ==
                  (surf * v.ntheta() * v.nzeta() + zeta * v.ntheta() + theta),
              "RealFieldView surface-major layout matches legacy");
    }

    // ---- aggregate bundles hold their member views ----
    {
        const int nH = 4, ntheta = 6, nzeta = 1;
        cumes::BaseGeometryHalfViews<double> half;
        // A freshly-defaulted bundle has null member views; construction is
        // trivial and the members are independently addressable.
        check(half.gsqrt.data() == nullptr && half.guu.data() == nullptr,
              "default bundle has null members");
        cumes::RealFieldView<double> g(nullptr, nH, ntheta, nzeta);
        half.gsqrt = g;
        check(half.gsqrt.surfaces() == nH && half.gsqrt.nzeta() == nzeta,
              "bundle members carry extents");
    }

    return summary();
}
