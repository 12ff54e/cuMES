// real_fields.cuh — typed real-space view aggregates (blueprint §6.3, §6.7).
//
// The spectral side already has SpectralView (component-major, domain-tagged).
// This header adds the real-space side: the parity-split geometry/force views,
// the half-grid metric/field views, the radial-profile views, and the distinct
// reduced-theta quadrature view. Each aggregate groups the individual
// RealFieldView<T>s an operator boundary consumes, so a kernel or operator
// signature names one typed bundle instead of a dozen raw pointers.
//
// Views never allocate or free; they carry a raw pointer plus the extents
// needed to index it on host or device. They are constructed from the owning
// raw-pointer bundles (RealSpaceStorage/the operator workspaces) at the
// operator boundary; the layout is identical, so indexing is bit-for-bit the
// legacy `surface*nZnT + zeta*ntheta + theta` arithmetic.
#ifndef CUMES_INCLUDE_CUMES_STATE_REAL_FIELDS_CUH_
#define CUMES_INCLUDE_CUMES_STATE_REAL_FIELDS_CUH_

#include "cumes/config/device_params.hpp"
#include "cumes/core/tensor_view.cuh"

namespace cumes {

// Defined in real_space_storage.hpp (which includes this header back, so the
// shared view-bundle factory below only needs the forward declaration — its
// body is a template and instantiates where RealSpaceStorage is complete).
template <class T>
struct RealSpaceStorage;

// Reduced-theta quadrature view (blueprint §4.1): the compact [0, pi] poloidal
// subset used by the forward quadrature and the constraint bandpass. It is a
// DISTINCT type — never an integer reinterpretation of a full-grid view — so
// code cannot silently mix the full `ntheta` grid with the `ntheta/2+1`
// reduced grid. Layout: [surface][zeta][reduced_theta], reduced_theta
// contiguous.
template <class T>
class ReducedThetaView {
   public:
    __host__ __device__ ReducedThetaView() = default;
    __host__ __device__
    ReducedThetaView(T* data, int surfaces, int ntheta_reduced, int nzeta)
        : data_(data),
          surfaces_(surfaces),
          ntheta_reduced_(ntheta_reduced),
          nzeta_(nzeta) {}

    __host__ __device__ T& operator()(int surface, int zeta, int theta_red) {
        return data_[surface * ntheta_reduced_ * nzeta_ +
                     zeta * ntheta_reduced_ + theta_red];
    }
    __host__ __device__ const T& operator()(int surface,
                                            int zeta,
                                            int theta_red) const {
        return data_[surface * ntheta_reduced_ * nzeta_ +
                     zeta * ntheta_reduced_ + theta_red];
    }

    __host__ __device__ T* data() const { return data_; }
    __host__ __device__ int surfaces() const { return surfaces_; }
    __host__ __device__ int ntheta_reduced() const { return ntheta_reduced_; }
    __host__ __device__ int nzeta() const { return nzeta_; }

   private:
    T* data_ = nullptr;
    int surfaces_ = 0;
    int ntheta_reduced_ = 0;
    int nzeta_ = 0;
};

// Full-grid parity-split geometry (inverse-DFT output): R, Z, lambda and their
// poloidal (u) / toroidal (v) derivatives, split by poloidal m-parity
// (e = even m, o = odd m). Matches the RealSpaceStorage field groups.
template <class T>
struct GeometryParityViews {
    RealFieldView<T> r_e, z_e, l_e;
    RealFieldView<T> ru_e, zu_e, lu_e;
    RealFieldView<T> r_o, z_o, l_o;
    RealFieldView<T> ru_o, zu_o, lu_o;
    RealFieldView<T> rv_e, zv_e, lv_e;
    RealFieldView<T> rv_o, zv_o, lv_o;
};

// Shared factory for the parity-split geometry view bundle (single definition;
// previously duplicated byte-identically in kernels/geometry_impl.cuh and
// kernels/forces_impl.cuh — review finding 4.2). Constructed at the operator
// boundary; the kernels then read the raw pointers back out of the bundles,
// keeping the flat `surface*nZnT + zeta*ntheta + theta` arithmetic bit-for-bit
// identical.
template <typename T>
inline GeometryParityViews<T> geometryParityViews(const RealSpaceStorage<T>& rs,
                                                  const DeviceParams<T>& p) {
    auto f = [&](T* d) { return RealFieldView<T>(d, p.ns, p.ntheta, p.nzeta); };
    GeometryParityViews<T> v;
    v.r_e = f(rs.d_r_e);
    v.z_e = f(rs.d_z_e);
    v.l_e = f(rs.d_l_e);
    v.ru_e = f(rs.d_ru_e);
    v.zu_e = f(rs.d_zu_e);
    v.lu_e = f(rs.d_lu_e);
    v.r_o = f(rs.d_r_o);
    v.z_o = f(rs.d_z_o);
    v.l_o = f(rs.d_l_o);
    v.ru_o = f(rs.d_ru_o);
    v.zu_o = f(rs.d_zu_o);
    v.lu_o = f(rs.d_lu_o);
    v.rv_e = f(rs.d_rv_e);
    v.zv_e = f(rs.d_zv_e);
    v.lv_e = f(rs.d_lv_e);
    v.rv_o = f(rs.d_rv_o);
    v.zv_o = f(rs.d_zv_o);
    v.lv_o = f(rs.d_lv_o);
    return v;
}

// Full-grid radial profile views (ns-length 1D arrays).
template <class T>
struct RadialProfileViews {
    T* iota_F;
    T* phip_F;
    T* chi_F;
    T* sqrtS_F;
    T* iota_H;
    T* pres_H;
    T* phip_H;
    T* dVds_H;
    T* sqrtS_H;
    T* curr_H;
    T* chip_H;
};

// Half-grid base geometry (geometry.cu output): parity-staggered R/Z, Jacobian
// and the covariant metric. All (ns-1, ntheta, nzeta).
template <class T>
struct BaseGeometryHalfViews {
    RealFieldView<T> r12, ru12, zu12, rs, zs, tau;
    RealFieldView<T> gsqrt, guu, guv, gvv;
};

// Half-grid magnetic field + total pressure (the ncurr=0/1 finalization).
template <class T>
struct MagneticFieldViews {
    RealFieldView<T> bsupu, bsupv;
    RealFieldView<T> bsubu, bsubv;
    RealFieldView<T> total_pressure;
};

// Full-grid force parity views (forces.cu output, forward-DFT input). The
// a/b/c families are the radial/poloidal/toroidal weak-form contributions.
template <class T>
struct ForceParityViews {
    RealFieldView<T> armn_e, armn_o, azmn_e, azmn_o;
    RealFieldView<T> brmn_e, brmn_o, bzmn_e, bzmn_o;
    RealFieldView<T> blmn_e, blmn_o, clmn_e, clmn_o;
    RealFieldView<T> crmn_e, crmn_o, czmn_e, czmn_o;
};

// Full-grid constraint-force views (constraint.cu frcon/fzcon outputs): the
// forward DFT adds xmpq[m]*frcon to the R cos-term and xmpq[m]*fzcon to the Z
// sin-term (vmecpp frcon/fzcon in AddConstraintForces). Split by poloidal
// m-parity like the MHD force families. A separate bundle from ForceParityViews
// because these come from the constraint operator, not the MHD force operator
// (blueprint §5.1 dependency rule: the forward operator consumes both).
template <class T>
struct ConstraintForceViews {
    RealFieldView<T> frcon_e, frcon_o, fzcon_e, fzcon_o;
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_STATE_REAL_FIELDS_CUH_
