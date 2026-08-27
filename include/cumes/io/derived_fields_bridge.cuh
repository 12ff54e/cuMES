// derived_fields_bridge.cuh — final GPU geometry pass -> host field inputs.
#ifndef CUMES_INCLUDE_CUMES_IO_DERIVED_FIELDS_BRIDGE_CUH_
#define CUMES_INCLUDE_CUMES_IO_DERIVED_FIELDS_BRIDGE_CUH_

#include "cumes/core/checked_size.hpp"
#include "cumes/io/derived_fields.hpp"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/state/real_space_storage.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <string_view>
#include <vector>

namespace cumes {
namespace derived_bridge_detail {

template <class T>
std::vector<double> copy_to_double(const T* device,
                                   std::size_t count,
                                   std::string_view label) {
    std::vector<T> native(count);
    if (count != 0) {
        const auto bytes = checked_mul(count, sizeof(T));
        if (!bytes)
            throw CumesError("derived field copy: byte count overflows");
        check_cuda(
            cudaMemcpy(native.data(), device, *bytes, cudaMemcpyDeviceToHost),
            label);
    }
    std::vector<double> result(count);
    for (std::size_t i = 0; i < count; ++i)
        result[i] = static_cast<double>(native[i]);
    return result;
}

}  // namespace derived_bridge_detail

// The caller must first run a final inverse/base-geometry/magnetic-field pass
// for the exact state being published and synchronize its stream. This bridge
// then performs only D2H copies and CPU post-processing; it launches no work
// and cannot perturb the solver trajectory.
template <class T>
Status capture_derived_fields(const DeviceParams<T>& p,
                              const Profiles<T>& profiles,
                              const RealSpaceStorage<T>& rs,
                              const GeometryOperator<T>& geometry,
                              EquilibriumSnapshot& snapshot) {
    const auto points = checked_mul(static_cast<std::size_t>(p.ntheta),
                                    static_cast<std::size_t>(p.nzeta));
    if (!points)
        return Status("derived field capture: angular extent overflows");
    const auto full = checked_mul(static_cast<std::size_t>(p.ns), *points);
    const auto half = checked_mul(static_cast<std::size_t>(p.ns - 1), *points);
    if (!full || !half)
        return Status("derived field capture: radial extent overflows");

    DerivedFieldInputs in;
    in.ns = p.ns;
    in.ntheta = p.ntheta;
    in.nzeta = p.nzeta;
    in.nfp = p.nfp;
    in.delta_s = 1.0 / static_cast<double>(p.ns - 1);
    in.mu0 = static_cast<double>(DeviceParams<T>::MU_0);

    const auto radial = profiles.profile_views();
    const auto base = geometry.base_geometry_views(p);
    const auto field = geometry.magnetic_field_views(p);
    const auto geometry_full = rs.geometry_views(p);
    using derived_bridge_detail::copy_to_double;

    in.sqrt_s_full =
        copy_to_double(radial.sqrtS_F, p.ns, "copy output sqrtS_F");
    in.sqrt_s_half =
        copy_to_double(radial.sqrtS_H, p.ns - 1, "copy output sqrtS_H");

    in.r_e = copy_to_double(geometry_full.r_e.data(), *full, "copy output r_e");
    in.r_o = copy_to_double(geometry_full.r_o.data(), *full, "copy output r_o");
    in.z_e = copy_to_double(geometry_full.z_e.data(), *full, "copy output z_e");
    in.z_o = copy_to_double(geometry_full.z_o.data(), *full, "copy output z_o");
    in.ru_e =
        copy_to_double(geometry_full.ru_e.data(), *full, "copy output ru_e");
    in.ru_o =
        copy_to_double(geometry_full.ru_o.data(), *full, "copy output ru_o");
    in.zu_e =
        copy_to_double(geometry_full.zu_e.data(), *full, "copy output zu_e");
    in.zu_o =
        copy_to_double(geometry_full.zu_o.data(), *full, "copy output zu_o");
    in.rv_e =
        copy_to_double(geometry_full.rv_e.data(), *full, "copy output rv_e");
    in.rv_o =
        copy_to_double(geometry_full.rv_o.data(), *full, "copy output rv_o");
    in.zv_e =
        copy_to_double(geometry_full.zv_e.data(), *full, "copy output zv_e");
    in.zv_o =
        copy_to_double(geometry_full.zv_o.data(), *full, "copy output zv_o");

    in.rs = copy_to_double(base.rs.data(), *half, "copy output rs");
    in.zs = copy_to_double(base.zs.data(), *half, "copy output zs");
    in.ru12 = copy_to_double(base.ru12.data(), *half, "copy output ru12");
    in.zu12 = copy_to_double(base.zu12.data(), *half, "copy output zu12");
    in.sqrtg = copy_to_double(base.gsqrt.data(), *half, "copy output sqrtg");
    in.bsupu = copy_to_double(field.bsupu.data(), *half, "copy output bsupu");
    in.bsupv = copy_to_double(field.bsupv.data(), *half, "copy output bsupv");
    in.bsubu = copy_to_double(field.bsubu.data(), *half, "copy output bsubu");
    in.bsubv = copy_to_double(field.bsubv.data(), *half, "copy output bsubv");

    return populate_derived_fields(in, snapshot);
}

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_IO_DERIVED_FIELDS_BRIDGE_CUH_
