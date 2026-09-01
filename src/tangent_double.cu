// Explicit CUDA instantiations used by the precise-double forward-tangent
// path. Linear transforms remain ordinary double cuFFT operations; the dual
// scalar is used only by nonlinear pointwise/reduction operators.
#include "cumes/numerics/forward_dual.cuh"
#include "kernels/forces_impl.cuh"
#include "kernels/fourier_impl.cuh"
#include "kernels/geometry_impl.cuh"
#include "kernels/profiles_impl.cuh"
#include "kernels/tangent_impl.cuh"

using TangentDual = cumes::ForwardDual<double>;

template cumes::RealSpaceStorage<TangentDual> real_space_create<TangentDual>(
    const DeviceParams<TangentDual>&,
    const std::optional<std::reference_wrapper<cumes::DeviceArena>>&);
template void real_space_free<TangentDual>(
    cumes::RealSpaceStorage<TangentDual>&);
template class cumes::Profiles<TangentDual>;
template cumes::GeometryOperator<TangentDual>::GeometryOperator(
    const DeviceParams<TangentDual>&,
    const std::optional<std::reference_wrapper<cumes::DeviceArena>>&);
template cumes::GeometryOperator<TangentDual>::~GeometryOperator();
template void cumes::GeometryOperator<TangentDual>::enqueue(
    const cumes::RealSpaceStorage<TangentDual>&,
    const DeviceParams<TangentDual>&,
    const cumes::RadialProfileViews<TangentDual>&,
    cudaStream_t);
template cumes::BaseGeometryHalfViews<TangentDual> cumes::GeometryOperator<
    TangentDual>::base_geometry_views(const DeviceParams<TangentDual>&) const;
template cumes::MagneticFieldViews<TangentDual> cumes::GeometryOperator<
    TangentDual>::magnetic_field_views(const DeviceParams<TangentDual>&) const;
template class cumes::MagneticFieldOperator<TangentDual>;
template class cumes::ForceOperator<TangentDual>;
