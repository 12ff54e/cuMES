#include "fourier_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template FourierPlan<double> fourierCreate<double>(const GridParams<double>&, cumes::DeviceArena*);
template void fourierFree<double>(FourierPlan<double>&);
template void inverseDFT<double>(const FourierPlan<double>&, cumes::SpectralView<const double, cumes::PhysicalStateDomain>, const GridParams<double>&, bool);
template void forwardDFT<double>(const FourierPlan<double>&, cumes::SpectralView<double, cumes::DecomposedResidualDomain>, const GridParams<double>&, const ConstraintWorkspace<double>&);
template void fourierCombineParity<double>(const FourierPlan<double>&, const GridParams<double>&);
