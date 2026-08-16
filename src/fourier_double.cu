#include "fourier_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template cumes::DeviceModeTable cumes::modeTableCreate<double>(const GridParams<double>&, cumes::DeviceArena*);
template FourierPlan<double> fourierCreate<double>(const GridParams<double>&, cumes::DeviceArena*);
template void fourierFree<double>(FourierPlan<double>&);
template cumes::RealSpaceStorage<double> realSpaceCreate<double>(const GridParams<double>&, cumes::DeviceArena*);
template void realSpaceFree<double>(cumes::RealSpaceStorage<double>&);
template void inverseDFT<double>(const FourierPlan<double>&, cumes::RealSpaceStorage<double>&, cumes::SpectralView<const double, cumes::PhysicalStateDomain>, const GridParams<double>&, const int*, const int*, bool, cudaStream_t);
template void inverseDFTFused<double>(const FourierPlan<double>&, cumes::RealSpaceStorage<double>&, cumes::SpectralView<const double, cumes::PhysicalStateDomain>, const GridParams<double>&, const int*, const int*, bool, double*, double*, cudaStream_t);
template void forwardDFT<double>(const FourierPlan<double>&, cumes::RealSpaceStorage<double>&, cumes::SpectralView<double, cumes::DecomposedResidualDomain>, const GridParams<double>&, const int*, const int*, const ConstraintWorkspace<double>&, cudaStream_t);
template void fourierCombineParity<double>(const FourierPlan<double>&, cumes::RealSpaceStorage<double>&, const GridParams<double>&, cudaStream_t);

// ToroidalFftOperator (owns the FourierPlan; wraps inverse/forward).
template class cumes::ToroidalFftOperator<double>;
