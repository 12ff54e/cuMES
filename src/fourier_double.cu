#include "fourier_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template FourierPlan<double> fourierCreate<double>(const GridParams<double>&, cumes::DeviceArena*);
template void fourierFree<double>(FourierPlan<double>&);
template void inverseDFT<double>(const FourierPlan<double>&, cumes::SpectralView<const double, cumes::PhysicalStateDomain>, const GridParams<double>&, bool, cudaStream_t);
template void inverseDFTFused<double>(const FourierPlan<double>&, cumes::SpectralView<const double, cumes::PhysicalStateDomain>, const GridParams<double>&, bool, double*, double*, cudaStream_t);
template void forwardDFT<double>(const FourierPlan<double>&, cumes::SpectralView<double, cumes::DecomposedResidualDomain>, const GridParams<double>&, const ConstraintWorkspace<double>&, cudaStream_t);
template void fourierCombineParity<double>(const FourierPlan<double>&, const GridParams<double>&, cudaStream_t);
