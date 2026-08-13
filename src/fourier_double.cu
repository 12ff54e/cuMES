#include "fourier_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template FourierPlan<double> fourierCreate<double>(const GridParams<double>&);
template void fourierFree<double>(FourierPlan<double>&);
template void inverseDFT<double>(const FourierPlan<double>&, const SpectralState<double>&, const GridParams<double>&, bool);
template void forwardDFT<double>(const FourierPlan<double>&, double*, const GridParams<double>&, const ConstraintWorkspace<double>&);
template void fourierCombineParity<double>(const FourierPlan<double>&, const GridParams<double>&);
