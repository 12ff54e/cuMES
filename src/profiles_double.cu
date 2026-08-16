#include "profiles_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template RadialProfiles<double> profilesCreate<double>(DeviceParams<double>&, const cumes::ValidatedProblem&, cumes::DeviceArena*);
template void profilesFree<double>(RadialProfiles<double>&);
