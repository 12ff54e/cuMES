// forces.cuh — MHD force residuals in real space with even/odd parity.
#pragma once
#include "vmec_types.h"
#include "fourier.cuh"
#include "geometry.cuh"

// Evaluate MHD force balance residuals in real space.
// Computes even/odd parity force components (armn_e/o, azmn_e/o, etc.)
// stored in FourierPlan scratch arrays.
template <typename T>
void computeForces(const cumes::RealSpaceStorage<T>& rs, const DeviceParams<T>& p,
                   const RadialProfiles<T>& rp, const MetricWorkspace<T>& mw,
                   cudaStream_t stream = 0);
