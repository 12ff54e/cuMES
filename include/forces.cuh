// forces.cuh — MHD force residuals in real space with even/odd parity.
#pragma once
#include "vmec_types.h"
#include "fourier.cuh"
#include "geometry.cuh"

// Evaluate MHD force balance residuals in real space.
// Computes even/odd parity force components (armn_e/o, azmn_e/o, etc.)
// stored in FourierPlan scratch arrays.
void computeForces(const FourierPlan& fp, const GridParams& p,
                   const RadialProfiles& rp, const MetricWorkspace& mw);
