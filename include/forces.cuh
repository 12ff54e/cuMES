// forces.cuh — MHD force residuals in real space with even/odd parity.
#pragma once
#include "vmec_types.h"
#include "fourier.cuh"
#include "geometry.cuh"

// Evaluate MHD force balance residuals in real space.
// Computes even/odd parity force components (armn_e/o, azmn_e/o, etc.)
// stored in FourierPlan scratch arrays.
template <typename T>
void computeForces(const FourierPlan<T>& fp, const GridParams<T>& p,
                   const RadialProfiles<T>& rp, const MetricWorkspace<T>& mw,
                   cudaStream_t stream = 0);

// §8.10 split prototype: the R/Z force families (armn/azmn/brmn/bzmn/crmn/czmn)
// in one kernel and the lambda force (blmn/clmn) in another, so neither holds
// both working sets live at once. `computeForces` (the monolith) remains the
// production/reference path; this is the comparison backend behind it.
template <typename T>
void computeForcesSplit(const FourierPlan<T>& fp, const GridParams<T>& p,
                        const RadialProfiles<T>& rp, const MetricWorkspace<T>& mw,
                        cudaStream_t stream = 0);
