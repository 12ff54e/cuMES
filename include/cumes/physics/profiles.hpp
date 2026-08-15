// profiles.hpp — radial profile boundary (blueprint §6.4, §6.7).
//
// Profiles are immutable prescribed data evaluated on the HOST and uploaded once
// per stage (they do not change across iterations), so unlike the enqueue-only
// device operators they are a host-side build step. The legacy profilesCreate
// (src/profiles_impl.cuh) is the reference implementation; this header names the
// typed boundary and separates the immutable prescribed data (evaluated here)
// from the geometry-dependent evolving quantities that live in the metric/field
// operators.
#pragma once

#include "cumes/state/real_fields.cuh"
#include "vmec_types.h"

namespace cumes {

// The host-side profile build: fills the 11 radial arrays (4 full-grid, 7
// half-grid) from the validated input, and sets the lambda scale. Returns the
// immutable RadialProfileViews consumed by the geometry/B operators.
template <class T>
struct RadialProfilesResult {
  RadialProfileViews<T> views;
  T delta_s = T(0);
  T lamscale = T(0);
};

}  // namespace cumes
