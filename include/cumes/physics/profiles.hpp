// profiles.hpp — radial profile boundary (blueprint §6.4, §6.7).
//
// Profiles are immutable prescribed data evaluated on the HOST and uploaded once
// per stage (they do not change across iterations), so unlike the enqueue-only
// device operators they are a host-side build step. The legacy profilesCreate
// (src/profiles_impl.cuh) is the reference implementation; this operator owns
// the RadialProfiles workspace it builds.
#pragma once

#include "cumes/state/real_fields.cuh"
#include "input.h"
#include "profiles.cuh"

namespace cumes {

class DeviceArena;

template <class T>
class Profiles {
 public:
  Profiles(GridParams<T>& p, const InputParams& ip, DeviceArena* arena)
      : rp_(profilesCreate(p, ip, arena)) {}
  ~Profiles() { profilesFree(rp_); }

  Profiles(const Profiles&) = delete;
  Profiles& operator=(const Profiles&) = delete;
  Profiles(Profiles&&) noexcept = default;
  Profiles& operator=(Profiles&&) noexcept = default;

  const RadialProfiles<T>& workspace() const { return rp_; }

 private:
  RadialProfiles<T> rp_;
};

}  // namespace cumes
