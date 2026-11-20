// Common cell trace. The incidence table supplies the two oriented edges of each curl.
#pragma once

#include "yee_primitives.cuh"

namespace beamz::cuda::trace_native {

template<int Phase, int Component, int Edge, class Environment, class Read>
__device__ __forceinline__ float CorrectedDifference(Environment& env, Read read,
    int version, int z, int y, int x) {
  constexpr auto term = Edge == 0 ? yee::FirstCurlTerm(Component) : yee::SecondCurlTerm(Component);
  constexpr int axis = term.derivative_axis, source = (Phase == 0 ? 3 : 0) + term.source_component;
  constexpr int slot = 2 * (3 * Phase + Component) + Edge;
  constexpr int direction = Phase == 0 ? 1 : -1;
  const int time = Phase == 0 ? version - 1 : version;
  const float delta = env.Difference(Phase, source, axis, z, y, x, read(source, time, 0, 0, 0),
      read(source, time, (axis == 0) * direction, (axis == 1) * direction, (axis == 2) * direction));
  if (!env.LocalActive(slot, z, y, x)) return delta;
  const float next = yee::AdvanceCpmlPsi(env.Coefficient(3, slot, z, y, x),
      env.LoadLocal(slot, z, y, x), env.Coefficient(2, slot, z, y, x), delta);
  const float corrected = __fmaf_rn(delta, env.Coefficient(4, slot, z, y, x), next);
  env.StoreLocal(slot, z, y, x, next);
  return corrected;
}

template<int Phase, int Component, bool CheckSupport = true, bool UnitDecay = false,
         class Environment, class Read>
__device__ __forceinline__ float AdvanceNativeCell(Environment& env, Read read,
    int version, int z, int y, int x) {
  static_assert((Phase == 0 || Phase == 1) && Component >= 0 && Component < 3);
  constexpr int field = 3 * Phase + Component;
  if constexpr (CheckSupport) {
    if (!env.Valid(field, z, y, x)) return 0.f;
  }
  const float first = CorrectedDifference<Phase, Component, 0>(env, read, version, z, y, x);
  const float second = CorrectedDifference<Phase, Component, 1>(env, read, version, z, y, x);
  const float curl = __fsub_rn(first, second);
  const float source = env.Coefficient(1, field, z, y, x);
  const float decay = UnitDecay ? 1.f : env.Coefficient(0, field, z, y, x);
  const float value = yee::AdvanceYeeField<UnitDecay>(Phase, read(field, version - 1, 0, 0, 0),
                                                      decay, source, curl);
  return env.Finish(field, version, z, y, x, value);
}

}  // namespace beamz::cuda::trace_native
