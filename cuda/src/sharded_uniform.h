#ifndef BEAMZ_CUDA_SHARDED_UNIFORM_H_
#define BEAMZ_CUDA_SHARDED_UNIFORM_H_
#include "sharded_cell.h"
#ifdef __CUDACC__
#define BEAMZ_UNIFORM __host__ __device__ __forceinline__
#else
#define BEAMZ_UNIFORM inline
#endif
namespace beamz::cuda::sharded {

// A valid target differs from either curl source only along its derivative
// axis. Validate that axis once, then use local strides including the halo.
// This shares the streamed kernel's direct stencil arithmetic while retaining
// global physical supports and local CPML ownership.
template<int Component>
BEAMZ_UNIFORM float UniformSource(const BeamzLaunch& l, const int p[3],
                                  int axis, int derivative_axis, int delta) {
  const auto& source = l.inputs[3 + Component];
  const int q[3] = {
      p[0] + (axis == 0) + (derivative_axis == 0 ? delta : 0),
      p[1] + (axis == 1) + (derivative_axis == 1 ? delta : 0),
      p[2] + (axis == 2) + (derivative_axis == 2 ? delta : 0)};
  const int offset = (q[0] * int(source.dims[1]) + q[1]) * int(source.dims[2]) + q[2];
  return static_cast<const float*>(source.data)[offset];
}

template<int Phase, int Component, int Axis>
BEAMZ_UNIFORM float UniformDifference(const BeamzLaunch& l, const int p[3],
                                      const int global[3], int shard_axis) {
  const auto* g = static_cast<const int32_t*>(l.shard_geometry.data);
  const int coordinate = global[Axis], stop = g[12 + 3 * Component + Axis];
  if constexpr (Phase == 0) {
    if (coordinate + 1 >= stop) return 0;
    return (UniformSource<Component>(l, p, shard_axis, Axis, 1)
          - UniformSource<Component>(l, p, shard_axis, Axis, 0)) * l.inv_resolution;
  }
  if (coordinate == 0)
    return (l.metallic_edges & (1 << (2 * Axis)))
      ? UniformSource<Component>(l, p, shard_axis, Axis, 0) * l.inv_resolution : 0;
  if (coordinate == stop)
    return (l.metallic_edges & (1 << (2 * Axis + 1)))
      ? -UniformSource<Component>(l, p, shard_axis, Axis, -1) * l.inv_resolution : 0;
  return (UniformSource<Component>(l, p, shard_axis, Axis, 0)
        - UniformSource<Component>(l, p, shard_axis, Axis, -1)) * l.inv_resolution;
}

template<int Term, int Axis>
BEAMZ_UNIFORM float UniformCpml(const BeamzLaunch& l, float derivative,
                                const int p[3], const int global[3]) {
  const auto* g = static_cast<const int32_t*>(l.shard_geometry.data);
  const int thickness = l.uniform_cpml_thickness;
  const int stop = g[3 + 3 * (Term / 2) + Axis];
  const int coordinate = global[Axis];
  constexpr float sign = Term % 2 ? -1.0f : 1.0f;
  const int packed = coordinate < thickness ? coordinate
      : coordinate >= stop - thickness ? coordinate - stop + 2 * thickness : -1;
  if (packed < 0) return sign * derivative;
  int q[3] = {p[0], p[1], p[2]};
  q[Axis] = packed;
  const auto& psi = l.inputs[31 + Term];
  const int offset = Offset(psi, q);
  const float a = static_cast<const float*>(l.inputs[13 + 3*Term].data)[packed];
  const float b = static_cast<const float*>(l.inputs[14 + 3*Term].data)[packed];
  const float inv_kappa = static_cast<const float*>(l.inputs[15 + 3*Term].data)[packed];
  const float next = b * static_cast<const float*>(psi.data)[offset] + a * derivative;
  static_cast<float*>(l.outputs[3 + Term].data)[offset] = next;
  return sign * (derivative * inv_kappa + next);
}

template<int Phase, int Component>
BEAMZ_UNIFORM void UpdateUniformCell(const BeamzLaunch& l, int z, int y, int x) {
  const auto& output = l.outputs[Component];
  if (z >= output.dims[0] || y >= output.dims[1] || x >= output.dims[2]) return;
  const auto* g = static_cast<const int32_t*>(l.shard_geometry.data);
  const int p[3] = {z, y, x};
  const int global[3] = {z + (g[0] == 0 ? g[1] : 0),
                         y + (g[0] == 1 ? g[1] : 0),
                         x + (g[0] == 2 ? g[1] : 0)};
  const int offset = Offset(output, p);
  for (int d = 0; d < 3; ++d) {
    if (global[d] >= g[3 + 3*Component + d]) {
      static_cast<float*>(output.data)[offset] = 0;
      return;
    }
  }
  constexpr int source0[3] = {2, 0, 1}, source1[3] = {1, 2, 0};
  constexpr int axis0[3] = {1, 0, 2}, axis1[3] = {0, 2, 1};
  const float d0 = UniformDifference<Phase, source0[Component], axis0[Component]>(l, p, global, g[0]);
  const float d1 = UniformDifference<Phase, source1[Component], axis1[Component]>(l, p, global, g[0]);
  const float curl = UniformCpml<2*Component, axis0[Component]>(l, d0, p, global)
                   + UniformCpml<2*Component+1, axis1[Component]>(l, d1, p, global);
  if (g[2]) {
    static_cast<float*>(output.data)[offset] = curl;
    return;
  }
  bool constrained = false;
  for (int d = 0; d < 3; ++d) {
    if ((Phase == 0) != (d == 2 - Component)) continue;
    constrained |= (global[d] == 0 && (l.metallic_edges & (1 << (2*d)))) ||
      (global[d] == g[3 + 3*Component + d] - 1 && (l.metallic_edges & (1 << (2*d+1))));
  }
  const float decay = Read(l.inputs[6 + Component], Offset(l.inputs[6 + Component], p));
  const float source = Read(l.inputs[9 + Component], Offset(l.inputs[9 + Component], p));
  const float next = decay * static_cast<const float*>(l.inputs[Component].data)[offset]
      + (Phase == 0 ? -source : source) * curl;
  static_cast<float*>(output.data)[offset] = constrained ? 0 : next;
}

inline bool UniformSupported(const BeamzLaunch& l) {
  if (l.metric_kind != 0 || l.nterms != 6 || l.uniform_cpml_thickness <= 0) return false;
  for (int t = 0; t < 6; ++t)
    if (l.inputs[31+t].element_type != kBeamzF32) return false;
  return true;
}
}
#undef BEAMZ_UNIFORM
#endif
