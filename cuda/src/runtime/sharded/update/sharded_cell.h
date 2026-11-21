#ifndef BEAMZ_CUDA_SHARDED_CELL_H_
#define BEAMZ_CUDA_SHARDED_CELL_H_

#include <cstdint>
#include <limits>

#include "launch.h"
#include "sharded_numeric.h"

#ifdef __CUDACC__
#define BEAMZ_CELL __host__ __device__ inline
#else
#define BEAMZ_CELL inline
#endif

// This arithmetic is shared by the CUDA launch and the CPU FFI contract tests.
// Fields and transverse CPML slabs are local; one-cell source faces are passed
// separately from the contiguous owned fields. CPML slabs normal to that axis are small
// replicated packed arrays, with each rank owning disjoint recurrence entries.
namespace beamz::cuda::sharded {

BEAMZ_CELL constexpr int CurlSource(int term) {
  constexpr int sources[6] = {2, 1, 0, 2, 1, 0};
  return sources[term];
}
BEAMZ_CELL constexpr int CurlAxis(int term) {
  constexpr int axes[6] = {1, 0, 0, 2, 2, 1};
  return axes[term];
}

BEAMZ_CELL int Offset(const BeamzBuffer& a, const int p[3]) {
  if (a.rank == 0) return 0;
  int offset = 0;
  for (int d = 0; d < 3; ++d)
    offset = offset * static_cast<int>(a.dims[d]) + (a.dims[d] == 1 ? 0 : p[d]);
  return offset;
}

BEAMZ_CELL int FieldOffset(const BeamzBuffer& field, const int p[3]) {
  return BeamzSplitRows(field) ? BeamzOffset3D<false, true>(field, p[0], p[1], p[2])
                              : Offset(field, p);
}

BEAMZ_CELL float HaloValue(const BeamzBuffer& face, const int p[3], int axis) {
  // A direct peer descriptor retains its complete field and names the remote
  // plane in dims[3]. Staged descriptors contain one contiguous plane.
  const int plane = face.rank == 4 ? int(face.dims[3]) : 0;
  const int q[3] = {axis == 0 ? plane : p[0], axis == 1 ? plane : p[1],
                    axis == 2 ? plane : p[2]};
  return static_cast<const float*>(face.data)[FieldOffset(face, q)];
}

template<class Buffer>
BEAMZ_CELL float Read(const Buffer& a, int offset) {
  return static_cast<const float*>(a.data)[offset];
}

BEAMZ_CELL float Decay(const BeamzLaunch& l, int component, const int p[3]) {
  const auto& source = l.inputs[9 + component];
  const auto& decay = l.inputs[6 + component];
  return source.element_type == kBeamzS32 ? 1.0f : Read(decay, Offset(decay, p));
}

BEAMZ_CELL float Scale(const BeamzLaunch& l, int component, const int p[3], int index) {
  const auto& source = l.inputs[9 + component];
  if (source.element_type != kBeamzS32) return Read(source, Offset(source, p));
  const auto* headers = static_cast<const int32_t*>(source.data);
  const int header = headers[index >> 5];
  return Read(l.inputs[6 + component], header < 0 ? ~header : header + (index & 31));
}

BEAMZ_CELL int PsiBase(const BeamzLaunch& l, int term) {
  const auto& geometry = l.shard_geometry;
  return geometry.dims[0] == 9
      ? static_cast<const int32_t*>(geometry.data)[21 + term] : 0;
}

BEAMZ_CELL void Write(const BeamzBuffer& a, int offset, float value) {
  static_cast<float*>(a.data)[offset] = value;
}

BEAMZ_CELL float Source(const BeamzLaunch& l, int component, const int global[3]) {
  const auto* geometry = static_cast<const int32_t*>(l.shard_geometry.data);
  const int axis = geometry[0], origin = geometry[1];
  const int32_t* logical = geometry + 12 + 3 * component;
  const BeamzBuffer& input = l.inputs[3 + component];
  int local[3];
  for (int d = 0; d < 3; ++d) {
    if (global[d] < 0 || global[d] >= logical[d]) return 0.0f;
    local[d] = global[d] - (d == axis ? origin : 0);
    if (d != axis && (local[d] < 0 || local[d] >= input.dims[d])) return 0.0f;
  }
  if (local[axis] < 0 || local[axis] >= input.dims[axis]) {
    if (local[axis] != -1 && local[axis] != input.dims[axis]) return 0.0f;
    const auto& face = l.shard_halos[2 * component + (local[axis] >= 0)];
    return HaloValue(face, local, axis);
  }
  return Read(input, FieldOffset(input, local));
}

template <int Phase>
BEAMZ_CELL float Difference(const BeamzLaunch& l, int component, int axis,
                            const int global[3]) {
  const auto* geometry = static_cast<const int32_t*>(l.shard_geometry.data);
  const int size = geometry[12 + 3 * component + axis];
  const int coordinate = global[axis];
  float scale = l.inv_resolution;
  if (l.metric_kind != 0) {
    const int metric_index = l.metric_kind == 1 ? 0 : coordinate;
    if (metric_index < 0 || (l.metrics[axis].rank != 0 &&
        metric_index >= l.metrics[axis].dims[0])) return 0.0f;
    scale = Read(l.metrics[axis], metric_index);
  }
  int other[3] = {global[0], global[1], global[2]};
  if constexpr (Phase == 0) {
    if (coordinate < 0 || coordinate + 1 >= size) return 0.0f;
    ++other[axis];
    return FixedDifference(Source(l, component, other) - Source(l, component, global), scale);
  }
  if (coordinate < 0 || coordinate > size) return 0.0f;
  if (coordinate == 0)
    return (l.metallic_edges & (1 << (2 * axis)))
        ? FixedDifference(Source(l, component, global), scale) : 0.0f;
  --other[axis];
  if (coordinate == size)
    return (l.metallic_edges & (1 << (2 * axis + 1)))
        ? FixedDifference(-Source(l, component, other), scale) : 0.0f;
  return FixedDifference(Source(l, component, global) - Source(l, component, other), scale);
}

BEAMZ_CELL float CorrectCpml(const BeamzLaunch& l, int term, float derivative,
                             const int local[3], const int global[3]) {
  const auto* metadata = static_cast<const int32_t*>(l.inputs[12].data) + 5 * term;
  const auto* geometry = static_cast<const int32_t*>(l.shard_geometry.data);
  const int axis = metadata[1], low = metadata[2], high = metadata[3];
  const int stop = geometry[3 + 3 * (term / 2) + axis];
  const int coordinate = global[axis];
  const float sign = static_cast<float>(metadata[4]);
  const int packed = coordinate < low ? coordinate
      : (coordinate >= stop - high ? low + coordinate - (stop - high) : -1);
  if (packed < 0) return sign * derivative;
  int p[3] = {local[0], local[1], local[2]};
  p[axis] = packed - PsiBase(l, term);
  const BeamzBuffer& psi = l.inputs[13 + 3 * l.nterms + term];
  const int offset = Offset(psi, p);
  const int coefficient = 13 + 3 * term;
  const float next = FixedPsi(Read(l.inputs[coefficient + 1], packed), Read(psi, offset),
                              Read(l.inputs[coefficient], packed), derivative);
  Write(l.outputs[3 + term], offset, next);
  return FixedCorrection(sign, derivative, Read(l.inputs[coefficient + 2], packed), next);
}

template <int Phase, int component>
BEAMZ_CELL void UpdateCell(const BeamzLaunch& l, int z, int y, int x) {
  const BeamzBuffer& output = l.outputs[component];
  if (z >= output.dims[0] || y >= output.dims[1] || x >= output.dims[2]) return;
  const auto* geometry = static_cast<const int32_t*>(l.shard_geometry.data);
  const int axis = geometry[0];
  if (axis < 0 || axis > 2) return;
  const int local[3] = {z, y, x};
  int global[3] = {z, y, x};
  global[axis] += geometry[1];
  const int32_t* logical = geometry + 3 + 3 * component;
  const int offset = FieldOffset(output, local);
  for (int d = 0; d < 3; ++d) {
    if (global[d] < 0 || global[d] >= logical[d]) {
      Write(output, offset, 0.0f);
      return;
    }
  }
  const int first = 2 * component;
  const float d0 = Difference<Phase>(l, CurlSource(first), CurlAxis(first), global);
  const float d1 = Difference<Phase>(l, CurlSource(first+1), CurlAxis(first+1), global);
  const float curl = l.nterms == 0 ? d0 - d1
      : CorrectCpml(l, 2 * component, d0, local, global)
      + CorrectCpml(l, 2 * component + 1, d1, local, global);
  if (geometry[2]) {
    // Coupled tensor constitutive updates run in JAX after global assembly.
    // Preserve curls at constrained sites until that coupling is complete.
    Write(output, offset, curl);
    return;
  }
  bool constrained = false;
  for (int d = 0; d < 3; ++d) {
    if ((Phase == 0) != (d == 2 - component)) continue;
    constrained |= (global[d] == 0 && (l.metallic_edges & (1 << (2 * d)))) ||
        (global[d] == logical[d] - 1 && (l.metallic_edges & (1 << (2 * d + 1))));
  }
  const float decay = Decay(l, component, local);
  const float source = Scale(l, component, local, Offset(output, local));
  const float next = FixedAdvance(Phase, Read(l.inputs[component], offset), decay, source, curl);
  Write(output, offset, constrained ? 0.0f : next);
}

inline bool ValidBuffer(const BeamzBuffer& a) {
  if (a.rank < 0 || a.rank > 3) return false;
  int64_t count = 1;
  for (int i = 0; i < a.rank; ++i) {
    if (a.dims[i] < 0 || a.dims[i] > std::numeric_limits<int>::max()) return false;
    count *= a.dims[i];
    if (count > std::numeric_limits<int>::max()) return false;
  }
  return count == 0 || a.data != nullptr;
}

inline bool Validate(const BeamzLaunch& l) {
  if (l.phase < 0 || l.phase > 1 || (l.nterms != 0 && l.nterms != 6) ||
      l.metric_kind < 0 || l.metric_kind > 2) return false;
  const auto& g = l.shard_geometry;
  if (!ValidBuffer(g) || g.rank != 2 || (g.dims[0] != 7 && g.dims[0] != 9) || g.dims[1] != 3 ||
      g.element_type != kBeamzS32) return false;
  for (int i = 0; i < 13 + 4 * l.nterms; ++i)
    if (!ValidBuffer(l.inputs[i])) return false;
  for (int i = 0; i < 3 + l.nterms; ++i) {
    const auto& a = l.outputs[i];
    const auto& input = l.inputs[i < 3 ? i : 13 + 3 * l.nterms + i - 3];
    if (!ValidBuffer(a) || a.rank != 3 || input.rank != 3 ||
        a.element_type != input.element_type) return false;
    if (a.element_type != kBeamzF32) return false;
    for (int d = 0; d < 3; ++d) if (a.dims[d] != input.dims[d]) return false;
  }
  for (int i = 0; i < 6; ++i)
    if (l.inputs[i].rank != 3 || l.inputs[i].element_type != kBeamzF32) return false;
  for (int c = 0; c < 3; ++c) {
    for (int side = 0; side < 2; ++side) {
      const auto& face = l.shard_halos[2*c + side];
      if (!ValidBuffer(face) || face.rank != 3 || face.element_type != kBeamzF32)
        return false;
      bool compatible = false;
      for (int axis = 0; axis < 3; ++axis) {
        bool match = face.dims[axis] == 1;
        for (int d = 0; d < 3; ++d)
          if (d != axis) match &= face.dims[d] == l.inputs[3+c].dims[d];
        compatible |= match;
      }
      if (!compatible) return false;
      for (int d = 0; d < 3; ++d)
        if (face.dims[d] != l.shard_halos[2*c].dims[d]) return false;
    }
  }
  for (int c = 0; c < 3; ++c) {
    const auto& decay = l.inputs[6 + c];
    const auto& source = l.inputs[9 + c];
    if (source.element_type == kBeamzS32) {
      const auto& field = l.outputs[c];
      const int64_t cells = field.dims[0] * field.dims[1] * field.dims[2];
      if (g.dims[0] != 9 || l.phase != 1 || decay.element_type != kBeamzF32 ||
          decay.rank != 1 || decay.dims[0] < 1 || source.rank != 2 ||
          source.dims[0] * source.dims[1] < (cells + 31) / 32) return false;
    } else if (decay.element_type != kBeamzF32 || source.element_type != kBeamzF32 ||
               (decay.rank != 0 && decay.rank != 3) ||
               (source.rank != 0 && source.rank != 3)) return false;
  }
  if (l.nterms) {
    const auto& m = l.inputs[12];
    if (m.rank != 2 || m.element_type != kBeamzS32 || m.dims[0] != 6 || m.dims[1] != 5) return false;
    for (int i = 13; i < 31; ++i)
      if (l.inputs[i].rank != 3 || l.inputs[i].element_type != kBeamzF32) return false;
  }
  for (const auto& m : l.metrics)
    if (!ValidBuffer(m) || m.element_type != kBeamzF32 ||
        (l.metric_kind == 1 && m.rank != 0) ||
        (l.metric_kind == 2 && (m.rank != 1 || m.dims[0] < 1))) return false;
  return true;
}

}  // namespace beamz::cuda::sharded
#undef BEAMZ_CELL
#endif
