#ifndef BEAMZ_CUDA_SHARDED_CELL_H_
#define BEAMZ_CUDA_SHARDED_CELL_H_

#include <cstdint>
#include <limits>

#include "launch.h"

#ifdef __CUDACC__
#define BEAMZ_CELL __host__ __device__ inline
#else
#define BEAMZ_CELL inline
#endif

// This arithmetic is shared by the CUDA launch and the CPU FFI contract tests.
// Fields and transverse CPML slabs are local; source fields have one ghost cell
// at each end of the partition axis. CPML slabs normal to that axis are small
// replicated packed arrays, with each rank owning disjoint recurrence entries.
namespace beamz::cuda::sharded {

BEAMZ_CELL int Offset(const BeamzBuffer& a, const int p[3]) {
  if (a.rank == 0) return 0;
  int offset = 0;
  for (int d = 0; d < 3; ++d)
    offset = offset * static_cast<int>(a.dims[d]) + (a.dims[d] == 1 ? 0 : p[d]);
  return offset;
}

BEAMZ_CELL float Read(const BeamzBuffer& a, int offset) {
  if (a.element_type == kBeamzBF16) {
    union { uint32_t bits; float value; } data;
    data.bits = static_cast<uint32_t>(static_cast<const uint16_t*>(a.data)[offset]) << 16;
    return data.value;
  }
  return static_cast<const float*>(a.data)[offset];
}

BEAMZ_CELL void Write(const BeamzBuffer& a, int offset, float value) {
  if (a.element_type == kBeamzBF16) {
    union { uint32_t bits; float value; } data;
    data.value = value;
    const uint32_t magnitude = data.bits & 0x7fffffffU;
    // Round to nearest-even and retain NaNs instead of rounding them to infinity.
    const uint16_t rounded = magnitude > 0x7f800000U
        ? static_cast<uint16_t>((data.bits >> 16) | 0x40U)
        : static_cast<uint16_t>((data.bits + 0x7fffU + ((data.bits >> 16) & 1U)) >> 16);
    static_cast<uint16_t*>(a.data)[offset] = rounded;
  } else {
    static_cast<float*>(a.data)[offset] = value;
  }
}

BEAMZ_CELL float Source(const BeamzLaunch& l, int component, const int global[3]) {
  const auto* geometry = static_cast<const int32_t*>(l.shard_geometry.data);
  const int axis = geometry[0], origin = geometry[1];
  const int32_t* logical = geometry + 12 + 3 * component;
  const BeamzBuffer& input = l.inputs[3 + component];
  int local[3];
  for (int d = 0; d < 3; ++d) {
    if (global[d] < 0 || global[d] >= logical[d]) return 0.0f;
    local[d] = global[d] - (d == axis ? origin - 1 : 0);
    if (local[d] < 0 || local[d] >= input.dims[d]) return 0.0f;
  }
  return Read(input, Offset(input, local));
}

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
  if (l.phase == 0) {
    if (coordinate < 0 || coordinate + 1 >= size) return 0.0f;
    ++other[axis];
    return (Source(l, component, other) - Source(l, component, global)) * scale;
  }
  if (coordinate < 0 || coordinate > size) return 0.0f;
  if (coordinate == 0)
    return (l.metallic_edges & (1 << (2 * axis)))
        ? Source(l, component, global) * scale : 0.0f;
  --other[axis];
  if (coordinate == size)
    return (l.metallic_edges & (1 << (2 * axis + 1)))
        ? -Source(l, component, other) * scale : 0.0f;
  return (Source(l, component, global) - Source(l, component, other)) * scale;
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
  p[axis] = packed;
  const BeamzBuffer& psi = l.inputs[13 + 3 * l.nterms + term];
  const int offset = Offset(psi, p);
  const int coefficient = 13 + 3 * term;
  const float next = Read(l.inputs[coefficient + 1], packed) * Read(psi, offset)
                   + Read(l.inputs[coefficient], packed) * derivative;
  Write(l.outputs[3 + term], offset, next);
  return sign * (derivative * Read(l.inputs[coefficient + 2], packed) + next);
}

BEAMZ_CELL void UpdateCell(const BeamzLaunch& l, int component, int z, int y, int x) {
  const BeamzBuffer& output = l.outputs[component];
  if (z >= output.dims[0] || y >= output.dims[1] || x >= output.dims[2]) return;
  const auto* geometry = static_cast<const int32_t*>(l.shard_geometry.data);
  const int axis = geometry[0];
  if (axis < 0 || axis > 2) return;
  const int local[3] = {z, y, x};
  int global[3] = {z, y, x};
  global[axis] += geometry[1];
  const int32_t* logical = geometry + 3 + 3 * component;
  const int offset = Offset(output, local);
  for (int d = 0; d < 3; ++d) {
    if (global[d] < 0 || global[d] >= logical[d]) {
      Write(output, offset, 0.0f);
      return;
    }
  }
  constexpr int source0[3] = {2, 0, 1}, source1[3] = {1, 2, 0};
  constexpr int axis0[3] = {1, 0, 2}, axis1[3] = {0, 2, 1};
  const float d0 = Difference(l, source0[component], axis0[component], global);
  const float d1 = Difference(l, source1[component], axis1[component], global);
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
    if ((l.phase == 0) != (d == 2 - component)) continue;
    constrained |= (global[d] == 0 && (l.metallic_edges & (1 << (2 * d)))) ||
        (global[d] == logical[d] - 1 && (l.metallic_edges & (1 << (2 * d + 1))));
  }
  const float decay = Read(l.inputs[6 + component], Offset(l.inputs[6 + component], local));
  const float source = Read(l.inputs[9 + component], Offset(l.inputs[9 + component], local));
  const float next = decay * Read(l.inputs[component], offset)
      + (l.phase == 0 ? -source : source) * curl;
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
  if (!ValidBuffer(g) || g.rank != 2 || g.dims[0] != 7 || g.dims[1] != 3 ||
      g.element_type != kBeamzS32) return false;
  for (int i = 0; i < 13 + 4 * l.nterms; ++i)
    if (!ValidBuffer(l.inputs[i])) return false;
  for (int i = 0; i < 3 + l.nterms; ++i) {
    const auto& a = l.outputs[i];
    const auto& input = l.inputs[i < 3 ? i : 13 + 3 * l.nterms + i - 3];
    if (!ValidBuffer(a) || a.rank != 3 || input.rank != 3 ||
        a.element_type != input.element_type) return false;
    if (a.element_type != kBeamzF32 && (i < 3 || a.element_type != kBeamzBF16)) return false;
    for (int d = 0; d < 3; ++d) if (a.dims[d] != input.dims[d]) return false;
  }
  for (int i = 0; i < 6; ++i)
    if (l.inputs[i].rank != 3 || l.inputs[i].element_type != kBeamzF32) return false;
  for (int i = 6; i < 12; ++i)
    if ((l.inputs[i].rank != 0 && l.inputs[i].rank != 3) ||
        l.inputs[i].element_type != kBeamzF32) return false;
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
