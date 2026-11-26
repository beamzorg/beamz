#ifndef BEAMZ_CUDA_SHARDED_UNIFORM_H_
#define BEAMZ_CUDA_SHARDED_UNIFORM_H_

#include "update/sharded_cell.h"

#ifdef __CUDACC__
#define BEAMZ_UNIFORM __host__ __device__ __forceinline__
#else
#define BEAMZ_UNIFORM inline
#endif
namespace beamz::cuda::sharded {

struct UniformField {
  float* data;
  int extent[3], stride[3];
};
struct UniformPsi {
  UniformField output;
  const float *previous, *a, *b, *inverse_kappa;
  int stop, origin;
};
struct UniformFace {
  UniformField field;
  int plane;
};
struct UniformBounds {
  int begin[3], end[3];
};
struct UniformCoefficient {
  const void* data;
  int element_type, stride[3];
};
BEAMZ_UNIFORM UniformCoefficient CoefficientView(const BeamzBuffer& b) {
  const bool spatial = b.rank == 3;
  return {b.data,
          b.element_type,
          {spatial && b.dims[0] > 1 ? int(b.dims[1] * b.dims[2]) : 0,
           spatial && b.dims[1] > 1 ? int(b.dims[2]) : 0, int(spatial && b.dims[2] > 1)}};
}
BEAMZ_UNIFORM int CoefficientOffset(const UniformCoefficient& a, const int p[3]) {
  return p[0] * a.stride[0] + p[1] * a.stride[1] + p[2] * a.stride[2];
}

template <bool Simple>
struct UniformMaterial;
template <>
struct UniformMaterial<true> {
  const float* table[3];
  const void* scale[3];
};
template <>
struct UniformMaterial<false> {
  UniformCoefficient decay[3], scale[3];
};

template <bool Simple>
struct UniformPhase {
  UniformField output[3], source[3];
  const float* previous[3];
  UniformFace halo[3][2];
  UniformPsi psi[6];
  UniformMaterial<Simple> material;
  int target_extent[3][3], source_extent[3][3];
  UniformBounds bulk, inner;
  int axis, origin, metallic, thickness;
  bool return_curl;
  float inverse_spacing;
};

BEAMZ_UNIFORM UniformField FieldView(const BeamzBuffer& b) {
  const int nx = int(b.dims[2]), ny = int(b.dims[1]);
  return {static_cast<float*>(b.data),
          {int(b.dims[0]), ny, nx},
          {b.plane_stride ? b.plane_stride : nx * ny, b.row_stride ? b.row_stride : nx, 1}};
}
BEAMZ_UNIFORM int Smaller(int a, int b) { return a < b ? a : b; }
BEAMZ_UNIFORM int Larger(int a, int b) { return a > b ? a : b; }

template <bool Simple>
BEAMZ_UNIFORM UniformPhase<Simple> PrepareUniform(const BeamzLaunch& l, const int* g) {
  UniformPhase<Simple> p{};
  p.axis = g[0];
  p.origin = g[1];
  p.return_curl = bool(g[2]);
  p.metallic = l.metallic_edges;
  p.thickness = l.uniform_cpml_thickness;
  p.inverse_spacing = l.inv_resolution;
  for (int c = 0; c < 3; ++c) {
    p.output[c] = FieldView(l.outputs[c]);
    p.previous[c] = static_cast<const float*>(l.inputs[c].data);
    p.source[c] = FieldView(l.inputs[3 + c]);
    for (int side = 0; side < 2; ++side) {
      const auto& halo = l.shard_halos[2 * c + side];
      p.halo[c][side] = {FieldView(halo), halo.rank == 4 ? int(halo.dims[3]) : 0};
    }
    if constexpr (Simple) {
      p.material.table[c] = static_cast<const float*>(l.inputs[6 + c].data);
      p.material.scale[c] = l.inputs[9 + c].data;
    } else {
      p.material.decay[c] = CoefficientView(l.inputs[6 + c]);
      p.material.scale[c] = CoefficientView(l.inputs[9 + c]);
    }
    for (int d = 0; d < 3; ++d) {
      p.target_extent[c][d] = g[3 + 3 * c + d];
      p.source_extent[c][d] = g[12 + 3 * c + d];
    }
  }
  for (int t = 0; t < 6; ++t) {
    const int slot = 13 + 3 * t;
    p.psi[t] = {FieldView(l.outputs[3 + t]),
                static_cast<const float*>(l.inputs[31 + t].data),
                static_cast<const float*>(l.inputs[slot].data),
                static_cast<const float*>(l.inputs[slot + 1].data),
                static_cast<const float*>(l.inputs[slot + 2].data),
                p.target_extent[t / 2][CurlAxis(t)],
                l.shard_geometry.dims[0] == 9 ? g[21 + t] : 0};
  }
  for (int pass = 0; pass < 2; ++pass) {
    auto& bounds = pass ? p.bulk : p.inner;
    const int margin = pass ? p.thickness : 0;
    for (int d = 0; d < 3; ++d) {
      const bool partitioned = d == p.axis;
      const int origin = partitioned ? p.origin : 0;
      bounds.begin[d] = Larger(int(partitioned), margin + 1 - origin);
      bounds.end[d] = p.output[0].extent[d] - int(partitioned);
      for (int c = 0; c < 3; ++c) {
        bounds.end[d] = Smaller(bounds.end[d], p.output[c].extent[d]);
        bounds.end[d] = Smaller(bounds.end[d], p.target_extent[c][d] - margin - origin - 1);
        if (d == 2) {
          for (int which = 0; which != 2; ++which) {
            const auto& field = which ? p.output[c] : p.source[c];
            if (field.stride[1] < field.extent[2])
              bounds.end[d] = Smaller(bounds.end[d], field.stride[1] - 1);
          }
        }
      }
    }
  }
  return p;
}

BEAMZ_UNIFORM int RowAddress(const UniformField& field, const int p[3]) {
  const int pitch = field.stride[1];
  if (pitch < field.extent[2] && p[2] >= pitch)
    return BeamzRowTailOffset(p[0] * field.extent[1] + p[1], field.extent[0] * field.extent[1],
                              field.extent[2], pitch, p[2]);
  return p[0] * field.stride[0] + p[1] * pitch + p[2];
}

template <int Component, int Axis, int Delta, int Rank, bool Edge, bool Simple>
BEAMZ_UNIFORM float UniformRead(const UniformPhase<Simple>& phase, const int p[3]) {
  const auto& src = phase.source[Component];
  if constexpr (Edge) {
    const int rank = Rank < 0 ? phase.axis : Rank;
    int q[3] = {p[0], p[1], p[2]};
    q[Axis] += Delta;
    if (q[rank] < 0 || q[rank] >= src.extent[rank]) {
      const auto& halo = phase.halo[Component][q[rank] >= 0];
      q[rank] = halo.plane;
      return halo.field.data[RowAddress(halo.field, q)];
    }
    return src.data[RowAddress(src, q)];
  }
  return src.data[p[0] * src.stride[0] + p[1] * src.stride[1] + p[2] + Delta * src.stride[Axis]];
}

template <int Phase, int Term, int Rank, bool Edge, bool Simple>
BEAMZ_UNIFORM float UniformDifference(const UniformPhase<Simple>& s, const int p[3]) {
  constexpr int c = CurlSource(Term), a = CurlAxis(Term);
  if constexpr (Edge) {
    const int rank = Rank < 0 ? s.axis : Rank;
    const int x = p[a] + (a == rank ? s.origin : 0), stop = s.source_extent[c][a];
    if constexpr (Phase == 0) {
      if (x + 1 >= stop) return 0.0f;
    } else {
      if (x == 0)
        return (s.metallic & (1 << (2 * a)))
                   ? FixedDifference(UniformRead<c, a, 0, Rank, true>(s, p), s.inverse_spacing)
                   : 0.0f;
      if (x == stop)
        return (s.metallic & (1 << (2 * a + 1)))
                   ? FixedDifference(-UniformRead<c, a, -1, Rank, true>(s, p), s.inverse_spacing)
                   : 0.0f;
    }
  }
  const float center = UniformRead<c, a, 0, Rank, Edge>(s, p);
  const float neighbor = UniformRead<c, a, Phase == 0 ? 1 : -1, Rank, Edge>(s, p);
  return FixedDifference(Phase == 0 ? neighbor - center : center - neighbor, s.inverse_spacing);
}

template <int Term, int Rank, bool Simple>
BEAMZ_UNIFORM float UniformCpml(const UniformPhase<Simple>& s, const int p[3], float difference) {
  constexpr int axis = CurlAxis(Term);
  constexpr float sign = Term % 2 ? -1.f : 1.f;
  const auto& slab = s.psi[Term];
  const int rank = Rank < 0 ? s.axis : Rank;
  const int coord = p[axis] + (axis == rank ? s.origin : 0), t = s.thickness;
  const int packed = coord < t ? coord : coord >= slab.stop - t ? coord - slab.stop + 2 * t : -1;
  if (packed < 0) return sign * difference;
  const int offset = p[0] * slab.output.stride[0] + p[1] * slab.output.stride[1] + p[2] +
                     (packed - slab.origin - p[axis]) * slab.output.stride[axis];
  const float* previous = Simple && Rank >= 0 ? slab.output.data : slab.previous;
  const float next = FixedPsi(slab.b[packed], previous[offset], slab.a[packed], difference);
  slab.output.data[offset] = next;
  return FixedCorrection(sign, difference, slab.inverse_kappa[packed], next);
}

template <int Phase, int Component, int Rank, bool Edge, bool Shell, bool Simple>
BEAMZ_UNIFORM void UpdateUniformPoint(const UniformPhase<Simple>& s, const int p[3]) {
  const auto& f = s.output[Component];
  const int rank = Rank < 0 ? s.axis : Rank;
  if constexpr (Edge)
    for (int d = 0; d < 3; ++d)
      if (p[d] >= f.extent[d]) return;
  const int address = Edge ? RowAddress(f, p) : p[0] * f.stride[0] + p[1] * f.stride[1] + p[2];
  if constexpr (Edge)
    for (int d = 0; d < 3; ++d)
      if (p[d] + (d == rank ? s.origin : 0) >= s.target_extent[Component][d]) {
        f.data[address] = 0.f;
        return;
      }
  float a = UniformDifference<Phase, 2 * Component, Rank, Edge>(s, p);
  float b = UniformDifference<Phase, 2 * Component + 1, Rank, Edge>(s, p);
  float curl;
  if constexpr (Shell)
    curl = UniformCpml<2 * Component, Rank>(s, p, a) + UniformCpml<2 * Component + 1, Rank>(s, p, b);
  else
    curl = a - b;
  if constexpr (!Simple || Rank < 0)
    if (s.return_curl) {
      f.data[address] = curl;
      return;
    }
  const int logical = (p[0] * f.extent[1] + p[1]) * f.extent[2] + p[2];
  float decay, scale;
  if constexpr (Simple && Phase == 0) {
    decay = *s.material.table[Component];
    scale = *static_cast<const float*>(s.material.scale[Component]);
  } else if constexpr (Simple) {
    const int head = static_cast<const int*>(s.material.scale[Component])[logical >> 5];
    decay = 1.f;
    scale = s.material.table[Component][head < 0 ? ~head : head + (logical & 31)];
  } else {
    const auto& d = s.material.decay[Component];
    const auto& v = s.material.scale[Component];
    decay = v.element_type == kBeamzS32 ? 1.f : Read(d, CoefficientOffset(d, p));
    if (v.element_type == kBeamzS32) {
      const int head = static_cast<const int*>(v.data)[logical >> 5];
      scale = Read(d, head < 0 ? ~head : head + (logical & 31));
    } else
      scale = Read(v, CoefficientOffset(v, p));
  }
  const float* previous = Simple && Rank >= 0 ? f.data : s.previous[Component];
  float next = FixedAdvance(Phase, previous[address], decay, scale, curl);
  if constexpr (Edge) {
    for (int d = 0; d < 3; ++d) {
      if ((Phase == 0) != (d == 2 - Component)) continue;
      const int coord = p[d] + (d == rank ? s.origin : 0);
      if ((coord == 0 && (s.metallic & (1 << (2 * d)))) ||
          (coord == s.target_extent[Component][d] - 1 && (s.metallic & (1 << (2 * d + 1)))))
        next = 0.f;
    }
  }
  f.data[address] = next;
}

inline bool UniformSupported(const BeamzLaunch& l) {
  if (l.metric_kind != 0 || l.nterms != 6 || l.uniform_cpml_thickness <= 0) return false;
  for (int t = 0; t < 6; ++t)
    if (l.inputs[31 + t].element_type != kBeamzF32) return false;
  return true;
}
}  // namespace beamz::cuda::sharded
#undef BEAMZ_UNIFORM
#endif
