// Exact first-order transpose of the streamed Yee/CPML phase. Inputs and
// cotangents are read-only: reverse-mode residuals must survive recomputation.
#include <cuda_runtime_api.h>

#include <cstdint>
#include <initializer_list>
#include <limits>

#include "adjoint_replay.h"
#include "kernels.h"
#include "launch.h"
#include "yee_primitives.cuh"

namespace {

__device__ __forceinline__ void WarpAdd(float* address, float gradient) {
  // Live lanes are contiguous: only x support bounds can split a 32-wide
  // warp for y/z profiles. A one-entry x absorber profile has one live lane.
  const unsigned mask = __activemask();
  const int lane = threadIdx.x & 31;
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other = __shfl_down_sync(mask, gradient, offset);
    if (lane + offset < 32 && (mask & (1u << (lane + offset)))) gradient += other;
  }
  if (lane == __ffs(mask) - 1) atomicAdd(address, gradient);
}

int64_t Elements(const BeamzBuffer& value) {
  int64_t count = 1;
  for (int axis = 0; axis < value.rank; ++axis) count *= value.dims[axis];
  return count;
}

__host__ __device__ bool SameShape(const BeamzBuffer& a, const BeamzBuffer& b) {
  if (a.rank != b.rank) return false;
  for (int axis = 0; axis < a.rank; ++axis) {
    if (a.dims[axis] != b.dims[axis]) return false;
  }
  return true;
}

__device__ __forceinline__ int Offset(const BeamzBuffer& value, const int* p) {
  if (value.rank == 0) return 0;
  int offset = 0;
  #pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    const int size = static_cast<int>(value.dims[axis]);
    offset = offset * size + (size == 1 ? 0 : p[axis]);
  }
  return offset;
}

__device__ __forceinline__ float Read(const BeamzBuffer& value, const int* p) {
  return static_cast<const float*>(value.data)[Offset(value, p)];
}

__device__ __forceinline__ void Add(const BeamzBuffer& value, const int* p, float gradient) {
  if (value.rank == 1 && value.dims[0] == 0) return;
  auto* address = static_cast<float*>(value.data) + Offset(value, p);
  // Each warp has one y/z coordinate in the 32-wide tile. Profiles broadcast
  // along x share a destination, so issue one atomic per live warp, not per cell.
  if (value.rank == 0 || value.dims[2] == 1) WarpAdd(address, gradient);
  else atomicAdd(address, gradient);
}

// A coefficient with the whole target shape is owned by this thread. Only
// broadcast coefficients need a reduction; full material grids need no atomics.
__device__ __forceinline__ void Accumulate(const BeamzBuffer& gradient, const BeamzBuffer& field,
                           const int* p, float value) {
  if (gradient.rank == 1 && gradient.dims[0] == 0) return;
  if (gradient.rank == 3 && gradient.dims[0] == field.dims[0] &&
      gradient.dims[1] == field.dims[1] && gradient.dims[2] == field.dims[2]) {
    static_cast<float*>(gradient.data)[Offset(gradient, p)] = value;
  } else {
    Add(gradient, p, value);
  }
}

__host__ __device__ __forceinline__ bool Active(const BeamzAdjointLaunch& launch, int input) {
  return (launch.gradient_mask & (uint64_t{1} << input)) != 0;
}

template <int Phase, int Component>
__device__ __forceinline__ bool Masked(const BeamzLaunch& launch, const int* p) {
  constexpr int normal = 2 - Component;
  const auto& field = launch.inputs[Component];
  for (int axis = 0; axis < 3; ++axis) {
    if ((Phase == 0) != (axis == normal)) continue;
    if ((p[axis] == 0 && (launch.metallic_edges & (1 << (2 * axis)))) ||
        (p[axis] == field.dims[axis] - 1 &&
         (launch.metallic_edges & (1 << (2 * axis + 1))))) return true;
  }
  return false;
}

// Represent the actual forward stencil, including replicated open H boundaries
// and zero ghost samples at PEC boundaries, before transposing it.
template <int Phase>
__device__ __forceinline__ float Difference(const BeamzLaunch& launch, int input, int axis,
                            const int* p, int* high, int* low,
                            bool* has_high, bool* has_low, float* scale) {
  const auto& value = launch.inputs[input];
  for (int a = 0; a < 3; ++a) high[a] = low[a] = p[a];
  const int coord = p[axis];
  *has_high = *has_low = true;
  if constexpr (Phase == 0) {
    high[axis]++;
    if (high[axis] >= value.dims[axis]) *has_high = *has_low = false;
  } else if (coord == 0) {
    *has_low = false;
    *has_high = (launch.metallic_edges & (1 << (2 * axis))) != 0;
  } else if (coord == value.dims[axis]) {
    *has_high = false;
    low[axis]--;
    *has_low = (launch.metallic_edges & (1 << (2 * axis + 1))) != 0;
  } else {
    low[axis]--;
  }
  *scale = launch.inv_resolution;
  if (launch.metric_kind != 0) {
    const auto& metric = launch.metrics[axis];
    *scale = static_cast<const float*>(metric.data)[metric.rank == 0 ? 0 : coord];
  }
  return (*has_high ? Read(value, high) : 0.0f) -
         (*has_low ? Read(value, low) : 0.0f);
}

__device__ __forceinline__ int CotangentBank(const BeamzAdjointLaunch& launch) {
  return launch.reverse_iteration.data ?
      (*static_cast<const int32_t*>(launch.reverse_iteration.data) & 1) : 0;
}

__device__ __forceinline__ float ReadPsiCotangent(const BeamzAdjointLaunch& launch,
                                                  int term, const int* p) {
  const auto& value = launch.phase.outputs[3 + term];
  return static_cast<const float*>(value.data)[
      Offset(value, p) + CotangentBank(launch) * launch.psi_bank_stride[term]];
}

// Cotangent of one unscaled difference at a target cell. This needs neither
// the primal fields nor the old CPML memory. Each source cell gathers its
// neighboring target contributions, transposing the exact boundary stencil.
template <int Phase, int Term>
__device__ __forceinline__ float DifferenceCotangent(const BeamzAdjointLaunch& launch,
                                                     const int* p) {
  const auto& phase = launch.phase;
  constexpr int term = Term;
  constexpr int component = Term / 2;
  constexpr int axes[6] = {1, 0, 0, 2, 2, 1};
  const auto& field = phase.inputs[component];
  for (int axis = 0; axis < 3; ++axis) {
    if (p[axis] < 0 || p[axis] >= field.dims[axis]) return 0.0f;
  }
  const float sign = ((term & 1) ? -1.0f : 1.0f) *
                     (Phase == 0 ? -1.0f : 1.0f);
  const float g = Masked<Phase, component>(phase, p) ? 0.0f :
                  Read(phase.outputs[component], p);
  float gd = sign * Read(phase.inputs[9 + component], p) *
             g;
  const int axis = axes[term];
  if (phase.nterms) {
    const auto* metadata = static_cast<const int32_t*>(phase.inputs[12].data);
    const int nlow = metadata[5 * term + 2];
    const int nhigh = metadata[5 * term + 3];
    const int stop = field.dims[axis];
    const int packed = p[axis] < nlow ? p[axis] :
        (p[axis] >= stop - nhigh ? nlow + p[axis] - stop + nhigh : -1);
    if (packed >= 0) {
      int q[3] = {p[0], p[1], p[2]};
      q[axis] = packed;
      const int base = 13 + 3 * term;
      const float gp = ReadPsiCotangent(launch, term, q) + gd;
      gd = Read(phase.inputs[base + 2], q) * gd +
           Read(phase.inputs[base], q) * gp;
    }
  }
  const float scale = phase.metric_kind == 0 ? phase.inv_resolution :
      static_cast<const float*>(phase.metrics[axis].data)[
          phase.metrics[axis].rank == 0 ? 0 : p[axis]];
  return scale * gd;
}

template <int Phase, int Term>
__device__ __forceinline__ float GatherTerm(const BeamzAdjointLaunch& launch,
                                            const BeamzBuffer& source,
                                            const int* p) {
  constexpr int axes[6] = {1, 0, 0, 2, 2, 1};
  const auto& phase = launch.phase;
  constexpr int axis = axes[Term];
  int neighbor[3] = {p[0], p[1], p[2]};
  float gradient = 0.0f;
  if constexpr (Phase == 0) {
    if (p[axis] > 0) {
      neighbor[axis]--;
      gradient += DifferenceCotangent<Phase, Term>(launch, neighbor);
    }
    if (p[axis] + 1 < source.dims[axis])
      gradient -= DifferenceCotangent<Phase, Term>(launch, p);
  } else {
    if (p[axis] > 0 || (phase.metallic_edges & (1 << (2 * axis))))
      gradient += DifferenceCotangent<Phase, Term>(launch, p);
    if (p[axis] + 1 < source.dims[axis] ||
        (phase.metallic_edges & (1 << (2 * axis + 1)))) {
      neighbor[axis]++;
      gradient -= DifferenceCotangent<Phase, Term>(launch, neighbor);
    }
  }
  return gradient;
}

template <int Phase, int Component>
__device__ __forceinline__ void GatherSource(const BeamzAdjointLaunch& launch,
                                             const int* p) {
  if (!Active(launch, 3 + Component)) return;
  const auto& phase = launch.phase;
  const auto& source = phase.inputs[3 + Component];
  for (int axis = 0; axis < 3; ++axis)
    if (p[axis] >= static_cast<int>(source.dims[axis])) return;
  constexpr int first[3] = {2, 1, 0};
  constexpr int second[3] = {5, 4, 3};
  float gradient = GatherTerm<Phase, first[Component]>(launch, source, p) +
                   GatherTerm<Phase, second[Component]>(launch, source, p);
  if (launch.accumulate_sources) {
    float base = Read(launch.source_cotangents[Component], p);
    if constexpr (Phase == 0) {
      // E's diagonal transpose is added only after the H transpose has read
      // its magnetic cotangents. PEC constraints mask this diagonal, not the
      // pre-E fields consumed by the forward H stencil.
      constexpr int normal = 2 - Component;
      for (int axis = 0; axis < 3; ++axis) {
        if (axis == normal) continue;
        if ((p[axis] == 0 && (phase.metallic_edges & (1 << (2 * axis)))) ||
            (p[axis] == source.dims[axis] - 1 &&
             (phase.metallic_edges & (1 << (2 * axis + 1))))) base = 0.0f;
      }
      base *= Read(launch.source_decays[Component], p);
    }
    gradient += base;
  }
  static_cast<float*>(launch.gradients[3 + Component].data)[Offset(source, p)] = gradient;
}

template <int Phase, int Component, int CoefficientMode>
__device__ __forceinline__ void ComponentAdjoint(const BeamzAdjointLaunch& launch,
                                                 const int* p) {
  constexpr int component = Component;
  const auto& phase = launch.phase;
  const auto& field = phase.inputs[component];
  GatherSource<Phase, Component>(launch, p);
  for (int axis = 0; axis < 3; ++axis)
    if (p[axis] >= static_cast<int>(field.dims[axis])) return;
  const int index = Offset(field, p);
  const float g = Masked<Phase, component>(phase, p)
                      ? 0.0f : Read(phase.outputs[component], p);
  const float sign = Phase == 0 ? -1.0f : 1.0f;
  const float source = Read(phase.inputs[9 + component], p);
  const float gcurl = sign * source * g;
  if (Active(launch, component))
    static_cast<float*>(launch.gradients[component].data)[index] =
        Read(phase.inputs[6 + component], p) * g;
  if (CoefficientMode == 2 && Active(launch, 6 + component))
    Accumulate(launch.gradients[6 + component], field, p, Read(field, p) * g);

  constexpr int sources[6] = {2, 1, 0, 2, 1, 0};
  constexpr int axes[6] = {1, 0, 0, 2, 2, 1};
  float curl = 0.0f;
  #pragma unroll
  for (int which = 0; which < 2; ++which) {
    const int term = 2 * component + which;
    const int axis = axes[term];
    const int input = 3 + sources[term];
    const float curl_sign = which == 0 ? 1.0f : -1.0f;
    int high[3], low[3];
    bool has_high, has_low;
    float scale;
    const bool material_gradient = CoefficientMode == 1 ||
        (CoefficientMode == 2 && Active(launch, 9 + component));
    const bool metric_gradient =
        CoefficientMode == 2 && phase.metric_kind != 0 &&
        Active(launch, 13 + 4 * phase.nterms + axis);
    const int base = 13 + 3 * term;
    const bool derivative_needed = material_gradient || metric_gradient ||
        (CoefficientMode == 2 && phase.nterms &&
         (Active(launch, base) || Active(launch, base + 2)));
    const float raw = derivative_needed ?
        Difference<Phase>(phase, input, axis, p, high, low, &has_high, &has_low, &scale) : 0.0f;
    if (!derivative_needed) scale = 0.0f;
    const float derivative = raw * scale;
    float gd = curl_sign * gcurl;
    float corrected = derivative;
    if (phase.nterms) {
      const auto* metadata = static_cast<const int32_t*>(phase.inputs[12].data);
      const int nlow = metadata[5 * term + 2];
      const int nhigh = metadata[5 * term + 3];
      const int stop = field.dims[axis];
      const int packed = p[axis] < nlow ? p[axis] :
          (p[axis] >= stop - nhigh ? nlow + p[axis] - stop + nhigh : -1);
      if (packed >= 0) {
        int q[3] = {p[0], p[1], p[2]};
        q[axis] = packed;
        const int psi = 13 + 3 * phase.nterms + term;
        const float a = Read(phase.inputs[base], q);
        const float b = Read(phase.inputs[base + 1], q);
        const float k = Read(phase.inputs[base + 2], q);
        const float old = (material_gradient ||
            (CoefficientMode == 2 && Active(launch, base + 1))) ?
            Read(phase.inputs[psi], q) : 0.0f;
        const float gp = ReadPsiCotangent(launch, term, q) + gd;
        if (material_gradient) corrected = k * derivative + (b * old + a * derivative);
        if (CoefficientMode == 2 && Active(launch, base)) Add(launch.gradients[base], q, derivative * gp);
        if (CoefficientMode == 2 && Active(launch, base + 1)) Add(launch.gradients[base + 1], q, old * gp);
        if (CoefficientMode == 2 && Active(launch, base + 2)) Add(launch.gradients[base + 2], q, derivative * gd);
        if (Active(launch, psi))
          static_cast<float*>(launch.gradients[psi].data)[Offset(phase.inputs[psi], q) +
              (CotangentBank(launch) ^ 1) * launch.psi_bank_stride[term]] =
              b * gp;
        gd = k * gd + a * gp;
      }
    }
    if (material_gradient) curl += curl_sign * corrected;
    if (metric_gradient) {
      const auto& metric = launch.metric_gradients[axis];
      auto* address = static_cast<float*>(metric.data) +
                      (metric.rank == 0 ? 0 : p[axis]);
      if (metric.rank == 0 || axis != 2) WarpAdd(address, raw * gd);
      else atomicAdd(address, raw * gd);
    }
  }
  if (CoefficientMode == 1 || (CoefficientMode == 2 && Active(launch, 9 + component)))
    Accumulate(launch.gradients[9 + component], field, p,
               sign * curl * g +
               ((launch.accumulate_material &&
                 SameShape(launch.gradients[9 + component], field))
                ? Read(launch.gradients[9 + component], p) : 0.0f));
}

template <int Phase, int CoefficientMode>
__global__ void PhaseAdjoint(BeamzAdjointLaunch launch) {
  const int p[3] = {static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z),
                    static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y),
                    static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
  ComponentAdjoint<Phase, 0, CoefficientMode>(launch, p);
  ComponentAdjoint<Phase, 1, CoefficientMode>(launch, p);
  ComponentAdjoint<Phase, 2, CoefficientMode>(launch, p);
}

// The chunk transpose has fixed active columns and contiguous Yee fields.
// Prepare their addressing once, outside the kernel, including zero strides
// for broadcast coefficients. The generic kernel still handles all other VJPs.
struct AdjointView {
  float *data;
  int stride[3];
};

struct AdjointField {
  AdjointView view;
  int size[3];
};

struct MaterialAdjoint {
  AdjointField target[3], source[3];
  AdjointView target_cotangent[3], target_gradient[3];
  AdjointView source_cotangent[3], source_gradient[3], source_decay[3];
  AdjointView decay[3], coefficient[3], material_gradient[3], curl[3], old_e[3],
      decay_gradient[3];
  AdjointView psi_old[6], psi_cotangent[6], psi_gradient[6];
  AdjointView cpml_a[6], cpml_b[6], cpml_k[6];
  const float *metric[3];
  const int *metadata;
  int uniform_cpml_thickness;
  int metallic_edges;
  int psi_bank_stride[6];
  const int *reverse_iteration;
};

AdjointView View(const BeamzBuffer &buffer, bool broadcast = false) {
  AdjointView result{static_cast<float *>(buffer.data), {0, 0, 0}};
  if (buffer.rank == 3) {
    result.stride[0] =
        broadcast && buffer.dims[0] == 1 ? 0 : buffer.plane_stride;
    result.stride[1] = broadcast && buffer.dims[1] == 1 ? 0 : buffer.row_stride;
    result.stride[2] = broadcast && buffer.dims[2] == 1 ? 0 : 1;
  }
  return result;
}

AdjointField Field(const BeamzBuffer &buffer) {
  return {View(buffer),
          {int(buffer.dims[0]), int(buffer.dims[1]), int(buffer.dims[2])}};
}

MaterialAdjoint PrepareMaterialAdjoint(const BeamzAdjointLaunch &launch,
                                       const BeamzBuffer *curls,
                                       const BeamzBuffer *old_e,
                                       const BeamzBuffer *decay_gradients) {
  MaterialAdjoint result{};
  const auto &phase = launch.phase;
  result.metallic_edges = phase.metallic_edges;
  result.metadata = static_cast<const int *>(phase.inputs[12].data);
  result.uniform_cpml_thickness = phase.uniform_cpml_thickness;
  result.reverse_iteration =
      static_cast<const int *>(launch.reverse_iteration.data);
  for (int c = 0; c < 3; ++c) {
    result.target[c] = Field(phase.inputs[c]);
    result.source[c] = Field(phase.inputs[3 + c]);
    result.target_cotangent[c] = View(phase.outputs[c]);
    result.target_gradient[c] = View(launch.gradients[c]);
    result.source_cotangent[c] = View(launch.source_cotangents[c]);
    result.source_gradient[c] = View(launch.gradients[3 + c]);
    result.source_decay[c] = View(launch.source_decays[c], true);
    result.decay[c] = View(phase.inputs[6 + c], true);
    result.coefficient[c] = View(phase.inputs[9 + c], true);
    result.material_gradient[c] = View(launch.gradients[9 + c]);
    if (curls)
      result.curl[c] = View(curls[c]);
    if (old_e) {
      result.old_e[c] = View(old_e[c]);
      result.decay_gradient[c] = View(decay_gradients[c]);
    }
    const auto &metric = phase.metrics[c];
    result.metric[c] = phase.metric_kind == 0
                           ? nullptr
                           : static_cast<const float *>(metric.data);
  }
  for (int t = 0; t < phase.nterms; ++t) {
    const int psi = 13 + 3 * phase.nterms + t;
    result.psi_old[t] = View(phase.inputs[psi]);
    result.psi_cotangent[t] = View(phase.outputs[3 + t]);
    result.psi_gradient[t] = View(launch.gradients[psi]);
    result.psi_bank_stride[t] = launch.psi_bank_stride[t];
    result.cpml_a[t] = View(phase.inputs[13 + 3 * t], true);
    result.cpml_b[t] = View(phase.inputs[14 + 3 * t], true);
    result.cpml_k[t] = View(phase.inputs[15 + 3 * t], true);
  }
  return result;
}

template <int Term> __host__ __device__ constexpr auto CurlTerm() {
  if constexpr (Term & 1)
    return beamz::cuda::yee::SecondCurlTerm(Term / 2);
  return beamz::cuda::yee::FirstCurlTerm(Term / 2);
}

__device__ __forceinline__ int Index(const AdjointView &view, const int *p) {
  return p[0] * view.stride[0] + p[1] * view.stride[1] + p[2] * view.stride[2];
}

__device__ __forceinline__ float Read(const AdjointView &view, const int *p) {
  return view.data[Index(view, p)];
}

__device__ __forceinline__ bool Contains(const AdjointField &field,
                                         const int *p) {
  return p[0] < field.size[0] && p[1] < field.size[1] && p[2] < field.size[2];
}

template <int Phase, int Component>
__device__ __forceinline__ bool Wall(int edges, const AdjointField &field,
                                     const int *p) {
  return beamz::cuda::yee::PecConstrained(field.size, Phase, Component, edges,
                                          p[0], p[1], p[2]);
}

template <int Term, bool UniformSlabs>
__device__ __forceinline__ int PackedCoordinate(const MaterialAdjoint &data,
                                                int size, int p) {
  const int low =
      UniformSlabs ? data.uniform_cpml_thickness : data.metadata[5 * Term + 2];
  const int high = UniformSlabs ? low : data.metadata[5 * Term + 3];
  return p < low ? p : (p >= size - high ? low + p - size + high : -1);
}

template <int Axis, int Metric>
__device__ __forceinline__ float MetricAt(const MaterialAdjoint &data,
                                          float inv_dx, int p) {
  if constexpr (Metric == 0)
    return inv_dx;
  if constexpr (Metric == 1)
    return data.metric[Axis][0];
  return data.metric[Axis][p];
}

template <int Phase, int Term, int Metric, bool Cpml, bool UniformSlabs>
__device__ __forceinline__ float WeightedCotangent(const MaterialAdjoint &data,
                                                   const int *p, float inv_dx,
                                                   int bank) {
  constexpr int component = Term / 2;
  constexpr int axis = CurlTerm<Term>().derivative_axis;
  const auto &target = data.target[component];
  constexpr float sign =
      ((Term & 1) ? -1.0f : 1.0f) * (Phase == 0 ? -1.0f : 1.0f);
  const float g = Wall<Phase, component>(data.metallic_edges, target, p)
                      ? 0.0f
                      : Read(data.target_cotangent[component], p);
  float weighted = sign * Read(data.coefficient[component], p) * g;
  if constexpr (Cpml) {
    const int packed =
        PackedCoordinate<Term, UniformSlabs>(data, target.size[axis], p[axis]);
    if (packed >= 0) {
      int q[3] = {p[0], p[1], p[2]};
      q[axis] = packed;
      const auto &psi = data.psi_cotangent[Term];
      const float gp =
          psi.data[Index(psi, q) + bank * data.psi_bank_stride[Term]] +
          weighted;
      weighted = Read(data.cpml_k[Term], q) * weighted +
                 Read(data.cpml_a[Term], q) * gp;
    }
  }
  return MetricAt<axis, Metric>(data, inv_dx, p[axis]) * weighted;
}

template <int Phase, int Term, int Metric, bool Cpml, bool UniformSlabs>
__device__ __forceinline__ float Gather(const MaterialAdjoint &data,
                                        const AdjointField &source,
                                        const int *p, float inv_dx, int bank) {
  constexpr int axis = CurlTerm<Term>().derivative_axis;
  const int coord = p[axis];
  int neighbor[3] = {p[0], p[1], p[2]};
  float g = 0.0f;
  // Phase validation proves equal transverse extents and an exact one-cell
  // offset along this derivative axis. The source-coordinate tests below
  // therefore bound every target sample before WeightedCotangent reads it.
  if constexpr (Phase == 0) {
    if (coord > 0) {
      --neighbor[axis];
      g += WeightedCotangent<Phase, Term, Metric, Cpml, UniformSlabs>(
          data, neighbor, inv_dx, bank);
    }
    if (coord + 1 < source.size[axis])
      g -= WeightedCotangent<Phase, Term, Metric, Cpml, UniformSlabs>(
          data, p, inv_dx, bank);
  } else {
    if (coord > 0 || (data.metallic_edges & (1 << (2 * axis))))
      g += WeightedCotangent<Phase, Term, Metric, Cpml, UniformSlabs>(
          data, p, inv_dx, bank);
    if (coord + 1 < source.size[axis] ||
        (data.metallic_edges & (1 << (2 * axis + 1)))) {
      ++neighbor[axis];
      g -= WeightedCotangent<Phase, Term, Metric, Cpml, UniformSlabs>(
          data, neighbor, inv_dx, bank);
    }
  }
  return g;
}

template <int Term, int Metric>
__device__ __forceinline__ float
PrimalElectricDifference(const MaterialAdjoint &data, const int *p,
                         float inv_dx) {
  constexpr int axis = CurlTerm<Term>().derivative_axis;
  const auto &source = data.source[CurlTerm<Term>().source_component];
  const int coord = p[axis];
  int neighbor[3] = {p[0], p[1], p[2]};
  --neighbor[axis];
  const bool high = coord < source.size[axis] &&
                    (coord > 0 || (data.metallic_edges & (1 << (2 * axis))));
  const bool low = coord > 0 && (coord < source.size[axis] ||
                                 (data.metallic_edges & (1 << (2 * axis + 1))));
  const float raw = (high ? Read(source.view, p) : 0.0f) -
                    (low ? Read(source.view, neighbor) : 0.0f);
  return raw * MetricAt<axis, Metric>(data, inv_dx, coord);
}

template <int Phase, int Term, int Metric, bool Cpml, bool UniformSlabs,
          bool SavedCurl>
__device__ __forceinline__ float
TransposeRecurrence(const MaterialAdjoint &data, const AdjointField &target,
                    const int *p, float inv_dx, int bank, float gcurl) {
  constexpr int axis = CurlTerm<Term>().derivative_axis;
  const float derivative =
      Phase == 1 && !SavedCurl
          ? PrimalElectricDifference<Term, Metric>(data, p, inv_dx)
          : 0.0f;
  if constexpr (Cpml) {
    const int packed =
        PackedCoordinate<Term, UniformSlabs>(data, target.size[axis], p[axis]);
    if (packed >= 0) {
      int q[3] = {p[0], p[1], p[2]};
      q[axis] = packed;
      const float b = Read(data.cpml_b[Term], q);
      const auto &psi = data.psi_cotangent[Term];
      const auto &gradient = data.psi_gradient[Term];
      const int stride = data.psi_bank_stride[Term];
      const float gp = psi.data[Index(psi, q) + bank * stride] +
                       ((Term & 1) ? -gcurl : gcurl);
      gradient.data[Index(gradient, q) + (bank ^ 1) * stride] = b * gp;
      if constexpr (Phase == 1 && !SavedCurl)
        return Read(data.cpml_k[Term], q) * derivative +
               (b * Read(data.psi_old[Term], q) +
                Read(data.cpml_a[Term], q) * derivative);
    }
  }
  return derivative;
}

template <int Phase, int Component, int Metric, bool Cpml, bool UniformSlabs,
          bool SavedCurl>
__device__ __forceinline__ void
TransposeMaterialComponent(const MaterialAdjoint &data, const int *p,
                           float inv_dx, int bank) {
  const auto &source = data.source[Component];
  if (Contains(source, p)) {
    constexpr int first[3] = {2, 1, 0}, second[3] = {5, 4, 3};
    const float gathered =
        Gather<Phase, first[Component], Metric, Cpml, UniformSlabs>(
            data, source, p, inv_dx, bank) +
        Gather<Phase, second[Component], Metric, Cpml, UniformSlabs>(
            data, source, p, inv_dx, bank);
    float base = Read(data.source_cotangent[Component], p);
    if constexpr (Phase == 0) {
      if (Wall<1, Component>(data.metallic_edges, source, p))
        base = 0.0f;
      base *= Read(data.source_decay[Component], p);
    }
    const auto &gradient = data.source_gradient[Component];
    gradient.data[Index(gradient, p)] = gathered + base;
  }

  const auto &target = data.target[Component];
  if (!Contains(target, p))
    return;
  const float g = Wall<Phase, Component>(data.metallic_edges, target, p)
                      ? 0.0f
                      : Read(data.target_cotangent[Component], p);
  constexpr float sign = Phase == 0 ? -1.0f : 1.0f;
  const float gcurl = sign * Read(data.coefficient[Component], p) * g;
  const float first =
      TransposeRecurrence<Phase, 2 * Component, Metric, Cpml, UniformSlabs,
                          SavedCurl>(data, target, p, inv_dx, bank, gcurl);
  const float second =
      TransposeRecurrence<Phase, 2 * Component + 1, Metric, Cpml, UniformSlabs,
                          SavedCurl>(data, target, p, inv_dx, bank, gcurl);
  if constexpr (Phase == 0) {
    const auto &gradient = data.target_gradient[Component];
    gradient.data[Index(gradient, p)] = Read(data.decay[Component], p) * g;
  } else {
    const auto &gradient = data.material_gradient[Component];
    const float curl =
        SavedCurl ? Read(data.curl[Component], p) : first - second;
    gradient.data[Index(gradient, p)] += curl * g;
    if constexpr (SavedCurl) {
      const auto &decay_gradient = data.decay_gradient[Component];
      if (decay_gradient.data)
        decay_gradient.data[Index(decay_gradient, p)] +=
            Read(data.old_e[Component], p) * g;
    }
  }
}

template <int Phase, int Metric, bool Cpml, bool UniformSlabs, bool SavedCurl>
__global__ void
PhaseAdjointMaterial(const __grid_constant__ MaterialAdjoint data,
                     float inv_dx) {
  const int p[3] = {static_cast<int>(blockIdx.z),
                    static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y),
                    static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
  const int bank = data.reverse_iteration ? (*data.reverse_iteration & 1) : 0;
  TransposeMaterialComponent<Phase, 0, Metric, Cpml, UniformSlabs, SavedCurl>(
      data, p, inv_dx, bank);
  TransposeMaterialComponent<Phase, 1, Metric, Cpml, UniformSlabs, SavedCurl>(
      data, p, inv_dx, bank);
  TransposeMaterialComponent<Phase, 2, Metric, Cpml, UniformSlabs, SavedCurl>(
      data, p, inv_dx, bank);
}

template <int Phase, int Metric, bool SavedCurl>
void EnqueueMaterialAdjoint(cudaStream_t stream,
                            const BeamzAdjointLaunch &launch, dim3 blocks,
                            dim3 threads, const BeamzBuffer *curls,
                            const BeamzBuffer *old_e = nullptr,
                            const BeamzBuffer *decay_gradients = nullptr) {
  const auto data =
      PrepareMaterialAdjoint(launch, curls, old_e, decay_gradients);
  const float inv_dx = launch.phase.inv_resolution;
  if (!launch.phase.nterms)
    PhaseAdjointMaterial<Phase, Metric, false, false, SavedCurl>
        <<<blocks, threads, 0, stream>>>(data, inv_dx);
  else if (launch.phase.uniform_cpml_thickness)
    PhaseAdjointMaterial<Phase, Metric, true, true, SavedCurl>
        <<<blocks, threads, 0, stream>>>(data, inv_dx);
  else
    PhaseAdjointMaterial<Phase, Metric, true, false, SavedCurl>
        <<<blocks, threads, 0, stream>>>(data, inv_dx);
}

template <int Phase>
cudaError_t LaunchPhase(cudaStream_t stream, const BeamzAdjointLaunch &launch,
                        const BeamzBuffer *curls, const BeamzBuffer *old_e,
                        const BeamzBuffer *decay_gradients) {
  int sizes[3] = {};
  for (int field = 0; field < 6; ++field)
    for (int axis = 0; axis < 3; ++axis)
      if (launch.phase.inputs[field].dims[axis] > sizes[axis])
        sizes[axis] = static_cast<int>(launch.phase.inputs[field].dims[axis]);
  const dim3 threads(32, 4, 1);
  const dim3 blocks((sizes[2] + 31) / 32, (sizes[1] + 3) / 4, sizes[0]);
  // Specialize the common lossless inverse-design cases, while keeping the
  // complete coefficient/CPML/metric VJP for callers who train those inputs.
  constexpr uint64_t coefficient_bits =
      (((uint64_t{1} << 31) - 1) & ~((uint64_t{1} << 6) - 1) & ~(uint64_t{1} << 12)) |
      (uint64_t{7} << 37);
  const uint64_t coefficients = launch.gradient_mask & coefficient_bits;
  const uint64_t psi_mask = launch.phase.nterms ? uint64_t{63} << 31 : 0;
  const bool material =
      launch.accumulate_sources &&
      (Phase == 0
           ? launch.gradient_mask == (psi_mask | 63)
           : launch.accumulate_material &&
                 launch.gradient_mask == (psi_mask | 56 | (uint64_t{7} << 9)) &&
                 SameShape(launch.gradients[9], launch.phase.inputs[0]) &&
                 SameShape(launch.gradients[10], launch.phase.inputs[1]) &&
                 SameShape(launch.gradients[11], launch.phase.inputs[2]));
  if (material) {
    if constexpr (Phase == 1) {
      if (curls) {
        if (launch.phase.metric_kind == 0)
          EnqueueMaterialAdjoint<1, 0, true>(stream, launch, blocks, threads,
                                             curls, old_e, decay_gradients);
        else if (launch.phase.metric_kind == 1)
          EnqueueMaterialAdjoint<1, 1, true>(stream, launch, blocks, threads,
                                             curls, old_e, decay_gradients);
        else
          EnqueueMaterialAdjoint<1, 2, true>(stream, launch, blocks, threads,
                                             curls, old_e, decay_gradients);
        return cudaPeekAtLastError();
      }
    }
    if (launch.phase.metric_kind == 0)
      EnqueueMaterialAdjoint<Phase, 0, false>(stream, launch, blocks, threads,
                                              nullptr);
    else if (launch.phase.metric_kind == 1)
      EnqueueMaterialAdjoint<Phase, 1, false>(stream, launch, blocks, threads,
                                              nullptr);
    else
      EnqueueMaterialAdjoint<Phase, 2, false>(stream, launch, blocks, threads,
                                              nullptr);
    return cudaPeekAtLastError();
  }
  if (curls)
    return cudaErrorInvalidValue;
  if (coefficients == 0)
    PhaseAdjoint<Phase, 0><<<blocks, threads, 0, stream>>>(launch);
  else if (coefficients == (uint64_t{7} << 9))
    PhaseAdjoint<Phase, 1><<<blocks, threads, 0, stream>>>(launch);
  else
    PhaseAdjoint<Phase, 2><<<blocks, threads, 0, stream>>>(launch);
  return cudaPeekAtLastError();
}

cudaError_t Zero(cudaStream_t stream, const BeamzBuffer& value) {
  const int64_t count = Elements(value);
  return count == 0 ? cudaSuccess :
      cudaMemsetAsync(value.data, 0, count * sizeof(float), stream);
}

}  // namespace

static int LaunchAdjoint(void *stream, const BeamzAdjointLaunch &launch,
                         const BeamzBuffer *curls, const BeamzBuffer *old_e,
                         const BeamzBuffer *decay_gradients) {
  const auto& phase = launch.phase;
  if (auto error = BeamzValidatePhase(phase); error != cudaSuccess) return error;
  const int payload = 13 + 4 * phase.nterms;
  // Packed material codebooks have no exact material VJP.
  // The differentiable planner supplies ordinary float32 coefficients instead.
  for (int index = 0; index < payload; ++index) {
    if (index == 12) continue;
    if (phase.inputs[index].element_type != kBeamzF32 ||
        launch.gradients[index].element_type != kBeamzF32 ||
        (Active(launch, index) && !SameShape(phase.inputs[index], launch.gradients[index])) ||
        Elements(phase.inputs[index]) > std::numeric_limits<int>::max()) {
      return cudaErrorInvalidValue;
    }
  }
  for (int component = 0; component < 3; ++component) {
    for (int index : {6 + component, 9 + component}) {
      const auto& value = phase.inputs[index];
      if (value.rank != 0 && value.rank != 3) return cudaErrorInvalidValue;
      for (int axis = 0; value.rank == 3 && axis < 3; ++axis) {
        if (value.dims[axis] != 1 &&
            value.dims[axis] != phase.inputs[component].dims[axis]) {
          return cudaErrorInvalidValue;
        }
      }
    }
  }
  for (int axis = 0; axis < 3; ++axis) {
    if ((Active(launch, payload + axis) &&
         !SameShape(phase.metrics[axis], launch.metric_gradients[axis])) ||
        launch.metric_gradients[axis].element_type != kBeamzF32) {
      return cudaErrorInvalidValue;
    }
  }
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  for (int index = 0; index < payload; ++index) {
    if (index == 12) continue;
    if (launch.accumulate_material && index >= 9 && index < 12) continue;
    // Fields, recurrence memories, and whole-grid coefficients have exactly
    // one writer per element. Only broadcast reductions need initialization.
    if (index < 6 || index >= 13 + 3 * phase.nterms) continue;
    if (index >= 6 && index < 12 &&
        SameShape(launch.gradients[index], phase.inputs[(index - 6) % 3])) continue;
    if (auto error = Zero(cuda_stream, launch.gradients[index]);
        error != cudaSuccess) return error;
  }
  for (const auto& metric : launch.metric_gradients) {
    if (auto error = Zero(cuda_stream, metric); error != cudaSuccess) return error;
  }
  return phase.phase == 0 ? LaunchPhase<0>(cuda_stream, launch, curls, old_e,
                                           decay_gradients)
                          : LaunchPhase<1>(cuda_stream, launch, curls, old_e,
                                           decay_gradients);
}

int BeamzLaunchAdjoint(void *stream, const BeamzAdjointLaunch &launch) {
  return LaunchAdjoint(stream, launch, nullptr, nullptr, nullptr);
}

int BeamzLaunchAdjointFromCurl(void *stream, const BeamzAdjointLaunch &launch,
                               const BeamzBuffer *curls,
                               const BeamzBuffer *old_e,
                               const BeamzBuffer *decay_gradients) {
  return LaunchAdjoint(stream, launch, curls, old_e, decay_gradients);
}
