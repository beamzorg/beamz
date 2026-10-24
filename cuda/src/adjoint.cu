// Exact first-order transpose of the streamed Yee/CPML phase. Inputs and
// cotangents are read-only: reverse-mode residuals must survive recomputation.
#include <cuda_runtime_api.h>

#include <cstdint>
#include <initializer_list>
#include <limits>

#include "kernels.h"
#include "launch.h"
#include "boundary.h"

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
    *has_high = beamz::cuda::boundary::DerivativeFactor(launch.metallic_edges, launch.magnetic_edges, axis, false) != 0;
  } else if (coord == value.dims[axis]) {
    *has_high = false;
    low[axis]--;
    *has_low = beamz::cuda::boundary::DerivativeFactor(launch.metallic_edges, launch.magnetic_edges, axis, true) != 0;
  } else {
    low[axis]--;
  }
  *scale = launch.inv_resolution;
  if (launch.metric_kind != 0) {
    const auto& metric = launch.metrics[axis];
    *scale = static_cast<const float*>(metric.data)[metric.rank == 0 ? 0 : coord];
  }
  float factor = 1.0f;
  if constexpr (Phase == 1) {
    if (coord == 0 || coord == value.dims[axis])
      factor = beamz::cuda::boundary::DerivativeFactor(
          launch.metallic_edges, launch.magnetic_edges, axis, coord != 0);
  }
  return factor * ((*has_high ? Read(value, high) : 0.0f) -
                   (*has_low ? Read(value, low) : 0.0f));
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
    const float low_factor = p[axis] > 0 ? 1.0f :
        beamz::cuda::boundary::DerivativeFactor(
            phase.metallic_edges, phase.magnetic_edges, axis, false);
    const float high_factor = p[axis] + 1 < source.dims[axis] ? 1.0f :
        beamz::cuda::boundary::DerivativeFactor(
            phase.metallic_edges, phase.magnetic_edges, axis, true);
    if (low_factor)
      gradient += low_factor * DifferenceCotangent<Phase, Term>(launch, p);
    if (high_factor) {
      neighbor[axis]++;
      gradient -= high_factor * DifferenceCotangent<Phase, Term>(launch, neighbor);
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

template <int Phase>
cudaError_t LaunchPhase(cudaStream_t stream, const BeamzAdjointLaunch& launch) {
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

int BeamzLaunchAdjoint(void* stream, const BeamzAdjointLaunch& launch) {
  const auto& phase = launch.phase;
  if (auto error = BeamzValidatePhase(phase); error != cudaSuccess) return error;
  const int payload = 13 + 4 * phase.nterms;
  // Packed material codebooks and BF16 rounding have no exact material VJP.
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
  return phase.phase == 0 ? LaunchPhase<0>(cuda_stream, launch) :
                            LaunchPhase<1>(cuda_stream, launch);
}
