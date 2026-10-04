#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "kernels.h"
#include "launch.h"

namespace {

constexpr int kTileX = 32;
constexpr int kTileY = 4;
constexpr int kTileZ = 2;

dim3 SourceThreads(int64_t x_extent) {
  if (x_extent >= 32) {
    return dim3(kTileX, kTileY, kTileZ);
  }
  if (x_extent <= 1) return dim3(1, 64, 2);
  if (x_extent <= 2) return dim3(2, 16, 4);
  if (x_extent <= 4) return dim3(4, 16, 2);
  if (x_extent <= 8) return dim3(8, 8, 2);
  return dim3(16, 4, 2);
}

struct DftFields {
  BeamzBuffer values[6];
  float dt;
};

DftFields MakeDftFields(const BeamzLaunch& h_launch,
                        const BeamzLaunch& e_launch) {
  DftFields fields{};
  for (int component = 0; component < 3; ++component) {
    fields.values[component] = e_launch.outputs[component];
    fields.values[3 + component] = h_launch.outputs[component];
  }
  fields.dt = e_launch.dt;
  return fields;
}

__device__ __forceinline__ int64_t ElementCount(const BeamzBuffer& value) {
  int64_t elements = 1;
  for (int axis = 0; axis < value.rank; ++axis) elements *= value.dims[axis];
  return elements;
}

__device__ __forceinline__ bool SourceCellConstrained(
    const BeamzBuffer& target, int component, int phase, int metallic_edges,
    int z, int y, int x) {
  const int normal_axis = 2 - component;
  const int coordinates[3] = {z, y, x};
  if (phase == 0) {
    const int coordinate = coordinates[normal_axis];
    return (coordinate == 0 &&
            (metallic_edges & (1 << (2 * normal_axis)))) ||
           (coordinate == target.dims[normal_axis] - 1 &&
            (metallic_edges & (1 << (2 * normal_axis + 1))));
  }
  for (int axis = 0; axis < 3; ++axis) {
    if (axis == normal_axis) continue;
    const int coordinate = coordinates[axis];
    if ((coordinate == 0 && (metallic_edges & (1 << (2 * axis)))) ||
        (coordinate == target.dims[axis] - 1 &&
         (metallic_edges & (1 << (2 * axis + 1))))) {
      return true;
    }
  }
  return false;
}

template <bool Atomic>
__device__ __forceinline__ void ApplySourceGroupCell(
    BeamzBuffer target, BeamzSourceGroupLaunch group, int source_index,
    int step_offset, int metallic_edges, int z, int y, int x) {
  if (z >= group.coefficients.dims[1] ||
      y >= group.coefficients.dims[2] ||
      x >= group.coefficients.dims[3]) {
    return;
  }
  const auto* starts = static_cast<const int32_t*>(group.starts.data);
  const int64_t target_z = static_cast<int64_t>(starts[3 * source_index]) + z;
  const int64_t target_y =
      static_cast<int64_t>(starts[3 * source_index + 1]) + y;
  const int64_t target_x =
      static_cast<int64_t>(starts[3 * source_index + 2]) + x;
  if (target_z < 0 || target_z >= target.dims[0] || target_y < 0 ||
      target_y >= target.dims[1] || target_x < 0 ||
      target_x >= target.dims[2]) {
    return;
  }
  // Sources injected after a field update are followed by PEC restoration in the
  // canonical step. Skipping those constrained additions is equivalent because
  // the native field update has already written zero to every constrained cell.
  if (group.timing != 0 &&
      SourceCellConstrained(target, group.component,
                            group.timing == 1 ? 0 : 1, metallic_edges,
                            static_cast<int>(target_z),
                            static_cast<int>(target_y),
                            static_cast<int>(target_x))) {
    return;
  }
  int waveform_index =
      static_cast<const int32_t*>(group.current_step.data)[0] + step_offset;
  waveform_index = waveform_index < 0 ? 0 : waveform_index;
  waveform_index =
      waveform_index >= group.waveforms.dims[1]
          ? static_cast<int>(group.waveforms.dims[1]) - 1
          : waveform_index;
  const int waveform_offset =
      source_index * static_cast<int>(group.waveforms.dims[1]) +
      waveform_index;
  const int coefficient_offset =
      ((source_index * static_cast<int>(group.coefficients.dims[1]) + z) *
           static_cast<int>(group.coefficients.dims[2]) +
       y) *
          static_cast<int>(group.coefficients.dims[3]) +
      x;
  const int target_offset = BeamzOffset3D(target, target_z, target_y, target_x);
  const float contribution =
      static_cast<const float*>(group.coefficients.data)[coefficient_offset] *
      static_cast<const float*>(group.waveforms.data)[waveform_offset];
  if constexpr (Atomic) {
    atomicAdd(static_cast<float*>(target.data) + target_offset, contribution);
  } else {
    static_cast<float*>(target.data)[target_offset] += contribution;
  }
}

template <int Timing>
__global__ void ApplySingleSourceGroup(BeamzBuffer target,
                                       BeamzSourceGroupLaunch group,
                                       int step_offset,
                                       int metallic_edges) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (z >= group.coefficients.dims[1] ||
      y >= group.coefficients.dims[2] ||
      x >= group.coefficients.dims[3]) {
    return;
  }
  const auto* starts = static_cast<const int32_t*>(group.starts.data);
  const int64_t target_z = static_cast<int64_t>(starts[0]) + z;
  const int64_t target_y = static_cast<int64_t>(starts[1]) + y;
  const int64_t target_x = static_cast<int64_t>(starts[2]) + x;
  if (target_z < 0 || target_z >= target.dims[0] || target_y < 0 ||
      target_y >= target.dims[1] || target_x < 0 ||
      target_x >= target.dims[2]) {
    return;
  }
  if constexpr (Timing != 0) {
    if (SourceCellConstrained(target, group.component, Timing == 1 ? 0 : 1,
                              metallic_edges, static_cast<int>(target_z),
                              static_cast<int>(target_y),
                              static_cast<int>(target_x))) {
      return;
    }
  }
  int waveform_index =
      static_cast<const int32_t*>(group.current_step.data)[0] + step_offset;
  waveform_index = waveform_index < 0 ? 0 : waveform_index;
  waveform_index =
      waveform_index >= group.waveforms.dims[1]
          ? static_cast<int>(group.waveforms.dims[1]) - 1
          : waveform_index;
  const int coefficient_offset =
      (z * static_cast<int>(group.coefficients.dims[2]) + y) *
          static_cast<int>(group.coefficients.dims[3]) +
      x;
  const int target_offset = BeamzOffset3D(target, target_z, target_y, target_x);
  static_cast<float*>(target.data)[target_offset] +=
      static_cast<const float*>(group.coefficients.data)[coefficient_offset] *
      static_cast<const float*>(group.waveforms.data)[waveform_index];
}

__device__ __forceinline__ void ApplyCoincidentSourceGroupCell(
    BeamzBuffer target, BeamzSourceGroupLaunch group, int step_offset,
    int metallic_edges, int z, int y, int x) {
  if (z >= group.coefficients.dims[1] ||
      y >= group.coefficients.dims[2] ||
      x >= group.coefficients.dims[3]) {
    return;
  }
  const auto* starts = static_cast<const int32_t*>(group.starts.data);
  const int64_t target_z = static_cast<int64_t>(starts[0]) + z;
  const int64_t target_y = static_cast<int64_t>(starts[1]) + y;
  const int64_t target_x = static_cast<int64_t>(starts[2]) + x;
  if (target_z < 0 || target_z >= target.dims[0] || target_y < 0 ||
      target_y >= target.dims[1] || target_x < 0 ||
      target_x >= target.dims[2]) {
    return;
  }
  if (group.timing != 0 &&
      SourceCellConstrained(target, group.component,
                            group.timing == 1 ? 0 : 1, metallic_edges,
                            static_cast<int>(target_z),
                            static_cast<int>(target_y),
                            static_cast<int>(target_x))) {
    return;
  }
  int waveform_index =
      static_cast<const int32_t*>(group.current_step.data)[0] + step_offset;
  waveform_index = waveform_index < 0 ? 0 : waveform_index;
  waveform_index =
      waveform_index >= group.waveforms.dims[1]
          ? static_cast<int>(group.waveforms.dims[1]) - 1
          : waveform_index;
  const int target_offset = BeamzOffset3D(target, target_z, target_y, target_x);
  float value = static_cast<float*>(target.data)[target_offset];
  const int coefficient_stride =
      static_cast<int>(group.coefficients.dims[1] *
                       group.coefficients.dims[2] *
                       group.coefficients.dims[3]);
  const int coefficient_cell =
      (z * static_cast<int>(group.coefficients.dims[2]) + y) *
          static_cast<int>(group.coefficients.dims[3]) +
      x;
  const auto* coefficients =
      static_cast<const float*>(group.coefficients.data);
  const auto* waveforms = static_cast<const float*>(group.waveforms.data);
  const int waveform_stride = static_cast<int>(group.waveforms.dims[1]);
  for (int source = 0; source < group.coefficients.dims[0]; ++source) {
    // Source products are formed before the scatter-add in the JAX reference.
    value = __fadd_rn(
        value, coefficients[source * coefficient_stride + coefficient_cell] *
                   waveforms[source * waveform_stride + waveform_index]);
  }
  static_cast<float*>(target.data)[target_offset] = value;
}

template <bool Atomic>
__global__ void ApplySourceGroupBatched(BeamzBuffer target,
                                        BeamzSourceGroupLaunch group,
                                        int z_blocks, int step_offset,
                                        int metallic_edges) {
  const int source_index = blockIdx.z / z_blocks;
  const int source_block_z = blockIdx.z - source_index * z_blocks;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = source_block_z * blockDim.z + threadIdx.z;
  // The compiler marks only statically disjoint slabs as non-atomic. The
  // conservative default preserves additive semantics for every other group.
  ApplySourceGroupCell<Atomic>(target, group, source_index, step_offset,
                               metallic_edges, z, y, x);
}

__global__ void ApplyCoincidentSourceGroup(BeamzBuffer target,
                                            BeamzSourceGroupLaunch group,
                                            int step_offset,
                                            int metallic_edges) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  ApplyCoincidentSourceGroupCell(target, group, step_offset, metallic_edges, z,
                                 y, x);
}

struct SourcePhasePlan {
  BeamzBuffer targets[3];
  BeamzSourceGroupLaunch groups[3];
  int ends[3];
  int x_blocks[3];
  int y_blocks[3];
  int z_blocks[3];
  int low;
  int high[3];
};

__global__ void ApplySourcePhase(SourcePhasePlan plan, int step,
                                 int metallic_edges) {
  const int block = blockIdx.x;
  const int index = block < plan.ends[0] ? 0 : (block < plan.ends[1] ? 1 : 2);
  int local = block - (index == 0 ? 0 : plan.ends[index - 1]);
  const int bx = local % plan.x_blocks[index];
  local /= plan.x_blocks[index];
  const int by = local % plan.y_blocks[index];
  local /= plan.y_blocks[index];
  const auto &group = plan.groups[index];
  const int x = bx * blockDim.x + threadIdx.x;
  const int y = by * blockDim.y + threadIdx.y;
  const int source = local / plan.z_blocks[index];
  const int z = (local % plan.z_blocks[index]) * blockDim.z + threadIdx.z;
  if (plan.low > 0) {
    const auto *starts = static_cast<const int32_t *>(group.starts.data);
    const int tz = starts[3 * source] + z;
    const int ty = starts[3 * source + 1] + y;
    const int tx = starts[3 * source + 2] + x;
    if (tz >= plan.low && ty >= plan.low && tx >= plan.low &&
        tz < plan.high[0] && ty < plan.high[1] && tx < plan.high[2])
      return;
  }
  if (group.coincident && group.coefficients.dims[0] > 1) {
    ApplyCoincidentSourceGroupCell(plan.targets[index], group, step,
                                   metallic_edges, z, y, x);
  } else if (group.disjoint || group.coefficients.dims[0] == 1) {
    ApplySourceGroupCell<false>(plan.targets[index], group, source, step,
                                metallic_edges, z, y, x);
  } else {
    ApplySourceGroupCell<true>(plan.targets[index], group, source, step,
                               metallic_edges, z, y, x);
  }
}

cudaError_t LaunchSourceGroup(cudaStream_t stream, const BeamzLaunch& launch,
                              const BeamzBuffer& target,
                              const BeamzSourceGroupLaunch& group,
                              int32_t step) {
  const dim3 threads = SourceThreads(group.coefficients.dims[3]);
  const int z_blocks =
      (group.coefficients.dims[1] + threads.z - 1) / threads.z;
  const bool single = group.coefficients.dims[0] == 1;
  const bool coincident = !single && group.coincident != 0;
  const int launch_z_blocks =
      !single && !coincident ? z_blocks * group.coefficients.dims[0] : z_blocks;
  const dim3 blocks(
      (group.coefficients.dims[3] + threads.x - 1) / threads.x,
      (group.coefficients.dims[2] + threads.y - 1) / threads.y,
      launch_z_blocks);
  if (single) {
    if (group.timing == 0) {
      ApplySingleSourceGroup<0><<<blocks, threads, 0, stream>>>(
          target, group, step, launch.metallic_edges);
    } else if (group.timing == 1) {
      ApplySingleSourceGroup<1><<<blocks, threads, 0, stream>>>(
          target, group, step, launch.metallic_edges);
    } else {
      ApplySingleSourceGroup<2><<<blocks, threads, 0, stream>>>(
          target, group, step, launch.metallic_edges);
    }
    return cudaPeekAtLastError();
  }
  if (coincident) {
    ApplyCoincidentSourceGroup<<<blocks, threads, 0, stream>>>(
        target, group, step, launch.metallic_edges);
    return cudaPeekAtLastError();
  }
  if (group.disjoint != 0) {
    ApplySourceGroupBatched<false><<<blocks, threads, 0, stream>>>(
        target, group, z_blocks, step, launch.metallic_edges);
  } else {
    ApplySourceGroupBatched<true><<<blocks, threads, 0, stream>>>(
        target, group, z_blocks, step, launch.metallic_edges);
  }
  return cudaPeekAtLastError();
}

// Match the JAX invocation clock without rounding a new origin at each graph
// boundary. Source indices still use current_step; only DFT time uses this offset.
__device__ __forceinline__ float ObservationTime(
    const DftFields& fields, const BeamzDftGroupLaunch& monitors, int step_offset) {
  const int64_t step = static_cast<const int32_t*>(monitors.elapsed_steps.data)[0]
                      + static_cast<int64_t>(step_offset) + 1;
  return __fmaf_rn(static_cast<float>(step), fields.dt,
                  static_cast<const float*>(monitors.time.data)[0]);
}

__global__ void PrepareDftPhases(DftFields fields, BeamzDftGroupLaunch monitors,
                                 int step_offset) {
  const int frequency = blockIdx.x * blockDim.x + threadIdx.x;
  const int monitor = blockIdx.y;
  const int max_frequency_count =
      static_cast<int>(monitors.frequencies.dims[1]);
  if (monitor >= monitors.monitor_count || frequency >= max_frequency_count) {
    return;
  }
  const int frequency_offset = monitor * max_frequency_count + frequency;
  const int phase_offset = frequency_offset + blockIdx.z *
                                                  monitors.monitor_count *
                                                  max_frequency_count;
  step_offset += blockIdx.z;
  float window = 0.0f;
  float phase_sin = 0.0f;
  float phase_cos = 0.0f;
  const auto *counts = static_cast<const int32_t *>(monitors.counts.data);
  const int frequency_count = counts[5 * monitor];
  const int interval =
      counts[5 * monitor + 2] > 0 ? counts[5 * monitor + 2] : 1;
  const auto *codes = static_cast<const int32_t *>(monitors.codes.data);
  if (frequency < frequency_count && frequency_count > 0 &&
      frequency_count <= max_frequency_count &&
      (codes[2 * monitor] == 0 || codes[2 * monitor] == 1) &&
      (codes[2 * monitor + 1] == 0 || codes[2 * monitor + 1] == 1)) {
    const int64_t absolute_step =
        static_cast<int64_t>(
            static_cast<const int32_t *>(monitors.current_step.data)[0]) +
        step_offset;
    const float time = ObservationTime(fields, monitors, step_offset);
    const auto *windows = static_cast<const float *>(monitors.windows.data);
    const float start = windows[3 * monitor];
    const float end = windows[3 * monitor + 1];
    if (absolute_step % interval == 0 && time >= start && time <= end) {
      window = 1.0f;
      if (codes[2 * monitor] == 1 && isfinite(end) && end > start) {
        const float tau =
            fminf(fmaxf((time - start) / (end - start), 0.0f), 1.0f);
        window = 0.5f * (1.0f - cosf(6.2831853071795864769f * tau));
      }
      const float frequency_hz = static_cast<const float *>(
          monitors.frequencies.data)[frequency_offset];
      sincosf(6.2831853071795864769f * frequency_hz * time, &phase_sin,
              &phase_cos);
    }
  }
  static_cast<float *>(monitors.phase_window.data)[phase_offset] = window;
  static_cast<float *>(monitors.phase_sin.data)[phase_offset] = phase_sin;
  static_cast<float *>(monitors.phase_cos.data)[phase_offset] = phase_cos;
}

template <bool SingleMonitor, bool CachedPhase>
__global__ void AccumulateDftGroups(DftFields fields,
                                    BeamzDftGroupLaunch monitors,
                                    int step_offset) {
  const int point = blockIdx.x * blockDim.x + threadIdx.x;
  const int frequency = blockIdx.y * blockDim.y + threadIdx.y;
  const int lane = blockIdx.z * blockDim.z + threadIdx.z;
  const int monitor = SingleMonitor ? 0 : lane / 6;
  const int component = SingleMonitor ? lane : lane % 6;
  if constexpr (!SingleMonitor) {
    if (monitor >= monitors.monitor_count) return;
  }

  const auto* counts = static_cast<const int32_t*>(monitors.counts.data);
  const int frequency_count = counts[5 * monitor];
  const int point_count = counts[5 * monitor + 1];
  const int interval = counts[5 * monitor + 2] > 0
                           ? counts[5 * monitor + 2]
                           : 1;
  const int value_offset = counts[5 * monitor + 3];
  const int weight_offset = counts[5 * monitor + 4];
  const int max_frequency_count = static_cast<int>(monitors.frequencies.dims[1]);
  const int max_points = static_cast<int>(monitors.indices.dims[2]);
  if (frequency_count < 1 || frequency_count > max_frequency_count ||
      point_count < 1 || point_count > max_points || value_offset < 0 ||
      weight_offset < 0 || frequency >= frequency_count) {
    return;
  }
  const int64_t value_count =
      6LL * frequency_count * point_count;
  if (static_cast<int64_t>(value_offset) + value_count >
          monitors.dft_re.dims[0] ||
      static_cast<int64_t>(value_offset) + value_count >
          monitors.dft_im.dims[0] ||
      static_cast<int64_t>(weight_offset) + frequency_count >
          monitors.dft_weight.dims[0]) {
    return;
  }

  const auto* codes = static_cast<const int32_t*>(monitors.codes.data);
  if ((codes[2 * monitor] != 0 && codes[2 * monitor] != 1) ||
      (codes[2 * monitor + 1] != 0 && codes[2 * monitor + 1] != 1)) {
    return;
  }
  const auto* windows = static_cast<const float*>(monitors.windows.data);
  float window = 0.0f;
  float phase_sin = 0.0f;
  float phase_cos = 0.0f;
  if constexpr (CachedPhase) {
    const int phase_offset = monitor * max_frequency_count + frequency;
    window = static_cast<const float*>(monitors.phase_window.data)[phase_offset];
    phase_sin = static_cast<const float*>(monitors.phase_sin.data)[phase_offset];
    phase_cos = static_cast<const float*>(monitors.phase_cos.data)[phase_offset];
    if (window == 0.0f) return;
  } else if (threadIdx.x == 0) {
    const int64_t absolute_step =
        static_cast<int64_t>(
            static_cast<const int32_t*>(monitors.current_step.data)[0]) +
        step_offset;
    const float time = ObservationTime(fields, monitors, step_offset);
    const float start = windows[3 * monitor];
    const float end = windows[3 * monitor + 1];
    if (absolute_step % interval == 0 && time >= start && time <= end) {
      window = 1.0f;
      if (codes[2 * monitor] == 1 && isfinite(end) && end > start) {
        const float tau =
            fminf(fmaxf((time - start) / (end - start), 0.0f), 1.0f);
        window = 0.5f * (1.0f - cosf(6.2831853071795864769f * tau));
      }
      const float frequency_hz =
          static_cast<const float*>(monitors.frequencies.data)
              [monitor * max_frequency_count + frequency];
      sincosf(6.2831853071795864769f * frequency_hz * time, &phase_sin,
              &phase_cos);
    }
  }
  window = __shfl_sync(0xffffffff, window, 0);
  phase_sin = __shfl_sync(0xffffffff, phase_sin, 0);
  phase_cos = __shfl_sync(0xffffffff, phase_cos, 0);
  if (window == 0.0f) return;
  if (point >= point_count) return;
  if (component == 0 && point == 0) {
    static_cast<float*>(monitors.dft_weight.data)
        [weight_offset + frequency] += window;
  }
  const float mask = static_cast<const float*>(monitors.component_masks.data)
      [monitor * 6 + component];
  if (mask == 0.0f) return;

  const int neighbors = static_cast<int>(monitors.indices.dims[3]);
  const int plan_base = ((monitor * 6 + component) * max_points + point) *
                        neighbors;
  const BeamzBuffer& field = fields.values[component];
  float sample = 0.0f;
  for (int neighbor = 0; neighbor < neighbors; ++neighbor) {
    const int gather_offset = plan_base + neighbor;
    const int field_offset =
        static_cast<const int32_t*>(monitors.indices.data)[gather_offset];
    if (field_offset >= 0 && field_offset < ElementCount(field)) {
      sample += static_cast<const float*>(
                    field.data)[BeamzPhysicalIndex(field, field_offset)] *
                static_cast<const float*>(monitors.weights.data)[gather_offset];
    }
  }

  float scale = window;
  if (codes[2 * monitor + 1] == 1) {
    const float length_unit = windows[3 * monitor + 2];
    if (!isfinite(length_unit) || length_unit <= 0.0f) return;
    scale *= fields.dt * static_cast<float>(interval) * 299792458.0f /
             length_unit / sqrtf(6.2831853071795864769f);
  }
  const int accumulator_offset =
      value_offset +
      (component * frequency_count + frequency) * point_count +
      point;
  static_cast<float*>(monitors.dft_re.data)[accumulator_offset] +=
      scale * sample * phase_cos;
  static_cast<float*>(monitors.dft_im.data)[accumulator_offset] +=
      scale * sample * phase_sin;
}

// Interpolation geometry is shared by both times and every frequency. Gather
// once per point, retaining the original neighbor accumulation order.
__global__ void
GatherDftPair(const __grid_constant__ DftFields first,
              const __grid_constant__ DftFields second,
              const __grid_constant__ BeamzDftGroupLaunch monitors) {
  const int point = blockIdx.x * blockDim.x + threadIdx.x;
  const int monitor = blockIdx.y / 6, component = blockIdx.y % 6;
  const auto *counts = static_cast<const int *>(monitors.counts.data);
  const int nf = counts[5 * monitor], np = counts[5 * monitor + 1];
  const int max_np = monitors.indices.dims[2],
            max_nf = monitors.frequencies.dims[1];
  if (nf < 1 || nf > max_nf || np < 1 || np > max_np || point >= np ||
      static_cast<const float *>(
          monitors.component_masks.data)[monitor * 6 + component] == 0.f)
    return;
  const auto *windows = static_cast<const float *>(monitors.phase_window.data);
  const bool active0 = windows[monitor * max_nf] != 0.f;
  const bool active1 =
      windows[(monitor + monitors.monitor_count) * max_nf] != 0.f;
  if (!active0 && !active1)
    return;
  const auto &a = first.values[component];
  const auto &b = second.values[component];
  const int neighbors = monitors.indices.dims[3];
  const int base = ((monitor * 6 + component) * max_np + point) * neighbors;
  float sample0 = 0.f, sample1 = 0.f;
  for (int neighbor = 0; neighbor < neighbors; ++neighbor) {
    const int index =
        static_cast<const int *>(monitors.indices.data)[base + neighbor];
    const float weight =
        static_cast<const float *>(monitors.weights.data)[base + neighbor];
    if (index >= 0 && index < ElementCount(a)) {
      if (active0)
        sample0 +=
            static_cast<const float *>(a.data)[BeamzPhysicalIndex(a, index)] *
            weight;
      if (active1)
        sample1 +=
            static_cast<const float *>(b.data)[BeamzPhysicalIndex(b, index)] *
            weight;
    }
  }
  const int output = ((monitor * 6 + component) * max_np + point) * 2;
  static_cast<float *>(monitors.pair_samples.data)[output] = sample0;
  static_cast<float *>(monitors.pair_samples.data)[output + 1] = sample1;
}

// Accumulate both observations in chronological order, retaining the DFT
// accumulator in registers between them. The intermediate field bank is sparse
// but includes every interpolation neighbor in this gather plan.
__global__ void
AccumulateDftPair(const __grid_constant__ DftFields first,
                  const __grid_constant__ DftFields second,
                  const __grid_constant__ BeamzDftGroupLaunch monitors) {
  const int point = blockIdx.x * blockDim.x + threadIdx.x;
  const int frequency = blockIdx.y * blockDim.y + threadIdx.y;
  const int lane = blockIdx.z * blockDim.z + threadIdx.z;
  const int monitor = lane / 6, component = lane % 6;
  if (monitor >= monitors.monitor_count)
    return;
  const auto *counts = static_cast<const int *>(monitors.counts.data);
  const int nf = counts[5 * monitor], np = counts[5 * monitor + 1];
  const int interval = max(1, counts[5 * monitor + 2]);
  const int vo = counts[5 * monitor + 3], wo = counts[5 * monitor + 4];
  const int max_nf = monitors.frequencies.dims[1];
  const int max_np = monitors.indices.dims[2];
  if (nf < 1 || nf > max_nf || np < 1 || np > max_np || vo < 0 || wo < 0 ||
      frequency >= nf || point >= np ||
      static_cast<int64_t>(vo) + 6LL * nf * np > monitors.dft_re.dims[0] ||
      static_cast<int64_t>(vo) + 6LL * nf * np > monitors.dft_im.dims[0] ||
      static_cast<int64_t>(wo) + nf > monitors.dft_weight.dims[0])
    return;
  const int offset = vo + (component * nf + frequency) * np + point;
  const float mask = static_cast<const float *>(
      monitors.component_masks.data)[monitor * 6 + component];
  float re = 0.f, im = 0.f;
  if (mask != 0.f) {
    re = static_cast<const float *>(monitors.dft_re.data)[offset];
    im = static_cast<const float *>(monitors.dft_im.data)[offset];
  }
  float weight = 0.f;
  if (component == 0 && point == 0)
    weight =
        static_cast<const float *>(monitors.dft_weight.data)[wo + frequency];
#pragma unroll
  for (int sub = 0; sub < 2; ++sub) {
    const int po =
        (sub * monitors.monitor_count + monitor) * max_nf + frequency;
    const float window =
        static_cast<const float *>(monitors.phase_window.data)[po];
    if (window == 0.f)
      continue;
    if (component == 0 && point == 0)
      weight += window;
    if (mask == 0.f)
      continue;
    const auto &fields = sub == 0 ? first : second;
    const int sample_offset =
        ((monitor * 6 + component) * max_np + point) * 2 + sub;
    const float sample =
        static_cast<const float *>(monitors.pair_samples.data)[sample_offset];
    float scale = window;
    if (static_cast<const int *>(monitors.codes.data)[2 * monitor + 1] == 1) {
      const float length =
          static_cast<const float *>(monitors.windows.data)[3 * monitor + 2];
      if (!isfinite(length) || length <= 0.f)
        continue;
      scale *= fields.dt * static_cast<float>(interval) * 299792458.0f /
               length / sqrtf(6.2831853071795864769f);
    }
    re += scale * sample *
          static_cast<const float *>(monitors.phase_cos.data)[po];
    im += scale * sample *
          static_cast<const float *>(monitors.phase_sin.data)[po];
  }
  if (component == 0 && point == 0)
    static_cast<float *>(monitors.dft_weight.data)[wo + frequency] = weight;
  if (mask != 0.f) {
    static_cast<float *>(monitors.dft_re.data)[offset] = re;
    static_cast<float *>(monitors.dft_im.data)[offset] = im;
  }
}

// Each lane owns one interpolation point. Reuse its sample in registers across
// a small frequency batch; separate batches retain frequency parallelism.
// Small frequency plans keep the original fused kernel.
constexpr int kDftFrequencyBatch = 8;
__global__ void AccumulateDftReusedSamples(
    const __grid_constant__ DftFields fields,
    const __grid_constant__ BeamzDftGroupLaunch monitors) {
  const int point = blockIdx.x * blockDim.x + threadIdx.x;
  const int monitor = blockIdx.y / 6, component = blockIdx.y % 6;
  const int first_frequency = blockIdx.z * kDftFrequencyBatch;
  const int frequency_stop = first_frequency + kDftFrequencyBatch;
  const auto* counts = static_cast<const int32_t*>(monitors.counts.data);
  const int nf = counts[5 * monitor], np = counts[5 * monitor + 1];
  const int interval = max(1, counts[5 * monitor + 2]);
  const int vo = counts[5 * monitor + 3], wo = counts[5 * monitor + 4];
  const int max_nf = monitors.frequencies.dims[1];
  const int max_np = monitors.indices.dims[2];
  if (nf < 1 || nf > max_nf || np < 1 || np > max_np || point >= np ||
      first_frequency >= nf || vo < 0 || wo < 0 ||
      static_cast<int64_t>(vo) + 6LL * nf * np > monitors.dft_re.dims[0] ||
      static_cast<int64_t>(vo) + 6LL * nf * np > monitors.dft_im.dims[0] ||
      static_cast<int64_t>(wo) + nf > monitors.dft_weight.dims[0])
    return;
  const float window = static_cast<const float*>(monitors.phase_window.data)
      [monitor * max_nf];
  if (window == 0.f) return;
  if (component == 0 && point == 0) {
    for (int frequency = first_frequency; frequency < min(nf, frequency_stop);
         ++frequency)
      static_cast<float*>(monitors.dft_weight.data)[wo + frequency] += window;
  }
  if (static_cast<const float*>(monitors.component_masks.data)
          [monitor * 6 + component] == 0.f)
    return;
  const auto& field = fields.values[component];
  const int neighbors = monitors.indices.dims[3];
  const int base = ((monitor * 6 + component) * max_np + point) * neighbors;
  float sample = 0.f;
  for (int neighbor = 0; neighbor < neighbors; ++neighbor) {
    const int index =
        static_cast<const int32_t*>(monitors.indices.data)[base + neighbor];
    if (index >= 0 && index < ElementCount(field))
      sample += static_cast<const float*>(field.data)
                    [BeamzPhysicalIndex(field, index)] *
                static_cast<const float*>(monitors.weights.data)[base + neighbor];
  }
  float scale = window;
  if (static_cast<const int32_t*>(monitors.codes.data)[2 * monitor + 1] == 1) {
    const float length =
        static_cast<const float*>(monitors.windows.data)[3 * monitor + 2];
    if (!isfinite(length) || length <= 0.f) return;
    scale *= fields.dt * static_cast<float>(interval) * 299792458.0f /
             length / sqrtf(6.2831853071795864769f);
  }
  for (int frequency = first_frequency; frequency < min(nf, frequency_stop);
         ++frequency) {
    const int po = monitor * max_nf + frequency;
    const int offset = vo + (component * nf + frequency) * np + point;
    static_cast<float*>(monitors.dft_re.data)[offset] +=
        scale * sample * static_cast<const float*>(monitors.phase_cos.data)[po];
    static_cast<float*>(monitors.dft_im.data)[offset] +=
        scale * sample * static_cast<const float*>(monitors.phase_sin.data)[po];
  }
}

// Severe aperture/frequency mismatch makes a rectangular launch mostly empty.
// Traverse the exact packed accumulator arena instead, retaining frequency
// parallelism. Bound monitor count so the small descriptor search stays cheap.
__global__ void AccumulateDftPacked(
    const __grid_constant__ DftFields fields,
    const __grid_constant__ BeamzDftGroupLaunch monitors) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= monitors.dft_re.dims[0]) return;
  const auto* counts = static_cast<const int32_t*>(monitors.counts.data);
  int monitor = 0, nf = 0, np = 0, vo = 0;
  for (; monitor < monitors.monitor_count; ++monitor) {
    nf = counts[5 * monitor];
    np = counts[5 * monitor + 1];
    vo = counts[5 * monitor + 3];
    if (nf > 0 && np > 0 && offset >= vo &&
        static_cast<int64_t>(offset) < vo + 6LL * nf * np) break;
  }
  if (monitor == monitors.monitor_count) return;
  const int max_nf = monitors.frequencies.dims[1];
  const int max_np = monitors.indices.dims[2];
  const int interval = max(1, counts[5 * monitor + 2]);
  const int wo = counts[5 * monitor + 4];
  if (nf > max_nf || np > max_np || vo < 0 || wo < 0 ||
      vo + 6LL * nf * np > monitors.dft_re.dims[0] ||
      vo + 6LL * nf * np > monitors.dft_im.dims[0] ||
      static_cast<int64_t>(wo) + nf > monitors.dft_weight.dims[0] ||
      offset >= monitors.dft_im.dims[0]) return;
  const int component = (offset - vo) / (nf * np);
  const int frequency = ((offset - vo) / np) % nf;
  const int point = (offset - vo) % np;
  const float window = static_cast<const float*>(monitors.phase_window.data)
      [monitor * max_nf + frequency];
  if (window == 0.f) return;
  if (component == 0 && point == 0)
    static_cast<float*>(monitors.dft_weight.data)[wo + frequency] += window;
  if (static_cast<const float*>(monitors.component_masks.data)
          [monitor * 6 + component] == 0.f)
    return;
  const auto& field = fields.values[component];
  const int neighbors = monitors.indices.dims[3];
  const int base = ((monitor * 6 + component) * max_np + point) * neighbors;
  float sample = 0.f;
  for (int neighbor = 0; neighbor < neighbors; ++neighbor) {
    const int index =
        static_cast<const int32_t*>(monitors.indices.data)[base + neighbor];
    if (index >= 0 && index < ElementCount(field))
      sample += static_cast<const float*>(field.data)
                    [BeamzPhysicalIndex(field, index)] *
                static_cast<const float*>(monitors.weights.data)[base + neighbor];
  }
  float scale = window;
  if (static_cast<const int32_t*>(monitors.codes.data)[2 * monitor + 1] == 1) {
    const float length =
        static_cast<const float*>(monitors.windows.data)[3 * monitor + 2];
    if (!isfinite(length) || length <= 0.f) return;
    scale *= fields.dt * static_cast<float>(interval) * 299792458.0f /
             length / sqrtf(6.2831853071795864769f);
  }
  const int po = monitor * max_nf + frequency;
  static_cast<float*>(monitors.dft_re.data)[offset] +=
      scale * sample * static_cast<const float*>(monitors.phase_cos.data)[po];
  static_cast<float*>(monitors.dft_im.data)[offset] +=
      scale * sample * static_cast<const float*>(monitors.phase_sin.data)[po];
}

cudaError_t LaunchDftGroups(cudaStream_t stream, const BeamzLaunch &h_launch,
                            const BeamzLaunch &e_launch,
                            const BeamzDftGroupLaunch &monitors, int32_t step) {
  const DftFields fields = MakeDftFields(h_launch, e_launch);
  const int frequency_threads =
      monitors.monitor_count == 1 && monitors.frequencies.dims[1] == 3 ? 3 : 2;
  const dim3 threads(32, frequency_threads, 2);
  const dim3 blocks((monitors.indices.dims[2] + threads.x - 1) / threads.x,
                    (monitors.frequencies.dims[1] + threads.y - 1) / threads.y,
                    (monitors.monitor_count * 6 + threads.z - 1) / threads.z);
  // A phase is shared by every component and point of one monitor/frequency.
  // For monitor-heavy work, one tiny producer launch removes repeated sincosf
  // calls from every 32-point block. Small plans keep the original fused path
  // so an extra kernel launch cannot dominate a short gather.
  const bool cache_phase =
      monitors.indices.dims[2] > kTileX ||
      monitors.monitor_count * monitors.frequencies.dims[1] >= 32;
  if (cache_phase) {
    const dim3 phase_threads(kTileX);
    const dim3 phase_blocks(
        (monitors.frequencies.dims[1] + phase_threads.x - 1) / phase_threads.x,
        monitors.monitor_count);
    PrepareDftPhases<<<phase_blocks, phase_threads, 0, stream>>>(
        fields, monitors, step);
    if (cudaError_t error = cudaPeekAtLastError(); error != cudaSuccess) {
      return error;
    }
  }
  const int64_t rectangular_values = monitors.monitor_count * 6LL *
      monitors.indices.dims[2] * monitors.frequencies.dims[1];
  const bool packed_launch = cache_phase && monitors.monitor_count <= 8 &&
      monitors.dft_re.dims[0] > 0 &&
      monitors.dft_re.dims[0] <= std::numeric_limits<int>::max() &&
      monitors.dft_re.dims[0] * 4 < rectangular_values;
  if (packed_launch) {
    const int threads = 256;
    const int blocks = (monitors.dft_re.dims[0] + threads - 1) / threads;
    AccumulateDftPacked<<<blocks, threads, 0, stream>>>(fields, monitors);
  } else if (cache_phase && monitors.frequencies.dims[1] >= 16) {
    const dim3 reuse_threads(128);
    const dim3 reuse_blocks(
        (monitors.indices.dims[2] + reuse_threads.x - 1) / reuse_threads.x,
        monitors.monitor_count * 6,
        (monitors.frequencies.dims[1] + kDftFrequencyBatch - 1) / kDftFrequencyBatch);
    AccumulateDftReusedSamples<<<reuse_blocks, reuse_threads, 0, stream>>>(
        fields, monitors);
  } else if (monitors.monitor_count == 1 && cache_phase) {
    AccumulateDftGroups<true, true>
        <<<blocks, threads, 0, stream>>>(fields, monitors, step);
  } else if (monitors.monitor_count == 1) {
    AccumulateDftGroups<true, false>
        <<<blocks, threads, 0, stream>>>(fields, monitors, step);
  } else if (cache_phase) {
    AccumulateDftGroups<false, true><<<blocks, threads, 0, stream>>>(
        fields, monitors, step);
  } else {
    AccumulateDftGroups<false, false><<<blocks, threads, 0, stream>>>(
        fields, monitors, step);
  }
  return cudaPeekAtLastError();
}

}  // namespace

cudaError_t BeamzEnqueueSourceGroup(
    cudaStream_t stream, const BeamzLaunch& launch, const BeamzBuffer& target,
    const BeamzSourceGroupLaunch& group, int32_t step) {
  return LaunchSourceGroup(stream, launch, target, group, step);
}

cudaError_t BeamzEnqueueDftGroups(cudaStream_t stream,
                                  const BeamzLaunch& h_launch,
                                  const BeamzLaunch& e_launch,
                                  const BeamzDftGroupLaunch& monitors,
                                  int32_t step) {
  return LaunchDftGroups(stream, h_launch, e_launch, monitors, step);
}

cudaError_t BeamzEnqueueSourcePhase(cudaStream_t stream, const BeamzLaunch& h,
                                    const BeamzLaunch& e,
                                    const BeamzSourceGroupLaunch* groups,
                                    int32_t count, int timing, int32_t step,
                                    bool shell_only, int inner_margin) {
  if (groups == nullptr || count == 0)
    return cudaSuccess;
  if (count != 9 || timing < 0 || timing > 2)
    return cudaErrorInvalidValue;
  SourcePhasePlan plan{};
  if (shell_only || inner_margin >= 0) {
    plan.low = h.uniform_cpml_thickness + (shell_only ? 0 : inner_margin);
    for (int axis = 0; axis < 3; ++axis) {
      plan.high[axis] =
          std::min({h.outputs[0].dims[axis], h.outputs[1].dims[axis],
                    h.outputs[2].dims[axis]}) -
          plan.low;
    }
  }
  int64_t max_x_extent = 1;
  for (int component = 0; component < 3; ++component) {
    const auto &group = groups[timing * 3 + component];
    if (group.coefficients.dims[0] > 0)
      max_x_extent = std::max(max_x_extent, group.coefficients.dims[3]);
  }
  const dim3 threads = SourceThreads(max_x_extent);
  int total = 0, active = 0, last = -1;
  for (int component = 0; component < 3; ++component) {
    const auto &group = groups[timing * 3 + component];
    plan.groups[component] = group;
    plan.targets[component] = timing == 0   ? e.inputs[component]
                              : timing == 1 ? h.outputs[component]
                                            : e.outputs[component];
    if (group.coefficients.dims[0] > 0) {
      ++active;
      last = component;
      const int xb = (group.coefficients.dims[3] + threads.x - 1) / threads.x;
      const int yb = (group.coefficients.dims[2] + threads.y - 1) / threads.y;
      const int zb = (group.coefficients.dims[1] + threads.z - 1) / threads.z;
      plan.x_blocks[component] = xb;
      plan.y_blocks[component] = yb;
      plan.z_blocks[component] = zb;
      const int64_t blocks =
          static_cast<int64_t>(xb) * yb * zb *
          (group.coincident ? 1 : group.coefficients.dims[0]);
      if (blocks > std::numeric_limits<int>::max() - total)
        return cudaErrorInvalidValue;
      total += static_cast<int>(blocks);
    }
    plan.ends[component] = total;
  }
  if (active == 0)
    return cudaSuccess;
  if (active == 1 && !shell_only && inner_margin < 0)
    return BeamzEnqueueSourceGroup(stream, h, plan.targets[last],
                                   plan.groups[last], step);
  ApplySourcePhase<<<total, threads, 0, stream>>>(plan, step, h.metallic_edges);
  return cudaPeekAtLastError();
}

cudaError_t BeamzEnqueueDftPair(cudaStream_t stream, const BeamzLaunch &h1,
                                const BeamzLaunch &e1, const BeamzLaunch &h2,
                                const BeamzLaunch &e2,
                                const BeamzDftGroupLaunch &monitors, int step) {
  const auto &samples = monitors.pair_samples;
  if (monitors.phase_sin.dims[0] < 2 * monitors.monitor_count ||
      samples.rank != 4 || samples.element_type != kBeamzF32 ||
      samples.data == nullptr || samples.dims[0] != monitors.monitor_count ||
      samples.dims[1] != 6 || samples.dims[2] != monitors.indices.dims[2] ||
      samples.dims[3] != 2)
    return cudaErrorInvalidValue;
  const auto first = MakeDftFields(h1, e1), second = MakeDftFields(h2, e2);
  const dim3 phase_blocks((monitors.frequencies.dims[1] + 31) / 32,
                          monitors.monitor_count, 2);
  PrepareDftPhases<<<phase_blocks, 32, 0, stream>>>(first, monitors, step);
  if (auto error = cudaPeekAtLastError(); error != cudaSuccess)
    return error;
  GatherDftPair<<<dim3((monitors.indices.dims[2] + 127) / 128,
                       monitors.monitor_count * 6),
                  128, 0, stream>>>(first, second, monitors);
  if (auto error = cudaPeekAtLastError(); error != cudaSuccess)
    return error;
  const dim3 threads(32, 2, 2);
  const dim3 blocks((monitors.indices.dims[2] + 31) / 32,
                    (monitors.frequencies.dims[1] + 1) / 2,
                    (monitors.monitor_count * 6 + 1) / 2);
  AccumulateDftPair<<<blocks, threads, 0, stream>>>(first, second, monitors);
  return cudaPeekAtLastError();
}
