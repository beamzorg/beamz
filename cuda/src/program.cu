#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

#include "abi_layout.h"
#include "graph.h"
#include "kernels.h"
#include "launch.h"

namespace {

using namespace beamz::cuda::abi;

bool FlagEnabled(const BeamzLaunch& launch, int32_t flag) {
  return (launch.cuda_flags & flag) != 0;
}

bool ScheduleFlagEnabled(const BeamzProgramLaunch& program, int32_t flag) {
  return (program.schedule_flags & flag) != 0;
}

bool FitsIntOffsets(const BeamzBuffer& value) {
  if (value.rank < 0 || value.rank > 4) return false;
  int64_t elements = 1;
  for (int axis = 0; axis < value.rank; ++axis) {
    if (value.dims[axis] < 0 ||
        value.dims[axis] > std::numeric_limits<int>::max()) {
      return false;
    }
    if (value.dims[axis] != 0 &&
        elements > std::numeric_limits<int>::max() / value.dims[axis]) {
      return false;
    }
    elements *= value.dims[axis];
    if (elements > std::numeric_limits<int>::max()) return false;
  }
  return true;
}

bool HasType(const BeamzBuffer& value, BeamzElementType type) {
  return value.element_type == type;
}

cudaError_t ValidateSourceGroups(const BeamzSourceGroupLaunch* groups,
                                 int32_t count) {
  if (groups == nullptr) return count == 0 ? cudaSuccess : cudaErrorInvalidValue;
  if (count != kSourceGroupCount) return cudaErrorInvalidValue;
  for (int32_t index = 0; index < count; ++index) {
    const BeamzSourceGroupLaunch& group = groups[index];
    if (group.component != index % 3 || group.timing != index / 3 ||
        group.component < 0 || group.component > 2 || group.timing < 0 ||
        group.timing > 2 || group.coefficients.rank != 4 ||
        group.waveforms.rank != 2 || group.starts.rank != 2 ||
        group.current_step.rank != 0 || group.coincident < 0 ||
        group.coincident > 1 || group.disjoint < 0 || group.disjoint > 1 ||
        (group.coincident != 0 && group.disjoint != 0) ||
        !HasType(group.coefficients, kBeamzF32) ||
        !HasType(group.waveforms, kBeamzF32) ||
        !HasType(group.starts, kBeamzS32) ||
        !HasType(group.current_step, kBeamzS32) ||
        group.coefficients.dims[0] != group.waveforms.dims[0] ||
        group.coefficients.dims[0] != group.starts.dims[0] ||
        group.starts.dims[1] != 3 || group.waveforms.dims[1] < 1 ||
        group.coefficients.dims[1] < 1 || group.coefficients.dims[2] < 1 ||
        group.coefficients.dims[3] < 1 || group.current_step.data == nullptr ||
        (group.coefficients.dims[0] > 0 &&
         (group.coefficients.data == nullptr ||
          group.waveforms.data == nullptr || group.starts.data == nullptr)) ||
        !FitsIntOffsets(group.coefficients) ||
        !FitsIntOffsets(group.waveforms) || !FitsIntOffsets(group.starts) ||
        !FitsIntOffsets(group.current_step)) {
      return cudaErrorInvalidValue;
    }
  }
  return cudaSuccess;
}

bool HasPackedLosslessMaterial(const BeamzLaunch& launch) {
  if (launch.phase != 1) return false;
  for (int component = 0; component < 3; ++component) {
    if (launch.inputs[6 + component].rank != 1 ||
        (launch.inputs[9 + component].rank != 1 &&
         launch.inputs[9 + component].rank != 2) ||
        launch.inputs[9 + component].element_type != kBeamzS32) {
      return false;
    }
  }
  return true;
}

bool UniformCpml(const BeamzProgramLaunch& program) {
  if (program.h_ab.nterms != kCpmlTermCount ||
      program.h_ab.uniform_cpml_thickness <= 0 ||
      program.e_ab.uniform_cpml_thickness !=
          program.h_ab.uniform_cpml_thickness) {
    return false;
  }
  if (program.field_bank_count >= 2 &&
      (program.h_ba.uniform_cpml_thickness !=
           program.h_ab.uniform_cpml_thickness ||
       program.e_ba.uniform_cpml_thickness !=
           program.h_ab.uniform_cpml_thickness)) {
    return false;
  }
  return true;
}

bool CombinedCpmlCoreSupported(const BeamzProgramLaunch& program) {
  if (!UniformCpml(program) || program.h_ab.metric_kind != 0 ||
      program.e_ab.metric_kind != 0) {
    return false;
  }
  if (program.field_bank_count >= 2 &&
      (program.h_ba.metric_kind != 0 || program.e_ba.metric_kind != 0)) {
    return false;
  }
  const BeamzLaunch* h_launches[] = {
      &program.h_ab, program.field_bank_count >= 2 ? &program.h_ba : nullptr};
  for (const auto *h : h_launches) {
    if (h == nullptr)
      continue;
    for (int material = 6; material < 12; ++material) {
      if (h->inputs[material].rank != 0)
        return false;
    }
  }
  const int64_t thickness = program.h_ab.uniform_cpml_thickness;
  for (int axis = 0; axis < 3; ++axis) {
    int64_t extent = program.h_ab.outputs[0].dims[axis];
    for (int component = 1; component < 3; ++component) {
      if (program.h_ab.outputs[component].dims[axis] < extent) {
        extent = program.h_ab.outputs[component].dims[axis];
      }
    }
    if (extent <= 2 * thickness) return false;
  }
  return true;
}

cudaError_t ValidateSchedulePlan(const BeamzProgramLaunch& program) {
  if (program.e_ab.tensor &&
      (program.field_bank_count != 1 ||
       ScheduleFlagEnabled(program, kNativeScheduleCombinedCpmlCore)))
    return cudaErrorInvalidValue;
  constexpr int32_t kKnownFlags =
      kNativeScheduleCpml | kNativeScheduleTemporal |
      kNativeScheduleUniformCpml | kNativeSchedulePackedMaterial |
      kNativeScheduleCombinedCpmlCore | kNativeScheduleSources |
      kNativeScheduleMonitors | kNativeScheduleGraphCache;
  if (program.schedule_flags < 0 ||
      (program.schedule_flags & ~kKnownFlags) != 0) {
    return cudaErrorInvalidValue;
  }
  const bool cpml = program.h_ab.nterms == kCpmlTermCount;
  const bool temporal = program.field_bank_count >= 2;
  const bool sources = program.source_group_count != 0;
  const bool monitors = program.monitors != nullptr;
  const bool packed_material =
      HasPackedLosslessMaterial(program.e_ab) &&
      (program.field_bank_count == 1 ||
       HasPackedLosslessMaterial(program.e_ba));
  const bool graph_cache = FlagEnabled(program.h_ab, kBeamzGraphCache);
  if (ScheduleFlagEnabled(program, kNativeScheduleCpml) != cpml ||
      ScheduleFlagEnabled(program, kNativeScheduleTemporal) != temporal ||
      ScheduleFlagEnabled(program, kNativeScheduleUniformCpml) !=
          UniformCpml(program) ||
      ScheduleFlagEnabled(program, kNativeSchedulePackedMaterial) !=
          packed_material ||
      ScheduleFlagEnabled(program, kNativeScheduleSources) != sources ||
      ScheduleFlagEnabled(program, kNativeScheduleMonitors) != monitors ||
      ScheduleFlagEnabled(program, kNativeScheduleGraphCache) != graph_cache ||
      (ScheduleFlagEnabled(program, kNativeScheduleCombinedCpmlCore) &&
       !CombinedCpmlCoreSupported(program))) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t ValidateMonitors(const BeamzDftGroupLaunch* value) {
  if (value == nullptr) return cudaSuccess;
  const BeamzDftGroupLaunch& monitors = *value;
  if (monitors.monitor_count < 1 || monitors.indices.rank != 4 ||
      monitors.weights.rank != 4 || monitors.frequencies.rank != 2 ||
      monitors.component_masks.rank != 2 || monitors.counts.rank != 2 ||
      monitors.codes.rank != 2 || monitors.windows.rank != 2 ||
      monitors.dft_re.rank != 1 || monitors.dft_im.rank != 1 ||
      monitors.dft_weight.rank != 1 || monitors.time.rank != 0 ||
      monitors.current_step.rank != 0 || monitors.elapsed_steps.rank != 0 || monitors.phase_sin.rank != 2 ||
      monitors.phase_cos.rank != 2 || monitors.phase_window.rank != 2 ||
      !HasType(monitors.indices, kBeamzS32) ||
      !HasType(monitors.weights, kBeamzF32) ||
      !HasType(monitors.frequencies, kBeamzF32) ||
      !HasType(monitors.component_masks, kBeamzF32) ||
      !HasType(monitors.counts, kBeamzS32) ||
      !HasType(monitors.codes, kBeamzS32) ||
      !HasType(monitors.windows, kBeamzF32) ||
      !HasType(monitors.dft_re, kBeamzF32) ||
      !HasType(monitors.dft_im, kBeamzF32) ||
      !HasType(monitors.dft_weight, kBeamzF32) ||
      !HasType(monitors.phase_sin, kBeamzF32) ||
      !HasType(monitors.phase_cos, kBeamzF32) ||
      !HasType(monitors.phase_window, kBeamzF32) ||
      !HasType(monitors.time, kBeamzF32) ||
      !HasType(monitors.current_step, kBeamzS32) ||
      !HasType(monitors.elapsed_steps, kBeamzS32) ||
      monitors.indices.dims[0] < monitors.monitor_count ||
      monitors.indices.dims[1] != 6 ||
      monitors.weights.dims[0] != monitors.indices.dims[0] ||
      monitors.weights.dims[1] != monitors.indices.dims[1] ||
      monitors.weights.dims[2] != monitors.indices.dims[2] ||
      monitors.weights.dims[3] != monitors.indices.dims[3] ||
      monitors.frequencies.dims[0] < monitors.monitor_count ||
      monitors.component_masks.dims[0] < monitors.monitor_count ||
      monitors.component_masks.dims[1] != 6 ||
      monitors.counts.dims[0] < monitors.monitor_count ||
      monitors.counts.dims[1] != 5 ||
      monitors.codes.dims[0] < monitors.monitor_count ||
      monitors.codes.dims[1] != 2 ||
      monitors.windows.dims[0] < monitors.monitor_count ||
      monitors.windows.dims[1] != 3 || monitors.dft_re.dims[0] < 1 ||
      monitors.dft_im.dims[0] != monitors.dft_re.dims[0] ||
      monitors.dft_weight.dims[0] < 1 || monitors.indices.dims[2] < 1 ||
      monitors.indices.dims[3] < 1 || monitors.frequencies.dims[1] < 1 ||
      monitors.phase_sin.dims[0] < monitors.monitor_count ||
      monitors.phase_sin.dims[1] != monitors.frequencies.dims[1] ||
      monitors.phase_cos.dims[0] != monitors.phase_sin.dims[0] ||
      monitors.phase_cos.dims[1] != monitors.phase_sin.dims[1] ||
      monitors.phase_window.dims[0] != monitors.phase_sin.dims[0] ||
      monitors.phase_window.dims[1] != monitors.phase_sin.dims[1] ||
      monitors.indices.data == nullptr ||
      monitors.weights.data == nullptr || monitors.frequencies.data == nullptr ||
      monitors.component_masks.data == nullptr || monitors.counts.data == nullptr ||
      monitors.codes.data == nullptr || monitors.windows.data == nullptr ||
      monitors.dft_re.data == nullptr || monitors.dft_im.data == nullptr ||
      monitors.dft_weight.data == nullptr || monitors.time.data == nullptr ||
      monitors.current_step.data == nullptr || monitors.elapsed_steps.data == nullptr || monitors.phase_sin.data == nullptr ||
      monitors.phase_cos.data == nullptr ||
      monitors.phase_window.data == nullptr) {
    return cudaErrorInvalidValue;
  }
  const BeamzBuffer buffers[] = {
      monitors.indices,         monitors.weights, monitors.frequencies,
      monitors.component_masks, monitors.counts,  monitors.codes,
      monitors.windows,         monitors.dft_re,   monitors.dft_im,
      monitors.dft_weight,      monitors.phase_sin, monitors.phase_cos,
      monitors.phase_window,    monitors.time,     monitors.current_step,
      monitors.elapsed_steps};
  for (const BeamzBuffer& buffer : buffers) {
    if (!FitsIntOffsets(buffer)) return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t ValidateProgram(const BeamzProgramLaunch& program) {
  if (program.nsteps < 1 || program.graph_cache_capacity < 0 ||
      (program.field_bank_count < 1 || program.field_bank_count > 3) ||
      program.h_ab.phase != 0 || program.e_ab.phase != 1 ||
      program.h_ab.nterms != program.e_ab.nterms ||
      FlagEnabled(program.h_ab, kBeamzCpmlPair)) {
    return cudaErrorInvalidValue;
  }
  if (program.field_bank_count == 3) {
    if (program.source_group_count != 9 || !CombinedCpmlCoreSupported(program))
      return cudaErrorInvalidValue;
    for (int c = 0; c < 3; ++c) {
      if (program.source_groups[c].coefficients.dims[0] != 0)
        return cudaErrorInvalidValue;
      for (int axis = 0; axis < 3; ++axis)
        if (program.h_ab.outputs[c].dims[axis] <=
            2 * (program.h_ab.uniform_cpml_thickness + 2))
          return cudaErrorInvalidValue;
    }
    for (int c = 0; c < 6; ++c) {
      const auto& reference =
          c < 3 ? program.h_ab.inputs[c] : program.e_ab.inputs[c - 3];
      const auto& scratch = program.pair_fields[c];
      if (scratch.rank != 3 || scratch.element_type != kBeamzF32 ||
          !scratch.data || !FitsIntOffsets(scratch) ||
          scratch.data == reference.data)
        return cudaErrorInvalidValue;
      for (int axis = 0; axis < 3; ++axis)
        if (scratch.dims[axis] != reference.dims[axis])
          return cudaErrorInvalidValue;
    }
  }
  if (cudaError_t error = BeamzValidatePhase(program.h_ab);
      error != cudaSuccess) {
    return error;
  }
  if (cudaError_t error = BeamzValidatePhase(program.e_ab);
      error != cudaSuccess) {
    return error;
  }
  if (program.field_bank_count >= 2) {
    if (program.h_ba.phase != 0 || program.e_ba.phase != 1 ||
        program.h_ba.nterms != program.h_ab.nterms ||
        program.e_ba.nterms != program.e_ab.nterms) {
      return cudaErrorInvalidValue;
    }
    if (cudaError_t error = BeamzValidatePhase(program.h_ba);
        error != cudaSuccess) {
      return error;
    }
    if (cudaError_t error = BeamzValidatePhase(program.e_ba);
        error != cudaSuccess) {
      return error;
    }
    if (program.h_ab.nterms == 0) {
      if (program.nsteps < 4 || program.source_group_count != 0 ||
          program.monitors != nullptr ||
          program.e_ab.metric_kind != program.h_ab.metric_kind ||
          program.h_ba.metric_kind != program.h_ab.metric_kind ||
          program.e_ba.metric_kind != program.h_ab.metric_kind ||
          program.h_ab.metallic_edges != 63 ||
          program.e_ab.metallic_edges != 63 ||
          program.h_ba.metallic_edges != 63 ||
          program.e_ba.metallic_edges != 63) {
        return cudaErrorInvalidValue;
      }
      for (int material = 0; material < 6; ++material) {
        const int h_rank = program.h_ab.inputs[6 + material].rank;
        const int e_rank = program.e_ab.inputs[6 + material].rank;
        if ((h_rank != 0 && h_rank != 3) ||
            (e_rank != 0 && e_rank != 3)) {
          return cudaErrorInvalidValue;
        }
      }
    } else if (program.h_ab.nterms != kCpmlTermCount ||
               program.source_group_count != kSourceGroupCount) {
      return cudaErrorInvalidValue;
    }
  }
  if (cudaError_t error = ValidateSourceGroups(program.source_groups,
                                                program.source_group_count);
      error != cudaSuccess) {
    return error;
  }
  if (cudaError_t error = ValidateMonitors(program.monitors);
      error != cudaSuccess) {
    return error;
  }
  return ValidateSchedulePlan(program);
}

cudaError_t EnqueueProgramSources(cudaStream_t stream,
                                  const BeamzProgramLaunch& program,
                                  const BeamzLaunch& h, const BeamzLaunch& e,
                                  int timing, int step, bool shell_only = false,
                                  int inner_margin = -1) {
  return BeamzEnqueueSourcePhase(stream, h, e, program.source_groups,
                                 program.source_group_count, timing, step,
                                 shell_only, inner_margin);
}

cudaError_t ObserveProgramStep(cudaStream_t stream,
                               const BeamzProgramLaunch& program,
                               const BeamzLaunch& h, const BeamzLaunch& e,
                               int step) {
  return program.monitors
             ? BeamzEnqueueDftGroups(stream, h, e, *program.monitors, step)
             : cudaSuccess;
}

int LaunchInPlaceProgram(void* raw_stream, const BeamzProgramLaunch& program) {
  const BeamzLaunch& h_launch = program.h_ab;
  const BeamzLaunch& e_launch = program.e_ab;
  const BeamzSourceGroupLaunch* source_groups = program.source_groups;
  const int32_t source_group_count = program.source_group_count;
  const BeamzDftGroupLaunch* monitors = program.monitors;
  auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
  const std::string graph_key = BeamzGraphKey("in-place", raw_stream, program);
  const bool cache_enabled = FlagEnabled(h_launch, kBeamzGraphCache);
  auto enqueue = [&]() {
    cudaError_t error = cudaSuccess;
    const bool split_cpml =
        ScheduleFlagEnabled(program, kNativeScheduleCombinedCpmlCore);
    auto enqueue_sources = [&](int timing, int32_t step) {
      error = EnqueueProgramSources(stream, program, h_launch, e_launch, timing, step);
    };
    for (int32_t step = 0; step < program.nsteps; ++step) {
      if (source_groups != nullptr) {
        enqueue_sources(0, step);
        if (error != cudaSuccess) break;
      }
      if (split_cpml) {
        // H and E remain separate launches: graph outputs alias their inputs, so
        // cross-block H-to-E fusion would race while reading the old H halo.
        error = BeamzEnqueueCpmlPhase(stream, h_launch);
        if (error != cudaSuccess) break;
        if (source_groups != nullptr) {
          enqueue_sources(1, step);
          if (error != cudaSuccess) break;
        }
        error = BeamzEnqueueCpmlPhase(stream, e_launch);
      } else {
        error = BeamzEnqueuePhase(raw_stream, h_launch);
        if (error != cudaSuccess) break;
        if (source_groups != nullptr) {
          enqueue_sources(1, step);
          if (error != cudaSuccess) break;
        }
        error = BeamzEnqueuePhase(raw_stream, e_launch);
      }
      if (error != cudaSuccess) break;
      if (source_groups != nullptr) {
        enqueue_sources(2, step);
        if (error != cudaSuccess) break;
      }
      if (monitors != nullptr) {
        error = ObserveProgramStep(stream, program, h_launch, e_launch, step);
        if (error != cudaSuccess) break;
      }
    }
    return error;
  };
  return BeamzLaunchGraph(stream, graph_key, cache_enabled,
                          program.graph_cache_capacity, enqueue);
}

int LaunchTemporalYeeProgram(void* raw_stream,
                             const BeamzProgramLaunch& program) {
  const BeamzLaunch& h_ab = program.h_ab;
  const BeamzLaunch& e_ab = program.e_ab;
  const BeamzLaunch& h_ba = program.h_ba;
  const BeamzLaunch& e_ba = program.e_ba;
  auto stream = reinterpret_cast<cudaStream_t>(raw_stream);

  BeamzLaunch h_tail = h_ab;
  BeamzLaunch e_tail = e_ab;
  for (int component = 0; component < 3; ++component) {
    h_tail.outputs[component] = h_ba.outputs[component];
    e_tail.inputs[3 + component] = h_tail.outputs[component];
    e_tail.outputs[component] = e_ba.outputs[component];
  }

  const std::string graph_key =
      BeamzGraphKey("temporal-yee", raw_stream, program);
  const bool cache_enabled = FlagEnabled(h_ab, kBeamzGraphCache);
  auto enqueue = [&]() {
    cudaError_t error = cudaSuccess;
    for (int32_t pair = 0; pair < program.nsteps / 2; ++pair) {
      error = BeamzEnqueueFusedFullStep(stream, h_ab, e_ab);
      if (error != cudaSuccess) return error;
      error = BeamzEnqueueFusedFullStep(stream, h_ba, e_ba);
      if (error != cudaSuccess) return error;
    }
    for (int32_t step = 2 * (program.nsteps / 2); step < program.nsteps;
         ++step) {
      error = BeamzEnqueuePhase(raw_stream, h_tail);
      if (error != cudaSuccess) return error;
      error = BeamzEnqueuePhase(raw_stream, e_tail);
      if (error != cudaSuccess) return error;
    }
    return error;
  };
  return BeamzLaunchGraph(stream, graph_key, cache_enabled,
                          program.graph_cache_capacity, enqueue);
}

// The ring kernel is tuned on GA102. Keep the established phase queue for
// smaller/narrower grids and unmeasured architectures; those cases did not
// consistently benefit from fusion. The override also lets parity tests cover
// tile tails and shell intersections without allocating multi-million-cell
// grids.
bool FuseCpmlCore(const BeamzProgramLaunch &program) {
  if (!ScheduleFlagEnabled(program, kNativeScheduleCombinedCpmlCore))
    return false;
  const char *override = std::getenv("BEAMZ_CUDA_CPML_CORE_FUSION");
  if (override != nullptr && std::strcmp(override, "0") == 0)
    return false;
  if (override != nullptr && std::strcmp(override, "1") == 0)
    return true;
  const auto &h = program.h_ab;
  int64_t shape[3];
  for (int axis = 0; axis < 3; ++axis) {
    shape[axis] = h.outputs[0].dims[axis];
    for (int component = 1; component < 3; ++component) {
      if (h.outputs[component].dims[axis] < shape[axis])
        shape[axis] = h.outputs[component].dims[axis];
    }
  }
  if (shape[2] < 384 || shape[0] * shape[1] * shape[2] < 12 * 1024 * 1024)
    return false;
  int device, major, minor;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                             device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                             device) != cudaSuccess)
    return false;
  return major == 8 && minor == 6;
}

// Two completed timesteps share tile-local fields in the deep interior. The
// intermediate bank remains observable by DFTs; a disjoint final bank prevents
// inter-block read/write hazards. The CPML and a two-cell coupling band advance
// conventionally at both substeps.
cudaError_t EnqueueTemporalPairProgram(cudaStream_t stream,
                                       const BeamzProgramLaunch &program,
                                       int tile, int pair_tile,
                                       bool fused_core) {
  BeamzBuffer a[6], b[6];
  for (int c = 0; c < 3; ++c) {
    a[c] = program.h_ab.inputs[c];
    a[3 + c] = program.e_ab.inputs[c];
    b[c] = program.h_ab.outputs[c];
    b[3 + c] = program.e_ab.outputs[c];
  }
  auto bind = [](BeamzLaunch &h, BeamzLaunch &e, const BeamzBuffer *input,
                 const BeamzBuffer *output) {
    for (int c = 0; c < 3; ++c) {
      h.inputs[c] = input[c];
      h.inputs[3 + c] = input[3 + c];
      h.outputs[c] = output[c];
      e.inputs[c] = input[3 + c];
      e.inputs[3 + c] = output[c];
      e.outputs[c] = output[3 + c];
    }
  };
  auto boundary_step = [&](const BeamzLaunch &h, const BeamzLaunch &e, int step,
                           bool band, bool observe = true) -> cudaError_t {
    if (pair_tile >= 3 && !fused_core) {
      auto sources = [&](int timing) -> cudaError_t {
        for (int index = 0; index < program.source_group_count; ++index) {
          const auto &group = program.source_groups[index];
          if (group.timing != timing || group.coefficients.dims[0] == 0)
            continue;
          const auto &target = timing == 1 ? h.outputs[group.component]
                                           : e.outputs[group.component];
          auto error = BeamzEnqueueSourceGroup(stream, h, target, group, step);
          if (error != cudaSuccess)
            return error;
        }
        return cudaSuccess;
      };
      auto error = BeamzEnqueueCpmlPhase(stream, h);
      if (error != cudaSuccess)
        return error;
      error = sources(1);
      if (error != cudaSuccess)
        return error;
      error = BeamzEnqueueCpmlPhase(stream, e);
      if (error != cudaSuccess)
        return error;
      error = sources(2);
      if (error != cudaSuccess)
        return error;
      return observe && program.monitors
                 ? ObserveProgramStep(stream, program, h, e, step)
                 : cudaSuccess;
    }
    auto error = BeamzEnqueueCpmlShell(stream, h);
    if (error != cudaSuccess)
      return error;
    error = EnqueueProgramSources(stream, program, h, e, 1, step, true);
    if (error != cudaSuccess)
      return error;
    error = band ? BeamzEnqueueCpmlPairBand(stream, h, e, program.source_groups,
                                            step)
                 : BeamzEnqueueCpmlCore(stream, h, e, program.source_groups,
                                        step, tile, false);
    if (error != cudaSuccess)
      return error;
    error = BeamzEnqueueCpmlShell(stream, e);
    if (error != cudaSuccess)
      return error;
    error = EnqueueProgramSources(stream, program, h, e, 2, step, false,
                                    band ? 2 : -1);
    if (error != cudaSuccess)
      return error;
    return program.monitors && observe
               ? ObserveProgramStep(stream, program, h, e, step)
               : cudaSuccess;
  };
  for (int pair = 0; pair < program.nsteps / 2; ++pair) {
    const auto *input = (pair & 1) ? program.pair_fields : a;
    const auto *output = (pair & 1) ? a : program.pair_fields;
    auto h1 = program.h_ab, e1 = program.e_ab;
    auto h2 = program.h_ba, e2 = program.e_ba;
    bind(h1, e1, input, b);
    bind(h2, e2, b, output);
    const bool rolling = pair_tile < 3;
    auto error = cudaSuccess;
    if (rolling) {
      error = BeamzEnqueueCpmlPairCore(stream, h1, e1, output,
                                       program.source_groups, 2 * pair,
                                       program.pair_publication, pair_tile);
      if (error != cudaSuccess)
        return error;
    }
    error = boundary_step(h1, e1, 2 * pair, rolling, false);
    if (error != cudaSuccess)
      return error;
    error = boundary_step(h2, e2, 2 * pair + 1, rolling, false);
    if (error != cudaSuccess)
      return error;
    if (program.monitors) {
      error = BeamzEnqueueDftPair(stream, h1, e1, h2, e2, *program.monitors,
                                  2 * pair);
      if (error != cudaSuccess)
        return error;
    }
  }
  if (program.nsteps & 1) {
    const auto *input = ((program.nsteps / 2) & 1) ? program.pair_fields : a;
    auto h = program.h_ab;
    auto e = program.e_ab;
    bind(h, e, input, b);
    return boundary_step(h, e, program.nsteps - 1, false);
  }
  return cudaSuccess;
}

int LaunchTemporalCpmlProgram(void *raw_stream,
                              const BeamzProgramLaunch &program) {
  const BeamzLaunch &h_ab = program.h_ab;
  const BeamzLaunch &e_ab = program.e_ab;
  const BeamzLaunch &h_ba = program.h_ba;
  const BeamzLaunch &e_ba = program.e_ba;
  const BeamzSourceGroupLaunch *source_groups = program.source_groups;
  const int32_t source_group_count = program.source_group_count;
  const BeamzDftGroupLaunch *monitors = program.monitors;
  auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
  const bool fused_core = FuseCpmlCore(program);
  int tile = 0;
  if (const char *value = std::getenv("BEAMZ_CUDA_CPML_TILE")) {
    if (std::strcmp(value, "64x4x8") == 0)
      tile = 1;
    else if (std::strcmp(value, "32x4x8") == 0)
      tile = 2;
    else if (std::strcmp(value, "32x8x8") != 0 &&
             std::strcmp(value, "auto") != 0)
      return cudaErrorInvalidValue;
  }
  // Source batching regressed the large unfused graph by 3.5% on GA102.
  // Preserve its established launch sequence; small and fused runs benefit.
  const auto &shape = h_ab.outputs[0].dims;
  const bool batch_sources =
      fused_core || shape[0] * shape[1] * shape[2] < 1024 * 1024;
  std::string graph_key =
      BeamzGraphKey(fused_core ? "temporal-cpml-fused-core" : "temporal-cpml",
                    raw_stream, program);
  graph_key.append(reinterpret_cast<const char *>(&tile), sizeof(tile));
  int pair_tile = 0;
  if (program.field_bank_count == 3) {
    const char *choice = std::getenv("BEAMZ_CUDA_PAIR_TILE");
    if (choice && std::strcmp(choice, "32x4x16") == 0)
      pair_tile = 1;
    else if (choice && std::strcmp(choice, "32x8x16") == 0)
      pair_tile = 2;
    else if (choice && std::strcmp(choice, "single") == 0)
      pair_tile = 3;
    else if (choice && std::strcmp(choice, "16x8x16") != 0)
      return cudaErrorInvalidValue;
  }
  graph_key.append(reinterpret_cast<const char *>(&pair_tile),
                   sizeof(pair_tile));
  const bool cache_enabled = FlagEnabled(h_ab, kBeamzGraphCache);

  auto enqueue = [&]() {
    if (program.field_bank_count == 3)
      return EnqueueTemporalPairProgram(stream, program, tile, pair_tile,
                                        fused_core);
    cudaError_t error = cudaSuccess;
    auto enqueue_sources = [&](const BeamzLaunch &h_launch,
                               const BeamzLaunch &e_launch, int timing,
                               int32_t step) {
      if (batch_sources) {
        error = EnqueueProgramSources(stream, program, h_launch, e_launch,
            timing, step, timing == 1 && fused_core);
      } else {
        for (int index = 0; index < source_group_count; ++index) {
          const auto &group = source_groups[index];
          if (group.timing != timing || group.coefficients.dims[0] == 0)
            continue;
          const auto &target =
              timing == 0 ? e_launch.inputs[group.component]
                          : (timing == 1 ? h_launch.outputs[group.component]
                                         : e_launch.outputs[group.component]);
          error =
              BeamzEnqueueSourceGroup(stream, h_launch, target, group, step);
          if (error != cudaSuccess)
            return;
        }
      }
    };
    for (int32_t step = 0; step < program.nsteps; ++step) {
      const bool ab = (step & 1) == 0;
      const BeamzLaunch& h_launch = ab ? h_ab : h_ba;
      const BeamzLaunch& e_launch = ab ? e_ab : e_ba;
      enqueue_sources(h_launch, e_launch, 0, step);
      if (error != cudaSuccess)
        return error;
      if (fused_core) {
        error = BeamzEnqueueCpmlShell(stream, h_launch);
        if (error != cudaSuccess)
          return error;
        enqueue_sources(h_launch, e_launch, 1, step);
        if (error != cudaSuccess)
          return error;
        error = BeamzEnqueueCpmlCore(stream, h_launch, e_launch, source_groups,
                                     step, tile);
        if (error != cudaSuccess)
          return error;
        error = BeamzEnqueueCpmlShell(stream, e_launch);
      } else if (ScheduleFlagEnabled(program,
                                     kNativeScheduleCombinedCpmlCore)) {
        error = BeamzEnqueueCpmlPhase(stream, h_launch);
        if (error != cudaSuccess)
          return error;
        enqueue_sources(h_launch, e_launch, 1, step);
        if (error != cudaSuccess)
          return error;
        error = BeamzEnqueueCpmlPhase(stream, e_launch);
      } else {
        error = BeamzEnqueuePhase(raw_stream, h_launch);
        if (error != cudaSuccess)
          return error;
        enqueue_sources(h_launch, e_launch, 1, step);
        if (error != cudaSuccess)
          return error;
        error = BeamzEnqueuePhase(raw_stream, e_launch);
      }
      if (error != cudaSuccess)
        return error;
      enqueue_sources(h_launch, e_launch, 2, step);
      if (error != cudaSuccess)
        return error;
      if (monitors != nullptr) {
        error = ObserveProgramStep(stream, program, h_launch, e_launch, step);
        if (error != cudaSuccess)
          return error;
      }
    }
    return error;
  };
  return BeamzLaunchGraph(stream, graph_key, cache_enabled,
                          program.graph_cache_capacity, enqueue);
}

}  // namespace

int BeamzLaunchProgram(void* raw_stream, const BeamzProgramLaunch& program) {
  if (cudaError_t error = ValidateProgram(program); error != cudaSuccess) {
    return static_cast<int>(error);
  }
  if (program.field_bank_count == 1) {
    return LaunchInPlaceProgram(raw_stream, program);
  }
  if (program.h_ab.nterms == 0 && program.source_group_count == 0 &&
      program.monitors == nullptr) {
    return LaunchTemporalYeeProgram(raw_stream, program);
  }
  if (program.h_ab.nterms == 6 && program.source_group_count == 9) {
    return LaunchTemporalCpmlProgram(raw_stream, program);
  }
  return cudaErrorInvalidValue;
}

int BeamzLaunchMstep(void *raw_stream, const BeamzMstepLaunch &launch) {
  const auto &program = launch.program;
  using namespace beamz::cuda::abi;
  const int tiles = program.h_ab.cuda_flags &
      (kBeamzMstepTile8x32 | kBeamzMstepTile8x16 | kBeamzMstepTile16x16);
  const int segments = program.h_ab.cuda_flags &
      (kBeamzMstepSegment32 | kBeamzMstepSegment64 |
       kBeamzMstepSegment128 | kBeamzMstepSegment256);
  if ((tiles && (tiles & (tiles - 1))) ||
      (segments && (segments & (segments - 1))))
    return cudaErrorInvalidValue;
  if (auto error = ValidateProgram(program); error != cudaSuccess)
    return error;
  if (program.h_ab.metric_kind != 0 || program.field_bank_count != 1 ||
      program.source_group_count != 9)
    return cudaErrorInvalidValue;
  int extent[3] = {};
  for (int c = 0; c < 6; ++c) {
    const auto &reference =
        c < 3 ? program.h_ab.outputs[c] : program.e_ab.outputs[c - 3];
    const auto &final = launch.final_fields[c];
    const auto &scratch = launch.scratch_fields[c];
    if (final.rank != 3 || final.element_type != kBeamzF32 || !final.data ||
        scratch.rank != 1 || scratch.element_type != kBeamzF32 ||
        !scratch.data || scratch.dims[0] < 128 || scratch.dims[0] % 128 ||
        !FitsIntOffsets(final) || !FitsIntOffsets(scratch) ||
        final.data == reference.data || scratch.data == reference.data ||
        scratch.data == final.data)
      return cudaErrorInvalidValue;
    // The core bypasses compact/padded addressing for full field banks.
    for (const auto &field : {reference, final})
      if (field.compact_slots || field.row_stride != field.dims[2] ||
          field.plane_stride != field.dims[1] * field.dims[2])
        return cudaErrorInvalidValue;
    for (int a = 0; a < 3; ++a) {
      if (final.dims[a] != reference.dims[a])
        return cudaErrorInvalidValue;
      extent[a] = std::max(extent[a], static_cast<int>(reference.dims[a]));
    }
    if (scratch.dims[0] != launch.scratch_fields[0].dims[0])
      return cudaErrorInvalidValue;
  }
  const BeamzBuffer buffers[] = {launch.bounds, launch.core_masks, launch.slots,
                                 launch.first_cells, launch.edge_cells};
  for (const auto &b : buffers)
    if (b.element_type != kBeamzS32 || !b.data || !FitsIntOffsets(b))
      return cudaErrorInvalidValue;
  const int qy = ((extent[1] + 15) / 16) * 2, qx = ((extent[2] + 31) / 32) * 2;
  if (launch.bounds.rank != 1 || launch.bounds.dims[0] != 2 ||
      launch.core_masks.rank != 2 || launch.core_masks.dims[0] != qy / 2 ||
      launch.core_masks.dims[1] != qx / 2 || launch.slots.rank != 3 ||
      launch.slots.dims[0] != extent[0] || launch.slots.dims[1] != qy ||
      launch.slots.dims[2] != qx || launch.first_cells.rank != 1 ||
      launch.edge_cells.rank != 1 ||
      launch.first_cells.dims[0] >
          static_cast<int64_t>(extent[0]) * extent[1] * extent[2] ||
      launch.edge_cells.dims[0] > launch.first_cells.dims[0])
    return cudaErrorInvalidValue;
  BeamzBuffer original[6], scratch[6];
  for (int c = 0; c < 6; ++c) {
    original[c] = c < 3 ? program.h_ab.outputs[c] : program.e_ab.outputs[c - 3];
    scratch[c] = original[c];
    scratch[c].data = launch.scratch_fields[c].data;
    scratch[c].compact_slots = static_cast<const int *>(launch.slots.data);
    scratch[c].compact_ny = qy;
    scratch[c].compact_nx = qx;
  }
  auto phases = [&](const BeamzBuffer *old, const BeamzBuffer *next) {
    auto h = program.h_ab, e = program.e_ab;
    for (int c = 0; c < 3; ++c) {
      h.inputs[c] = old[c];
      h.inputs[3 + c] = old[3 + c];
      h.outputs[c] = next[c];
      e.inputs[c] = old[3 + c];
      e.inputs[3 + c] = next[c];
      e.outputs[c] = next[3 + c];
    }
    return std::pair<BeamzLaunch, BeamzLaunch>(h, e);
  };
  auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
  std::string key = BeamzGraphKey("mstep-column-k2", raw_stream, program);
  for (const auto &b : buffers)
    BeamzAppendBufferKey(&key, b);
  for (int c = 0; c < 6; ++c) {
    BeamzAppendBufferKey(&key, launch.final_fields[c]);
    BeamzAppendBufferKey(&key, launch.scratch_fields[c]);
  }
  auto enqueue = [&]() {
    const BeamzBuffer *current = original;
    const BeamzBuffer *dest = launch.final_fields;
    auto inject = [&](const auto &p, int timing, int step) {
      return EnqueueProgramSources(stream, program, p.first, p.second, timing, step);
    };
    auto observe = [&](const BeamzBuffer *f, int step) {
      if (!program.monitors)
        return cudaSuccess;
      auto p = phases(f, f);
      return ObserveProgramStep(stream, program, p.first, p.second, step);
    };
    for (int step = 0; step + 1 < program.nsteps; step += 2) {
      auto base = phases(current, current);
      if (auto err = inject(base, 0, step); err != cudaSuccess)
        return err;
      if (auto err =
              BeamzEnqueueMstepCore(stream, base.first, base.second, dest,
                                    launch.core_masks, launch.bounds);
          err != cudaSuccess)
        return err;
      auto first = phases(current, scratch);
      if (auto err = BeamzEnqueueMstepEdge(
              stream, first.first, launch.first_cells, extent[1], extent[2]);
          err != cudaSuccess)
        return err;
      if (auto err = inject(first, 1, step); err != cudaSuccess)
        return err;
      if (auto err = BeamzEnqueueMstepEdge(
              stream, first.second, launch.first_cells, extent[1], extent[2]);
          err != cudaSuccess)
        return err;
      if (auto err = inject(first, 2, step); err != cudaSuccess)
        return err;
      if (auto err = observe(scratch, step); err != cudaSuccess)
        return err;
      auto second = phases(scratch, dest);
      if (auto err = inject(second, 0, step + 1); err != cudaSuccess)
        return err;
      if (auto err = BeamzEnqueueMstepEdge(
              stream, second.first, launch.edge_cells, extent[1], extent[2]);
          err != cudaSuccess)
        return err;
      if (auto err = inject(second, 1, step + 1); err != cudaSuccess)
        return err;
      if (auto err = BeamzEnqueueMstepEdge(
              stream, second.second, launch.edge_cells, extent[1], extent[2]);
          err != cudaSuccess)
        return err;
      if (auto err = inject(second, 2, step + 1); err != cudaSuccess)
        return err;
      if (auto err = observe(dest, step + 1); err != cudaSuccess)
        return err;
      current = dest;
      dest = current == original ? launch.final_fields : original;
    }
    if (program.nsteps % 2) {
      const int step = program.nsteps - 1;
      auto tail = phases(current, current);
      if (auto err = inject(tail, 0, step); err != cudaSuccess)
        return err;
      if (auto err = BeamzEnqueuePhase(stream, tail.first); err != cudaSuccess)
        return err;
      if (auto err = inject(tail, 1, step); err != cudaSuccess)
        return err;
      if (auto err = BeamzEnqueuePhase(stream, tail.second); err != cudaSuccess)
        return err;
      if (auto err = inject(tail, 2, step); err != cudaSuccess)
        return err;
      if (auto err = observe(current, step); err != cudaSuccess)
        return err;
    }
    return cudaSuccess;
  };
  return BeamzLaunchGraph(stream, key,
                          FlagEnabled(program.h_ab, kBeamzGraphCache),
                          program.graph_cache_capacity, enqueue);
}
