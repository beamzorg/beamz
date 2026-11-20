#include "native_launch.h"

using namespace beamz::cuda::abi;

void BeamzNativePrepare(const BeamzNativeAttributes& a,
                        const BeamzBuffer* inputs, const BeamzBuffer* outputs,
                        int source_offset, int monitor_offset,
                        int workspace_count, bool mstep,
                        BeamzNativeLaunchStorage* storage) {
  auto initialize = [&](int phase) {
    BeamzLaunch value{};
    value.abi_version = a.abi_version;
    value.cuda_flags = a.cuda_flags;
    value.phase = phase;
    value.nterms = a.cpml_enabled ? kCpmlTermCount : 0;
    value.metric_kind = a.metric_kind;
    value.dt = a.dt;
    value.resolution = a.resolution;
    value.inv_resolution = 1.0f / a.resolution;
    value.dt_over_eps = a.dt / 8.8541878128e-12f;
    value.dt_over_mu = a.dt / 1.25663706212e-6f;
    value.metallic_edges = a.boundary_code & 0x3f;
    value.uniform_cpml_thickness = a.boundary_code >> 8;
    return value;
  };
  auto& program = storage->program;
  program = {};
  program.h_ab = initialize(0);
  program.e_ab = initialize(1);
  auto& h = program.h_ab;
  auto& e = program.e_ab;
  for (int c = 0; c < 3; ++c) {
    h.inputs[c] = inputs[c];
    h.inputs[3 + c] = inputs[3 + c];
    h.outputs[c] = outputs[c];
    e.inputs[c] = inputs[3 + c];
    e.inputs[3 + c] = outputs[c];
    e.outputs[c] = outputs[3 + c];
  }
  if (a.cpml_enabled) {
    for (int i = 0; i < static_cast<int>(kCpmlPhaseInputCount); ++i) {
      h.inputs[kFieldCount + i] = inputs[kFieldCount + i];
      e.inputs[kFieldCount + i] = inputs[kCpmlEInputOffset + i];
    }
    for (int term = 0; term < static_cast<int>(kCpmlTermCount); ++term) {
      h.outputs[3 + term] = outputs[kFieldCount + term];
      e.outputs[3 + term] = outputs[kFieldCount + kCpmlTermCount + term];
    }
    for (int axis = 0; axis < 3; ++axis) {
      h.metrics[axis] = inputs[kCpmlHMetricOffset + axis];
      e.metrics[axis] = inputs[kCpmlEMetricOffset + axis];
    }
  } else {
    for (int i = 0; i < 6; ++i) {
      h.inputs[6 + i] = inputs[6 + i];
      e.inputs[6 + i] = inputs[12 + i];
    }
    for (int axis = 0; axis < 3; ++axis) {
      h.metrics[axis] = inputs[18 + axis];
      e.metrics[axis] = inputs[21 + axis];
    }
  }
  for (int i = 0; i < static_cast<int>(kSourceGroupCount); ++i) {
    auto& group = storage->groups[i];
    group = {};
    group.coefficients = inputs[source_offset + 3 * i];
    group.waveforms = inputs[source_offset + 3 * i + 1];
    group.starts = inputs[source_offset + 3 * i + 2];
    group.current_step = inputs[monitor_offset + kMonitorCurrentStepInput];
    group.component = i % 3;
    group.timing = i / 3;
    group.coincident = (a.coincident_source_group_mask >> i) & 1;
    group.disjoint = (a.disjoint_source_group_mask >> i) & 1;
  }
  const int state_count = a.cpml_enabled ? kCpmlGraphOutputCount : kYeeGraphOutputCount;
  const int dft_offset = state_count + workspace_count;
  auto& monitors = storage->monitors;
  monitors = {};
  monitors.indices = inputs[monitor_offset];
  monitors.weights = inputs[monitor_offset + 1];
  monitors.frequencies = inputs[monitor_offset + 2];
  monitors.component_masks = inputs[monitor_offset + 3];
  monitors.counts = inputs[monitor_offset + 4];
  monitors.codes = inputs[monitor_offset + 5];
  monitors.windows = inputs[monitor_offset + 6];
  monitors.dft_re = outputs[dft_offset];
  monitors.dft_im = outputs[dft_offset + 1];
  monitors.dft_weight = outputs[dft_offset + 2];
  monitors.phase_sin = outputs[dft_offset + 3];
  monitors.phase_cos = outputs[dft_offset + 4];
  monitors.phase_window = outputs[dft_offset + 5];
  monitors.time = inputs[monitor_offset + 13];
  monitors.current_step = inputs[monitor_offset + 14];
  monitors.elapsed_steps = inputs[monitor_offset + 15];
  monitors.monitor_count = a.monitor_count;
  program.source_groups = storage->groups;
  program.source_group_count = kSourceGroupCount;
  program.monitors = a.monitor_count ? &monitors : nullptr;
  program.field_bank_count = 1;
  program.nsteps = a.nsteps;
  program.graph_cache_capacity = a.graph_cache_capacity;
  program.schedule_flags = a.schedule_flags;
  if (mstep) {
    auto& launch = storage->mstep;
    launch = {};
    launch.program = program;
    for (int c = 0; c < 6; ++c) {
      launch.final_fields[c] = outputs[state_count + c];
      launch.scratch_fields[c] = outputs[state_count + 6 + c];
    }
    const int plan_offset = monitor_offset + kMonitorInputCount;
    launch.bounds = inputs[plan_offset];
    launch.core_masks = inputs[plan_offset + 1];
    launch.slots = inputs[plan_offset + 2];
    launch.first_cells = inputs[plan_offset + 3];
    launch.edge_cells = inputs[plan_offset + 4];
  }
}
