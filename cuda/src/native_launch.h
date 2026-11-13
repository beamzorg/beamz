#ifndef BEAMZ_CUDA_NATIVE_LAUNCH_H_
#define BEAMZ_CUDA_NATIVE_LAUNCH_H_

#include "abi_layout.h"
#include "launch.h"

// Independent front-end descriptors. These mirror the FFI decoder, with no
// dependency on XLA/JAX types or ownership conventions.
struct BeamzNativeAttributes {
  int abi_version = beamz::cuda::abi::kAbiVersion;
  int cuda_flags = beamz::cuda::abi::kBeamzGraphCache;
  int nsteps = 1, boundary_code = 0, metric_kind = 0;
  int monitor_count = 0, coincident_source_group_mask = 0;
  int disjoint_source_group_mask = 0, graph_cache_capacity = 32;
  int schedule_flags = 0;
  bool cpml_enabled = false;
  float dt = 0.f, resolution = 0.f;
};

struct BeamzNativeLaunchStorage {
  BeamzSourceGroupLaunch groups[beamz::cuda::abi::kSourceGroupCount]{};
  BeamzDftGroupLaunch monitors{};
  BeamzProgramLaunch program{};
  BeamzMstepLaunch mstep{};
};

void BeamzNativePrepare(const BeamzNativeAttributes& attributes,
                        const BeamzBuffer* inputs, const BeamzBuffer* outputs,
                        int source_offset, int monitor_offset,
                        int workspace_count, bool mstep,
                        BeamzNativeLaunchStorage* storage);

#endif
