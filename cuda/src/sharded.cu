#include <cuda_runtime_api.h>

#include "sharded_schedule.h"
#include "sharded_uniform.h"
#include "yee_primitives.cuh"

namespace {
namespace local = beamz::cuda::sharded;

__global__ void UpdateSharded(BeamzLaunch launch) {
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  local::UpdateThread(launch, first, stride);
}

// Classify whole tiles, so the bulk kernel has neither packed CPML state nor
// per-neighbor logical-bound checks in its register footprint. Conservative
// bounds send staggered edges and storage padding to the general kernel.
__device__ __forceinline__ bool BulkTile(const BeamzLaunch& l) {
  const auto* g = static_cast<const int32_t*>(l.shard_geometry.data);
  if (g[2]) return false;  // Tensor curls retain the general constitutive path.
  const int first[3] = {int(blockIdx.z * blockDim.z),
                        int(blockIdx.y * blockDim.y),
                        int(blockIdx.x * blockDim.x)};
  const int width[3] = {int(blockDim.z), int(blockDim.y), int(blockDim.x)};
  const int thickness = l.uniform_cpml_thickness;
  for (int d = 0; d < 3; ++d) {
    const int global_first = first[d] + (d == g[0] ? g[1] : 0);
    if (d == g[0] && (first[d] < 1 || first[d] + width[d] >= l.outputs[0].dims[d])) return false;
    if (global_first < thickness + 1) return false;
    for (int c = 0; c < 3; ++c) {
      if (first[d] + width[d] > l.outputs[c].dims[d] ||
          global_first + width[d] >= g[3 + 3*c + d] - thickness)
        return false;
    }
  }
  return true;
}

template<int Phase, int Component, int Axis>
__device__ __forceinline__ float BulkDifference(const BeamzLaunch& l,
                                               const int p[3], int shard_axis) {
  const auto& value = l.inputs[3 + Component];
  const int q[3] = {p[0], p[1], p[2]};
  const int offset = (q[0] * int(value.dims[1]) + q[1]) * int(value.dims[2]) + q[2];
  constexpr int sign = Phase == 0 ? 1 : -1;
  const int stride = Axis == 0 ? int(value.dims[1] * value.dims[2])
                   : Axis == 1 ? int(value.dims[2]) : 1;
  const auto* input = static_cast<const float*>(value.data);
  return (Phase == 0 ? input[offset + sign*stride] - input[offset]
                     : input[offset] - input[offset + sign*stride]) * l.inv_resolution;
}

template<int Phase, int Component>
__device__ __forceinline__ void BulkComponent(const BeamzLaunch& l,
                                             const int p[3], int shard_axis) {
  constexpr auto first = beamz::cuda::yee::FirstCurlTerm(Component);
  constexpr auto second = beamz::cuda::yee::SecondCurlTerm(Component);
  const float curl = BulkDifference<Phase, first.source_component, first.derivative_axis>(l, p, shard_axis)
                   - BulkDifference<Phase, second.source_component, second.derivative_axis>(l, p, shard_axis);
  const int offset = local::Offset(l.outputs[Component], p);
  const float decay = local::Read(l.inputs[6 + Component], local::Offset(l.inputs[6 + Component], p));
  const float scale = local::Read(l.inputs[9 + Component], local::Offset(l.inputs[9 + Component], p));
  const float old = local::Read(l.inputs[Component], offset);
  static_cast<float*>(l.outputs[Component].data)[offset] =
      beamz::cuda::yee::AdvanceYeeField(Phase, old, decay, scale, curl);
}

template<int Phase, bool Bulk>
__global__ void UpdateShardedTile(BeamzLaunch launch) {
  if (BulkTile(launch) != Bulk) return;
  const int p[3] = {int(blockIdx.z * blockDim.z + threadIdx.z),
                    int(blockIdx.y * blockDim.y + threadIdx.y),
                    int(blockIdx.x * blockDim.x + threadIdx.x)};
  if constexpr (Bulk) {
    const int axis = static_cast<const int32_t*>(launch.shard_geometry.data)[0];
    BulkComponent<Phase, 0>(launch, p, axis);
    BulkComponent<Phase, 1>(launch, p, axis);
    BulkComponent<Phase, 2>(launch, p, axis);
  } else {
    local::UpdateUniformCell<Phase, 0>(launch, p[0], p[1], p[2]);
    local::UpdateUniformCell<Phase, 1>(launch, p[0], p[1], p[2]);
    local::UpdateUniformCell<Phase, 2>(launch, p[0], p[1], p[2]);
  }
}
}

int BeamzLaunchSharded(void* raw_stream, const BeamzLaunch& launch) {
  if (!local::Validate(launch)) return cudaErrorInvalidValue;
  auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
  if (local::UniformSupported(launch)) {
    int extent[3] = {};
    for (int c = 0; c < 3; ++c)
      for (int d = 0; d < 3; ++d)
        extent[d] = max(extent[d], int(launch.outputs[c].dims[d]));
    if (extent[0] == 0 || extent[1] == 0 || extent[2] == 0) return cudaSuccess;
    const dim3 threads(32, 4, 2);
    const dim3 blocks((extent[2]+31)/32, (extent[1]+3)/4, (extent[0]+1)/2);
    if (blocks.y <= 65535 && blocks.z <= 65535) {
      if (launch.phase == 0) {
        UpdateShardedTile<0, true><<<blocks, threads, 0, stream>>>(launch);
        UpdateShardedTile<0, false><<<blocks, threads, 0, stream>>>(launch);
      } else {
        UpdateShardedTile<1, true><<<blocks, threads, 0, stream>>>(launch);
        UpdateShardedTile<1, false><<<blocks, threads, 0, stream>>>(launch);
      }
      return static_cast<int>(cudaPeekAtLastError());
    }
  }
  const int block_count = local::LaunchBlocks(launch);
  if (block_count == 0) return cudaSuccess;
  UpdateSharded<<<block_count, local::kThreads, 0, stream>>>(launch);
  return static_cast<int>(cudaPeekAtLastError());
}
