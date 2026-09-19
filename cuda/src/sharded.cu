#include <cuda_runtime_api.h>

#include "sharded_schedule.h"

namespace {
__global__ void UpdateSharded(BeamzLaunch launch) {
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  beamz::cuda::sharded::UpdateThread(launch, first, stride);
}
}

int BeamzLaunchSharded(void* raw_stream, const BeamzLaunch& launch) {
  if (!beamz::cuda::sharded::Validate(launch)) return cudaErrorInvalidValue;
  const dim3 threads(beamz::cuda::sharded::kThreads);
  const int block_count = beamz::cuda::sharded::LaunchBlocks(launch);
  if (block_count == 0) return cudaSuccess;
  const dim3 blocks(block_count);
  UpdateSharded<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(raw_stream)>>>(launch);
  return static_cast<int>(cudaPeekAtLastError());
}
