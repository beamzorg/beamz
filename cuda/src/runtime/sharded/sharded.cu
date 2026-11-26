#include <cuda_runtime_api.h>

#include <algorithm>

#include "halo/packets.h"
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

__device__ __forceinline__ bool Contains(const local::UniformBounds& r) {
  const int z = blockIdx.z * blockDim.z, y = blockIdx.y * blockDim.y, x = blockIdx.x * blockDim.x;
  return z >= r.begin[0] && z + int(blockDim.z) <= r.end[0] && y >= r.begin[1] &&
         y + int(blockDim.y) <= r.end[1] && x >= r.begin[2] && x + int(blockDim.x) <= r.end[2];
}

template <int Phase, int Rank, bool Edge, bool Shell, bool Simple>
__device__ __forceinline__ void UpdateThree(const local::UniformPhase<Simple>& view, const int p[3]) {
  local::UpdateUniformPoint<Phase, 0, Rank, Edge, Shell>(view, p);
  local::UpdateUniformPoint<Phase, 1, Rank, Edge, Shell>(view, p);
  local::UpdateUniformPoint<Phase, 2, Rank, Edge, Shell>(view, p);
}

template <int Phase, int Rank, bool Simple>
__device__ __forceinline__ void UpdateTile(const local::UniformPhase<Simple>& view) {
  const int p[3] = {int(blockIdx.z * blockDim.z + threadIdx.z),
                    int(blockIdx.y * blockDim.y + threadIdx.y),
                    int(blockIdx.x * blockDim.x + threadIdx.x)};
  if (Contains(view.bulk))
    UpdateThree<Phase, Rank, false, false>(view, p);
  else if (Contains(view.inner))
    UpdateThree<Phase, Rank, false, true>(view, p);
  else
    UpdateThree<Phase, Rank, true, true>(view, p);
}

template <int Phase, int Rank, bool Simple>
__global__ void UpdatePrepared(const __grid_constant__ local::UniformPhase<Simple> view) {
  UpdateTile<Phase, Rank>(view);
}

template <int Phase, bool Simple>
__global__ void UpdateIndirect(const __grid_constant__ BeamzLaunch launch) {
  // FFI callers own geometry on the GPU. All threads of a tile share its decoded view.
  __shared__ local::UniformPhase<Simple> view;
  if (!threadIdx.x && !threadIdx.y && !threadIdx.z)
    view = local::PrepareUniform<Simple>(launch, static_cast<const int*>(launch.shard_geometry.data));
  __syncthreads();
  UpdateTile<Phase, -1>(view);
}

template <int Phase, bool Simple>
void EnqueueUniform(cudaStream_t stream, const BeamzLaunch& l, const int* geometry, dim3 blocks,
                    dim3 threads) {
  if (!geometry) {
    UpdateIndirect<Phase, Simple><<<blocks, threads, 0, stream>>>(l);
    return;
  }
  const auto view = local::PrepareUniform<Simple>(l, geometry);
  if (view.axis == 0)
    UpdatePrepared<Phase, 0><<<blocks, threads, 0, stream>>>(view);
  else if (view.axis == 1)
    UpdatePrepared<Phase, 1><<<blocks, threads, 0, stream>>>(view);
  else
    UpdatePrepared<Phase, 2><<<blocks, threads, 0, stream>>>(view);
}

__global__ void PackFace(BeamzBuffer field, BeamzBuffer face, int axis, int plane) {
  const int n = int(face.dims[0] * face.dims[1] * face.dims[2]);
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    const int row = int(i) / int(face.dims[2]);
    int p[3] = {row / int(face.dims[1]), row % int(face.dims[1]), int(i) % int(face.dims[2])};
    p[axis] = plane;
    static_cast<float*>(face.data)[i] = local::Read(field, local::FieldOffset(field, p));
  }
}
}

cudaError_t BeamzEnqueueShardFace(cudaStream_t stream, BeamzBuffer field, BeamzBuffer face, int axis,
                                  int plane) {
  const int64_t count = local::CellCount(face);
  if (!count) return cudaSuccess;
  const int blocks = int(std::min<int64_t>((count + 255) / 256, 65535));
  PackFace<<<blocks, 256, 0, stream>>>(field, face, axis, plane);
  return cudaPeekAtLastError();
}

bool BeamzValidateNativeShard(const BeamzLaunch& launch) { return local::Validate(launch); }

int BeamzEnqueueNativeShard(void* raw_stream, const BeamzLaunch& launch, const int* g) {
  auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
  if (local::UniformSupported(launch)) {
    int extent[3] = {};
    for (int c = 0; c < 3; ++c)
      for (int d = 0; d < 3; ++d) extent[d] = max(extent[d], int(launch.outputs[c].dims[d]));
    if (extent[0] == 0 || extent[1] == 0 || extent[2] == 0) return cudaSuccess;
    const dim3 threads(32, 4, 2);
    const dim3 blocks((extent[2] + 31) / 32, (extent[1] + 3) / 4,
                      (extent[0] + threads.z - 1) / threads.z);
    if (blocks.y <= 65535 && blocks.z <= 65535) {
      bool simple = !g || !g[2];
      if (g) {
        for (int c = 0; c < 3; ++c) simple &= launch.inputs[c].data == launch.outputs[c].data;
        for (int t = 0; t < 6; ++t) simple &= launch.inputs[31 + t].data == launch.outputs[3 + t].data;
      }
      for (int c = 0; c < 3; ++c) {
        const auto& decay = launch.inputs[6 + c];
        const auto& scale = launch.inputs[9 + c];
        simple &= launch.phase == 0
                      ? (decay.rank == 0 && scale.rank == 0 && decay.element_type == kBeamzF32 &&
                         scale.element_type == kBeamzF32)
                      : (scale.element_type == kBeamzS32 && decay.element_type == kBeamzF32);
      }
      if (simple && launch.phase == 0)
        EnqueueUniform<0, true>(stream, launch, g, blocks, threads);
      else if (simple)
        EnqueueUniform<1, true>(stream, launch, g, blocks, threads);
      else if (launch.phase == 0)
        EnqueueUniform<0, false>(stream, launch, g, blocks, threads);
      else
        EnqueueUniform<1, false>(stream, launch, g, blocks, threads);
      return static_cast<int>(cudaPeekAtLastError());
    }
  }
  const int block_count = local::LaunchBlocks(launch);
  if (block_count == 0) return cudaSuccess;
  UpdateSharded<<<block_count, local::kThreads, 0, stream>>>(launch);
  return static_cast<int>(cudaPeekAtLastError());
}

int BeamzLaunchSharded(void* raw_stream, const BeamzLaunch& launch) {
  if (!BeamzValidateNativeShard(launch)) return cudaErrorInvalidValue;
  return BeamzEnqueueNativeShard(raw_stream, launch, nullptr);
}

namespace beamz::cuda::halo {
namespace {
constexpr unsigned kThreads = kWords / 2;
// Test the bit representation so signed zero, subnormals and NaN payloads survive.
__global__ void PackBlocks(const unsigned* src, size_t n, Packet* packets, unsigned* count) {
  const unsigned lane = threadIdx.x;
  const size_t i = size_t(blockIdx.x) * kWords + lane;
  const unsigned a = i < n ? src[i] : 0, b = i + kThreads < n ? src[i + kThreads] : 0;
  if (!__syncthreads_or(a | b)) return;
  __shared__ unsigned slot;
  if (!lane) slot = atomicAdd(count, 1u);
  __syncthreads();
  Packet& p = packets[slot];
  if (!lane) p.index = blockIdx.x;
  p.words[lane] = a;
  p.words[lane + kThreads] = b;
}
__global__ void RestoreBlocks(const Packet* packets, unsigned* dst, size_t n) {
  const Packet& p = packets[blockIdx.x];
  const unsigned lane = threadIdx.x;
  const size_t i = size_t(p.index) * kWords + lane;
  if (i < n) dst[i] = p.words[lane];
  if (i + kThreads < n) dst[i + kThreads] = p.words[lane + kThreads];
}
}  // namespace
cudaError_t Pack(const void* src, size_t n, void* packets, void* count, cudaStream_t stream) {
  auto e = cudaMemsetAsync(count, 0, sizeof(unsigned), stream);
  if (e != cudaSuccess) return e;
  PackBlocks<<<(n + kWords - 1) / kWords, kThreads, 0, stream>>>(
      static_cast<const unsigned*>(src), n, static_cast<Packet*>(packets), static_cast<unsigned*>(count));
  return cudaGetLastError();
}
cudaError_t Restore(const void* packets, unsigned count, void* dst, size_t n, cudaStream_t stream) {
  auto e = cudaMemsetAsync(dst, 0, n * sizeof(unsigned), stream);
  if (e != cudaSuccess || !count) return e;
  RestoreBlocks<<<count, kThreads, 0, stream>>>(static_cast<const Packet*>(packets),
                                                static_cast<unsigned*>(dst), n);
  return cudaGetLastError();
}
}  // namespace beamz::cuda::halo
