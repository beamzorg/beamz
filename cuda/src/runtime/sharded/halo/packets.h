#pragma once
#include <cuda_runtime_api.h>

#include <cstddef>

namespace beamz::cuda::halo {
constexpr unsigned kWords = 128;
struct Packet {
  unsigned index, words[kWords];
};
constexpr size_t kChunk = 4u << 20;
constexpr size_t kBufferBytes = kChunk / (sizeof(unsigned) * kWords) * sizeof(Packet);
constexpr size_t kEndpointBytes = 2 * kBufferBytes + sizeof(unsigned);
cudaError_t Pack(const void* src, size_t words, void* packets, void* count, cudaStream_t stream);
cudaError_t Restore(const void* packets, unsigned count, void* dst, size_t words, cudaStream_t stream);
}  // namespace beamz::cuda::halo
