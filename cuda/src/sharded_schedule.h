#ifndef BEAMZ_CUDA_SHARDED_SCHEDULE_H_
#define BEAMZ_CUDA_SHARDED_SCHEDULE_H_

#include "sharded_cell.h"

#ifdef __CUDACC__
#define BEAMZ_SCHEDULE __host__ __device__ inline
#else
#define BEAMZ_SCHEDULE inline
#endif

namespace beamz::cuda::sharded {

constexpr int kThreads = 128;
constexpr int kMaxBlocks = 65535;

BEAMZ_SCHEDULE int64_t CellCount(const BeamzBuffer& buffer) {
  return buffer.dims[0] * buffer.dims[1] * buffer.dims[2];
}

inline int LaunchBlocks(const BeamzLaunch& launch) {
  int64_t count = 0;
  for (int c = 0; c < 3; ++c) {
    const int64_t cells = CellCount(launch.outputs[c]);
    if (cells > count) count = cells;
  }
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  return static_cast<int>(blocks > kMaxBlocks ? kMaxBlocks : blocks);
}

// Use 64-bit offsets for the loop increment, including its final iteration.
// Each component has its own Yee support and therefore its own linear bounds.
BEAMZ_SCHEDULE void UpdateThread(const BeamzLaunch& launch, int64_t first,
                                 int64_t stride) {
  for (int c = 0; c < 3; ++c) {
    const auto& output = launch.outputs[c];
    const int64_t count = CellCount(output);
    for (int64_t index = first; index < count; index += stride) {
      const int x = static_cast<int>(index % output.dims[2]);
      const int64_t row = index / output.dims[2];
      const int y = static_cast<int>(row % output.dims[1]);
      const int z = static_cast<int>(row / output.dims[1]);
      UpdateCell(launch, c, z, y, x);
    }
  }
}

}  // namespace beamz::cuda::sharded
#undef BEAMZ_SCHEDULE
#endif
