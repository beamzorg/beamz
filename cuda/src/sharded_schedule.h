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

// ValidBuffer restricts every buffer to INT_MAX elements. Coordinates can use
// 32-bit division while the grid-stride increment stays 64-bit to avoid overflow
// after the final iteration. Phase and component are fixed for each traversal.
template <int Phase, int Component>
BEAMZ_SCHEDULE void UpdateComponent(const BeamzLaunch& launch, int64_t first,
                                    int64_t stride) {
  const auto& output = launch.outputs[Component];
  const int64_t count = CellCount(output);
  const int nx = static_cast<int>(output.dims[2]);
  const int ny = static_cast<int>(output.dims[1]);
  for (int64_t index = first; index < count; index += stride) {
    const int cell = static_cast<int>(index);
    const int x = cell % nx;
    const int row = cell / nx;
    const int y = row % ny;
    const int z = row / ny;
    UpdateCell<Phase, Component>(launch, z, y, x);
  }
}

BEAMZ_SCHEDULE void UpdateThread(const BeamzLaunch& launch, int64_t first,
                                 int64_t stride) {
  if (launch.phase == 0) {
    UpdateComponent<0, 0>(launch, first, stride);
    UpdateComponent<0, 1>(launch, first, stride);
    UpdateComponent<0, 2>(launch, first, stride);
  } else {
    UpdateComponent<1, 0>(launch, first, stride);
    UpdateComponent<1, 1>(launch, first, stride);
    UpdateComponent<1, 2>(launch, first, stride);
  }
}

}  // namespace beamz::cuda::sharded
#undef BEAMZ_SCHEDULE
#endif
