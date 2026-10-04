#include <algorithm>
#include <cassert>
#include <climits>
#include <vector>

#include "sharded_schedule.h"

int main() {
  using namespace beamz::cuda::sharded;
  BeamzLaunch launch{};
  int32_t geometry[21]{};
  launch.shard_geometry.data = geometry;
  float decay = 2.0f, source = 0.0f;
  std::vector<float> fields[3];
  for (int c = 0; c < 3; ++c) {
    // Distinct Yee supports, all beyond the old grid.z limit.
    const int z = 65536 + c, y = 2, x = 3 - c;
    geometry[3 + 3 * c] = z;
    geometry[4 + 3 * c] = y;
    geometry[5 + 3 * c] = x;
    fields[c].resize(z * y * x);
    launch.outputs[c] = {fields[c].data(), 3, kBeamzF32, {z, y, x, 0}};
    launch.inputs[c] = launch.outputs[c];
    launch.inputs[6 + c] = {&decay, 0, kBeamzF32, {}};
    launch.inputs[9 + c] = {&source, 0, kBeamzF32, {}};
  }
  const int blocks = LaunchBlocks(launch);
  assert(blocks > 0 && blocks <= kMaxBlocks);
  // Exercise both the actual launch width and repeated grid-stride iterations.
  for (int64_t stride : {int64_t(blocks) * kThreads, int64_t(kThreads)}) {
    for (auto& field : fields) std::fill(field.begin(), field.end(), 1.0f);
    for (int64_t first = 0; first < stride; ++first)
      UpdateThread(launch, first, stride);
    // In-place doubling detects missed cells AND duplicate visits.
    for (const auto& field : fields)
      for (float value : field) assert(value == 2.0f);
  }
  // Verify the cap and overflow-safe rounding without allocating a huge field.
  launch.outputs[0].dims[0] = INT_MAX;
  launch.outputs[0].dims[1] = launch.outputs[0].dims[2] = 1;
  assert(LaunchBlocks(launch) == kMaxBlocks);
  for (auto& output : launch.outputs) output.dims[0] = 0;
  assert(LaunchBlocks(launch) == 0);
}
