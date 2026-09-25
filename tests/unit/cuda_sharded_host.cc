// Test-only launcher for the production sharded arithmetic and FFI decoder.
// No Python numerical substitute and no CUDA runtime are involved.
#include "sharded_schedule.h"

int BeamzLaunchStreamed(void*, const BeamzLaunch&) { return 1; }
int BeamzLaunchProgram(void*, const BeamzProgramLaunch&) { return 1; }

int BeamzLaunchSharded(void*, const BeamzLaunch& launch) {
  if (!beamz::cuda::sharded::Validate(launch)) return 1;
  beamz::cuda::sharded::UpdateThread(launch, 0, 1);
  return 0;
}
