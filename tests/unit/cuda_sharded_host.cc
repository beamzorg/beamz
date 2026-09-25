// Test-only launcher for the production sharded arithmetic and FFI decoder.
// No Python numerical substitute and no CUDA runtime are involved.
#include "sharded_schedule.h"
#include "sharded_uniform.h"

int BeamzLaunchStreamed(void*, const BeamzLaunch&) { return 1; }
int BeamzLaunchProgram(void*, const BeamzProgramLaunch&) { return 1; }

int BeamzLaunchSharded(void*, const BeamzLaunch& launch) {
  if (!beamz::cuda::sharded::Validate(launch)) return 1;
  if (beamz::cuda::sharded::UniformSupported(launch)) {
    for (int c = 0; c < 3; ++c) {
      const auto& out = launch.outputs[c];
      for (int z = 0; z < out.dims[0]; ++z)
        for (int y = 0; y < out.dims[1]; ++y)
          for (int x = 0; x < out.dims[2]; ++x) {
#define CELL(P, C) beamz::cuda::sharded::UpdateUniformCell<P,C>(launch,z,y,x)
            if (launch.phase == 0) {
              if (c == 0) CELL(0,0); else if (c == 1) CELL(0,1); else CELL(0,2);
            } else {
              if (c == 0) CELL(1,0); else if (c == 1) CELL(1,1); else CELL(1,2);
            }
#undef CELL
          }
    }
  } else {
    beamz::cuda::sharded::UpdateThread(launch, 0, 1);
  }
  return 0;
}
