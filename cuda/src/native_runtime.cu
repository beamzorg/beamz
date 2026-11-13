#include "native_runtime.h"

namespace {
__global__ void AdvanceClocks(int* current_step, int* elapsed_steps,
                              const float* origin, float* current_time,
                              float dt, int nsteps) {
  *current_step += nsteps;
  *elapsed_steps += nsteps;
  // Match the existing state-time expression's two rounding steps. In
  // particular a nonzero origin can distinguish this from a contracted FMA.
  const float elapsed_time = __fmul_rn(static_cast<float>(*elapsed_steps), dt);
  *current_time = __fadd_rn(*origin, elapsed_time);
}
}  // namespace

cudaError_t BeamzNativeAdvanceClocks(cudaStream_t stream, int* current_step,
                                   int* elapsed_steps, const float* origin,
                                   float* current_time, float dt, int nsteps) {
  AdvanceClocks<<<1, 1, 0, stream>>>(current_step, elapsed_steps, origin,
                                    current_time, dt, nsteps);
  return cudaGetLastError();
}
