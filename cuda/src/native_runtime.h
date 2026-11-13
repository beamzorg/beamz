#ifndef BEAMZ_CUDA_NATIVE_RUNTIME_H_
#define BEAMZ_CUDA_NATIVE_RUNTIME_H_

#include <cuda_runtime_api.h>

cudaError_t BeamzNativeAdvanceClocks(cudaStream_t stream, int* current_step,
                                   int* elapsed_steps, const float* origin,
                                   float* current_time, float dt, int nsteps);

#endif
