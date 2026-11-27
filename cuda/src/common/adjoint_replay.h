#ifndef BEAMZ_CUDA_ADJOINT_REPLAY_H_
#define BEAMZ_CUDA_ADJOINT_REPLAY_H_

#include <cuda_runtime_api.h>

#include "launch.h"

// Optional replay packet stores the corrected electric curl on each Yee
// support. The H fields and CPML memories are scratch in this packet.
cudaError_t BeamzEnqueueElectricCurl(cudaStream_t stream, const BeamzLaunch& launch,
                                    const BeamzBuffer* curls, const BeamzBuffer* old_e,
                                    bool last_step);
int BeamzLaunchCurlTape(void* stream, const BeamzProgramLaunch& program,
                       const BeamzBuffer* curls, const BeamzBuffer* old_e);
int BeamzLaunchAdjointFromCurl(void* stream, const BeamzAdjointLaunch& launch,
                             const BeamzBuffer* curls, const BeamzBuffer* old_e,
                             const BeamzBuffer* decay_gradients);

#endif  // BEAMZ_CUDA_ADJOINT_REPLAY_H_
