#ifndef BEAMZ_CUDA_KERNELS_H_
#define BEAMZ_CUDA_KERNELS_H_

#include <cuda_runtime_api.h>

#include "launch.h"

cudaError_t BeamzEnqueueTemporalCpml(cudaStream_t stream, const BeamzLaunch &h,
    const BeamzLaunch &e, const BeamzBuffer *final_fields,
    const BeamzSourceGroupLaunch *groups, int step, BeamzBuffer publication, int tile = 0);

cudaError_t BeamzEnqueueSpatialCpml(cudaStream_t stream, const BeamzLaunch &h,
    const BeamzLaunch &e, const BeamzSourceGroupLaunch *groups, int step);

// Leaf launchers enqueue one operation and own no graph or timestep policy.
cudaError_t BeamzValidatePhase(const BeamzLaunch &launch);
cudaError_t BeamzEnqueuePhase(void *stream, const BeamzLaunch &launch);
cudaError_t BeamzEnqueueCpmlPhase(cudaStream_t stream,
                                  const BeamzLaunch &launch);
cudaError_t BeamzEnqueueCpmlShell(cudaStream_t stream,
                                  const BeamzLaunch &launch);
cudaError_t BeamzEnqueueCpmlCore(cudaStream_t stream, const BeamzLaunch &h,
                                 const BeamzLaunch &e,
                                 const BeamzSourceGroupLaunch *groups, int step,
                                 int tile = 0, bool band = false);
cudaError_t BeamzEnqueueCpmlPairBand(cudaStream_t stream, const BeamzLaunch &h,
                                     const BeamzLaunch &e,
                                     const BeamzSourceGroupLaunch *groups,
                                     int step);
cudaError_t BeamzEnqueueCpmlPairCore(cudaStream_t stream, const BeamzLaunch &h,
                                     const BeamzLaunch &e,
                                     const BeamzBuffer *final_fields,
                                     const BeamzSourceGroupLaunch *groups,
                                     int step, BeamzBuffer publication = {},
                                     int tile = 0);
cudaError_t BeamzEnqueueFusedFullStep(cudaStream_t stream,
                                      const BeamzLaunch &h_launch,
                                      const BeamzLaunch &e_launch);
cudaError_t BeamzEnqueueSourceGroup(cudaStream_t stream,
                                    const BeamzLaunch &launch,
                                    const BeamzBuffer &target,
                                    const BeamzSourceGroupLaunch &group,
                                    int32_t step);
cudaError_t BeamzEnqueueSourcePhase(cudaStream_t stream,
                                    const BeamzLaunch &h_launch,
                                    const BeamzLaunch &e_launch,
                                    const BeamzSourceGroupLaunch *groups,
                                    int32_t count, int timing, int32_t step,
                                    bool shell_only = false,
                                    int inner_margin = -1);
cudaError_t BeamzEnqueueDftGroups(cudaStream_t stream,
                                  const BeamzLaunch &h_launch,
                                  const BeamzLaunch &e_launch,
                                  const BeamzDftGroupLaunch &monitors,
                                  int32_t step);

cudaError_t BeamzEnqueueDftPair(cudaStream_t stream, const BeamzLaunch &h1,
                                const BeamzLaunch &e1, const BeamzLaunch &h2,
                                const BeamzLaunch &e2,
                                const BeamzDftGroupLaunch &monitors, int step);

#endif // BEAMZ_CUDA_KERNELS_H_
