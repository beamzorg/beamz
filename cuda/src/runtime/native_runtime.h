#ifndef BEAMZ_CUDA_NATIVE_RUNTIME_H_
#define BEAMZ_CUDA_NATIVE_RUNTIME_H_

#include <cuda_runtime_api.h>
#include <cstdint>
#include <string>

#include "launch.h"

cudaError_t BeamzNativeAdvanceClocks(cudaStream_t stream, int* current_step,
                                   int* elapsed_steps, const float* origin,
                                   float* current_time, float dt, int nsteps);

// Materials are scalar, dense, or run-length encoded on logical Yee supports.
// Geometry stays factorized: no full-volume diagnostic weights are allocated.
struct BeamzDiagnosticMaterial {
  const float* values = nullptr;
  const int32_t* ends = nullptr;
  int count = 0;
  int encoding = 0;  // 0 scalar, 1 dense, 2 exclusive-end RLE
  const double* widths[3] = {};
};

struct BeamzDiagnosticPartial {
  double sum;
  float maximum;
  uint32_t nonfinite;
};

constexpr int kBeamzDiagnosticBlocks = 1024;

cudaError_t BeamzEnqueueFieldDiagnostics(
    cudaStream_t stream, const BeamzBuffer& field,
    const BeamzDiagnosticMaterial& material, double domain_measure,
    BeamzDiagnosticPartial* partials);

cudaError_t BeamzEnqueueTensorDiagnostics(
    cudaStream_t stream, const BeamzBuffer& left, int left_axis,
    const BeamzBuffer& right, int right_axis, const BeamzBuffer& nodes,
    const BeamzDiagnosticMaterial& material, const double* left_widths,
    const double* right_widths, double domain_measure,
    BeamzDiagnosticPartial* partials);

struct BeamzDftNorms {
  double current, previous, difference;
  unsigned nonfinite;
};

struct BeamzDftResidual {
  unsigned long long maximum_bits;
  unsigned nonfinite;
};

constexpr int kBeamzDftDiagnosticTile = 4096;

cudaError_t BeamzEnqueueDftDiagnostics(
    cudaStream_t stream, const float* real, const float* imag,
    float* previous_real, float* previous_imag, int frequencies, int points,
    bool compare, BeamzDftNorms* partials, BeamzDftResidual* residual);

// Optional ordinary-schedule effects. Stage 0/1/2 follows pre-E/H/E
// injection respectively; stage 3 follows the completed DFT acquisition.
struct BeamzNativeStepEffects {
  using Enqueue = cudaError_t (*)(cudaStream_t, const BeamzProgramLaunch&,
                                 int stage, int offset, const void*);
  Enqueue enqueue;
  const void* context;
  std::string key;
};

struct BeamzNativeScatter {
  BeamzBuffer indices, ends, coefficients, waveform;
  int timing, component;
};

struct BeamzNativeRecorder {
  BeamzBuffer indices, weights, frames;
  int component, interval;
  float sign;
};

int BeamzLaunchProgramEffects(void*, const BeamzProgramLaunch&,
                              const BeamzNativeStepEffects*);
int BeamzLaunchTracedProgramEffects(void*, const BeamzProgramLaunch&,
                                    const BeamzNativeStepEffects*);
int BeamzEnqueueProgramStepAtWithEffects(void*, const BeamzProgramLaunch&, int,
                                        const BeamzNativeStepEffects&);
bool BeamzEnqueueTracedReplayStep(void*, const BeamzProgramLaunch&, int step,
                                 bool magnetic_only, cudaError_t* status);
cudaError_t BeamzEnqueueNativeScatter(cudaStream_t, const BeamzProgramLaunch&,
                                      const BeamzNativeScatter&, int offset);
cudaError_t BeamzEnqueueNativeRecorder(cudaStream_t, const BeamzProgramLaunch&,
                                       const BeamzNativeRecorder&, int frame);
cudaError_t BeamzEnqueueNativeRecorderFrame(cudaStream_t, const BeamzBuffer&,
                                            const BeamzNativeRecorder&, int frame);

#endif
