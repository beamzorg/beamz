// Private experimental native API. No public FFI or production defaults change.
#pragma once
#include "launch.h"
#include <cstdint>

namespace beamz::cuda::deep {
struct Observation {
  const int64_t *keys = nullptr;   // sorted unique native component+logical-cell keys
  const uint32_t *publication_words = nullptr; // four six-bit cell masks per word
  float *values = nullptr;        // [4,key_count] raw field values, no preweighting
  int64_t field_base[6]{};         // Hx,Hy,Hz,Ex,Ey,Ez full-support prefix sums
  int key_count = 0;
  int mask_nz = 0, mask_ny = 0, mask_nx = 0;
};
}  // namespace beamz::cuda::deep

// One four-step group. All initial full fields and packed psi are immutable;
// final_fields and psi outputs must be distinct banks. Caller advances clocks.
int BeamzDeepD4AdmissionKind();
int BeamzEnqueueDeepWarpD4(void *stream, const BeamzProgramLaunch &program,
                         const BeamzBuffer *final_fields, int step,
                         const beamz::cuda::deep::Observation &observation);
// Four chronological calls to the existing native DFT arithmetic, using raw
// staged values and remapped logical gather indices. Caller advances clocks.
int BeamzEnqueueDeepWarpDft4(void *stream, const BeamzProgramLaunch &program,
                           int step,
                           const beamz::cuda::deep::Observation &observation,
                           const BeamzBuffer &remapped_indices);
