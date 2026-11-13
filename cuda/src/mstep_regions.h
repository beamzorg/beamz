#ifndef BEAMZ_CUDA_MSTEP_REGIONS_H_
#define BEAMZ_CUDA_MSTEP_REGIONS_H_

#include <cstdint>

#ifdef __CUDACC__
#define BEAMZ_MSTEP_REGION_HD __host__ __device__ __forceinline__
#else
#define BEAMZ_MSTEP_REGION_HD inline
#endif

// A conservative union over the three staggered component supports. If this
// returns false, every valid component at this coordinate is outside the
// symmetric absorbing slab. Included axes still use component-specific checks.
BEAMZ_MSTEP_REGION_HD bool MstepCpmlAxisMayBeActive(
    int coordinate, int thickness, int64_t extent0, int64_t extent1,
    int64_t extent2) {
  const int64_t first = extent0 < extent1 ? extent0 : extent1;
  const int64_t smallest = first < extent2 ? first : extent2;
  return coordinate < thickness || coordinate >= smallest - thickness;
}

#undef BEAMZ_MSTEP_REGION_HD
#endif
