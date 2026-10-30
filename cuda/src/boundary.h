#ifndef BEAMZ_CUDA_BOUNDARY_H_
#define BEAMZ_CUDA_BOUNDARY_H_

#include "launch.h"

#ifdef __CUDACC__
#define BEAMZ_BOUNDARY __host__ __device__ __forceinline__
#else
#define BEAMZ_BOUNDARY inline
#endif

namespace beamz::cuda::boundary {

// Tangential H lives half a cell inside the physical wall. Its odd PMC image
// doubles the backward derivative at the E node. PEC retains a zero ghost;
// an open face retains the existing even ghost (zero boundary derivative).
BEAMZ_BOUNDARY float DerivativeFactor(int pec, int pmc, int axis, bool high) {
  const int bit = 1 << (2 * axis + static_cast<int>(high));
  return (pmc & bit) ? 2.0f : ((pec & bit) ? 1.0f : 0.0f);
}

BEAMZ_BOUNDARY bool ValidFaces(int pec, int pmc) {
  return pec >= 0 && pec <= 63 && pmc >= 0 && pmc <= 63 && !(pec & pmc);
}

// The production streamed stencil is portable so CPU-only development can
// exercise its physical endpoints and pitched storage without a CUDA runtime.
template <bool HasWalls = true, bool Compact = false>
BEAMZ_BOUNDARY float BackwardDifference(
    const BeamzBuffer& value, int axis, int z, int y, int x,
    int pec, int pmc, float inv_spacing) {
  const int coordinate = axis == 0 ? z : (axis == 1 ? y : x);
  const int size = static_cast<int>(value.dims[axis]);
  int low_z = z, low_y = y, low_x = x;
  if (axis == 0) --low_z;
  if (axis == 1) --low_y;
  if (axis == 2) --low_x;
  const auto* data = static_cast<const float*>(value.data);
  float difference;
  if (coordinate == 0 || coordinate == size) {
    if constexpr (!HasWalls) return 0.0f;
    const bool high = coordinate == size;
    const float factor = DerivativeFactor(pec, pmc, axis, high);
    if (!factor) return 0.0f;
    const int offset = high ? BeamzOffset3D<Compact>(value, low_z, low_y, low_x)
                            : BeamzOffset3D<Compact>(value, z, y, x);
    difference = (high ? -factor : factor) * data[offset];
  } else {
    difference = data[BeamzOffset3D<Compact>(value, z, y, x)]
                 - data[BeamzOffset3D<Compact>(value, low_z, low_y, low_x)];
  }
#ifdef __CUDA_ARCH__
  return __fmul_rn(difference, inv_spacing);
#else
  return difference * inv_spacing;
#endif
}

}  // namespace beamz::cuda::boundary
#undef BEAMZ_BOUNDARY
#endif  // BEAMZ_CUDA_BOUNDARY_H_
