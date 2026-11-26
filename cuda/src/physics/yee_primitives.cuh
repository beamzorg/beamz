#ifndef BEAMZ_CUDA_YEE_PRIMITIVES_CUH_
#define BEAMZ_CUDA_YEE_PRIMITIVES_CUH_

#include <cstdint>

#include "launch.h"

// Shared numerical conventions for every CUDA Yee implementation.  Launchers
// may choose different staging and tiling, but they must agree on these fixed
// curl, CPML, and material-update definitions.
namespace beamz::cuda::yee {

#include "curl_terms.inc"

// Returns false before any packed CPML index is formed from untrusted metadata.
__host__ __device__ __forceinline__ bool CpmlPackedCoordinate(
    int coordinate, int axis_size, int low, int high, int* packed) {
  if (low < 0 || high < 0 || static_cast<int64_t>(low) + high > axis_size) {
    return false;
  }
  if (coordinate < low) {
    *packed = coordinate;
    return true;
  }
  if (coordinate >= axis_size - high) {
    *packed = low + coordinate - (axis_size - high);
    return true;
  }
  return false;
}

__device__ __forceinline__ float ScaleYeeDifference(float difference,
                                                   float inv_spacing) {
  // Round each derivative before curl subtraction or CPML correction. An
  // inlined multiply must not contract with a different stage's arithmetic.
  return __fmul_rn(difference, inv_spacing);
}

template <bool UnitDecay = false>
__device__ __forceinline__ float AdvanceYeeField(int phase, float old_field,
                                                  float decay, float source,
                                                  float curl) {
  // Fix the multiply/FMA order across ordinary, fused and temporal kernels.
  // Compiler contraction otherwise depends on inlining and tile specialization.
  return __fmaf_rn(phase == 0 ? -source : source, curl,
                   UnitDecay ? old_field : __fmul_rn(decay, old_field));
}

__device__ __forceinline__ float AdvanceCpmlPsi(float b, float old_psi,
                                                 float a, float derivative) {
  return __fmaf_rn(b, old_psi, __fmul_rn(a, derivative));
}

__device__ __forceinline__ float CorrectCpmlDerivative(float sign,
                                                        float derivative,
                                                        float inv_kappa,
                                                        float next_psi) {
  return __fmul_rn(sign, __fmaf_rn(derivative, inv_kappa, next_psi));
}

template <typename Index>
__device__ __forceinline__ float PackedMaterialSource(
    const float* codebook, int64_t codebook_size, const uint32_t* packed,
    Index linear) {
  const uint32_t word = packed[linear >> 2];
  const uint32_t code = (word >> (8 * (linear & 3))) & 0xffu;
  return code < codebook_size ? codebook[code] : 0.0f;
}

__device__ __forceinline__ float PackedMaterialSource(
    const BeamzBuffer& codebook, const BeamzBuffer& packed_codes,
    int64_t linear) {
  if (packed_codes.rank == 2) {
    // One cached header/constant per 32 Yee samples in homogeneous interiors.
    // Mixed blocks retain every original coefficient without quantization.
    const int header = static_cast<const int32_t*>(packed_codes.data)[linear >> 5];
    const int index = header < 0 ? ~header : header + (linear & 31);
    return static_cast<const float*>(codebook.data)[index];
  }
  return PackedMaterialSource(static_cast<const float*>(codebook.data),
                              codebook.dims[0],
                              static_cast<const uint32_t*>(packed_codes.data),
                              linear);
}

template <class Dimension>
__host__ __device__ __forceinline__ bool PecConstrained(
    const Dimension* dims, int phase, int component, int metallic_edges,
    int z, int y, int x) {
  const int normal_axis = 2 - component;
  const int coordinates[3] = {z, y, x};
  if (phase == 0) {
    const int coordinate = coordinates[normal_axis];
    return (coordinate == 0 &&
            (metallic_edges & (1 << (2 * normal_axis)))) ||
           (coordinate == dims[normal_axis] - 1 &&
            (metallic_edges & (1 << (2 * normal_axis + 1))));
  }
  for (int axis = 0; axis < 3; ++axis) {
    if (axis == normal_axis) continue;
    const int coordinate = coordinates[axis];
    if ((coordinate == 0 && (metallic_edges & (1 << (2 * axis)))) ||
        (coordinate == dims[axis] - 1 &&
         (metallic_edges & (1 << (2 * axis + 1))))) {
      return true;
    }
  }
  return false;
}

__host__ __device__ __forceinline__ bool PecConstrained(
    const BeamzBuffer& output, int phase, int component, int metallic_edges,
    int z, int y, int x) {
  return PecConstrained(output.dims,phase,component,metallic_edges,z,y,x);
}

}  // namespace beamz::cuda::yee

#endif  // BEAMZ_CUDA_YEE_PRIMITIVES_CUH_
