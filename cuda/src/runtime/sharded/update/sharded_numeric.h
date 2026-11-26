#ifndef BEAMZ_CUDA_SHARDED_NUMERIC_H_
#define BEAMZ_CUDA_SHARDED_NUMERIC_H_
#ifdef __CUDACC__
#include "yee_primitives.cuh"
#define BEAMZ_NUMERIC __host__ __device__ __forceinline__
#else
#define BEAMZ_NUMERIC inline
#endif
namespace beamz::cuda::sharded {
BEAMZ_NUMERIC float FixedAdvance(int phase, float old, float decay, float source, float curl) {
#ifdef __CUDA_ARCH__
  return beamz::cuda::yee::AdvanceYeeField(phase, old, decay, source, curl);
#else
  return decay * old + (phase == 0 ? -source : source) * curl;
#endif
}
BEAMZ_NUMERIC float FixedDifference(float difference, float inverse_spacing) {
#ifdef __CUDA_ARCH__
  return beamz::cuda::yee::ScaleYeeDifference(difference, inverse_spacing);
#else
  return difference * inverse_spacing;
#endif
}
BEAMZ_NUMERIC float FixedPsi(float b, float old, float a, float derivative) {
#ifdef __CUDA_ARCH__
  return beamz::cuda::yee::AdvanceCpmlPsi(b, old, a, derivative);
#else
  return b * old + a * derivative;
#endif
}
BEAMZ_NUMERIC float FixedCorrection(float sign, float derivative, float inverse_kappa, float psi) {
#ifdef __CUDA_ARCH__
  return beamz::cuda::yee::CorrectCpmlDerivative(sign, derivative, inverse_kappa, psi);
#else
  return sign * (derivative * inverse_kappa + psi);
#endif
}
}
#undef BEAMZ_NUMERIC
#endif
