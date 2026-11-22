#ifndef BEAMZ_CUDA_MSTEP_INDEX_H_
#define BEAMZ_CUDA_MSTEP_INDEX_H_

#include <cstdint>

// The compact queue stores nonnegative row IDs below INT32_MAX. For this
// bounded numerator range a rounded-up 32-bit reciprocal produces either the
// exact quotient or one above it; a multiply/compare corrects that last case.
// No field-value approximation or changed coordinate representation is used.
struct MstepDivisor {
  uint32_t divisor;
  uint32_t multiplier;
};

inline MstepDivisor MakeMstepDivisor(uint32_t divisor) {
  return {divisor,
          divisor > 1
              ? static_cast<uint32_t>(((uint64_t{1} << 32) + divisor - 1) /
                                      divisor)
              : 0};
}

struct MstepIndexDecode {
  MstepDivisor x;
  MstepDivisor plane;
};

inline MstepIndexDecode MakeMstepIndexDecode(int ny, int nx) {
  return {MakeMstepDivisor(static_cast<uint32_t>(nx)),
          MakeMstepDivisor(static_cast<uint32_t>(ny) * nx)};
}

#ifdef __CUDACC__
#define BEAMZ_MSTEP_INDEX_HD __host__ __device__ __forceinline__
#else
#define BEAMZ_MSTEP_INDEX_HD inline
#endif

BEAMZ_MSTEP_INDEX_HD uint32_t MstepQuotient(uint32_t value,
                                           MstepDivisor divisor) {
  if (divisor.divisor == 1) return value;
#ifdef __CUDA_ARCH__
  const uint32_t quotient = __umulhi(value, divisor.multiplier);
#else
  const uint32_t quotient = static_cast<uint32_t>(
      (static_cast<uint64_t>(value) * divisor.multiplier) >> 32);
#endif
  // value <= INT32_MAX and divisor <= INT32_MAX imply this product cannot
  // overflow uint32_t, even when the initial quotient is one too large.
  return quotient - static_cast<uint32_t>(quotient * divisor.divisor > value);
}

#undef BEAMZ_MSTEP_INDEX_HD
#endif
