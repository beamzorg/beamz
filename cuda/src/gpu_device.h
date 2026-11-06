#ifndef BEAMZ_GPU_DEVICE_H_
#define BEAMZ_GPU_DEVICE_H_

// Device-side platform prelude. Kernels are written in CUDA C++; this header
// supplies neutral names only where CUDA and HIP disagree. Every macro expands
// to the original CUDA tokens on CUDA. It is safe to include from host-only
// C++ translation units, which see only BEAMZ_GPU_COMPILER and
// BEAMZ_DEVICE_COMPILE (both undefined there).

// Compiling with a GPU compiler (either pass): replaces __CUDACC__.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define BEAMZ_GPU_COMPILER 1
#endif

// Compiling device code: replaces __CUDA_ARCH__. hip-clang never defines
// __CUDA_ARCH__, so a bare check would silently pick host code in kernels.
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define BEAMZ_DEVICE_COMPILE 1
#endif

#if defined(BEAMZ_GPU_COMPILER)

#if defined(BEAMZ_PLATFORM_HIP)

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#if defined(BEAMZ_DEVICE_COMPILE) && defined(__GFX9__)
// Warp reductions and lane masks assume 32-wide waves (RDNA). CDNA's 64-wide
// waves need width-generic code first.
#error "BeamZ HIP kernels support wave32 (RDNA) targets only"
#endif

// CUDA has no grid-constant concept on AMD: kernel arguments already live in
// a read-only kernarg segment.
#define __grid_constant__

typedef __hip_bfloat16 beamz_bf16;
// The float -> __bf16 conversion rounds to nearest even, like CUDA's _rn.
__host__ __device__ inline beamz_bf16 beamz_float_to_bf16_rn(float value) {
  return __float2bfloat16(value);
}

// HIP's *_sync builtins take 64-bit lane masks.
typedef unsigned long long beamz_lane_mask;
#define BEAMZ_FULL_LANE_MASK 0xffffffffull
__device__ inline int beamz_first_lane(beamz_lane_mask mask) {
  return __ffsll(mask);
}

// HIP's second argument means minimum waves per execution unit, not resident
// blocks per multiprocessor. Keep only the thread bound until AMD is tuned.
#define BEAMZ_LAUNCH_BOUNDS(threads, min_blocks) __launch_bounds__(threads)

// No asynchronous global-to-shared copy on RDNA. A synchronous copy is a valid
// completion of the asynchronous contract: callers already wait before reading.
#define BEAMZ_ASYNC_COPY_F32(shared_destination, global_source) \
  (*(shared_destination) = *(global_source))
#define BEAMZ_ASYNC_COMMIT() ((void)0)
#define BEAMZ_ASYNC_WAIT_ALL() ((void)0)

#else  // CUDA

#include <cuda_bf16.h>
#include <cuda_pipeline.h>

typedef __nv_bfloat16 beamz_bf16;
__host__ __device__ __forceinline__ beamz_bf16 beamz_float_to_bf16_rn(
    float value) {
  return __float2bfloat16_rn(value);
}

typedef unsigned beamz_lane_mask;
#define BEAMZ_FULL_LANE_MASK 0xffffffffu
__device__ __forceinline__ int beamz_first_lane(beamz_lane_mask mask) {
  return __ffs(mask);
}

#define BEAMZ_LAUNCH_BOUNDS(threads, min_blocks) \
  __launch_bounds__(threads, min_blocks)

#define BEAMZ_ASYNC_COPY_F32(shared_destination, global_source) \
  __pipeline_memcpy_async((shared_destination), (global_source), sizeof(float))
#define BEAMZ_ASYNC_COMMIT() __pipeline_commit()
#define BEAMZ_ASYNC_WAIT_ALL() __pipeline_wait_prior(0)

#endif  // BEAMZ_PLATFORM_HIP

#endif  // BEAMZ_GPU_COMPILER

#endif  // BEAMZ_GPU_DEVICE_H_
