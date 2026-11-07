#ifndef BEAMZ_GPU_RUNTIME_H_
#define BEAMZ_GPU_RUNTIME_H_

// Host-side platform prelude. CUDA spelling is the source dialect; HIP builds
// (CMake defines BEAMZ_PLATFORM_HIP) alias every runtime name the component
// uses onto its HIP twin, in the way Futhark's HIP backend reuses its CUDA
// prelude. Add a name here before using a new runtime API anywhere else.

#if defined(BEAMZ_PLATFORM_HIP)

#include <hip/hip_runtime_api.h>

#define BEAMZ_GPU_PLATFORM_NAME "rocm"

#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaError_t hipError_t
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventQuery hipEventQuery
#define cudaEventRecord hipEventRecord
#define cudaEvent_t hipEvent_t
#define cudaFuncAttributeMaxDynamicSharedMemorySize \
  hipFuncAttributeMaxDynamicSharedMemorySize
// CUDA's runtime header overloads this for kernel symbols; HIP takes void*.
template <typename Kernel>
inline hipError_t BeamzHipFuncSetAttribute(Kernel* kernel,
                                           hipFuncAttribute attribute,
                                           int value) {
  return hipFuncSetAttribute(reinterpret_cast<const void*>(kernel), attribute,
                             value);
}
#define cudaFuncSetAttribute BeamzHipFuncSetAttribute
#define cudaGetDevice hipGetDevice
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphExec_t hipGraphExec_t
// CUDA 12's three-argument form; HIP keeps the CUDA 11 five-argument name.
#define cudaGraphInstantiate hipGraphInstantiateWithFlags
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraph_t hipGraph_t
#define cudaMemsetAsync hipMemsetAsync
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess

#else

#include <cuda_runtime_api.h>

#define BEAMZ_GPU_PLATFORM_NAME "cuda"

#endif

#endif  // BEAMZ_GPU_RUNTIME_H_
