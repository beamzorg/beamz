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

#elif defined(BEAMZ_PLATFORM_CPU)

// Host execution: kernels run synchronously through the OpenMP block
// scheduler in cpu_launch.h, so streams, events and device queries are
// trivial. CUDA graphs do not exist; graph.cu enqueues directly.
#include <cstddef>
#include <cstring>

#define BEAMZ_GPU_PLATFORM_NAME "cpu"

enum cudaError_t {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorNotReady = 600,
  cudaErrorNotSupported = 801,
};
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
typedef void* cudaGraph_t;
typedef void* cudaGraphExec_t;
enum cudaDeviceAttr {
  cudaDevAttrComputeCapabilityMajor,
  cudaDevAttrComputeCapabilityMinor,
};
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize };
enum cudaStreamCaptureMode { cudaStreamCaptureModeThreadLocal };
#define cudaEventDisableTiming 0u

// Launch errors are per calling thread, as with the CUDA runtime.
inline thread_local cudaError_t beamz_cpu_last_error = cudaSuccess;

inline cudaError_t cudaGetLastError() {
  const cudaError_t error = beamz_cpu_last_error;
  beamz_cpu_last_error = cudaSuccess;
  return error;
}
inline cudaError_t cudaPeekAtLastError() { return beamz_cpu_last_error; }
inline const char* cudaGetErrorString(cudaError_t error) {
  switch (error) {
    case cudaSuccess: return "no error";
    case cudaErrorInvalidValue: return "invalid argument";
    case cudaErrorMemoryAllocation: return "out of memory";
    case cudaErrorNotReady: return "not ready";
    case cudaErrorNotSupported: return "operation not supported on CPU";
  }
  return "unknown error";
}
inline cudaError_t cudaGetDevice(int* device) {
  *device = 0;
  return cudaSuccess;
}
// No compute capability: tuning gates for NVIDIA architectures stay closed.
inline cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr, int) {
  *value = 0;
  return cudaSuccess;
}
template <typename Kernel>
inline cudaError_t cudaFuncSetAttribute(Kernel*, cudaFuncAttribute, int) {
  return cudaSuccess;
}
inline cudaError_t cudaMemsetAsync(void* data, int value, size_t bytes,
                                   cudaStream_t = nullptr) {
  std::memset(data, value, bytes);
  return cudaSuccess;
}
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned) {
  *event = nullptr;
  return cudaSuccess;
}
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr) {
  return cudaSuccess;
}
inline cudaError_t cudaEventQuery(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode) {
  return cudaErrorNotSupported;
}
inline cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t*) {
  return cudaErrorNotSupported;
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t*, cudaGraph_t,
                                        unsigned long long) {
  return cudaErrorNotSupported;
}
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) {
  return cudaErrorNotSupported;
}
inline cudaError_t cudaGraphDestroy(cudaGraph_t) { return cudaSuccess; }
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t) { return cudaSuccess; }

#else

#include <cuda_runtime_api.h>

#define BEAMZ_GPU_PLATFORM_NAME "cuda"

#endif

#endif  // BEAMZ_GPU_RUNTIME_H_
