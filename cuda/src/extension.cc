#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "abi_layout.h"
#include "ffi_handler.h"

namespace nb = nanobind;
using namespace beamz::cuda::abi;

// CMake names the module _cuda or _hip. NB_MODULE pastes its first argument,
// so expand the definition through one extra macro level first.
#define BEAMZ_NB_MODULE(name, variable) NB_MODULE(name, variable)

#if defined(BEAMZ_PLATFORM_HIP)
constexpr const char* kGpuPlatform = "rocm";
#elif defined(BEAMZ_PLATFORM_CPU)
constexpr const char* kGpuPlatform = "cpu";
#else
constexpr const char* kGpuPlatform = "cuda";
#endif

BEAMZ_NB_MODULE(BEAMZ_EXTENSION_NAME, module) {
  module.attr("__version__") = kComponentVersion;
  module.attr("__gpu_platform__") = kGpuPlatform;
  module.attr("__abi_version__") = kAbiVersion;
  module.def("registrations", []() {
    nb::dict registrations;
    registrations[kStreamedTarget] =
        nb::capsule(reinterpret_cast<void*>(beamz_cuda_streamed));
#if !defined(BEAMZ_PLATFORM_CPU)
    // Simulated multi-device CPU runs gave wrong results that disappeared
    // when launches were serialised across XLA device threads; sharding is a
    // GPU feature here (see docs/superpowers/plans/2026-11-06-cpu-prelude.md).
    registrations[kShardedTarget] =
        nb::capsule(reinterpret_cast<void*>(beamz_cuda_sharded));
#endif
    registrations[kProgramTarget] =
        nb::capsule(reinterpret_cast<void*>(beamz_cuda_program));
#if !defined(BEAMZ_PLATFORM_CPU)
    // Whole-column two-step tiles need shared memory and barriers.
    registrations[kMstepTarget] =
        nb::capsule(reinterpret_cast<void *>(beamz_cuda_mstep));
#endif
    registrations[kAdjointTarget] =
        nb::capsule(reinterpret_cast<void*>(beamz_cuda_adjoint));
    registrations[kMaterialAdjointTarget] =
        nb::capsule(reinterpret_cast<void*>(beamz_cuda_material_adjoint));
    return registrations;
  });
}
