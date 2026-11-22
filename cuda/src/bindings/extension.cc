#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "abi_layout.h"
#include "ffi_handler.h"

namespace nb = nanobind;
using namespace beamz::cuda::abi;

void RegisterNativeCuda(nb::module_& module);

NB_MODULE(_cuda, module) {
  RegisterNativeCuda(module);
  module.attr("coherent_backend_version")=1;
  module.attr("compiled_adjoint_replay") = true;
  module.attr("compiled_adjoint_chunk") = true;
  module.attr("__version__") = kComponentVersion;
  module.attr("__abi_version__") = kAbiVersion;
  module.def("registrations", []() {
    nb::dict registrations;
    registrations[kStreamedTarget] =
        nb::capsule(reinterpret_cast<void*>(beamz_cuda_streamed));
    registrations[kShardedTarget] =
        nb::capsule(reinterpret_cast<void*>(beamz_cuda_sharded));
    registrations[kProgramTarget] =
        nb::capsule(reinterpret_cast<void*>(beamz_cuda_program));
    registrations[kMstepTarget] =
        nb::capsule(reinterpret_cast<void *>(beamz_cuda_mstep));
#if BEAMZ_CUDA_MSTEP_FULL_INTERMEDIATES
    // Additive capability contract: this implementation never dereferences
    // compact field scratch. Older callers/registrations remain unchanged.
    registrations[kMstepFullIntermediatesTarget] =
        nb::capsule(reinterpret_cast<void *>(beamz_cuda_mstep));
#endif
    registrations[kAdjointTarget] =
        nb::capsule(reinterpret_cast<void*>(beamz_cuda_adjoint));
    registrations[kMaterialAdjointTarget] =
        nb::capsule(reinterpret_cast<void*>(beamz_cuda_material_adjoint));
    return registrations;
  });
}
