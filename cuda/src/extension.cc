#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "abi_layout.h"
#include "ffi_handler.h"

namespace nb = nanobind;
using namespace beamz::cuda::abi;

NB_MODULE(_cuda, module) {
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
    registrations[kHopperTarget] =
        nb::capsule(reinterpret_cast<void*>(beamz_cuda_hopper));
    return registrations;
  });
}
