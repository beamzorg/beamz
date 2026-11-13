"""Isolate header-contained temporal kernels in a non-RDC CUDA translation unit.

Keep the ordinary CUDA core/native runtime in their existing RDC mode. Only the
moving consumer changes compilation mode, avoiding outlined shuffle convergence
calls and their live-state ABI spills. All shader/math/admission bytes remain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT = "12486a02201fcd072e1f1aabd5e66b20e1e8049176f2043afa878f9cd5ace12d"
BLOCK = """#if BEAMZ_CUDA_NATIVE_MOVING
#include "native_moving.h"
#include "compact_full_edge.cuh"
#include "asymmetric_native_cell_generated.cuh"
#include "asymmetric_native.cuh"
#endif
"""
TU = (
    """// Private temporal consumer: executable device code, no cross-TU device calls.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include "abi_layout.h"
#include "kernels.h"
#include "launch.h"
#include "yee_primitives.cuh"
#include "mstep_index.h"
"""
    + BLOCK
)
TARGET = """  # Moving device helpers are header-contained. Keep ordinary core RDC;
  # compile only this consumer in whole-program mode to avoid shuffle ABI calls.
  add_library(beamz_cuda_moving OBJECT src/native_moving_kernel.cu)
  target_include_directories(beamz_cuda_moving PRIVATE src)
  target_compile_features(beamz_cuda_moving PRIVATE cxx_std_17)
  set_target_properties(beamz_cuda_moving PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF)
  set_property(TARGET beamz_cuda_moving PROPERTY CUDA_ARCHITECTURES "${BEAMZ_CUDA_ARCHITECTURES}")
  target_compile_definitions(beamz_cuda_moving PRIVATE
    $<TARGET_PROPERTY:beamz_cuda_core,COMPILE_DEFINITIONS>)
  target_compile_options(beamz_cuda_moving PRIVATE
    $<TARGET_PROPERTY:beamz_cuda_core,COMPILE_OPTIONS>)
"""
SOURCE_PROPERTIES = """
# Native research definitions/options are attached to update.cu as SOURCE
# properties by the existing build. Copy them only after that setup is final.
if(TARGET beamz_cuda_moving)
  get_source_file_property(_moving_source_defs src/update.cu COMPILE_DEFINITIONS)
  if(_moving_source_defs)
    target_compile_definitions(beamz_cuda_moving PRIVATE ${_moving_source_defs})
  endif()
  get_source_file_property(_moving_source_options src/update.cu COMPILE_OPTIONS)
  if(_moving_source_options)
    target_compile_options(beamz_cuda_moving PRIVATE ${_moving_source_options})
  endif()
endif()
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one private moving-TU anchor: {old!r}")
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    src, out = args.source.resolve(), args.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (
        out.exists()
        or out == canonical
        or src == out
        or src in out.parents
        or out in src.parents
    ):
        parser.error("New disjoint private source required")
    consumer = (src / "src/asymmetric_native.cuh").read_bytes()
    if sha(consumer) != PARENT:
        raise ValueError("Exact partial-Ey bulk-v3 consumer required")
    update = (src / "src/update.cu").read_text()
    cmake = (src / "CMakeLists.txt").read_text()
    changed_update = once(
        update,
        BLOCK,
        "// Moving consumer compiled separately in native_moving_kernel.cu.\n",
    )
    changed_cmake = once(
        cmake,
        "if(BEAMZ_CUDA_BUILD_NATIVE)\n  nanobind_add_module(",
        "if(BEAMZ_CUDA_BUILD_NATIVE)\n" + TARGET + "  nanobind_add_module(",
    )
    changed_cmake = once(
        changed_cmake,
        "target_link_libraries(_cuda_native PRIVATE beamz_cuda_core)",
        "target_link_libraries(_cuda_native PRIVATE beamz_cuda_core beamz_cuda_moving)",
    )
    changed_cmake += SOURCE_PROPERTIES
    shutil.copytree(src, out)
    (out / "src/update.cu").write_text(changed_update)
    (out / "src/native_moving_kernel.cu").write_text(TU)
    (out / "CMakeLists.txt").write_text(changed_cmake)
    assert (out / "src/asymmetric_native.cuh").read_bytes() == consumer
    proof = {
        "consumer_sha256_unchanged": PARENT,
        "core_update_parent_sha256": sha(update.encode()),
        "core_update_sha256": sha(changed_update.encode()),
        "moving_TU_sha256": sha(TU.encode()),
        "cmake_parent_sha256": sha(cmake.encode()),
        "cmake_sha256": sha(changed_cmake.encode()),
        "ordinary_core_RDC_unchanged": True,
        "moving_only_RDC": False,
        "scope": "Compilation boundary only; all temporal point/schedule/host-admission bytes unchanged. Full linking/resources/numerical gates required",
    }
    (out.parent / "separate-moving-source-proof.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
