"""Add a separate persistent P5 tile-step entry without changing old kernels.

The copied P5 body retains ordered arithmetic, five phase barriers and its
field-storage mapping. Scheduler hooks use cold volatile shared state. The
new host entry and workspace are private and opt-in; session integration lives
in the parent's separate native_extension.cc patcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

try:
    from scripts.patch_cuda_permanent_family_frontier import PARENT_HASHES, once
except ModuleNotFoundError:
    from patch_cuda_permanent_family_frontier import PARENT_HASHES, once


def sha(data):
    return hashlib.sha256(data).hexdigest()


def clone_body(original):
    start = original.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int BlockT=0,int SourceMask=-1,bool PlanePair=false>\n__device__ __forceinline__ void SourcePendingPairedBody")
    end = original.index("constexpr size_t SourcePendingSharedBytes=", start)
    body = original[start:end]
    body = once(body,
        "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int BlockT=0,int SourceMask=-1,bool PlanePair=false>\n__device__ __forceinline__ void SourcePendingPairedBody(const Parameters&p,const int32_t*tiles){",
        "template<int Depth>\n__device__ __forceinline__ void PersistentPairedBody(const Parameters&p){\n constexpr int Layout=1,Policy=5,BlockT=1024,SourceMask=20;\n constexpr bool Fast=false,IsotropicH=true,UniformE=false,PlanePair=false;")
    body = once(body, "const int tile=tiles[blockIdx.x]", "const int tile=C().tile")
    body = once(body, "const uint64_t epoch=uint64_t(*p.epoch)<<32;", "const uint64_t epoch=uint64_t(C().epoch)<<32;")
    body = once(body,
        " if(!threadIdx.x){const int owner=(rank&3)*p.frontier+p.slots[tile];p.owner_tags[owner]=epoch|uint32_t(tile);}\n __syncthreads();",
        " if(!threadIdx.x)Claim();\n __syncthreads();\n if(C().abort_local)return;")
    body = once(body,
        "   window[1]=min(Depth,min(oz+BZ-1,min(oy+BY-1,ox+BX-1)));\n  }\n  __syncthreads();if(window[0]>window[1])return;",
        "   window[1]=min(Depth,min(oz+BZ-1,min(oy+BY-1,ox+BX-1)));\n   // Existing window barrier broadcasts this wait before ANY base/late-start read.\n   if(window[0]<=window[1])WaitUpstream(window[0]);\n  }\n  __syncthreads();if(C().abort_local||window[0]>window[1])return;")
    body = once(body,
        "if(bits&&p.owner_tags[owner]!=(epoch|uint32_t(producer)))",
        "if(bits&&Acquire(p.owner_tags+owner)!=(epoch|uint32_t(producer)))")
    body = once(body,
        " Environment<Layout,Fast,IsotropicH,Policy,UniformE,SourceMask,PlanePair>env{p,flags};",
        " PersistentEnvironment env{p,flags};")
    body = once(body,
        "  __syncthreads(); // RLpair retires BOTH final planes' H XY readers.",
        "  // Every field/psi/observation/trail writer orders its own global stores.\n  __threadfence();\n  __syncthreads(); // RLpair retires BOTH final planes' H XY readers.")
    body = once(body,
        "  if(!threadIdx.x)*phase_step=n+1;\n  __syncthreads(); // RE: publish all current E and retire scratch readers for next H.",
        "  if(!threadIdx.x){*phase_step=n+1;PublishAndWait(n,window[1]);}\n  __syncthreads(); // RE: publish all current E and retire scratch readers for next H.\n  if(C().abort_local)return;")
    # Every original scalar call remains in its original per-point/component
    # order. No source, DFT, CPML round or field-lifetime expression is changed.
    parent_body = original[start:end]
    for phase in (0, 1):
        for c in range(3):
            token=f"beamz::cuda::trace_native::AdvanceNativeCell<{phase},{c},false>"
            if parent_body.count(token) != body.count(token):
                raise ValueError("Ordered canonical cell call changed")
    if parent_body.count("__syncthreads()") != body.count("__syncthreads()"):
        raise ValueError("Per-body CTA barrier count changed")
    if "tiles[blockIdx.x]" in body or "p.owner_tags[owner]=" in body:
        raise ValueError("Original nonpersistent tile/claim operation remains")
    return body


def patch_copy(source, output):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1]/"cuda"
    if (output.exists() or source==output or source in output.parents or output in source.parents or
            output==canonical or canonical in output.parents):
        raise ValueError("New disjoint noncanonical source required")
    for name, expected in PARENT_HASHES.items():
        if sha((source/"src"/name).read_bytes()) != expected:
            raise ValueError(f"Exact selected52865 CUDA/cell/workspace parent required: {name}")
    cmake_parent=(source/"CMakeLists.txt").read_bytes()
    if sha(cmake_parent)!="f18fb2d7182a0fae93278e5f1b59af9d69cc98f86fee02ccaf122794f8f0c981":
        raise ValueError("Exact selected52865 CMake compilation context required")
    cmake=once(cmake_parent.decode(),
        "  target_include_directories(beamz_cuda_moving PRIVATE src)",
        """  target_include_directories(beamz_cuda_moving PRIVATE src)
  # CUDA atomics live in CCCL; distro-split toolkits may install it outside opt/cuda/include.
  find_path(BEAMZ_CUDA_CCCL_INCLUDE cuda/atomic
    HINTS "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/include"
          "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/../../usr/include/cccl"
    REQUIRED)
  target_include_directories(beamz_cuda_moving PRIVATE "${BEAMZ_CUDA_CCCL_INCLUDE}")""")
    original=(source/"src/asymmetric_native.cuh").read_text()
    body=clone_body(original)
    here=Path(__file__).resolve().parent
    header=(here/"cuda_native_persistent.h").read_bytes()
    template=(here/"cuda_native_persistent.cuh").read_text()
    generated=once(template,"// BEAMZ_PERSISTENT_PAIRED_BODY_INSERT",body)
    shutil.copytree(source,output)
    (output/"src/native_persistent.h").write_bytes(header)
    (output/"src/native_persistent.cuh").write_text(generated)
    (output/"src/asymmetric_native.cuh").write_text(original+'\n#include "native_persistent.cuh"\n')
    (output/"CMakeLists.txt").write_text(cmake)
    proof={"parent_module":"52865a3ac8371e2390749953eb3af99746df108fe7bcb00fc4be8b35f4259b92",
           "parent_hashes":PARENT_HASHES,"old_consumer_prefix_byte_identical":True,
           "cloned_p5_body_sha256":sha(body.encode()),"persistent_header_sha256":sha(header),
           "persistent_consumer_sha256":sha(generated.encode()),"patcher_sha256":sha(Path(__file__).read_bytes()),
           "expected_moving_entries_including_reset":65,"persistent_depth_entries":2,"new_reset_entries":1,"source_presence_admission":[0,20],
           "cmake_parent_sha256":sha(cmake_parent),"cmake_result_sha256":sha(cmake.encode()),
           "CCCL_include_policy":"find_path cuda/atomic under toolkit/include or distro-split ../../usr/include/cccl, REQUIRED; moving target only",
           "storage":"original paired1024 P5, fourZ/thread, FP32, no psi cache, distinct A/B",
           "claim_protocol":"explicit old-holder+all7old-consumer completion; ready0 before release tag",
           "readiness":"all-thread threadfence before existingRLpair; releaseSTEP then waitnext beforeexistingRE",
           "mutable_psi_cache":"coherent L2 __ldcg after acquire dependency, never readonly __ldg",
           "abort":"Workspace timeout_ns ABI default60s; host may select5s; progress watchdog/global abort/uniform shared CTA exit",
           "gpu_executed":False,"requires":["CPU FIFO/RAW/WAR/epoch/window/watchdog proof","zero-local actual compilation","all30 exact first/replay","sanitizers","complete paired benchmarks"]}
    (output.parent/"persistent-p5-proof.json").write_text(json.dumps(proof,indent=2)+"\n")
    return proof


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args();print(json.dumps(patch_copy(a.source,a.output)))


if __name__=="__main__":main()
