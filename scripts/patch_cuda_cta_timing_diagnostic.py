"""DIAGNOSTIC ONLY: last-macro per-tile globaltimer/start-SMID intervals.

This copy adds timerstores and one converged terminal CTA barrier to each body
exit. It changes resources and timing and MUST NOT be a performance candidate.
The iteration bodies and ordered physics expressions remain byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.patch_cuda_shared_profiles import PARENT


TIMERS = r'''
// DIAGNOSTIC ONLY. %globaltimer is sampled by thread0; SMID is start attribution.
__device__ __forceinline__ void BeginTileInterval(const Parameters&p,int tile){
 if(!threadIdx.x){
  uint64_t now;uint32_t sm;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now) :: "memory");
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm) :: "memory");
  p.diag_start_ns[tile]=now;p.diag_smid[tile]=static_cast<int32_t>(sm);
 }
}
__device__ __forceinline__ void EndTileInterval(const Parameters&p,const int32_t*tiles){
 // All warps retire physics and trail writes before the terminal sample.
 __syncthreads();
 if(!threadIdx.x){
  // This opaque late reload must not extend the entry tile ID through solver
  // registers. Both diagnostic pointer and index are consumed only here.
  const int tile=static_cast<const volatile int32_t*>(tiles)[blockIdx.x];
  uint64_t now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now) :: "memory");
  p.diag_end_ns[tile]=now;
 }
}
'''

DOWNLOAD = r'''
  nb::dict MovingTileDiagnostics(){
#if !BEAMZ_CUDA_NATIVE_MOVING
    throw std::invalid_argument("moving timing diagnostic support not compiled");
#else
    auto guard=Guard();RequireOpen();
    if(!moving_ready_)throw std::invalid_argument("run a moving invocation before diagnostic download");
    const size_t count=moving_plan_.tile_flags.size();
    std::vector<uint64_t>start(count),end(count);std::vector<int32_t>smid(count);
    Check(cudaMemcpyAsync(start.data(),moving_workspace_.diag_start_ns,count*8,cudaMemcpyDeviceToHost,stream_.value),"download tile start diagnostics");
    Check(cudaMemcpyAsync(end.data(),moving_workspace_.diag_end_ns,count*8,cudaMemcpyDeviceToHost,stream_.value),"download tile end diagnostics");
    Check(cudaMemcpyAsync(smid.data(),moving_workspace_.diag_smid,count*4,cudaMemcpyDeviceToHost,stream_.value),"download tile SM diagnostics");
    Check(cudaStreamSynchronize(stream_.value),"finish tile diagnostic download");
    int device=0,sms=0;Check(cudaGetDevice(&device),"diagnostic current device");
    Check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device),"diagnostic SM count");
    nb::dict r;r["diagnostic_only"]=true;
    r["start_ns"]=CopyHostArray(numpy_,nb::make_tuple(count),"uint64",start.data(),count*8);
    r["end_ns"]=CopyHostArray(numpy_,nb::make_tuple(count),"uint64",end.data(),count*8);
    r["smid"]=CopyHostArray(numpy_,nb::make_tuple(count),"int32",smid.data(),count*4);
    r["tile_flags"]=CopyHostArray(numpy_,nb::make_tuple(count),"uint8",moving_plan_.tile_flags.data(),count);
    r["tile_ids"]=CopyHostArray(numpy_,nb::make_tuple(moving_plan_.tile_ids.size()),"int32",moving_plan_.tile_ids.data(),moving_plan_.tile_ids.size()*4);
    r["slots"]=CopyHostArray(numpy_,nb::make_tuple(moving_plan_.slots.size()),"int32",moving_plan_.slots.data(),moving_plan_.slots.size()*4);
    r["rank_offsets"]=CopyHostArray(numpy_,nb::make_tuple(moving_plan_.rank_offsets.size()),"int32",moving_plan_.rank_offsets.data(),moving_plan_.rank_offsets.size()*4);
    r["counts_zyx"]=nb::make_tuple(moving_plan_.counts[0],moving_plan_.counts[1],moving_plan_.counts[2]);
    r["physical_SM_count"]=sms;r["epochs"]=moving_plan_.epochs;r["max_frontier"]=moving_plan_.max_frontier;r["depth"]=moving_plan_.depth;
    r["scope"]="last complete moving macro; thread0entry toterminalCTAbarrier; emptytilesincluded; excludesDFT/tails/clocks; startSMID only";
    return r;
#endif
  }

'''


def sha(value):
    return hashlib.sha256(value).hexdigest()


def transformed(files):
    for name, expected in PARENT.items():
        if sha(files[name]) != expected:
            raise ValueError("Exact selected52865 diagnostic parent required:" + name)
    result, edits = dict(files), {n: [] for n in PARENT}

    def change(name, before, after, count=1):
        text = result[name].decode()
        if text.count(before) != count:
            raise ValueError(f"Expected{count} CTA diagnostic anchors in{name}:{before!r}")
        result[name] = text.replace(before, after).encode()
        edits[name].append((before, after, count))

    name = "asymmetric_native.cuh"
    change(name, " int nz,ny,nx,frontier,macro_step;int physical_n[3];float h_source;float common_e[3];", " int nz,ny,nx,frontier,macro_step;int physical_n[3];float h_source;float common_e[3];\n uint64_t*diag_start_ns;uint64_t*diag_end_ns;int32_t*diag_smid;")
    change(name, "__device__ __forceinline__ int WrapZ(int z)", TIMERS + "\n__device__ __forceinline__ int WrapZ(int z)")
    tile = " const int tile=tiles[blockIdx.x],bz=tile/(p.nx*p.ny),by=(tile/p.nx)%p.ny,bx=tile%p.nx;"
    change(name, tile, tile + "\n BeginTileInterval(p,tile);", count=2)
    change(name, "if(window[0]>window[1])return;", "if(window[0]>window[1]){EndTileInterval(p,tiles);return;}", count=2)
    # The two bodies end just before their next top-level template/constant.
    change(name, "\n}\n\ntemplate<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__", "\n EndTileInterval(p,tiles);\n}\n\ntemplate<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__")
    change(name, "\n}\n\n\nconstexpr size_t SourcePendingSharedBytes", "\n EndTileInterval(p,tiles);\n}\n\n\nconstexpr size_t SourcePendingSharedBytes")
    change(name, "!w.trails||!w.owner_tags||!w.epoch||!w.bad)return false;", "!w.trails||!w.owner_tags||!w.epoch||!w.bad||!w.diag_start_ns||!w.diag_end_ns||!w.diag_smid)return false;")
    change(name, " out->observation=w.observation;out->slots=w.slots_device;", " out->diag_start_ns=w.diag_start_ns;out->diag_end_ns=w.diag_end_ns;out->diag_smid=w.diag_smid;\n out->observation=w.observation;out->slots=w.slots_device;")
    name = "native_moving.h"
    change(name, "  int32_t* bad=nullptr;", "  uint64_t*diag_start_ns=nullptr;uint64_t*diag_end_ns=nullptr;int32_t*diag_smid=nullptr;\n  int32_t* bad=nullptr;")
    name = "native_extension.cc"
    change(name, 'r["extra_gpu_bytes"]=p.metadata_bytes+p.trail_floats*4+observation_bytes;', 'r["diagnostic_tile_bytes"]=p.tile_flags.size()*20;r["extra_gpu_bytes"]=p.metadata_bytes+p.trail_floats*4+observation_bytes+p.tile_flags.size()*20;')
    change(name, "moving_extra_bytes_=moving_plan_.metadata_bytes+moving_plan_.trail_floats*sizeof(float)+moving_observation_bytes_;", "moving_extra_bytes_=moving_plan_.metadata_bytes+moving_plan_.trail_floats*sizeof(float)+moving_observation_bytes_+moving_plan_.tile_flags.size()*20;")
    change(name, "    moving_workspace_.trails=static_cast<float*>(upload(nullptr,moving_plan_.trail_floats*sizeof(float)));", "    moving_workspace_.diag_start_ns=static_cast<uint64_t*>(upload(nullptr,moving_plan_.tile_flags.size()*8,true));\n    moving_workspace_.diag_end_ns=static_cast<uint64_t*>(upload(nullptr,moving_plan_.tile_flags.size()*8,true));\n    moving_workspace_.diag_smid=static_cast<int32_t*>(upload(nullptr,moving_plan_.tile_flags.size()*4,true));\n    moving_workspace_.trails=static_cast<float*>(upload(nullptr,moving_plan_.trail_floats*sizeof(float)));")
    change(name, '      Check(cudaMemsetAsync(moving_workspace_.bad,0,sizeof(int32_t),stream_.value),"clear moving invocation diagnostic");', '      Check(cudaMemsetAsync(moving_workspace_.diag_start_ns,0,moving_plan_.tile_flags.size()*8,stream_.value),"clear tile start diagnostic");\n      Check(cudaMemsetAsync(moving_workspace_.diag_end_ns,0,moving_plan_.tile_flags.size()*8,stream_.value),"clear tile end diagnostic");\n      Check(cudaMemsetAsync(moving_workspace_.diag_smid,0xff,moving_plan_.tile_flags.size()*4,stream_.value),"clear tile SM diagnostic");\n      Check(cudaMemsetAsync(moving_workspace_.bad,0,sizeof(int32_t),stream_.value),"clear moving invocation diagnostic");')
    change(name, '"native-asymmetric-moving-v1"', '"native-asymmetric-moving-cta-diagnostic-v1"')
    change(name, "        append_pointer(moving_workspace_.trails);", "        append_pointer(moving_workspace_.diag_start_ns);append_pointer(moving_workspace_.diag_end_ns);append_pointer(moving_workspace_.diag_smid);\n        append_pointer(moving_workspace_.trails);")
    change(name, "    append_bytes(&moving_workspace_.source_presence_mask,sizeof(uint32_t));", "    const bool timing_diagnostic_only=true;append_bytes(&timing_diagnostic_only,sizeof(bool));\n    append_bytes(&moving_workspace_.source_presence_mask,sizeof(uint32_t));")
    reset = '    UpdateClocks(0);\n    Check(cudaStreamSynchronize(stream_.value), "finish native reset");'
    reset_clear = '''    UpdateClocks(0);
#if BEAMZ_CUDA_NATIVE_MOVING
    if(moving_ready_){
      Check(cudaMemsetAsync(moving_workspace_.diag_start_ns,0,moving_plan_.tile_flags.size()*8,stream_.value),"reset tile start diagnostic");
      Check(cudaMemsetAsync(moving_workspace_.diag_end_ns,0,moving_plan_.tile_flags.size()*8,stream_.value),"reset tile end diagnostic");
      Check(cudaMemsetAsync(moving_workspace_.diag_smid,0xff,moving_plan_.tile_flags.size()*4,stream_.value),"reset tile SM diagnostic");
    }
#endif
    Check(cudaStreamSynchronize(stream_.value), "finish native reset");'''
    change(name, reset, reset_clear)
    change(name, "  nb::dict Clocks() {", DOWNLOAD + "  nb::dict Clocks() {")
    change(name, '    result["moving_source_presence_mask"]=moving_workspace_.source_presence_mask;', '    result["moving_cta_timing_diagnostic_only"]=true;result["diagnostic_tile_bytes"]=moving_plan_.tile_flags.size()*20;\n    result["moving_source_presence_mask"]=moving_workspace_.source_presence_mask;')
    change(name, 'NB_MODULE(_cuda_native, module) {', 'NB_MODULE(_cuda_native, module) {\n  module.attr("moving_cta_timing_diagnostic_only")=true;')
    change(name, '      .def("clocks", &NativeSession::Clocks)', '      .def("moving_tile_diagnostics",&NativeSession::MovingTileDiagnostics)\n      .def("clocks", &NativeSession::Clocks)')
    for name, replacements in edits.items():
        text = result[name].decode()
        for before, after, count in reversed(replacements):
            if text.count(after) != count:
                raise ValueError("CTA diagnostic source inverse is ambiguous")
            text = text.replace(after, before)
        if text.encode() != files[name]:
            raise ValueError("CTA diagnostic source inverse differs")
    return result, {"diagnostic_only": True, "must_not_be_performance_candidate": True,
                    "exact_source_inverse": True, "iteration_bodies_byte_identical": True,
                    "new_terminal_barrier_per_taken_body_exit": 1,
                    "start_end_smid_bytes_per_tile": 20,
                    "last_macro_overwrites_previous_macros": True,
                    "per_invocation_arrays_clear_before_graph_execution": True,
                    "private_allocations_and_graph_pointer_bindings": True,
                    "expected_moving_kernels": 62,
                    "limitations": ["Added timerstores/parameter pointers/barriers perturb timing and compiled resources",
                                    "CTA interval presence is not SM instruction utilization or occupancy",
                                    "StartSMID does not detect preemption migration",
                                    "Lastmacro scope excludesDFT, ordinary tails andclock kernels"]}


def patch_copy(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if output.exists() or output == canonical or canonical in output.parents or source in output.parents or output in source.parents:
        raise ValueError("New disjoint private diagnostic source required")
    before = {name: (source / "src" / name).read_bytes() for name in PARENT}
    after, proof = transformed(before)
    shutil.copytree(source, output)
    for name, value in after.items():
        (output / "src" / name).write_bytes(value)
    proof.update(schema="beamz.cta-timing-diagnostic.v1", parent_hashes=PARENT,
                 result_hashes={name: sha(v) for name, v in after.items()},
                 producer_sha256=sha(Path(__file__).read_bytes()), GPU_executed=False)
    (output.parent / "cta-timing-diagnostic-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output), indent=2))


if __name__ == "__main__":
    main()
