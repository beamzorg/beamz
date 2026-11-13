"""Append a source-only private-Ey schedule to the frozen small Obs engine.

Original NoIO/bulk/observer consumers retain their four-barrier schedule. Only
byte-packed/lossless-isotropic programs with source presence20 use the new
source3/4/5 entries. No source coefficient values or floating operations change.
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

from scripts.patch_cuda_native_source_mask20 import PRESENCE_HELPER, once

PARENT_HASHES = {
    "src/asymmetric_native.cuh": "f0657105ea5a57219d48a7a067d1ef53f6246676579e722ffd0670a14d3d8f49",
    "src/native_extension.cc": "8f122cdbb850374f44c8e59c3282274b744ebff0ef5a907540a245a53a730bcf",
    "src/native_moving.h": "3932dc6c86898d1090e9d9efba2a176ad9300a8f63cefe6430dfa0901dc7345b",
    "src/native_moving.cc": "44e47b882864f2338764592a5911e2fe9f37ad9c49862dc62ed27d93e099bfd1",
    "src/asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
    "src/compact_full_edge.cuh": "62d4fa66106469e7a89c1848361ac72fab6aef9b01bb8cb09dca0fbcafa17da4",
    "src/update.cu": "ffe39d7d139dfe49d2124d32006f75db503177c90d0da730c9ce0a97c6982af0",
    "CMakeLists.txt": "925493680a0578cf69f22a355d4663cbc35ff5b998be03b3195e03f0f4421bd8",
}

LAUNCH = """template<int Depth,int Layout,bool IsotropicH,int Policy>
inline cudaError_t LaunchSourcePending(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count,uint32_t presence){
 if(!count)return cudaSuccess;
 if constexpr(Layout==1&&IsotropicH){
  if(BeamzNativeMovingSourceMask20Class(Layout,IsotropicH,presence)){
   const auto kernel=AsymmetricSourcePending20<Depth,Policy>;
   auto status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SourcePendingSharedBytes);if(status!=cudaSuccess)return status;
   AsymmetricSourcePending20<Depth,Policy><<<count,1024,SourcePendingSharedBytes,stream>>>(p,ids);
   return cudaGetLastError();
  }
 }
 return Launch<Depth,Layout,false,IsotropicH,Policy>(stream,p,ids,count);
}
"""

KERNEL = """constexpr size_t SourcePendingSharedBytes=(3ULL*PAD+2048)*sizeof(float)+8*sizeof(size_t)+3*sizeof(int)+3ULL*LowXPoints*sizeof(float);
static_assert(SourcePendingSharedBytes==58680,"Proved source-only small cache layout");
template<int Depth,int Policy>
__global__ __launch_bounds__(1024,1) void AsymmetricSourcePending20(
 const __grid_constant__ Parameters p,const int32_t*ids){
 static_assert(Policy>=3&&Policy<=5,"Only source-bearing policies");
 SourcePendingBody<Depth,1,false,true,Policy,false,1024,20>(p,ids);
}

"""

MOVING_BLOCK = """#if BEAMZ_CUDA_NATIVE_MOVING
#include "native_moving.h"
#include "compact_full_edge.cuh"
#include "asymmetric_native_cell_generated.cuh"
#include "asymmetric_native.cuh"
#endif"""

TU = """// Private moving consumer; no cross-TU device calls.
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
""" + MOVING_BLOCK + "\n"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def transformed(parent, body):
    for name, expected in PARENT_HASHES.items():
        if sha(parent[name]) != expected:
            raise ValueError(f"Exact frozen Obs9f6a parent required: {name}")
    files = {name: data.decode() for name, data in parent.items()}
    c = files["src/asymmetric_native.cuh"]
    c = once(c, "template<int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5),bool UniformE=false> struct Environment {",
             "template<int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5),bool UniformE=false,int SourceMask=-1> struct Environment {")
    c = once(c, "if constexpr(HasSources)if(flags&beamz::cuda::moving::kMayHaveSources){",
             "if constexpr(HasSources)if((SourceMask<0||(unsigned(SourceMask)&(1u<<field)))&&(flags&beamz::cuda::moving::kMayHaveSources)){")
    c = once(c, "inline bool MakeSource(", body + "\n" + KERNEL + "inline bool MakeSource(")
    anchor = " for(int field=0;field<6;++field)if(!MakeSource(program.source_groups[3+field],&out->sources[field]))return false;"
    c = once(c, anchor, anchor + "\n uint32_t actual_source_presence=0;for(int field=0;field<6;++field)if(out->sources[field].count>0)actual_source_presence|=1u<<field;\n if(actual_source_presence!=w.source_presence_mask)return false;")
    anchor = "template<int Depth,int Layout,bool IsotropicH>\ninline cudaError_t LaunchRanks"
    c = once(c, anchor, LAUNCH + anchor)
    for policy in (3,4,5):
        c = once(c, f"case {policy}:status=Launch<Depth,Layout,false,IsotropicH,{policy}>(branch,p,w.tile_ids_device+start,count);break;",
                 f"case {policy}:status=LaunchSourcePending<Depth,Layout,IsotropicH,{policy}>(branch,p,w.tile_ids_device+start,count,w.source_presence_mask);break;")
    c = once(c, "status=Launch<Depth,Layout,false,IsotropicH>(stream,p,w.tile_ids_device+start+bulk,count-bulk);",
             "status=LaunchSourcePending<Depth,Layout,IsotropicH,5>(stream,p,w.tile_ids_device+start+bulk,count-bulk,w.source_presence_mask);")
    files["src/asymmetric_native.cuh"] = c
    h = files["src/native_moving.h"]
    h = once(h, "#include <vector>", "#include <vector>\n#include <stdexcept>")
    h = once(h, "  uint32_t data_flags=0;", "  uint32_t data_flags=0;\n  uint32_t source_presence_mask=0;")
    h = once(h, "// CUDA consumer owns", PRESENCE_HELPER + "\n// CUDA consumer owns")
    files["src/native_moving.h"] = h
    e = files["src/native_extension.cc"]
    anchor='    Check(static_cast<cudaError_t>(BeamzValidateNativeProgram(validate.program)),"validate moving packet");'
    e=once(e,anchor,anchor+"\n    moving_workspace_.source_presence_mask=BeamzNativeMovingSourcePresenceMask(validate.program.source_groups,validate.program.source_group_count);")
    anchor="    append_bytes(&obs.key_count,sizeof(int));"
    e=once(e,anchor,"    append_bytes(&moving_workspace_.source_presence_mask,sizeof(uint32_t));\n"+anchor)
    anchor='    result["moving_uniform_noio_tiles"]=moving_uniform_noio_;'
    metadata="""    result["moving_source_presence_mask"]=moving_workspace_.source_presence_mask;
    const int source_layout=(moving_workspace_.data_flags&beamz::cuda::moving::kBytePackedE)?1:2;
    const bool source_class=BeamzNativeMovingSourceMask20Class(source_layout,bool(moving_workspace_.data_flags&beamz::cuda::moving::kLosslessIsotropicH),moving_workspace_.source_presence_mask);
    std::array<int64_t,3> source_tiles{};
    if(source_class)for(size_t rank=0;rank<moving_plan_.bulk_counts.size();++rank){
     if(!moving_plan_.policy_offsets.empty())for(int policy=3;policy<=5;++policy)source_tiles[policy-3]+=moving_plan_.policy_offsets[rank*7+policy+1]-moving_plan_.policy_offsets[rank*7+policy];
     else source_tiles[2]+=moving_plan_.rank_offsets[rank+1]-moving_plan_.rank_offsets[rank]-moving_plan_.bulk_counts[rank];
    }
    result["moving_source_mask20_policy_tiles"]=nb::make_tuple(source_tiles[0],source_tiles[1],source_tiles[2]);
    result["moving_source_mask20_dispatch_active"]=source_class&&(source_tiles[0]+source_tiles[1]+source_tiles[2]>0);
    result["moving_source_pending_shared_bytes"]=58680;result["moving_original_shared_bytes"]=51748;
"""
    e=once(e,anchor,metadata+anchor)
    anchor='  module.attr("moving_noio_uniform_1024")=true;'
    e=once(e,anchor,'  module.attr("moving_source_pending_mask20_1024")=true;\n  module.attr("moving_source_mask20_policies")=nb::make_tuple(3,4,5);\n'+anchor)
    files["src/native_extension.cc"]=e
    files["src/update.cu"]=once(files["src/update.cu"],MOVING_BLOCK,"// Moving kernels are isolated in native_moving_kernel.cu.")
    files["src/native_moving_kernel.cu"]=TU
    cm=files["CMakeLists.txt"]
    anchor="if(BEAMZ_CUDA_BUILD_NATIVE)\n"
    target="""if(BEAMZ_CUDA_BUILD_NATIVE)
  add_library(beamz_cuda_moving OBJECT src/native_moving_kernel.cu)
  target_include_directories(beamz_cuda_moving PRIVATE src)
  target_compile_features(beamz_cuda_moving PRIVATE cxx_std_17)
  set_target_properties(beamz_cuda_moving PROPERTIES POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAMZ_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF)
  target_compile_options(beamz_cuda_moving PRIVATE $<TARGET_PROPERTY:beamz_cuda_core,COMPILE_OPTIONS>)
"""
    cm=once(cm,anchor,target)
    cm=once(cm,"target_link_libraries(_cuda_native PRIVATE beamz_cuda_core)","target_link_libraries(_cuda_native PRIVATE beamz_cuda_core beamz_cuda_moving)")
    cm+="""
# Copy the exact update.cu source settings only after all options are established.
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
    files["CMakeLists.txt"]=cm
    return {name:text.encode() for name,text in files.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    source,output=args.source.resolve(),args.output.resolve()
    canonical=Path(__file__).resolve().parents[1]/"cuda"
    if output.exists() or source==output or source in output.parents or output in source.parents or output==canonical or canonical in output.parents:
        parser.error("New disjoint noncanonical output required")
    bodyfile=Path(__file__).with_name("cuda_native_source_pending_body.cuh")
    body=bodyfile.read_text()
    parent={name:(source/name).read_bytes() for name in PARENT_HASHES}
    files=transformed(parent,body)
    shutil.copytree(source,output)
    for name,data in files.items():
        (output/name).write_bytes(data)
    proof={"parent_module_sha256":"9f6ade0cfb908fb1a71c883bfd2b106321a0401a2b8ec1bc87472047a7cb22ec","parent_hashes":PARENT_HASHES,
           "result_hashes":{n:sha(d) for n,d in files.items()},"body_sha256":sha(body.encode()),"patcher_sha256":sha(Path(__file__).read_bytes()),
           "scope":"Original52small4barrier consumers retained; onlySourceMask20byte/isotropic usesnew7barrier1024body. Samehostqueues/trails/Params/sourceFP, nonRDCmovingTU; GPU gates required."}
    (output.parent/"hybrid-source-proof.json").write_text(json.dumps(proof,indent=2)+"\n")
    print(json.dumps(proof))


if __name__=="__main__":
    main()
