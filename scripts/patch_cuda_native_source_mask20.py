"""Copy-only source-presence specialization on the private partial-Ey engine.

Only globally empty source groups are erased. Source coefficients, products,
rounding, clocks, CPML and observations remain unchanged. The presence fact is
captured from validated private descriptors and checked again by the launcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT_HASHES = {
    "src/asymmetric_native.cuh": "12486a02201fcd072e1f1aabd5e66b20e1e8049176f2043afa878f9cd5ace12d",
    "src/native_extension.cc": "904a52c95c04818bd43a724e57896d9e6053150cb74fee0bf90eac835c2b3170",
    "src/native_moving.h": "ab7d3505fc036b95628fbf0d18daa85eaccdcde3178f9777dee2aa5a878eae10",
    "src/native_moving.cc": "2f75cb1ec4463c288e6f7e4a28acec54dc9215bc596f9a45b35c9f69d46c3f2c",
    "src/asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
    "src/compact_full_edge.cuh": "62d4fa66106469e7a89c1848361ac72fab6aef9b01bb8cb09dca0fbcafa17da4",
    "src/native_moving_kernel.cu": "029de2a5357c219a5ad00e36d800f8ad9d94a681eca906f7f9e2581833a9890d",
    "CMakeLists.txt": "6da194d16212c10b1e2819c70e891f0f8fa73622b7533b003cd20b76a7018a5c",
}

PRESENCE_HELPER = """// Counts are immutable validated descriptor facts, not coefficient values.
inline uint32_t BeamzNativeMovingSourcePresenceMask(const BeamzSourceGroupLaunch*groups,int count){
 if(!groups||count!=9)throw std::invalid_argument("moving source presence requires nine groups");
 uint32_t mask=0;
 for(int field=0;field<6;++field){const auto sources=groups[3+field].coefficients.dims[0];
  if(sources<0)throw std::invalid_argument("negative source count");
  if(sources>0)mask|=uint32_t(1u<<field);
 }
 return mask;
}
inline constexpr bool BeamzNativeMovingSourceMask20Class(int layout,bool isotropic,uint32_t presence){
 return layout==1&&isotropic&&presence==20u;
}
"""

KERNEL = """template<int Depth,int Policy>
__global__ __launch_bounds__(512,1) void AsymmetricSourceMask20T512(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 static_assert(Policy>=3&&Policy<=5,"Source presence applies only to admitted IO policies");
 UniformBody<Depth,1,false,true,Policy,false,512,20>(p,tiles);
}

"""

LAUNCH = """template<int Depth,int Layout,bool IsotropicH,int Policy>
inline cudaError_t LaunchSource(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count,uint32_t source_presence){
 if(!count)return cudaSuccess;
 if constexpr(Layout==1&&IsotropicH){
  if(BeamzNativeMovingSourceMask20Class(Layout,IsotropicH,source_presence)){
   const auto kernel=AsymmetricSourceMask20T512<Depth,Policy>;
   auto status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes);if(status!=cudaSuccess)return status;
   AsymmetricSourceMask20T512<Depth,Policy><<<count,512,SharedBytes,stream>>>(p,ids);
   return cudaGetLastError();
  }
 }
 return Launch<Depth,Layout,false,IsotropicH,Policy>(stream,p,ids,count);
}
"""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one exact source-mask20 anchor: {old!r}")
    return text.replace(old, new, 1)


def transformed(parent):
    for name, expected in PARENT_HASHES.items():
        if digest(parent[name]) != expected:
            raise ValueError(f"Exact a0da partial-Ey separate-TU v7 parent required: {name}")
    files = {name: data.decode() for name, data in parent.items()}
    c = files["src/asymmetric_native.cuh"]
    c = once(c, "template<int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5),bool UniformE=false> struct Environment {",
             "template<int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5),bool UniformE=false,int SourceMask=-1> struct Environment {")
    c = once(c, "if constexpr(HasSources)if(flags&beamz::cuda::moving::kMayHaveSources){",
             "if constexpr(HasSources)if((SourceMask<0||(unsigned(SourceMask)&(1u<<field)))&&(flags&beamz::cuda::moving::kMayHaveSources)){")
    c = once(c, "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int BlockT=0>",
             "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int BlockT=0,int SourceMask=-1>")
    c = once(c, "Environment<Layout,Fast,IsotropicH,Policy,UniformE>env{p,flags};",
             "Environment<Layout,Fast,IsotropicH,Policy,UniformE,SourceMask>env{p,flags};")
    c = once(c, "inline bool MakeSource(", KERNEL + "inline bool MakeSource(")
    old = " for(int field=0;field<6;++field)if(!MakeSource(program.source_groups[3+field],&out->sources[field]))return false;"
    c = once(c, old, old + "\n uint32_t actual_source_presence=0;for(int field=0;field<6;++field)if(out->sources[field].count>0)actual_source_presence|=1u<<field;\n if(actual_source_presence!=w.source_presence_mask)return false;")
    anchor = "template<int Depth,int Layout,bool IsotropicH>\ninline cudaError_t LaunchRanks"
    c = once(c, anchor, LAUNCH + anchor)
    for policy in (3, 4, 5):
        c = once(c, f"case {policy}:status=Launch<Depth,Layout,false,IsotropicH,{policy}>(branch,p,w.tile_ids_device+start,count);break;",
                 f"case {policy}:status=LaunchSource<Depth,Layout,IsotropicH,{policy}>(branch,p,w.tile_ids_device+start,count,w.source_presence_mask);break;")
    c = once(c, "status=Launch<Depth,Layout,false,IsotropicH>(stream,p,w.tile_ids_device+start+bulk,count-bulk);",
             "status=LaunchSource<Depth,Layout,IsotropicH,5>(stream,p,w.tile_ids_device+start+bulk,count-bulk,w.source_presence_mask);")
    files["src/asymmetric_native.cuh"] = c
    h = files["src/native_moving.h"]
    h = once(h, "#include <vector>", "#include <vector>\n#include <stdexcept>")
    h = once(h, "  uint32_t data_flags=0;", "  uint32_t data_flags=0;\n  uint32_t source_presence_mask=0; // six validated immutable post-source counts (HOST)")
    h = once(h, "// CUDA consumer owns", PRESENCE_HELPER + "\n// CUDA consumer owns")
    files["src/native_moving.h"] = h
    e = files["src/native_extension.cc"]
    anchor = '    Check(static_cast<cudaError_t>(BeamzValidateNativeProgram(validate.program)),"validate moving packet");'
    e = once(e, anchor, anchor + "\n    moving_workspace_.source_presence_mask=BeamzNativeMovingSourcePresenceMask(validate.program.source_groups,validate.program.source_group_count);")
    anchor = "    append_bytes(&obs.key_count,sizeof(int));"
    e = once(e, anchor, "    append_bytes(&moving_workspace_.source_presence_mask,sizeof(uint32_t));\n" + anchor)
    anchor = '    result["moving_uniform_noio_tiles"]=moving_uniform_noio_;'
    metadata = """    result["moving_source_presence_mask"]=moving_workspace_.source_presence_mask;
    const int source_layout=(moving_workspace_.data_flags&beamz::cuda::moving::kBytePackedE)?1:2;
    const bool source_class=BeamzNativeMovingSourceMask20Class(source_layout,bool(moving_workspace_.data_flags&beamz::cuda::moving::kLosslessIsotropicH),moving_workspace_.source_presence_mask);
    std::array<int64_t,3> source_policy_tiles{};
    if(source_class)for(size_t rank=0;rank<moving_plan_.bulk_counts.size();++rank){
      if(!moving_plan_.policy_offsets.empty())for(int policy=3;policy<=5;++policy)source_policy_tiles[policy-3]+=moving_plan_.policy_offsets[rank*7+policy+1]-moving_plan_.policy_offsets[rank*7+policy];
      else source_policy_tiles[2]+=moving_plan_.rank_offsets[rank+1]-moving_plan_.rank_offsets[rank]-moving_plan_.bulk_counts[rank];
    }
    result["moving_source_mask20_policy_tiles"]=nb::make_tuple(source_policy_tiles[0],source_policy_tiles[1],source_policy_tiles[2]);
    result["moving_source_mask20_dispatch_active"]=source_class&&(source_policy_tiles[0]+source_policy_tiles[1]+source_policy_tiles[2]>0);
"""
    e = once(e, anchor, metadata + anchor)
    anchor = '  module.attr("moving_noio_uniform_1024")=true;'
    e = once(e, anchor, '  module.attr("moving_source_mask20_512")=true;\n  module.attr("moving_source_mask20_policies")=nb::make_tuple(3,4,5);\n' + anchor)
    files["src/native_extension.cc"] = e
    return {name: text.encode() for name, text in files.items()}


def patch_copy(source, output):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents
            or output in source.parents or output == canonical or canonical in output.parents):
        raise ValueError("Output must be a new disjoint noncanonical source tree")
    parent = {name: (source / name).read_bytes() for name in PARENT_HASHES}
    files = transformed(parent)
    shutil.copytree(source, output)
    for name, data in files.items():
        (output / name).write_bytes(data)
    report = {"parent_module_sha256": "a0daf779081324e31c09538e961d639c5321310fff49acaee84f8346a53fd4ad",
              "parent_hashes": PARENT_HASHES,
              "result_hashes": {name: digest(data) for name, data in files.items()},
              "patcher_sha256": digest(Path(__file__).read_bytes()),
              "global_presence_fact_not_coefficient_values": True,
              "no_fact_bit_reuse": True,
              "scope": "Source policies3/4/5 mask20 byte/isotropic512; same queues/rank joins/FP/CPML/observations, private4byte fact/key only; GPU validation required"}
    (output.parent / "source-mask20-source-proof.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(patch_copy(args.source, args.output)))


if __name__ == "__main__":
    main()
