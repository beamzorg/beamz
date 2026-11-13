"""Copy-only constexpr single-axis NoIO cohorts on the exact9f6a observer parent."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT_HASHES = {
    "asymmetric_native.cuh": "f0657105ea5a57219d48a7a067d1ef53f6246676579e722ffd0670a14d3d8f49",
    "native_extension.cc": "8f122cdbb850374f44c8e59c3282274b744ebff0ef5a907540a245a53a730bcf",
    "native_moving.h": "3932dc6c86898d1090e9d9efba2a176ad9300a8f63cefe6430dfa0901dc7345b",
    "native_moving.cc": "44e47b882864f2338764592a5911e2fe9f37ad9c49862dc62ed27d93e099bfd1",
    "asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
}
HOST_FUNCTION = r"""
void BeamzCompactNativeMovingNoIOAxes(BeamzNativeMovingPlan&plan,bool enabled){
 using namespace beamz::cuda::moving;const size_t ranks=plan.bulk_counts.size();
 if(plan.rank_offsets.size()!=ranks+1||plan.uniform_noio_counts.size()!=ranks)throw std::invalid_argument("NoIO axis rank metadata differs");
 plan.uniform_noio_axis_counts.assign(ranks*3,0);
 if(plan.policy_offsets.empty())return;
 if(plan.policy_offsets.size()!=ranks*7)throw std::invalid_argument("NoIO axis policy metadata differs");
 const auto category=[&](int tile){const int bits=plan.tile_flags[tile]&14;return bits==2?0:bits==4?1:bits==8?2:3;};
 for(size_t rank=0;rank<ranks;++rank){
  const int rb=plan.rank_offsets[rank],re=plan.rank_offsets[rank+1],first=plan.policy_offsets[rank*7+1],count=plan.uniform_noio_counts[rank];
  if(count<0||rb>first||first>plan.policy_offsets[rank*7+2]||count>plan.policy_offsets[rank*7+2]-first||re>int(plan.tile_ids.size()))throw std::invalid_argument("NoIO axis prefix outside policy");
  const int last=first+count;
  for(int i=first;i<last;++i){const int tile=plan.tile_ids[i];
   if(tile<0||size_t(tile)>=plan.tile_flags.size()||Policy(plan.tile_flags[tile])!=1||!(plan.tile_flags[tile]&64))throw std::invalid_argument("NoIO axis prefix lacks immutable certificate");
  }
  if(enabled){std::stable_sort(plan.tile_ids.begin()+first,plan.tile_ids.begin()+last,[&](int a,int b){return category(a)<category(b);});
   for(int i=first;i<last;++i){const int kind=category(plan.tile_ids[i]);if(kind<3)++plan.uniform_noio_axis_counts[rank*3+kind];}
  }
  for(int i=rb;i<re;++i)plan.slots[plan.tile_ids[i]]=i-rb;
 }
}
"""
AXIS_LAUNCH = """template<int Depth,int Axes>
inline cudaError_t LaunchNoIOAxis(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count){
 if(!count)return cudaSuccess;
 const auto kernel=AsymmetricNoIOSingleAxis1024<Depth,Axes>;
 auto status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes);if(status!=cudaSuccess)return status;
 AsymmetricNoIOSingleAxis1024<Depth,Axes><<<count,1024,SharedBytes,stream>>>(p,ids);return cudaGetLastError();
}
"""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one exact NoIO-axis anchor: {old!r}")
    return text.replace(old, new, 1)


def transformed(parent):
    for name, expected in PARENT_HASHES.items():
        if digest(parent[name]) != expected:
            raise ValueError(f"Exact9f6a parent required: {name}")
    files = {name: data.decode() for name, data in parent.items()}
    c = files["asymmetric_native.cuh"]
    c = once(
        c,
        "bool UniformE=false> struct Environment",
        "bool UniformE=false,int ActiveAxes=-1> struct Environment",
    )
    c = once(
        c,
        "  if(!(flags&(2u<<axis)))return false;",
        "  if constexpr(ActiveAxes>=0){if(!(ActiveAxes&(1u<<axis)))return false;}\n  else if(!(flags&(2u<<axis)))return false;",
    )
    c = once(
        c,
        "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE>\n__device__",
        "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__",
    )
    c = once(
        c,
        " Environment<Layout,Fast,IsotropicH,Policy,UniformE>env{p,flags};",
        " Environment<Layout,Fast,IsotropicH,Policy,UniformE,ActiveAxes>env{p,flags};",
    )
    c = once(
        c,
        "inline bool MakeSource(",
        """template<int Depth,int Axes>
__global__ __launch_bounds__(1024,1) void AsymmetricNoIOSingleAxis1024(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 UniformBody<Depth,1,false,true,1,true,Axes>(p,tiles);
}

inline bool MakeSource(""",
    )
    anchor = (
        "template<int Depth,int Layout,bool IsotropicH>\ninline cudaError_t LaunchNoIO"
    )
    c = once(c, anchor, AXIS_LAUNCH + anchor)
    c = once(
        c,
        "int count,int uniform_count){\n if(uniform_count<0||uniform_count>count)return cudaErrorInvalidValue;\n if constexpr(Layout==1&&IsotropicH){\n  // Existing policy",
        "int count,int uniform_count,const int32_t*axis_counts){\n if(uniform_count<0||uniform_count>count||!axis_counts)return cudaErrorInvalidValue;\n int single=0;for(int axis=0;axis<3;++axis){if(axis_counts[axis]<0||axis_counts[axis]>uniform_count-single)return cudaErrorInvalidValue;single+=axis_counts[axis];}\n if constexpr(Layout==1&&IsotropicH){\n  // Existing policy",
    )
    anchor = "  const auto kernel=AsymmetricNoIOUniform1024<Depth>;\n"
    insert = """  int offset=0;
  status=LaunchNoIOAxis<Depth,1>(stream,p,ids+offset,axis_counts[0]);if(status!=cudaSuccess)return status;offset+=axis_counts[0];
  status=LaunchNoIOAxis<Depth,2>(stream,p,ids+offset,axis_counts[1]);if(status!=cudaSuccess)return status;offset+=axis_counts[1];
  status=LaunchNoIOAxis<Depth,4>(stream,p,ids+offset,axis_counts[2]);if(status!=cudaSuccess)return status;offset+=axis_counts[2];
  if(offset==uniform_count)return cudaSuccess;
"""
    c = once(c, anchor, insert + anchor)
    c = once(
        c,
        "AsymmetricNoIOUniform1024<Depth><<<uniform_count,1024,SharedBytes,stream>>>(p,ids);",
        "AsymmetricNoIOUniform1024<Depth><<<uniform_count-offset,1024,SharedBytes,stream>>>(p,ids+offset);",
    )
    c = once(
        c,
        "count,w.uniform_noio_counts_host[rank]);break;",
        "count,w.uniform_noio_counts_host[rank],w.uniform_noio_axis_counts_host+rank*3);break;",
    )
    c = once(
        c,
        "if(!w.uniform_noio_counts_host||!w.uniform_observer_counts_host)",
        "if(!w.uniform_noio_counts_host||!w.uniform_observer_counts_host||!w.uniform_noio_axis_counts_host)",
    )
    files["asymmetric_native.cuh"] = c
    h = files["native_moving.h"]
    h = once(
        h,
        "  const int32_t* uniform_observer_counts_host=nullptr;",
        "  const int32_t* uniform_noio_axis_counts_host=nullptr; // [rank,3] Z/Y/X certified subranges\n  const int32_t* uniform_observer_counts_host=nullptr;",
    )
    h = once(
        h,
        "uniform_noio_counts,uniform_observer_counts;",
        "uniform_noio_counts,uniform_observer_counts,uniform_noio_axis_counts;",
    )
    h = once(
        h,
        "void BeamzCompactNativeMovingUniformObservers(BeamzNativeMovingPlan&plan,bool enabled);",
        "void BeamzCompactNativeMovingUniformObservers(BeamzNativeMovingPlan&plan,bool enabled);\nvoid BeamzCompactNativeMovingNoIOAxes(BeamzNativeMovingPlan&plan,bool enabled);",
    )
    files["native_moving.h"] = h
    files["native_moving.cc"] += HOST_FUNCTION
    e = files["native_extension.cc"]
    e = once(
        e,
        "    BeamzCompactNativeMovingUniformObservers(moving_plan_,compact_enabled);",
        "    BeamzCompactNativeMovingUniformObservers(moving_plan_,compact_enabled);\n    BeamzCompactNativeMovingNoIOAxes(moving_plan_,compact_enabled);",
    )
    e = once(
        e,
        "    moving_workspace_.uniform_observer_counts_host=",
        "    moving_workspace_.uniform_noio_axis_counts_host=moving_plan_.uniform_noio_axis_counts.data();\n    moving_workspace_.uniform_observer_counts_host=",
    )
    e = once(
        e,
        "    append_bytes(moving_plan_.uniform_observer_counts.data()",
        "    append_bytes(moving_plan_.uniform_noio_axis_counts.data(),moving_plan_.uniform_noio_axis_counts.size()*4);\n    append_bytes(moving_plan_.uniform_observer_counts.data()",
    )
    e = once(
        e,
        "(moving_plan_.policy_offsets.size()+moving_plan_.uniform_noio_counts.size()+moving_plan_.uniform_observer_counts.size())*4",
        "(moving_plan_.policy_offsets.size()+moving_plan_.uniform_noio_counts.size()+moving_plan_.uniform_observer_counts.size()+moving_plan_.uniform_noio_axis_counts.size())*4",
    )
    e = once(
        e,
        "(p.policy_offsets.size()+2*p.bulk_counts.size())*4;",
        '(p.policy_offsets.size()+5*p.bulk_counts.size())*4;r["noio_axis_host_bytes"]=3*p.bulk_counts.size()*4;',
    )
    e = once(
        e,
        'result["moving_observation_bytes"]=moving_observation_bytes_;',
        """size_t axis_tiles[3]{};for(size_t i=0;i<moving_plan_.uniform_noio_axis_counts.size();++i)axis_tiles[i%3]+=moving_plan_.uniform_noio_axis_counts[i];
    result["moving_noio_axis_tiles_zyx"]=nb::make_tuple(axis_tiles[0],axis_tiles[1],axis_tiles[2]);result["moving_noio_axis_host_bytes"]=moving_plan_.uniform_noio_axis_counts.size()*4;
    result["moving_observation_bytes"]=moving_observation_bytes_;""",
    )
    e = once(
        e,
        '  module.attr("moving_observer_uniform_1024")=true;',
        '  module.attr("moving_noio_single_axis_dispatch")=true;\n  module.attr("moving_observer_uniform_1024")=true;',
    )
    files["native_extension.cc"] = e
    return {name: data.encode() for name, data in files.items()}


def patch_copy(source, output):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (
        output.exists()
        or source == output
        or source in output.parents
        or output in source.parents
    ):
        raise ValueError("Output must be a new disjoint research tree")
    if output == canonical or canonical in output.parents:
        raise ValueError("Canonical source must remain unchanged")
    parent = {name: (source / "src" / name).read_bytes() for name in PARENT_HASHES}
    files = transformed(parent)
    shutil.copytree(source, output)
    for name, data in files.items():
        (output / "src" / name).write_bytes(data)
    proof = {
        "parent_module_sha256": "9f6ade0cfb908fb1a71c883bfd2b106321a0401a2b8ec1bc87472047a7cb22ec",
        "parent_hashes": PARENT_HASHES,
        "result_hashes": {name: digest(data) for name, data in files.items()},
        "patcher_sha256": digest(Path(__file__).read_bytes()),
        "scope": "Only certified NoIO prefix subdivided by EXACT axis flag masks2/4/8; canonicalcell/F32math unaffected; genericmulti-axis andallotherpolicies unchanged",
    }
    output.with_name(output.name + "-axis-proof.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    patch_copy(args.source, args.output)


if __name__ == "__main__":
    main()
