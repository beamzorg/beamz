"""Copy-only certified observer UniformE dispatch on exact compact7497 source."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT_HASHES = {
    "asymmetric_native.cuh": "a111536eba602a638aeec536ae58af5728f64af8bad0609c7b64e4f672369c62",
    "native_extension.cc": "0f812bcf1e1fa8fdb70e3f1e15b20dc921612d85e19f0682be091035f75e97f5",
    "native_moving.h": "657ac01e5bc3acf02133bafa601f5cf4ea279ad501a6abdddb61e322ea470f8b",
    "native_moving.cc": "1282f12e4dbd572e9889a8a85407436b468dd7df002314d95344d988a5b2971a",
    "asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
}

HOST_FUNCTION = r"""
// A second stable prefix, confined to policy2; NoIO queues remain unchanged.
void BeamzCompactNativeMovingUniformObservers(BeamzNativeMovingPlan&plan,bool enabled){
  using namespace beamz::cuda::moving;
  const size_t ranks=plan.bulk_counts.size();
  if(plan.rank_offsets.size()!=ranks+1)throw std::invalid_argument("compact observer rank offsets differ");
  plan.uniform_observer_counts.assign(ranks,0);
  if(plan.policy_offsets.empty())return;
  if(plan.policy_offsets.size()!=ranks*7)throw std::invalid_argument("compact observer policy offsets differ");
  for(size_t rank=0;rank<ranks;++rank){
    const int begin=plan.rank_offsets[rank],end=plan.rank_offsets[rank+1];
    const int first=plan.policy_offsets[rank*7+2],last=plan.policy_offsets[rank*7+3];
    if(begin<0||begin>first||first>last||last>end||end>int(plan.tile_ids.size()))throw std::invalid_argument("compact observer range outside rank");
    for(int i=first;i<last;++i){const int tile=plan.tile_ids[i];
      if(tile<0||size_t(tile)>=plan.tile_flags.size()||Policy(plan.tile_flags[tile])!=2)throw std::invalid_argument("compact observer range contains another policy");
    }
    if(enabled){const auto start=plan.tile_ids.begin()+first,finish=plan.tile_ids.begin()+last;
      const auto split=std::stable_partition(start,finish,[&](int tile){return bool(plan.tile_flags[tile]&64);});
      plan.uniform_observer_counts[rank]=split-start;
    }
    for(int i=begin;i<end;++i)plan.slots[plan.tile_ids[i]]=i-begin;
  }
}
"""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one exact observer UniformE anchor: {old!r}")
    return text.replace(old, new, 1)


def transformed(parent):
    for name, expected in PARENT_HASHES.items():
        if digest(parent[name]) != expected:
            raise ValueError(f"Exact compact7497 parent required: {name}")
    files = {name: data.decode() for name, data in parent.items()}
    c = files["asymmetric_native.cuh"]
    c = once(
        c,
        "(Layout==1&&Policy==1&&IsotropicH&&UniformE)",
        "(Layout==1&&(Policy==1||Policy==2)&&IsotropicH&&UniformE)",
    )
    c = once(
        c,
        """ }else if constexpr(Fast||(Policy==1&&IsotropicH)){
""",
        """ }else if constexpr(Layout==1&&Policy==2&&IsotropicH){
  // Noncertified observer subrange; canonical spatial E lookup remains.
  UniformBody<Depth,Layout,false,true,2,false>(p,tiles);
 }else if constexpr(Fast||(Policy==1&&IsotropicH)){
""",
    )
    c = once(
        c,
        "inline bool MakeSource(",
        """template<int Depth>
__global__ __launch_bounds__(1024,1) void AsymmetricObserverUniform1024(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 UniformBody<Depth,1,false,true,2,true>(p,tiles);
}

inline bool MakeSource(""",
    )
    anchor = (
        "template<int Depth,int Layout,bool IsotropicH>\ninline cudaError_t LaunchRanks"
    )
    helper = """template<int Depth,int Layout,bool IsotropicH>
inline cudaError_t LaunchObservers(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count,int uniform_count){
 if(uniform_count<0||uniform_count>count)return cudaErrorInvalidValue;
 if constexpr(Layout==1&&IsotropicH){
  auto status=Launch<Depth,Layout,false,IsotropicH,2>(stream,p,ids+uniform_count,count-uniform_count);
  if(status!=cudaSuccess||!uniform_count)return status;
  const auto kernel=AsymmetricObserverUniform1024<Depth>;
  status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes);if(status!=cudaSuccess)return status;
  AsymmetricObserverUniform1024<Depth><<<uniform_count,1024,SharedBytes,stream>>>(p,ids);
  return cudaGetLastError();
 }else{
  if(uniform_count)return cudaErrorInvalidValue;
  return Launch<Depth,Layout,false,IsotropicH,2>(stream,p,ids,count);
 }
}
"""
    c = once(c, anchor, helper + anchor)
    c = once(
        c,
        "case 2:status=Launch<Depth,Layout,false,IsotropicH,2>(branch,p,w.tile_ids_device+start,count);break;",
        "case 2:status=LaunchObservers<Depth,Layout,IsotropicH>(branch,p,w.tile_ids_device+start,count,w.uniform_observer_counts_host[rank]);break;",
    )
    c = once(
        c,
        " if(!w.uniform_noio_counts_host)return cudaErrorInvalidValue;",
        " if(!w.uniform_noio_counts_host||!w.uniform_observer_counts_host)return cudaErrorInvalidValue;",
    )
    files["asymmetric_native.cuh"] = c

    h = files["native_moving.h"]
    h = once(
        h,
        "  const int32_t* uniform_noio_counts_host=nullptr;",
        "  const int32_t* uniform_observer_counts_host=nullptr; // certified policy2 prefix per rank (HOST)\n  const int32_t* uniform_noio_counts_host=nullptr;",
    )
    h = once(
        h,
        "  std::vector<int32_t> bulk_counts,policy_offsets,uniform_noio_counts;",
        "  std::vector<int32_t> bulk_counts,policy_offsets,uniform_noio_counts,uniform_observer_counts;",
    )
    h = once(
        h,
        "void BeamzCompactNativeMovingUniformNoIO(BeamzNativeMovingPlan&plan,bool enabled);",
        "void BeamzCompactNativeMovingUniformNoIO(BeamzNativeMovingPlan&plan,bool enabled);\nvoid BeamzCompactNativeMovingUniformObservers(BeamzNativeMovingPlan&plan,bool enabled);",
    )
    files["native_moving.h"] = h
    files["native_moving.cc"] += HOST_FUNCTION

    e = files["native_extension.cc"]
    e = once(
        e,
        "size_t certified_bulk=0,certified_noio=0;",
        "size_t certified_bulk=0,certified_noio=0,certified_observer=0;",
    )
    e = once(
        e,
        "||beamz::cuda::moving::Policy(flags)!=1)continue;",
        "||(beamz::cuda::moving::Policy(flags)!=1&&beamz::cuda::moving::Policy(flags)!=2))continue;",
    )
    e = once(
        e,
        "if(has_e_update){moving_plan_.tile_flags[tile]|=64;++certified_noio;}",
        "if(has_e_update){moving_plan_.tile_flags[tile]|=64;if(beamz::cuda::moving::Policy(flags)==1)++certified_noio;else ++certified_observer;}",
    )
    e = once(
        e,
        "moving_uniform_bulk_=certified_bulk;moving_uniform_noio_=certified_noio;",
        "moving_uniform_bulk_=certified_bulk;moving_uniform_noio_=certified_noio;moving_uniform_observer_=certified_observer;",
    )
    e = once(
        e,
        "    BeamzCompactNativeMovingUniformNoIO(moving_plan_,compact_enabled);",
        "    BeamzCompactNativeMovingUniformNoIO(moving_plan_,compact_enabled);\n    BeamzCompactNativeMovingUniformObservers(moving_plan_,compact_enabled);",
    )
    anchor = '    if(BEAMZ_CUDA_NATIVE_MOVING_POLICIES&&compact_enabled&&compact_count!=certified_noio)throw std::logic_error("compact NoIO certificate count differs");'
    e = once(
        e,
        anchor,
        anchor
        + """
    size_t observer_count=0;for(int count:moving_plan_.uniform_observer_counts)observer_count+=count;
    if(BEAMZ_CUDA_NATIVE_MOVING_POLICIES&&compact_enabled&&observer_count!=certified_observer)throw std::logic_error("compact observer certificate count differs");""",
    )
    e = once(
        e,
        "    moving_workspace_.uniform_noio_counts_host=",
        "    moving_workspace_.uniform_observer_counts_host=moving_plan_.uniform_observer_counts.data();\n    moving_workspace_.uniform_noio_counts_host=",
    )
    e = once(
        e,
        "    append_bytes(moving_plan_.uniform_noio_counts.data()",
        "    append_bytes(moving_plan_.uniform_observer_counts.data(),moving_plan_.uniform_observer_counts.size()*4);\n    append_bytes(moving_plan_.uniform_noio_counts.data()",
    )
    e = once(
        e,
        "(moving_plan_.policy_offsets.size()+moving_plan_.uniform_noio_counts.size())*4",
        "(moving_plan_.policy_offsets.size()+moving_plan_.uniform_noio_counts.size()+moving_plan_.uniform_observer_counts.size())*4",
    )
    e = once(
        e,
        'result["moving_compact_noio_tiles"]=moving_uniform_noio_;',
        'result["moving_compact_noio_tiles"]=moving_uniform_noio_;result["moving_uniform_observer_tiles"]=moving_uniform_observer_;result["moving_compact_observer_host_bytes"]=moving_plan_.uniform_observer_counts.size()*4;',
    )
    e = once(
        e,
        '(p.policy_offsets.size()+p.bulk_counts.size())*4;r["compact_noio_host_bytes"]=p.bulk_counts.size()*4;',
        '(p.policy_offsets.size()+2*p.bulk_counts.size())*4;r["compact_noio_host_bytes"]=p.bulk_counts.size()*4;r["compact_observer_host_bytes"]=p.bulk_counts.size()*4;',
    )
    e = once(
        e,
        "size_t moving_uniform_bulk_=0,moving_uniform_noio_=0,moving_uniform_prefix_bytes_=0;",
        "size_t moving_uniform_bulk_=0,moving_uniform_noio_=0,moving_uniform_observer_=0,moving_uniform_prefix_bytes_=0;",
    )
    e = once(
        e,
        '  module.attr("moving_compact_noio_dispatch")=true;',
        '  module.attr("moving_observer_uniform_1024")=true;\n  module.attr("moving_compact_observer_dispatch")=true;\n  module.attr("moving_compact_noio_dispatch")=true;',
    )
    files["native_extension.cc"] = e
    return {name: text.encode() for name, text in files.items()}


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
        "parent_module_sha256": "749766062df96a9d1d0dfa1e27f93499534c6d5baa032f7504104ef6e50fe1f0",
        "parent_hashes": PARENT_HASHES,
        "result_hashes": {name: digest(data) for name, data in files.items()},
        "patcher_sha256": digest(Path(__file__).read_bytes()),
        "scope": "Exact supported-E certificate extends only to lossless-isotropic bytepacked policy2; compact prefix in same cohort; canonicalcell/Finish/CPML/source/DFT unchanged",
    }
    output.with_name(output.name + "-observer-proof.json").write_text(
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
