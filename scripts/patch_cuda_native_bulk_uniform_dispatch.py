"""Private copy-only uniform bulk queue partition for a larger-cache experiment.

This partitions existing rank-local bulk IDs after immutable material
certification. It preserves the six cohorts and all rank joins, and recomputes
every producer slot against the reordered queue. No device compaction occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT_HASHES = {
    "asymmetric_native.cuh": "37b1083bc50de42ae8799919e1199efcd2af64fab46c4d1fc41d544c85001c18",
    "native_extension.cc": "c60e5ba98962764413eb6d16752ac270a7d24b0328a60b99a6686af7f99d4a61",
    "native_moving.h": "6ff14ef5b7132393a57d952853d0419e5d80e60985943977ba600d433a3ad66b",
    "native_moving.cc": "d88443d5d642ba2ef4475941d2c01457e7eee3c08d1ae28a400ea721869f3a44",
    "asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
}

LAUNCH_HELPER = """template<int Depth,int Layout,bool IsotropicH>
inline cudaError_t LaunchBulk(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count,int uniform_count){
 if(uniform_count<0||uniform_count>count)return cudaErrorInvalidValue;
 if constexpr(Layout==1&&IsotropicH){
  // One existing policy branch/join; disjoint immutable rank-local subranges.
  auto status=Launch<Depth,Layout,true,IsotropicH,0>(stream,p,ids+uniform_count,count-uniform_count);
  if(status!=cudaSuccess||!uniform_count)return status;
  const auto kernel=AsymmetricPartialUniformBulk1024<Depth>;
  status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes);if(status!=cudaSuccess)return status;
  AsymmetricPartialUniformBulk1024<Depth><<<uniform_count,1024,SharedBytes,stream>>>(p,ids);
  return cudaGetLastError();
 }else{
  if(uniform_count)return cudaErrorInvalidValue;
  return Launch<Depth,Layout,true,IsotropicH,0>(stream,p,ids,count);
 }
}
"""

HOST_FUNCTION = r"""
// Run only after the immutable per-tile bit64 material certificate is complete.
void BeamzCompactNativeMovingUniformBulk(BeamzNativeMovingPlan&plan,bool enabled){
  using namespace beamz::cuda::moving;
  const size_t ranks=plan.bulk_counts.size();
  if(plan.rank_offsets.size()!=ranks+1)throw std::invalid_argument("compact bulk rank offsets differ");
  if(!plan.policy_offsets.empty()&&plan.policy_offsets.size()!=ranks*7)
    throw std::invalid_argument("compact bulk policy offsets differ");
  plan.uniform_bulk_counts.assign(ranks,0);
  for(size_t rank=0;rank<ranks;++rank){
    const int begin=plan.rank_offsets[rank],end=plan.rank_offsets[rank+1];
    const int first=begin,last=begin+plan.bulk_counts[rank];
    if(begin<0||begin>last||last>end||end>int(plan.tile_ids.size()))
      throw std::invalid_argument("compact bulk range outside rank");
    if(!plan.policy_offsets.empty()&&(plan.policy_offsets[rank*7]!=first||plan.policy_offsets[rank*7+1]!=last))
      throw std::invalid_argument("compact bulk policy range differs");
    for(int i=first;i<last;++i){const int tile=plan.tile_ids[i];
      if(tile<0||size_t(tile)>=plan.tile_flags.size()||Policy(plan.tile_flags[tile])!=0)
        throw std::invalid_argument("compact bulk range contains another policy");
    }
    if(enabled){
      const auto start=plan.tile_ids.begin()+first,finish=plan.tile_ids.begin()+last;
      const auto split=std::stable_partition(start,finish,[&](int tile){return bool(plan.tile_flags[tile]&64);});
      plan.uniform_bulk_counts[rank]=split-start;
    }
    // This is the same private slot table used by every later-rank consumer.
    for(int i=begin;i<end;++i)plan.slots[plan.tile_ids[i]]=i-begin;
  }
}
"""


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one exact uniform-bulk anchor: {old!r}")
    return text.replace(old, new, 1)


def transform_host(files):
    """Transform native private host metadata; consumer dispatch is separate."""
    files = dict(files)
    h = files["native_moving.h"]
    h = once(
        h,
        "  const int32_t* bulk_counts_host=nullptr;",
        "  const int32_t* uniform_bulk_counts_host=nullptr; // certified policy0 prefix per rank (HOST)\n  const int32_t* bulk_counts_host=nullptr;",
    )
    h = once(
        h,
        "  std::vector<int32_t> bulk_counts,policy_offsets;",
        "  std::vector<int32_t> bulk_counts,policy_offsets,uniform_bulk_counts;",
    )
    h = once(
        h,
        "// CUDA consumer owns",
        "// Private host-only queue partition after immutable certification.\nvoid BeamzCompactNativeMovingUniformBulk(BeamzNativeMovingPlan&,bool);\n\n// CUDA consumer owns",
    )
    files["native_moving.h"] = h
    files["native_moving.cc"] += HOST_FUNCTION
    e = files["native_extension.cc"]
    anchor = "    moving_uniform_bulk_=certified_bulk;moving_uniform_noio_=certified_noio;moving_uniform_prefix_bytes_=certificate.peak_scratch_bytes;"
    e = once(
        e,
        anchor,
        anchor + """
    const bool compact_bulk_enabled=(moving_workspace_.data_flags&beamz::cuda::moving::kBytePackedE)&&(moving_workspace_.data_flags&beamz::cuda::moving::kLosslessIsotropicH);
    BeamzCompactNativeMovingUniformBulk(moving_plan_,compact_bulk_enabled);
    size_t compact_bulk_count=0;for(int count:moving_plan_.uniform_bulk_counts)compact_bulk_count+=count;
    if(compact_bulk_enabled&&compact_bulk_count!=certified_bulk)throw std::logic_error("compact bulk certificate count differs");""",
    )
    e = once(
        e,
        "    moving_workspace_.rank_offsets_host=moving_plan_.rank_offsets.data();",
        "    moving_workspace_.uniform_bulk_counts_host=moving_plan_.uniform_bulk_counts.data();\n    moving_workspace_.rank_offsets_host=moving_plan_.rank_offsets.data();",
    )
    e = once(
        e,
        "    append_bytes(moving_plan_.policy_offsets.data()",
        "    append_bytes(moving_plan_.uniform_bulk_counts.data(),moving_plan_.uniform_bulk_counts.size()*4);\n    append_bytes(moving_plan_.policy_offsets.data()",
    )
    e = once(
        e,
        'result["moving_policy_host_bytes"]=moving_plan_.policy_offsets.size()*4;',
        'result["moving_policy_host_bytes"]=(moving_plan_.policy_offsets.size()+moving_plan_.uniform_bulk_counts.size())*4;result["moving_compact_uniform_bulk_host_bytes"]=moving_plan_.uniform_bulk_counts.size()*4;size_t compact_bulk=0;for(int count:moving_plan_.uniform_bulk_counts)compact_bulk+=count;result["moving_compact_uniform_bulk_tiles"]=compact_bulk;',
    )
    e = once(
        e,
        'r["policy_host_bytes"]=p.policy_offsets.size()*4;',
        'r["policy_host_bytes"]=(p.policy_offsets.size()+p.bulk_counts.size())*4;r["compact_uniform_bulk_host_bytes"]=p.bulk_counts.size()*4;',
    )
    e = once(
        e,
        '  module.attr("moving_noio_uniform_1024")=true;',
        '  module.attr("moving_compact_uniform_bulk_dispatch")=true;\n  module.attr("moving_noio_uniform_1024")=true;',
    )
    files["native_extension.cc"] = e
    return files


def transform_consumer(c):
    anchor = "template<int Depth,int Layout,bool IsotropicH>\ninline cudaError_t LaunchRanks"
    c = once(c, anchor, LAUNCH_HELPER + anchor)
    c = once(
        c,
        "case 0:status=Launch<Depth,Layout,true,IsotropicH,0>(branch,p,w.tile_ids_device+start,count);break;",
        "case 0:status=LaunchBulk<Depth,Layout,IsotropicH>(branch,p,w.tile_ids_device+start,count,w.uniform_bulk_counts_host[rank]);break;",
    )
    c = once(
        c,
        "auto status=Launch<Depth,Layout,true,IsotropicH>(stream,p,w.tile_ids_device+start,bulk);",
        "auto status=LaunchBulk<Depth,Layout,IsotropicH>(stream,p,w.tile_ids_device+start,bulk,w.uniform_bulk_counts_host[rank]);",
    )
    c = once(
        c,
        "!w.rank_offsets_host||!w.bulk_counts_host||",
        "!w.uniform_bulk_counts_host||!w.rank_offsets_host||!w.bulk_counts_host||",
    )
    c = once(
        c,
        "w.bulk_counts_host[rank]>count)return false;",
        "w.bulk_counts_host[rank]>count||w.uniform_bulk_counts_host[rank]<0||w.uniform_bulk_counts_host[rank]>w.bulk_counts_host[rank])return false;",
    )
    return c


def digest(data):
    return hashlib.sha256(data).hexdigest()


def transformed(parent):
    for name, expected in PARENT_HASHES.items():
        if digest(parent[name]) != expected:
            raise ValueError(f"Exact partial-Ey8x32 thread-v2 parent required: {name}")
    files = transform_host({name: value.decode() for name, value in parent.items()})
    files["asymmetric_native.cuh"] = transform_consumer(files["asymmetric_native.cuh"])
    # The entire CUDA prefix, including every body and wrapper, is immutable.
    anchor = "inline bool MakeSource("
    assert files["asymmetric_native.cuh"].split(anchor)[0] == parent["asymmetric_native.cuh"].decode().split(anchor)[0]
    return {name: value.encode() for name, value in files.items()}


def patch_copy(source, output):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents
            or output in source.parents or output == canonical or canonical in output.parents):
        raise ValueError("Output must be a new disjoint noncanonical source tree")
    parent = {name: (source / "src" / name).read_bytes() for name in PARENT_HASHES}
    files = transformed(parent)
    shutil.copytree(source, output)
    for name, data in files.items():
        (output / "src" / name).write_bytes(data)
    proof = {
        "parent_hashes": PARENT_HASHES,
        "result_hashes": {name: digest(data) for name, data in files.items()},
        "patcher_sha256": digest(Path(__file__).read_bytes()),
        "cuda_prefix_unchanged": True,
        "scope": "Only policy0 host partition/private counts/LaunchBulk; original CUDA bodies and six joins unchanged; GPU validation required",
    }
    (output.parent / "bulk-dispatch-source-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(patch_copy(args.source, args.output)))


if __name__ == "__main__":
    main()
