"""Copy-only compact host dispatch for certified NoIO kernels on exact 4941.

The immutable certificate partitions only policy1 inside each existing rank.
All six policy boundaries and joins remain unchanged; no GPU compaction occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT_HASHES = {
    "asymmetric_native.cuh": "3ed4a4f837df1ca60f409a243c0dd8a63778c0b2b983bac0ee895b56da04d6aa",
    "native_extension.cc": "844217c56d96bbe827bcfa36522075d7f84658bd238f739dbf757858904ea596",
    "native_moving.h": "27bbb9feff90d91474d1429453d32985ec8fec908c24b0b8bafc927ee462a773",
    "native_moving.cc": "d88443d5d642ba2ef4475941d2c01457e7eee3c08d1ae28a400ea721869f3a44",
    "asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
}

HOST_FUNCTION = r"""
// Certificate bits are immutable preparation facts, never GPU-generated queues.
void BeamzCompactNativeMovingUniformNoIO(BeamzNativeMovingPlan&plan,bool enabled){
  using namespace beamz::cuda::moving;
  const size_t ranks=plan.bulk_counts.size();
  if(plan.rank_offsets.size()!=ranks+1)throw std::invalid_argument("compact NoIO rank offsets differ");
  plan.uniform_noio_counts.assign(ranks,0);
  if(plan.policy_offsets.empty())return; // legacy policy-off path stays unchanged
  if(plan.policy_offsets.size()!=ranks*7)throw std::invalid_argument("compact NoIO policy offsets differ");
  for(size_t rank=0;rank<ranks;++rank){
    const int begin=plan.rank_offsets[rank],end=plan.rank_offsets[rank+1];
    const int first=plan.policy_offsets[rank*7+1],last=plan.policy_offsets[rank*7+2];
    if(begin<0||begin>first||first>last||last>end||end>int(plan.tile_ids.size()))
      throw std::invalid_argument("compact NoIO range outside rank");
    for(int i=first;i<last;++i){const int tile=plan.tile_ids[i];
      if(tile<0||size_t(tile)>=plan.tile_flags.size()||Policy(plan.tile_flags[tile])!=1)
        throw std::invalid_argument("compact NoIO range contains another policy");
    }
    if(enabled){
      const auto start=plan.tile_ids.begin()+first,finish=plan.tile_ids.begin()+last;
      const auto split=std::stable_partition(start,finish,[&](int tile){return bool(plan.tile_flags[tile]&64);});
      plan.uniform_noio_counts[rank]=split-start;
    }
    // All ranks preserve their antichain and cardinality. Every producer lookup
    // uses this same private slot array, including upper-rank consumers.
    for(int i=begin;i<end;++i)plan.slots[plan.tile_ids[i]]=i-begin;
  }
}
"""

LAUNCH_OLD = """ if constexpr(Layout==1&&Policy==1&&IsotropicH){
  const auto uniform_kernel=AsymmetricNoIOUniform1024<Depth>;
  status=cudaFuncSetAttribute(uniform_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes);if(status!=cudaSuccess)return status;
  AsymmetricNoIOUniform1024<Depth><<<count,1024,SharedBytes,stream>>>(p,ids);
  return cudaGetLastError();
 }
"""
LAUNCH_HELPER = """template<int Depth,int Layout,bool IsotropicH>
inline cudaError_t LaunchNoIO(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count,int uniform_count){
 if(uniform_count<0||uniform_count>count)return cudaErrorInvalidValue;
 if constexpr(Layout==1&&IsotropicH){
  // Existing policy branch/join; both subranges are disjoint rank-local IDs.
  auto status=Launch<Depth,Layout,false,IsotropicH,1>(stream,p,ids+uniform_count,count-uniform_count);
  if(status!=cudaSuccess||!uniform_count)return status;
  const auto kernel=AsymmetricNoIOUniform1024<Depth>;
  status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes);if(status!=cudaSuccess)return status;
  AsymmetricNoIOUniform1024<Depth><<<uniform_count,1024,SharedBytes,stream>>>(p,ids);
  return cudaGetLastError();
 }else{
  if(uniform_count)return cudaErrorInvalidValue;
  return Launch<Depth,Layout,false,IsotropicH,1>(stream,p,ids,count);
 }
}
"""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one exact compact-NoIO anchor: {old!r}")
    return text.replace(old, new, 1)


def transformed(parent):
    for name, expected in PARENT_HASHES.items():
        if digest(parent[name]) != expected:
            raise ValueError(f"Exact 4941 parent required: {name}")
    files = {name: data.decode() for name, data in parent.items()}
    c = files["asymmetric_native.cuh"]
    c = once(
        c,
        """  // Disjoint immutable CTA-uniform gate BEFORE any owner tag/body/barrier.
  if(p.flags[tiles[blockIdx.x]]&64)return;
""",
        """  // Host compact range contains only noncertified NoIO IDs.
""",
    )
    c = once(
        c,
        """ // Same cohort; complementary gate prevents all duplicate ownership/writes.
 if(!(p.flags[tiles[blockIdx.x]]&64))return;
""",
        """ // Host compact range contains only certified NoIO IDs.
""",
    )
    c = once(c, LAUNCH_OLD, "")
    anchor = (
        "template<int Depth,int Layout,bool IsotropicH>\ninline cudaError_t LaunchRanks"
    )
    c = once(c, anchor, LAUNCH_HELPER + anchor)
    c = once(
        c,
        "case 1:status=Launch<Depth,Layout,false,IsotropicH,1>(branch,p,w.tile_ids_device+start,count);break;",
        "case 1:status=LaunchNoIO<Depth,Layout,IsotropicH>(branch,p,w.tile_ids_device+start,count,w.uniform_noio_counts_host[rank]);break;",
    )
    c = once(
        c,
        " Parameters p;int layout=0;if(!Prepare(program,final_fields,w,macro_step,&p,&layout))return cudaErrorInvalidValue;",
        " Parameters p;int layout=0;if(!Prepare(program,final_fields,w,macro_step,&p,&layout))return cudaErrorInvalidValue;\n#if BEAMZ_CUDA_NATIVE_MOVING_POLICIES\n if(!w.uniform_noio_counts_host)return cudaErrorInvalidValue;\n#endif",
    )
    files["asymmetric_native.cuh"] = c

    h = files["native_moving.h"]
    h = once(
        h,
        "  const int32_t* bulk_counts_host=nullptr;",
        "  const int32_t* uniform_noio_counts_host=nullptr; // certified policy1 prefix per rank (HOST)\n  const int32_t* bulk_counts_host=nullptr;",
    )
    h = once(
        h,
        "  std::vector<int32_t> bulk_counts,policy_offsets;",
        "  std::vector<int32_t> bulk_counts,policy_offsets,uniform_noio_counts;",
    )
    h = once(
        h,
        "// CUDA consumer owns",
        "// Private host-only stable partition after immutable material certification.\nvoid BeamzCompactNativeMovingUniformNoIO(BeamzNativeMovingPlan&plan,bool enabled);\n\n// CUDA consumer owns",
    )
    files["native_moving.h"] = h
    files["native_moving.cc"] += HOST_FUNCTION

    e = files["native_extension.cc"]
    anchor = "    moving_uniform_bulk_=certified_bulk;moving_uniform_noio_=certified_noio;moving_uniform_prefix_bytes_=certificate.peak_scratch_bytes;"
    e = once(
        e,
        anchor,
        anchor
        + """
    const bool compact_enabled=(moving_workspace_.data_flags&beamz::cuda::moving::kBytePackedE)&&(moving_workspace_.data_flags&beamz::cuda::moving::kLosslessIsotropicH);
    BeamzCompactNativeMovingUniformNoIO(moving_plan_,compact_enabled);
    size_t compact_count=0;for(int count:moving_plan_.uniform_noio_counts)compact_count+=count;
    if(BEAMZ_CUDA_NATIVE_MOVING_POLICIES&&compact_enabled&&compact_count!=certified_noio)throw std::logic_error("compact NoIO certificate count differs");""",
    )
    e = once(
        e,
        "    moving_workspace_.policy_offsets_host=",
        "    moving_workspace_.uniform_noio_counts_host=moving_plan_.uniform_noio_counts.data();\n    moving_workspace_.policy_offsets_host=",
    )
    e = once(
        e,
        "    append_bytes(moving_plan_.policy_offsets.data()",
        "    append_bytes(moving_plan_.uniform_noio_counts.data(),moving_plan_.uniform_noio_counts.size()*4);\n    append_bytes(moving_plan_.policy_offsets.data()",
    )
    e = once(
        e,
        'result["moving_policy_host_bytes"]=moving_plan_.policy_offsets.size()*4;',
        'result["moving_policy_host_bytes"]=(moving_plan_.policy_offsets.size()+moving_plan_.uniform_noio_counts.size())*4;result["moving_compact_noio_host_bytes"]=moving_plan_.uniform_noio_counts.size()*4;result["moving_compact_noio_tiles"]=moving_uniform_noio_;',
    )
    e = once(
        e,
        'r["policy_host_bytes"]=p.policy_offsets.size()*4;',
        'r["policy_host_bytes"]=(p.policy_offsets.size()+p.bulk_counts.size())*4;r["compact_noio_host_bytes"]=p.bulk_counts.size()*4;',
    )
    e = once(
        e,
        '  module.attr("moving_noio_uniform_1024")=true;',
        '  module.attr("moving_compact_noio_dispatch")=true;\n  module.attr("moving_noio_uniform_gated")=false;\n  module.attr("moving_noio_uniform_1024")=true;',
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
        raise ValueError("Output must be a new disjoint source tree")
    if output == canonical or canonical in output.parents:
        raise ValueError("Cannot modify installed/canonical CUDA sources")
    parent = {name: (source / "src" / name).read_bytes() for name in PARENT_HASHES}
    files = transformed(parent)
    shutil.copytree(source, output)
    for name, data in files.items():
        (output / "src" / name).write_bytes(data)
    proof = {
        "parent_module_sha256": "4941e6728b10e803d03d0f09e137cbbecd3b187eb4ac9686fe9fa7161da18e80",
        "parent_hashes": PARENT_HASHES,
        "result_hashes": {name: digest(data) for name, data in files.items()},
        "patcher_sha256": digest(Path(__file__).read_bytes()),
        "scope": "Host-only stable partition within policy1; six rank cohorts and joins unchanged; canonical body/env/cell unchanged; GPU validation required",
    }
    output.with_name(output.name + "-compact-proof.json").write_text(
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
