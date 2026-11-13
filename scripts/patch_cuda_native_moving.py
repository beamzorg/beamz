#!/usr/bin/env python3
"""Install the complete asymmetric research runner into a copied CUDA tree."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(
            f"expected one patch anchor, found {text.count(old)}: {old[:80]}"
        )
    return text.replace(old, new, 1)


CPU_PLAN = r"""
#if BEAMZ_CUDA_NATIVE_MOVING
nb::dict MovingResourcePlan(nb::sequence inputs,int depth){
  if(depth!=4&&depth!=16)throw std::invalid_argument("compiled native moving depth must be4 or16");
  if(nb::len(inputs)!=134)throw std::invalid_argument("moving plan requires134-input CPML packet");
  auto numpy=nb::module_::import_("numpy");std::vector<HostBuffer> host;host.reserve(134);
  std::vector<BeamzBuffer> fields;fields.reserve(134);
  for(size_t i=0;i<134;++i){host.push_back(Describe(inputs[i],numpy,false));fields.push_back(host.back().descriptor);}
  auto obs=BeamzBuildNativeDeepPlan(fields.data(),fields[113],static_cast<const int32_t*>(host[113].data));
  std::vector<std::array<int,3>>points;int c=0;
  for(int64_t key:obs.keys){while(c<5&&key>=obs.field_base[c+1])++c;const auto&f=fields[c];int64_t p=key-obs.field_base[c];points.push_back({int(p/(f.dims[1]*f.dims[2])),int((p/f.dims[2])%f.dims[1]),int(p%f.dims[2])});}
  BeamzSourceGroupLaunch groups[9]{};const int32_t*starts[9]{};
  for(int g=0;g<9;++g){groups[g].coefficients=fields[86+3*g];starts[g]=static_cast<const int32_t*>(host[88+3*g].data);}
  const auto p=BeamzBuildNativeMovingPlan(fields.data(),depth,points,groups,starts,12,bool(BEAMZ_CUDA_NATIVE_MOVING_POLICIES));
  size_t lookup_bytes=0;
#if BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION
  const auto sparse=BeamzBuildNativeMovingSparseObservation(fields.data(),obs.keys);lookup_bytes=(sparse.coarse.size()+sparse.leaf.size())*4;
#endif
  const size_t observation_bytes=obs.keys.size()*8+obs.publication_words.size()*4+obs.remapped_indices.size()*4+size_t(depth)*obs.keys.size()*4+lookup_bytes;
  nb::dict r;r["depth"]=depth;r["counts_zyx"]=nb::make_tuple(p.counts[0],p.counts[1],p.counts[2]);r["tile_count"]=p.tile_ids.size();r["rank_count"]=p.bulk_counts.size();r["max_frontier"]=p.max_frontier;r["epochs"]=p.epochs;r["surface_points"]=p.surface_points;r["trail_bytes"]=p.trail_floats*4;r["metadata_bytes"]=p.metadata_bytes;r["observation_bytes"]=observation_bytes;r["observation_keys"]=obs.keys.size();r["direct_lookup_bytes"]=lookup_bytes;r["extra_gpu_bytes"]=p.metadata_bytes+p.trail_floats*4+observation_bytes;
  std::vector<int64_t>policy_counts(6,0);
  for(size_t rank=0;rank<p.bulk_counts.size();++rank)if(!p.policy_offsets.empty())for(int policy=0;policy<6;++policy)policy_counts[policy]+=p.policy_offsets[7*rank+policy+1]-p.policy_offsets[7*rank+policy];
  r["policy_tile_counts"]=nb::cast(policy_counts);r["policy_host_bytes"]=p.policy_offsets.size()*4;
  size_t bulk=0;for(int count:p.bulk_counts)bulk+=count;r["bulk_tile_count"]=bulk;return r;
}
#endif
"""


def patch(source, cell_header, consumer):
    source = source.resolve()
    if source == (ROOT / "cuda").resolve():
        raise ValueError("only copied candidate trees may be patched")
    for name in ("native_moving.h", "native_moving.cc"):
        shutil.copyfile(ROOT / "cuda/src" / name, source / "src" / name)
    for src, name in (
        (cell_header, "asymmetric_native_cell_generated.cuh"),
        (consumer, "asymmetric_native.cuh"),
        (ROOT / "scripts/cuda_compact_full_edge.cuh", "compact_full_edge.cuh"),
    ):
        shutil.copyfile(src, source / "src" / name)
    p = source / "CMakeLists.txt"
    s = p.read_text()
    s = once(
        s,
        "option(BEAMZ_CUDA_DEEP_D4",
        'option(BEAMZ_CUDA_NATIVE_MOVING "Research complete asymmetric point-loop" OFF)\noption(BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION "Research direct sparse raw observation lookup" OFF)\noption(BEAMZ_CUDA_NATIVE_MOVING_POLICIES "Research moving tile policy specialization" OFF)\noption(BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES "Research same-rank policy fork/join" OFF)\noption(BEAMZ_CUDA_DEEP_D4',
    )
    s += '\nset_property(SOURCE src/update.cu src/native_extension.cc APPEND PROPERTY COMPILE_DEFINITIONS "BEAMZ_CUDA_NATIVE_MOVING=$<BOOL:${BEAMZ_CUDA_NATIVE_MOVING}>")\n'
    s = once(
        s,
        "    src/native_deep.cc\n",
        "    src/native_deep.cc\n    src/native_moving.cc\n",
    )
    s += '\nset_property(SOURCE src/update.cu src/native_extension.cc APPEND PROPERTY COMPILE_DEFINITIONS "BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION=$<BOOL:${BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION}>")\n'
    s += '\nset_property(SOURCE src/update.cu src/native_extension.cc APPEND PROPERTY COMPILE_DEFINITIONS "BEAMZ_CUDA_NATIVE_MOVING_POLICIES=$<BOOL:${BEAMZ_CUDA_NATIVE_MOVING_POLICIES}>")\n'
    s += '\nset_property(SOURCE src/update.cu src/native_extension.cc APPEND PROPERTY COMPILE_DEFINITIONS "BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES=$<BOOL:${BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES}>")\n'
    s += "\nset_property(SOURCE src/update.cu APPEND PROPERTY COMPILE_OPTIONS --extended-lambda)\n"
    p.write_text(s)
    p = source / "src/update.cu"
    p.write_text(
        p.read_text()
        + '\n#if BEAMZ_CUDA_NATIVE_MOVING\n#include "native_moving.h"\n#include "compact_full_edge.cuh"\n#include "asymmetric_native_cell_generated.cuh"\n#include "asymmetric_native.cuh"\n#endif\n'
    )
    p = source / "src/native_extension.cc"
    s = p.read_text()
    s = once(
        s,
        '#include "native_runtime.h"',
        '#include "native_runtime.h"\n#include "native_moving.h"\n#include "kernels.h"\n#ifndef BEAMZ_CUDA_NATIVE_MOVING\n#define BEAMZ_CUDA_NATIVE_MOVING 0\n#endif',
    )
    s = once(s, "class NativeSession {", CPU_PLAN + "\nclass NativeSession {")
    s = once(
        s,
        "struct HostBuffer {",
        r"""
#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
struct MovingDependencyEvent {
  cudaEvent_t value=nullptr;
  MovingDependencyEvent(){Check(cudaEventCreateWithFlags(&value,cudaEventDisableTiming),"create moving dependency event");}
  ~MovingDependencyEvent(){if(value)cudaEventDestroy(value);}
};
#endif
struct HostBuffer {""",
    )
    s = once(
        s,
        "  ~NativeSession() { if (!closed_) cudaStreamSynchronize(stream_.value); }",
        r"""
  ~NativeSession(){if(!closed_){
    cudaStreamSynchronize(stream_.value);
#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
    for(const auto&stream:moving_policy_streams_)if(stream&&stream->value)cudaStreamSynchronize(stream->value);
#endif
  }}""",
    )

    old = "(BEAMZ_CUDA_DEEP_D4 && i == static_cast<size_t>(monitor_offset_))"
    new = "(BEAMZ_CUDA_DEEP_D4 && i == static_cast<size_t>(monitor_offset_)) ||\n          (BEAMZ_CUDA_NATIVE_MOVING && ((i>=6&&i<12)||i==static_cast<size_t>(monitor_offset_)||\n           (i>=static_cast<size_t>(source_offset_)&&i<static_cast<size_t>(monitor_offset_)&&(i-source_offset_)%3==2)))"
    s = once(s, old, new)
    s = once(
        s,
        "  nb::dict RunDeep4(",
        (ROOT / "scripts/cuda_native_moving_run.inc").read_text()
        + "\n  nb::dict RunDeep4(",
    )
    s = once(
        s,
        " private:\n",
        " private:\n"
        + (ROOT / "scripts/cuda_native_moving_prepare.inc").read_text()
        + "\n",
    )
    s = once(
        s,
        '    Check(cudaStreamSynchronize(stream_.value), "close native session");',
        r"""    Check(cudaStreamSynchronize(stream_.value), "close native session");
#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
    for(const auto&stream:moving_policy_streams_)if(stream&&stream->value)Check(cudaStreamSynchronize(stream->value),"close moving policy stream");
#endif""",
    )
    s = once(
        s,
        "    allocations_.clear();",
        r"""#if BEAMZ_CUDA_NATIVE_MOVING
#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
    moving_policy_done_.clear();moving_policy_fork_.reset();moving_policy_streams_.clear();
#endif
    moving_allocations_.clear();moving_ready_=false;
#endif
    allocations_.clear();""",
    )
    s = once(
        s,
        '    result["deep_d4_compiled"]',
        """    result["moving_compiled"]=bool(BEAMZ_CUDA_NATIVE_MOVING);
#if BEAMZ_CUDA_NATIVE_MOVING
    result["moving_parallel_policies_compiled"]=bool(BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES);result["moving_parallel_policies_active"]=moving_workspace_.parallel_policies;result["moving_ready"]=moving_ready_;result["moving_policies_compiled"]=bool(BEAMZ_CUDA_NATIVE_MOVING_POLICIES);result["moving_policy_host_bytes"]=moving_plan_.policy_offsets.size()*4;result["moving_extra_bytes"]=moving_extra_bytes_;
    result["moving_observation_bytes"]=moving_observation_bytes_;result["moving_direct_lookup_bytes"]=moving_lookup_bytes_;result["moving_trail_bytes"]=moving_plan_.trail_floats*4;
    result["moving_depth"]=moving_workspace_.depth;result["moving_rank_count"]=moving_workspace_.rank_count;
    result["moving_epoch_policy"]="monotone_operational_epoch_not_restored_with_physics";
#endif
    result["deep_d4_compiled"]""",
    )
    s = once(
        s,
        "  BeamzNativeAttributes attributes_;",
        """#if BEAMZ_CUDA_NATIVE_MOVING
  bool moving_ready_=false;
  size_t moving_extra_bytes_=0,moving_observation_bytes_=0,moving_lookup_bytes_=0;
  BeamzNativeMovingPlan moving_plan_;
  BeamzNativeMovingWorkspace moving_workspace_{};
  BeamzBuffer moving_remapped_{};
  std::vector<std::unique_ptr<DeviceMemory>> moving_allocations_;
  std::string moving_fact_key_;
#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
  std::vector<std::unique_ptr<Stream>>moving_policy_streams_;
  std::vector<std::unique_ptr<MovingDependencyEvent>>moving_policy_done_;
  std::unique_ptr<MovingDependencyEvent>moving_policy_fork_;
#endif
#endif
  BeamzNativeAttributes attributes_;""",
    )
    s = once(
        s,
        '  module.attr("jax_independent")',
        """  module.attr("moving")=bool(BEAMZ_CUDA_NATIVE_MOVING);
#if BEAMZ_CUDA_NATIVE_MOVING
  module.def("moving_resource_plan",&MovingResourcePlan,"inputs"_a,"depth"_a=16);
#endif
  module.attr("jax_independent")""",
    )
    s = once(
        s,
        '      .def("run_deep4"',
        '      .def("run_moving",&NativeSession::RunMoving,"chunks"_a,"depth"_a=16)\n      .def("run_deep4"',
    )
    p.write_text(s)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--cell-header", type=Path, required=True)
    p.add_argument(
        "--consumer", type=Path, default=ROOT / "scripts/cuda_asymmetric_native.cuh"
    )
    a = p.parse_args()
    patch(a.source, a.cell_header, a.consumer)


if __name__ == "__main__":
    main()
