"""Factor certified NoIO axes into concurrent code families on selected52865.

The rank antichain and complete rank joins stay unchanged. Three child streams
fork from policy1's stream; same-rank one-hot axis kernels run concurrently with
the generic/nonuniform NoIO ranges, then join policy1 before its outer done
event. Streams/events live in NativeSession and every decision enters its key.
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

from scripts.patch_cuda_permanent_family_frontier import PARENT_HASHES as BASE_HASHES, once
from scripts.patch_cuda_native_noio_axis_cohorts import HOST_FUNCTION


PARENT_HASHES = dict(BASE_HASHES, **{
    "native_extension.cc": "e863084f0680d25981585fd4bd0f5ce1a1dd2bde8dd2e507f6ec5895352fef27"})

SCHEDULE = r"""// Generated/reviewed host-only schedule. Also included by the CPU event mock.
template<int Depth,int Layout,bool IsotropicH>
inline cudaError_t LaunchNoIO(cudaStream_t stream,const Parameters&p,const int32_t*ids,
 int count,int uniform_count,const int32_t*axis_counts,const BeamzNativeMovingWorkspace&w){
 if(uniform_count<0||uniform_count>count||!axis_counts)return cudaErrorInvalidValue;
 int singles=0;
 for(int axis=0;axis<3;++axis){if(axis_counts[axis]<0||axis_counts[axis]>uniform_count-singles)return cudaErrorInvalidValue;singles+=axis_counts[axis];}
 if constexpr(Layout==1&&IsotropicH){
  const bool parallel=w.parallel_noio_axes&&singles>0;
  cudaError_t status=cudaSuccess;
  if(parallel){
   if(!w.noio_axis_fork_event)return cudaErrorInvalidValue;
   for(int axis=0;axis<3;++axis)if(axis_counts[axis]&&(!w.noio_axis_streams[axis]||!w.noio_axis_done_events[axis]))return cudaErrorInvalidValue;
   status=cudaEventRecord(static_cast<cudaEvent_t>(w.noio_axis_fork_event),stream);if(status!=cudaSuccess)return status;
  }
  int offset=0;
  for(int axis=0;axis<3;++axis){
   const int n=axis_counts[axis];
   if(n){
    cudaStream_t child=parallel?static_cast<cudaStream_t>(w.noio_axis_streams[axis]):stream;
    if(parallel){status=cudaStreamWaitEvent(child,static_cast<cudaEvent_t>(w.noio_axis_fork_event),0);if(status!=cudaSuccess)return status;}
    if(axis==0)status=LaunchNoIOAxis<Depth,1>(child,p,ids+offset,n);
    else if(axis==1)status=LaunchNoIOAxis<Depth,2>(child,p,ids+offset,n);
    else status=LaunchNoIOAxis<Depth,4>(child,p,ids+offset,n);
    if(status!=cudaSuccess)return status;
    if(parallel){status=cudaEventRecord(static_cast<cudaEvent_t>(w.noio_axis_done_events[axis]),child);if(status!=cudaSuccess)return status;}
   }
   offset+=n;
  }
  // Generic and nonuniform regions remain on the policy stream. They are
  // independent of every child axis range and may overlap those child grids.
  status=Launch<Depth,Layout,false,IsotropicH,1>(stream,p,ids+uniform_count,count-uniform_count);if(status!=cudaSuccess)return status;
  status=LaunchNoIOUniformRange<Depth>(stream,p,ids+singles,uniform_count-singles);if(status!=cudaSuccess)return status;
  if(parallel)for(int axis=0;axis<3;++axis)if(axis_counts[axis]){
   status=cudaStreamWaitEvent(stream,static_cast<cudaEvent_t>(w.noio_axis_done_events[axis]),0);if(status!=cudaSuccess)return status;
  }
  return cudaSuccess;
 }else{
  if(uniform_count||singles)return cudaErrorInvalidValue;
  return Launch<Depth,Layout,false,IsotropicH,1>(stream,p,ids,count);
 }
}
"""

AXIS_KERNEL = r"""template<int Depth,int Axes>
__global__ __launch_bounds__(1024,1) void AsymmetricNoIOSingleAxis1024(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 static_assert(Axes==1||Axes==2||Axes==4,"One-hot immutable NoIO axis class required");
 UniformBody<Depth,1,false,true,1,true,Axes>(p,tiles);
}

"""

AXIS_LAUNCH = r"""template<int Depth,int Axes>
inline cudaError_t LaunchNoIOAxis(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count){
 if(!count)return cudaSuccess;
 const auto kernel=AsymmetricNoIOSingleAxis1024<Depth,Axes>;
 auto status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes);if(status!=cudaSuccess)return status;
 AsymmetricNoIOSingleAxis1024<Depth,Axes><<<count,1024,SharedBytes,stream>>>(p,ids);return cudaGetLastError();
}
template<int Depth>
inline cudaError_t LaunchNoIOUniformRange(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count){
 if(!count)return cudaSuccess;
 const auto kernel=AsymmetricNoIOUniform1024<Depth>;
 auto status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes);if(status!=cudaSuccess)return status;
 AsymmetricNoIOUniform1024<Depth><<<count,1024,SharedBytes,stream>>>(p,ids);return cudaGetLastError();
}

"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def transformed(parent, *, concurrent=True):
    for name, expected in PARENT_HASHES.items():
        if sha(parent[name]) != expected:
            raise ValueError(f"Exact selected52865 parent required: {name}")
    files = {name: data.decode() for name, data in parent.items()}
    c = files["asymmetric_native.cuh"]
    old = """ // Original full certified NoIO grid; selector uniform before any barrier.
 switch(p.flags[tiles[blockIdx.x]]&14){
  case 2:UniformBody<Depth,1,false,true,1,true,1>(p,tiles);return;
  case 4:UniformBody<Depth,1,false,true,1,true,2>(p,tiles);return;
  case 8:UniformBody<Depth,1,false,true,1,true,4>(p,tiles);return;
  default:UniformBody<Depth,1,false,true,1,true>(p,tiles);return;
 }"""
    c = once(c, old,
        " // Certified non-one-hot remainder; one generic body avoids axis code duplication.\n UniformBody<Depth,1,false,true,1,true>(p,tiles);")
    c = once(c,
        "template<int Depth>\n__global__ __launch_bounds__(1024,1) void AsymmetricObserverUniform1024(",
        AXIS_KERNEL+"template<int Depth>\n__global__ __launch_bounds__(1024,1) void AsymmetricObserverUniform1024(")
    start = c.index("template<int Depth,int Layout,bool IsotropicH>\ninline cudaError_t LaunchNoIO(")
    end = c.index("template<int Depth,int Layout,bool IsotropicH>\ninline cudaError_t LaunchObservers", start)
    c = c[:start]+AXIS_LAUNCH+'#include "concurrent_noio_axis_schedule.inc"\n\n'+c[end:]
    c = once(c,
        "count,w.uniform_noio_counts_host[rank]);break;",
        "count,w.uniform_noio_counts_host[rank],w.uniform_noio_axis_counts_host+rank*3,w);break;")
    c = once(c,
        "if(!w.uniform_noio_counts_host||!w.uniform_observer_counts_host)",
        "if(!w.uniform_noio_counts_host||!w.uniform_observer_counts_host||!w.uniform_noio_axis_counts_host)")
    c = once(c,
        " if(!w.policy_offsets_host)return false;",
        " if(!w.policy_offsets_host)return false;\n if(w.parallel_noio_axes){\n  if(!w.noio_axis_fork_event)return false;\n  for(int axis=0;axis<3;++axis)if(!w.noio_axis_streams[axis]||!w.noio_axis_done_events[axis])return false;\n }")
    files["asymmetric_native.cuh"] = c
    h = files["native_moving.h"]
    h = once(h,
        "extern const bool BeamzResearchSourcePlanePairEnabled __attribute__((visibility(\"hidden\")));",
        "extern const bool BeamzResearchSourcePlanePairEnabled __attribute__((visibility(\"hidden\")));\nextern const bool BeamzResearchConcurrentNoIOAxesEnabled __attribute__((visibility(\"hidden\")));")
    h = once(h, "  const int32_t* uniform_noio_counts_host=nullptr;",
        "  const int32_t* uniform_noio_axis_counts_host=nullptr; // [rank,3] exact Z/Y/X one-hot prefixes(HOST)\n  const int32_t* uniform_noio_counts_host=nullptr;")
    h = once(h, "  bool parallel_policies=false;",
        "  bool parallel_policies=false;\n  void* noio_axis_streams[3]{};\n  void* noio_axis_fork_event=nullptr;\n  void* noio_axis_done_events[3]{};\n  bool parallel_noio_axes=false;")
    h = once(h, "uniform_noio_counts,uniform_observer_counts;",
        "uniform_noio_counts,uniform_observer_counts,uniform_noio_axis_counts;")
    h = once(h,
        "void BeamzCompactNativeMovingUniformObservers(BeamzNativeMovingPlan&plan,bool enabled);",
        "void BeamzCompactNativeMovingUniformObservers(BeamzNativeMovingPlan&plan,bool enabled);\nvoid BeamzCompactNativeMovingNoIOAxes(BeamzNativeMovingPlan&plan,bool enabled);")
    files["native_moving.h"] = h
    files["native_moving.cc"] += HOST_FUNCTION
    e = files["native_extension.cc"]
    e = once(e, "const bool BeamzResearchSourcePlanePairEnabled=true;",
        "const bool BeamzResearchSourcePlanePairEnabled=true;\nconst bool BeamzResearchConcurrentNoIOAxesEnabled="+("true" if concurrent else "false")+";")
    e = once(e,
        "    BeamzCompactNativeMovingUniformObservers(moving_plan_,compact_enabled);",
        "    BeamzCompactNativeMovingUniformObservers(moving_plan_,compact_enabled);\n    BeamzCompactNativeMovingNoIOAxes(moving_plan_,compact_enabled);")
    e = once(e, "    moving_workspace_.uniform_observer_counts_host=",
        "    moving_workspace_.uniform_noio_axis_counts_host=moving_plan_.uniform_noio_axis_counts.data();\n    moving_workspace_.uniform_observer_counts_host=")
    e = once(e,
        "    append_bytes(moving_plan_.uniform_observer_counts.data()",
        "    append_bytes(moving_plan_.uniform_noio_axis_counts.data(),moving_plan_.uniform_noio_axis_counts.size()*4);\n    append_bytes(moving_plan_.uniform_observer_counts.data()")
    e = once(e,
        "    append_bytes(&moving_workspace_.parallel_policies,sizeof(bool));",
        "    append_bytes(&moving_workspace_.parallel_policies,sizeof(bool));\n    append_bytes(&moving_workspace_.parallel_noio_axes,sizeof(bool));\n    const bool axis_selector=BeamzResearchConcurrentNoIOAxesEnabled;append_bytes(&axis_selector,sizeof(bool));")
    e = once(e,
        "        append_pointer(moving_workspace_.policy_fork_event);",
        "        append_pointer(moving_workspace_.policy_fork_event);\n        for(int axis=0;axis<3;++axis){append_pointer(moving_workspace_.noio_axis_streams[axis]);append_pointer(moving_workspace_.noio_axis_done_events[axis]);}\n        append_pointer(moving_workspace_.noio_axis_fork_event);")
    e = once(e,
        "    moving_workspace_.policy_fork_event=moving_policy_fork_->value;moving_workspace_.parallel_policies=true;",
        """    moving_workspace_.policy_fork_event=moving_policy_fork_->value;moving_workspace_.parallel_policies=true;
    while(moving_noio_axis_streams_.size()<3)moving_noio_axis_streams_.push_back(std::make_unique<Stream>());
    while(moving_noio_axis_done_.size()<3)moving_noio_axis_done_.push_back(std::make_unique<MovingDependencyEvent>());
    if(!moving_noio_axis_fork_)moving_noio_axis_fork_=std::make_unique<MovingDependencyEvent>();
    for(int axis=0;axis<3;++axis){moving_workspace_.noio_axis_streams[axis]=moving_noio_axis_streams_[axis]->value;moving_workspace_.noio_axis_done_events[axis]=moving_noio_axis_done_[axis]->value;}
    moving_workspace_.noio_axis_fork_event=moving_noio_axis_fork_->value;
    moving_workspace_.parallel_noio_axes=BeamzResearchConcurrentNoIOAxesEnabled;""")
    e = once(e,
        "  std::unique_ptr<MovingDependencyEvent>moving_policy_fork_;",
        "  std::unique_ptr<MovingDependencyEvent>moving_policy_fork_;\n  std::vector<std::unique_ptr<Stream>>moving_noio_axis_streams_;\n  std::vector<std::unique_ptr<MovingDependencyEvent>>moving_noio_axis_done_;\n  std::unique_ptr<MovingDependencyEvent>moving_noio_axis_fork_;")
    e = once(e,
        "    for(const auto&stream:moving_policy_streams_)if(stream&&stream->value)cudaStreamSynchronize(stream->value);",
        "    for(const auto&stream:moving_policy_streams_)if(stream&&stream->value)cudaStreamSynchronize(stream->value);\n    for(const auto&stream:moving_noio_axis_streams_)if(stream&&stream->value)cudaStreamSynchronize(stream->value);")
    e = once(e,
        "    for(const auto&stream:moving_policy_streams_)if(stream&&stream->value)Check(cudaStreamSynchronize(stream->value),\"close moving policy stream\");",
        "    for(const auto&stream:moving_policy_streams_)if(stream&&stream->value)Check(cudaStreamSynchronize(stream->value),\"close moving policy stream\");\n    for(const auto&stream:moving_noio_axis_streams_)if(stream&&stream->value)Check(cudaStreamSynchronize(stream->value),\"close moving NoIO axis stream\");")
    e = once(e,
        "    moving_policy_done_.clear();moving_policy_fork_.reset();moving_policy_streams_.clear();",
        "    moving_noio_axis_done_.clear();moving_noio_axis_fork_.reset();moving_noio_axis_streams_.clear();\n    moving_policy_done_.clear();moving_policy_fork_.reset();moving_policy_streams_.clear();")
    e = once(e,
        "(moving_plan_.policy_offsets.size()+moving_plan_.uniform_noio_counts.size()+moving_plan_.uniform_observer_counts.size())*4",
        "(moving_plan_.policy_offsets.size()+moving_plan_.uniform_noio_counts.size()+moving_plan_.uniform_observer_counts.size()+moving_plan_.uniform_noio_axis_counts.size())*4")
    e = once(e,
        "(p.policy_offsets.size()+2*p.bulk_counts.size())*4;",
        '(p.policy_offsets.size()+5*p.bulk_counts.size())*4;r["noio_axis_host_bytes"]=3*p.bulk_counts.size()*4;')
    e = once(e,
        'result["moving_observation_bytes"]=moving_observation_bytes_;',
        """size_t axis_tiles[3]{};for(size_t i=0;i<moving_plan_.uniform_noio_axis_counts.size();++i)axis_tiles[i%3]+=moving_plan_.uniform_noio_axis_counts[i];
    result["moving_noio_axis_tiles_zyx"]=nb::make_tuple(axis_tiles[0],axis_tiles[1],axis_tiles[2]);result["moving_noio_axis_host_bytes"]=moving_plan_.uniform_noio_axis_counts.size()*4;
    result["moving_noio_axis_parallel_active"]=moving_workspace_.parallel_noio_axes;
    result["moving_observation_bytes"]=moving_observation_bytes_;""")
    e = once(e,
        '  module.attr("moving_noio_axis_bundle")=true;',
        '  module.attr("moving_noio_axis_bundle")=false;\n  module.attr("moving_noio_concurrent_axis_families")=true;')
    files["native_extension.cc"] = e
    # Cell arithmetic, source geometry and source/observer kernel bodies remain.
    marker = "// Private source-only storage schedule; source-bound"
    original_c = parent["asymmetric_native.cuh"].decode()
    cell_begin = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody"
    cell_end = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__"
    if original_c[original_c.index(cell_begin):original_c.index(cell_end)] != c[c.index(cell_begin):c.index(cell_end)]:
        raise ValueError("Original uniform arithmetic/storage body changed")
    if original_c[original_c.index(marker):original_c.index("inline bool MakeSource")] != c[c.index(marker):c.index("inline bool MakeSource")]:
        raise ValueError("Paired source schedule changed")
    files["concurrent_noio_axis_schedule.inc"] = SCHEDULE
    return {name: data.encode() for name, data in files.items()}


def patch_copy(source, output, *, concurrent=True):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents or output in source.parents
            or output == canonical or canonical in output.parents):
        raise ValueError("New disjoint noncanonical output required")
    parent = {name: (source/"src"/name).read_bytes() for name in PARENT_HASHES}
    files = transformed(parent, concurrent=concurrent)
    shutil.copytree(source, output)
    for name, data in files.items():
        (output/"src"/name).write_bytes(data)
    proof = {"parent_module": "52865a3ac8371e2390749953eb3af99746df108fe7bcb00fc4be8b35f4259b92",
             "parent_hashes": PARENT_HASHES, "result_hashes": {n: sha(d) for n, d in files.items()},
             "patcher_sha256": sha(Path(__file__).read_bytes()),
             "immutable_concurrent_selector": concurrent, "CUDA_input_sources_identical_between_selectors": True,
             "binary_CUDA_payload_identity_requires_offline_hash_audit": True,
             "host_partition_function_sha256": sha(HOST_FUNCTION.encode()),
             "expected_moving_kernel_entries": 68,
             "axis_cohorts": "stable exactflags2/4/8 in certified NoIO prefix; IDs/slots/counts serialized",
             "nested_DAG": "outerRankFork -> NoIOfork -> axisKernels -> axisDone -> NoIOjoin -> outerPolicyDone -> rankJoin",
             "generic_and_nonuniform_NoIO_overlap_children": True,
             "cell_and_source_bodies_byte_identical": True, "gpu_executed": False,
             "requires": ["CPU actual host partition/launch event mock", "exact-context zero-local resources",
                          "all30 numerical/clock/input parity", "memcheck/synccheck/racecheck", "full paired large-domain measurements"]}
    (output.parent/"concurrent-noio-axes-proof.json").write_text(json.dumps(proof, indent=2)+"\n")
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--serial-control", action="store_true")
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output, concurrent=not a.serial_control)))


if __name__ == "__main__":
    main()
