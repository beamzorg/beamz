"""NativeSession integration for the private persistent tile-step prototype.

Normal run_moving retains its original kernel path. The new method has an
independent graph namespace, exact prepared schedule key and private workspace.
This pass does not generate or certify the persistent CUDA scheduler itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT = "e863084f0680d25981585fd4bd0f5ce1a1dd2bde8dd2e507f6ec5895352fef27"

ENSURE = r'''
  void EnsurePersistentResources(const std::string&order,int requested_workers){
    if(!moving_ready_)throw std::invalid_argument("persistent moving plan is not prepared");
    if(order!="rank"&&order!="locality")throw std::invalid_argument("persistent order must be rank or locality");
    if(requested_workers<0)throw std::invalid_argument("persistent worker count cannot be negative");
    const auto&mw=moving_workspace_;
    using namespace beamz::cuda::moving;
    if(!(mw.data_flags&kBytePackedE)||!(mw.data_flags&kLosslessIsotropicH)||
       (mw.source_presence_mask!=0&&mw.source_presence_mask!=20))
      throw std::invalid_argument("persistent prototype requires byte-packed E, lossless isotropic H and source mask0 or20");
    int device=0,sms=0;Check(cudaGetDevice(&device),"persistent current device");
    Check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device),"persistent SM count");
    if(requested_workers>sms)throw std::invalid_argument("persistent workers exceed physical SM count");
    const size_t count=moving_plan_.tile_ids.size(),owners=size_t(mw.epochs)*mw.max_frontier;
    if(count<1||count>size_t(INT32_MAX-1024))throw std::invalid_argument("persistent queue count exceeds bounded int32");
    const int workers=std::min<int>(requested_workers?requested_workers:sms,int(count));
    if(workers<1)throw std::invalid_argument("persistent worker count must be positive");
    if(persistent_ready_){
      if(order!=persistent_order_||workers!=persistent_workspace_.worker_count)
        throw std::invalid_argument("persistent order/workers are fixed per session");
      return;
    }
    if(!persistent_allocations_.empty()){
      Check(cudaStreamSynchronize(stream_.value),"retire incomplete persistent upload");
      persistent_allocations_.clear();
    }
    persistent_workspace_={};persistent_fact_key_.clear();
    auto plan=BeamzBuildNativePersistentPlan(moving_plan_,order=="locality");
    const size_t bytes=count*16+owners*4+8;
    size_t free=0,total=0;Check(cudaMemGetInfo(&free,&total),"query persistent admission");
    if(bytes+(256ULL<<20)>free)
      throw std::invalid_argument("persistent metadata plus256MiB reserve exceeds actual free GPU memory");
    auto upload=[&](const void*data,size_t size,bool zero=false){
      auto memory=std::make_unique<DeviceMemory>(size);
      if(size){
        if(zero)Check(cudaMemsetAsync(memory->data,0,size,stream_.value),"initialize persistent metadata");
        else if(data)Check(cudaMemcpyAsync(memory->data,data,size,cudaMemcpyHostToDevice,stream_.value),"upload persistent metadata");
      }
      void*pointer=memory->data;persistent_allocations_.push_back(std::move(memory));return pointer;
    };
    try{
    auto&pw=persistent_workspace_;
    pw.job_ids_device=static_cast<int32_t*>(upload(plan.job_ids.data(),count*4));
    pw.previous_holder_device=static_cast<int32_t*>(upload(plan.previous_holder.data(),count*4));
    pw.next_job=static_cast<int32_t*>(upload(nullptr,4,true));
    pw.abort=static_cast<int32_t*>(upload(nullptr,4,true));
    pw.claimed_epoch=static_cast<uint32_t*>(upload(nullptr,count*4,true));
    pw.completed_epoch=static_cast<uint32_t*>(upload(nullptr,count*4,true));
    pw.ready_step=static_cast<uint32_t*>(upload(nullptr,owners*4,true));
    pw.job_count=int(count);pw.worker_count=workers;pw.timeout_ns=5000000000ULL;
    const auto append=[&](const void*data,size_t size){persistent_fact_key_.append(static_cast<const char*>(data),size);};
    append(plan.job_ids.data(),count*4);append(plan.previous_holder.data(),count*4);
    append(&pw.job_count,sizeof(pw.job_count));append(&pw.worker_count,sizeof(pw.worker_count));
    append(&pw.timeout_ns,sizeof(pw.timeout_ns));
    persistent_fact_key_.append(order);persistent_order_=order;
    Check(cudaStreamSynchronize(stream_.value),"finish persistent metadata preparation");
    persistent_extra_bytes_=bytes;resident_bytes_+=bytes;persistent_ready_=true;
    }catch(...){
      cudaStreamSynchronize(stream_.value);
      persistent_allocations_.clear();persistent_workspace_={};
      persistent_fact_key_.clear();persistent_order_.clear();persistent_extra_bytes_=0;
      throw;
    }
  }
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Expected one persistent-session anchor: " + old)
    return text.replace(old, new, 1)


def transform(text):
    if sha(text.encode()) != PARENT:
        raise ValueError("Exact selected52865 NativeSession required")
    edits = []

    def change(old, new):
        nonlocal text
        text = once(text, old, new)
        edits.append((old, new))

    change('#include "native_moving.h"', '#include "native_moving.h"\n#include "native_persistent.h"')
    change('  nb::dict RunMoving(const std::vector<int>&chunks,int depth){', r'''
  nb::dict RunMoving(const std::vector<int>&chunks,int depth){
    return RunMovingImpl(chunks,depth,false,"rank",0);
  }
  nb::dict RunMovingPersistent(const std::vector<int>&chunks,int depth,const std::string&order,int workers){
    return RunMovingImpl(chunks,depth,true,order,workers);
  }
  nb::dict RunMovingImpl(const std::vector<int>&chunks,int depth,bool persistent,const std::string&order,int workers){''')
    change('    EnsureMovingResources(depth);\n    const auto started=',
           '    EnsureMovingResources(depth);\n'
           '    if(persistent){\n'
           '      if(order!="rank"&&order!="locality"||workers<0)throw std::invalid_argument("invalid persistent schedule");\n'
           '      if(std::any_of(chunks.begin(),chunks.end(),[&](int n){return n>=depth;}))EnsurePersistentResources(order,workers);\n'
           '    }\n    const auto started=')
    change('BeamzGraphKey("native-asymmetric-moving-v1",stream_.value,identity.program)',
           'BeamzGraphKey(persistent?"native-asymmetric-moving-persistent-v1":"native-asymmetric-moving-v1",stream_.value,identity.program)')
    change('        append_pointer(clock_time_->data);', r'''        if(persistent){
          key.append(persistent_fact_key_);
          append_pointer(persistent_workspace_.job_ids_device);append_pointer(persistent_workspace_.previous_holder_device);
          append_pointer(persistent_workspace_.next_job);append_pointer(persistent_workspace_.abort);
          append_pointer(persistent_workspace_.claimed_epoch);append_pointer(persistent_workspace_.completed_epoch);append_pointer(persistent_workspace_.ready_step);
        }
        append_pointer(clock_time_->data);''')
    change('int error=BeamzEnqueueNativeAsymmetric(stream_.value,launch.program,other_.data(),offset,moving_workspace_);',
           'int error=persistent?BeamzEnqueueNativePersistent(stream_.value,launch.program,other_.data(),offset,moving_workspace_,persistent_workspace_):BeamzEnqueueNativeAsymmetric(stream_.value,launch.program,other_.data(),offset,moving_workspace_);')
    change('result["algorithm"]="asymmetric_moving";',
           'result["algorithm"]=persistent?"persistent_tile_step":"asymmetric_moving";result["persistent_tile_step"]=persistent;result["persistent_order"]=persistent?order:"";')
    change('    moving_allocations_.clear();moving_ready_=false;',
           '    persistent_allocations_.clear();persistent_ready_=false;\n    moving_allocations_.clear();moving_ready_=false;')
    change('  void EnsureMovingResources(int depth){', ENSURE + '\n  void EnsureMovingResources(int depth){')
    change('  bool moving_ready_=false;', r'''  bool persistent_ready_=false;
  size_t persistent_extra_bytes_=0;
  BeamzNativePersistentWorkspace persistent_workspace_{};
  std::vector<std::unique_ptr<DeviceMemory>>persistent_allocations_;
  std::string persistent_fact_key_,persistent_order_;
  bool moving_ready_=false;''')
    change('    result["moving_source_presence_mask"]=', r'''    result["persistent_tile_step_ready"]=persistent_ready_;
    result["persistent_tile_step_order"]=persistent_order_;
    result["persistent_tile_step_metadata_bytes"]=persistent_extra_bytes_;
    result["persistent_tile_step_workers"]=persistent_workspace_.worker_count;
    result["persistent_tile_step_jobs"]=persistent_workspace_.job_count;
    result["persistent_tile_step_timeout_ns"]=persistent_workspace_.timeout_ns;
    result["moving_source_presence_mask"]=''')
    change('  module.attr("moving_noio_uniform_e")=true;',
           '  module.attr("moving_persistent_tile_step")=bool(BEAMZ_CUDA_NATIVE_MOVING);\n  module.attr("moving_noio_uniform_e")=true;')
    change('      .def("run_moving",&NativeSession::RunMoving,"chunks"_a,"depth"_a=16)',
           '      .def("run_moving",&NativeSession::RunMoving,"chunks"_a,"depth"_a=16)\n'
           '      .def("run_moving_persistent",&NativeSession::RunMovingPersistent,"chunks"_a,"depth"_a=16,"order"_a="rank","workers"_a=0)')
    inverse = text
    for old, new in reversed(edits):
        inverse = once(inverse, new, old)
    if sha(inverse.encode()) != PARENT:
        raise ValueError("Persistent-session inverse failed")
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    source, output = a.source.resolve(), a.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if output.exists() or output == canonical or canonical in output.parents or source in output.parents or output in source.parents:
        p.error("New disjoint noncanonical source required")
    text = transform((source / "src/native_extension.cc").read_text())
    shutil.copytree(source, output)
    (output / "src/native_extension.cc").write_text(text)
    record = {"schema": "beamz.persistent-session-integration.v1", "host_parent_sha256": PARENT,
              "host_result_sha256": sha(text.encode()), "exact_source_inverse": True,
              "session_lock_held_once_inside_impl": True, "normal_method_kernel_path_retained": True,
              "persistent_graph_namespace_and_all_pointers_bound": True,
              "metadata_bytes": "16*tiles+4*(epochs*frontier)+8", "initial_final_fields_stay_separate": True,
              "GPU_executed": False, "producer_sha256": sha(Path(__file__).read_bytes())}
    (output.parent / "persistent-session-proof.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
