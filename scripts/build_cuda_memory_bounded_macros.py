"""Bound eight-macro overlap using rolling observations and lazy Mstep scratch.

Preserve numerical kernels and contributor order. A macro's chronological DFT
is drained after its last affine level, before its observation slot is reused.
Mstep-only scratch remains a privately captured CPU value until a method needs
it; moving tails are ordinary program steps and never consume that scratch.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_trace_phase_cycle import block


def stage(source,output,expected,raw_slots=3):
    source,output=Path(source).resolve(),Path(output).resolve();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if output.exists() or source in output.parents or output in source.parents or raw_slots not in (2,3):raise ValueError('New disjoint bounded-ring source required')
    if sha(source/'src/asymmetric_native.cuh')!=expected:raise ValueError('Frozen eight-macro source required')
    shutil.copytree(source,output);src=output/'src';c=(src/'asymmetric_native.cuh').read_text();old_c=c;e=(src/'native_extension.cc').read_text();old_e=e;h=(src/'native_moving.h').read_text();planner=(src/'native_macro_slab.h').read_text()
    planner=once(planner,' if(ranks+size_t(macros-1)*stride>',f' stride=std::max(stride,int((ranks+{raw_slots}-1)/{raw_slots}));\n if(ranks+size_t(macros-1)*stride>')
    c=once(c,'int step_offset=0;','int step_offset=0,raw_step_offset=0;')
    c=once(c,'Observe(p,field,version+step_offset,z,y,x,value);','Observe(p,field,version+raw_step_offset,z,y,x,value);')
    binding='env.step_offset=MacroStepOffset<Depth>(p,tiles[blockIdx.x]);'
    if c.count(binding)!=3:raise ValueError('Three native body time bindings required')
    c=c.replace(binding,binding+f'env.raw_step_offset=(MacroStepOffset<1>(p,tiles[blockIdx.x])%{raw_slots})*Depth;')
    a,b,entry=block(c,'int BeamzEnqueueNativeMacroSlab(')
    entry=once(entry,'const BeamzNativeMovingWorkspace&base){','const BeamzNativeMovingWorkspace&base,int level_begin,int level_end,bool begin_epoch){')
    entry=once(entry,' slab.rank_count=base.slab_level_count;const auto&w=slab;',
               ''' if(level_begin<0||level_end<=level_begin||level_end>base.slab_level_count)return cudaErrorInvalidValue;
 slab.policy_offsets_host+=size_t(level_begin)*7;slab.uniform_noio_counts_host+=level_begin;slab.uniform_observer_counts_host+=level_begin;
 slab.rank_count=level_end-level_begin;const auto&w=slab;''')
    anchor=' BeginEpoch<<<1,1,0,stream>>>(w.epoch,w.bad);auto status=cudaGetLastError();if(status!=cudaSuccess)return status;'
    entry=once(entry,anchor,' if(begin_epoch){'+anchor.strip()+'}')
    c=c[:a]+entry+c[b:]
    h=once(h,'int BeamzEnqueueNativeMacroSlab(void*stream,const BeamzProgramLaunch&program,\n const BeamzBuffer*final_fields,int macro_step,const BeamzNativeMovingWorkspace&workspace);','int BeamzEnqueueNativeMacroSlab(void*stream,const BeamzProgramLaunch&program,\n const BeamzBuffer*final_fields,int macro_step,const BeamzNativeMovingWorkspace&workspace,int level_begin,int level_end,bool begin_epoch);')
    for before,after in (
        ('+size_t(depth)*8*obs.keys.size()*4+lookup_bytes;',f'+size_t(depth)*{raw_slots}*obs.keys.size()*4+lookup_bytes;'),
        ('+size_t(depth)*8*observation.keys.size()*sizeof(float)+moving_lookup_bytes_;',f'+size_t(depth)*{raw_slots}*observation.keys.size()*sizeof(float)+moving_lookup_bytes_;'),
        ('upload(nullptr,size_t(depth)*8*observation.keys.size()*4)',f'upload(nullptr,size_t(depth)*{raw_slots}*observation.keys.size()*4)')):e=once(e,before,after)
    call='''              int error=BeamzEnqueueNativeMacroSlab(stream_.value,launch.program,other_.data(),offset,moving_workspace_);if(error)return static_cast<cudaError_t>(error);
              for(int m=0;m<8;++m){error=EnqueueMovingDft(launch.program,offset+m*depth,m*depth);if(error)return static_cast<cudaError_t>(error);}'''
    replacement=f'''              int level_begin=0;
              for(int m=0;m<8;++m){{
                const int level_end=int(moving_plan_.bulk_counts.size())+m*moving_macro_slab_.macro_level_stride;
                int error=BeamzEnqueueNativeMacroSlab(stream_.value,launch.program,other_.data(),offset,moving_workspace_,level_begin,level_end,m==0);if(error)return static_cast<cudaError_t>(error);
                error=EnqueueMovingDft(launch.program,offset+m*depth,(m%{raw_slots})*depth);if(error)return static_cast<cudaError_t>(error);
                level_begin=level_end;
              }}'''
    e=once(e,call,replacement)
    # Capture deferred bytes privately: download must keep the original value
    # even if a caller later mutates its unused workspace ndarray.
    anchor='''    for (size_t i = 0; i < input_count; ++i) {
      host_.push_back(Describe(inputs[i], numpy_, persistent_.count(i) != 0 ||'''
    e=once(e,anchor,'''    for (size_t i = 0; i < input_count; ++i) {
      const bool defer_scratch=workspace_count_&&i>=size_t(graph_count_+6)&&i<size_t(graph_count_+12);
      if(defer_scratch)deferred_scratch_.insert(int(i));
      host_.push_back(Describe(inputs[i], numpy_, defer_scratch||persistent_.count(i) != 0 ||''')
    e=once(e,'auto memory = std::make_unique<DeviceMemory>(host_.back().bytes);','auto memory = std::make_unique<DeviceMemory>(defer_scratch?0:host_.back().bytes);')
    e=once(e,'      if (host_.back().bytes)\n        Check(cudaMemcpyAsync(memory->data, host_.back().data, host_.back().bytes,','      if (!defer_scratch&&host_.back().bytes)\n        Check(cudaMemcpyAsync(memory->data, host_.back().data, host_.back().bytes,')
    e=once(e,'  std::vector<std::unique_ptr<DeviceMemory>> allocations_;','  std::set<int>deferred_scratch_;\n  std::vector<std::unique_ptr<DeviceMemory>> allocations_;')
    ensure='''  void DeferScratchForMoving(){
    std::vector<int>retire;
    if(workspace_count_)for(int c=0;c<6;++c){const int index=graph_count_+6+c;if(!deferred_scratch_.count(index))retire.push_back(index);}
    if(retire.empty())return;
    for(int index:retire)if(host_[index].bytes)Check(cudaMemcpyAsync(const_cast<void*>(host_[index].data),base_inputs_[index].data,host_[index].bytes,cudaMemcpyDeviceToHost,stream_.value),"preserve idle Mstep scratch on host");
    Check(cudaStreamSynchronize(stream_.value),"finish idle scratch preservation");
    for(int index:retire){resident_bytes_-=allocations_[index]->bytes;allocations_[index]=std::make_unique<DeviceMemory>(0);base_inputs_[index].data=nullptr;inputs_[index]=base_inputs_[index];deferred_scratch_.insert(index);}
    BindOutputs();
  }
  void EnsureScratchResources(){
    if(deferred_scratch_.empty())return;
    while(!deferred_scratch_.empty()){const int index=*deferred_scratch_.begin();
      auto memory=std::make_unique<DeviceMemory>(host_[index].bytes);
      if(host_[index].bytes)Check(cudaMemcpyAsync(memory->data,host_[index].data,host_[index].bytes,cudaMemcpyHostToDevice,stream_.value),"materialize deferred Mstep scratch");
      resident_bytes_+=memory->bytes;base_inputs_[index].data=memory->data;inputs_[index]=base_inputs_[index];allocations_[index]=std::move(memory);
      deferred_scratch_.erase(index);BindOutputs();
    }
    Check(cudaStreamSynchronize(stream_.value),"finish deferred Mstep scratch upload");
  }

'''
    e=once(e,'  void RequireOpen() const {',ensure+'  void RequireOpen() const {')
    e=once(e,'    EnsureMovingResources(depth);','    DeferScratchForMoving();\n    EnsureMovingResources(depth);')
    a,b,run=block(e,'nb::dict RunSchedule(');run=once(run,'    float cuda_ms = 0, physics_ms = 0, copy_ms = 0;','    if(mstep_)EnsureScratchResources();\n    float cuda_ms = 0, physics_ms = 0, copy_ms = 0;');e=e[:a]+run+e[b:]
    a,b,download=block(e,'nb::list Download(')
    download=once(download,'      if (host.bytes)\n        Check(cudaMemcpyAsync',
                  '      if(deferred_scratch_.count(output_aliases_[i])){if(host.bytes)std::memcpy(const_cast<void*>(destination.data),host.data,host.bytes);}\n      else if (host.bytes)\n        Check(cudaMemcpyAsync');e=e[:a]+download+e[b:]
    a,b,download=block(e,'void DownloadInto(')
    download=once(download,'    auto* target = static_cast<unsigned char*>(const_cast<void*>(destination.data));',
                  '''    if(deferred_scratch_.count(output_aliases_[output_index])){
      nb::gil_scoped_release release;if(expected.bytes)std::memcpy(const_cast<void*>(destination.data),expected.data,expected.bytes);return;
    }
    auto* target = static_cast<unsigned char*>(const_cast<void*>(destination.data));''');e=e[:a]+download+e[b:]
    e=once(e,'    result["scratch_bytes"] = scratch_bytes;','    result["scratch_bytes"] = scratch_bytes;\n    size_t deferred_bytes=0;for(int index:deferred_scratch_)deferred_bytes+=host_[index].bytes;\n    result["deferred_scratch_bytes"]=deferred_bytes;result["allocated_scratch_bytes"]=scratch_bytes-deferred_bytes;')
    e=once(e,' module.attr("moving_macro_slab_count")=8;',f' module.attr("moving_macro_slab_count")=8;module.attr("moving_raw_ring_slots")={raw_slots};module.attr("native_lazy_mstep_scratch")=true;')
    e=once(e,'    result["moving_macro_slab_count"]=8;',f'    result["moving_raw_ring_slots"]={raw_slots};result["moving_macro_level_stride"]=moving_macro_slab_.macro_level_stride;\n    result["moving_macro_slab_count"]=8;')
    e=e.replace('native-banked-macro-slab8-v1',f'native-banked-memory-bounded8-ring{raw_slots}-v1')
    for name,text in [('asymmetric_native.cuh',c),('native_extension.cc',e),('native_moving.h',h),('native_macro_slab.h',planner)]: (src/name).write_text(text)
    proof=dict(schema='beamz.memory-bounded-native-macro.v1',native_banked=True,macro_count=8,raw_ring_slots=raw_slots,expected_kernels=68,parent_consumer_sha256=expected,consumer_sha256=sha(src/'asymmetric_native.cuh'),extension_sha256=sha(src/'native_extension.cc'),header_sha256=sha(src/'native_moving.h'),planner_sha256=sha(src/'native_macro_slab.h'),source_sha256=sha(Path(__file__)),deferred_scratch=True,source_time='Full integer macro offset unchanged',raw_time=f'Independent modulo{raw_slots} raw slot, drained in original macro/version order at each macro last level',memory_order='ring_slots*macro_level_stride>=physical_rank_count; future same-slot first writer follows previous macro DFT on main stream and policy fork/join events',GPU_executed=False)
    (output.parent/'macro-pipeline-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);p.add_argument('--raw-slots',type=int,choices=(2,3),default=3);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256,a.raw_slots)))
