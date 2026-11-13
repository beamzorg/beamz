"""Stage schedule-first ownership and policy-specialized partial-step workers."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from scripts.build_cuda_partial_plane_pipeline import stage as stage_partial,once,ENQUEUE
from scripts.cuda_trace_phase_cycle import block


CLOSED_ENQUEUE='''
int BeamzEnqueueNativeClosedPipeline(void*opaque,const BeamzProgramLaunch&program,
 const BeamzBuffer*final_fields,int macro_step,const BeamzNativeMovingWorkspace&w,
 const BeamzNativeClosedPipelineWorkspace&q){
 using namespace beamz::cuda::moving_native;
 const auto validation=BeamzValidateNativeProgram(program);if(validation)return validation;
 if(!(w.data_flags&beamz::cuda::moving::kBytePackedE)||!(w.data_flags&beamz::cuda::moving::kLosslessIsotropicH)||
    !(w.data_flags&8u)||w.source_presence_mask!=20u||!q.cpml_uniform||q.tile_count!=w.tile_count||q.cohort_count<1||q.max_workers<1||
    !q.next||!q.pending||!q.claimed||!q.ready||!q.initial_next||!q.initial_pending||!q.initial_heads||!q.root_counts||
    !q.tile_cohort||!q.owner_slots||!q.finish_offsets||q.finish_count<0||(q.finish_count&&!q.finish_children)||!q.heads||!q.enqueued||!q.started||!q.completed||!q.exited||!q.abort||
    !q.cohort_offsets_host||!q.cohort_policies_host||q.timeout_ns<1000000ULL)return cudaErrorInvalidValue;
 if(q.cohort_offsets_host[0]!=0||q.cohort_offsets_host[q.cohort_count]!=q.tile_count)return cudaErrorInvalidValue;
 Parameters p;int layout=0;if(!Prepare(program,final_fields,w,macro_step,&p,&layout)||layout!=1)return cudaErrorInvalidValue;
 const auto stream=static_cast<cudaStream_t>(opaque);
 BeginEpoch<<<1,1,0,stream>>>(w.epoch,w.bad);auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
 AsymmetricClosedReset<<<(q.tile_count+255)/256,256,0,stream>>>(q,w.bad);status=cudaGetLastError();if(status!=cudaSuccess)return status;
 if(w.depth==4)return LaunchClosedCohorts<4>(stream,p,q);
 if(w.depth==16)return LaunchClosedCohorts<16>(stream,p,q);
 return cudaErrorInvalidValue;
}
'''


def stage(source,output,expected,*,generic_noio=False,front_startup=False,single_kernel=False,outline_policies=False):
    if outline_policies and not single_kernel:raise ValueError('Outlined policy trial requires explicit global worker')
    source,output=Path(source).resolve(),Path(output).resolve()
    intermediate=stage_partial(source,output,expected)
    c=(output/'src/asymmetric_native.cuh').read_text()
    c=once(c,ENQUEUE,CLOSED_ENQUEUE)
    c=c.replace('native_plane_pipeline','native_closed_pipeline').replace('plane_pipeline::','closed_pipeline::')
    c=c.replace('closed_pipeline::Started();','closed_pipeline::Started(p);')
    c=c.replace('closed_pipeline::WaitParents(prev);','closed_pipeline::WaitParents(p,prev,true);').replace('closed_pipeline::WaitParents(n);','closed_pipeline::WaitParents(p,n);')
    if front_startup:c=c.replace('closed_pipeline::WaitParents(p,prev,true);','closed_pipeline::WaitParents(p,prev);')
    e=(source/'src/native_extension.cc').read_text()
    e=once(e,'#include "native_moving.h"','#include "native_moving.h"\n#include "native_closed_cohorts.h"\n#include "native_closed_pipeline.h"')
    e=once(e,'  BeamzNativeMovingPlan moving_plan_;','  BeamzNativeClosedCohorts moving_closed_plan_;BeamzNativeClosedPipelineWorkspace moving_closed_workspace_;bool moving_closed_active_=false,moving_closed_cpml_uniform_=false;\n  BeamzNativeMovingPlan moving_plan_;')
    static='  const auto p=BeamzBuildNativeMovingPlan(fields.data(),depth,points,groups,starts,12,bool(BEAMZ_CUDA_NATIVE_MOVING_POLICIES));'
    e=once(e,static,'''  auto p=BeamzBuildNativeMovingPlan(fields.data(),depth,points,groups,starts,12,bool(BEAMZ_CUDA_NATIVE_MOVING_POLICIES));
  const auto closed=BeamzBuildNativeClosedCohorts(p,4);
  const size_t old_owners=size_t(p.epochs)*p.max_frontier,owners=std::max(old_owners,size_t(closed.owner_count));
  p.metadata_bytes+=(owners-old_owners)*sizeof(uint64_t);p.trail_floats=owners*depth*6*p.surface_points;
  const size_t closed_bytes_bound=closed.CapacityBytes();''')
    e=once(e,'r["extra_gpu_bytes"]=p.metadata_bytes+p.trail_floats*4+observation_bytes;',
           'r["extra_gpu_bytes"]=p.metadata_bytes+p.trail_floats*4+observation_bytes+closed_bytes_bound;r["closed_pipeline_metadata_bound"]=closed_bytes_bound;r["closed_pipeline_owner_count"]=closed.owner_count;r["closed_pipeline_cohort_count"]=closed.cohort_policies.size();')
    marker='    moving_observation_bytes_=observation.keys.size()*sizeof(int64_t)'
    e=once(e,marker,'''    moving_closed_plan_=BeamzBuildNativeClosedCohorts(moving_plan_,4);
    moving_closed_cpml_uniform_=certificate.supported;
    for(size_t t=0;t<moving_plan_.tile_flags.size();++t){const int policy=beamz::cuda::moving::Policy(moving_plan_.tile_flags[t]);
      if((policy==1||policy==5)&&(!certificate.supported||!certificate.admitted[t]))moving_closed_cpml_uniform_=false;
    }
    const size_t original_owners=size_t(moving_plan_.epochs)*moving_plan_.max_frontier;
    const size_t allocated_closed_owners=std::max(original_owners,size_t(moving_closed_plan_.owner_count));
    moving_plan_.metadata_bytes+=(allocated_closed_owners-original_owners)*sizeof(uint64_t);
    moving_plan_.trail_floats=allocated_closed_owners*depth*6*moving_plan_.surface_points;
'''+marker)
    e=once(e,'moving_extra_bytes_=moving_plan_.metadata_bytes+moving_plan_.trail_floats*sizeof(float)+moving_observation_bytes_;',
           'moving_extra_bytes_=moving_plan_.metadata_bytes+moving_plan_.trail_floats*sizeof(float)+moving_observation_bytes_+moving_closed_plan_.CapacityBytes();')
    e=once(e,'size_t(moving_plan_.epochs)*moving_plan_.max_frontier*sizeof(uint64_t),true)','allocated_closed_owners*sizeof(uint64_t),true)')
    allocation='    moving_workspace_.epoch=static_cast<uint32_t*>(upload(nullptr,4,true));moving_workspace_.bad=static_cast<int32_t*>(upload(nullptr,4,true));'
    e=once(e,allocation,allocation+'''
    auto&closed=moving_closed_workspace_;const size_t task_bytes=moving_plan_.tile_ids.size()*sizeof(int32_t),cohort_bytes=moving_closed_plan_.cohort_policies.size()*sizeof(int32_t);
    auto ints=[&](const std::vector<int32_t>&v){return static_cast<int32_t*>(upload(v.data(),v.size()*sizeof(int32_t)));};
    closed.next=static_cast<int32_t*>(upload(nullptr,task_bytes));closed.pending=static_cast<int32_t*>(upload(nullptr,task_bytes));
    closed.claimed=static_cast<int32_t*>(upload(nullptr,task_bytes));closed.ready=static_cast<int32_t*>(upload(nullptr,task_bytes));
    closed.initial_next=ints(moving_closed_plan_.initial_next);closed.initial_pending=ints(moving_closed_plan_.initial_pending);
    closed.initial_heads=ints(moving_closed_plan_.initial_heads);closed.root_counts=ints(moving_closed_plan_.root_counts);
    closed.tile_cohort=ints(moving_closed_plan_.tile_cohort);closed.owner_slots=ints(moving_closed_plan_.owner_slots);
    closed.finish_offsets=ints(moving_closed_plan_.finish_offsets);closed.finish_children=static_cast<int32_t*>(upload(nullptr,task_bytes*8));
    if(!moving_closed_plan_.finish_children.empty())Check(cudaMemcpyAsync(const_cast<int32_t*>(closed.finish_children),moving_closed_plan_.finish_children.data(),moving_closed_plan_.finish_children.size()*sizeof(int32_t),cudaMemcpyHostToDevice,stream_.value),"upload closed completion CSR");
    closed.finish_count=moving_closed_plan_.finish_children.size();
    closed.cpml_uniform=moving_closed_cpml_uniform_;
    closed.heads=static_cast<int32_t*>(upload(nullptr,cohort_bytes));closed.enqueued=static_cast<int32_t*>(upload(nullptr,cohort_bytes));
    closed.started=static_cast<int32_t*>(upload(nullptr,cohort_bytes));closed.completed=static_cast<int32_t*>(upload(nullptr,cohort_bytes));closed.exited=static_cast<int32_t*>(upload(nullptr,cohort_bytes));
    closed.abort=static_cast<int32_t*>(upload(nullptr,4));closed.tile_count=moving_plan_.tile_ids.size();closed.cohort_count=moving_closed_plan_.cohort_policies.size();
    closed.cohort_offsets_host=moving_closed_plan_.cohort_offsets.data();closed.cohort_policies_host=moving_closed_plan_.cohort_policies.data();
    int closed_device=0;Check(cudaGetDevice(&closed_device),"closed pipeline current device");Check(cudaDeviceGetAttribute(&closed.max_workers,cudaDevAttrMultiProcessorCount,closed_device),"closed pipeline SM count");
    moving_closed_active_=moving_closed_cpml_uniform_&&(moving_workspace_.data_flags&beamz::cuda::moving::kBytePackedE)&&(moving_workspace_.data_flags&beamz::cuda::moving::kLosslessIsotropicH)&&(moving_workspace_.data_flags&8u)&&moving_workspace_.source_presence_mask==20u;
''')
    key='    append_bytes(&moving_workspace_.parallel_policies,sizeof(bool));'
    e=once(e,key,key+'''
    append_bytes(&moving_closed_active_,sizeof(bool));append_bytes(&closed.max_workers,sizeof(int));append_bytes(&closed.timeout_ns,sizeof(uint64_t));
    append_bytes(&closed.tile_count,sizeof(int));append_bytes(&closed.cohort_count,sizeof(int));append_bytes(&closed.finish_count,sizeof(int));
    append_bytes(&closed.cpml_uniform,sizeof(int));
    const std::vector<int32_t>*closed_vectors[]={&moving_closed_plan_.owner_slots,&moving_closed_plan_.tile_cohort,&moving_closed_plan_.initial_pending,&moving_closed_plan_.initial_next,&moving_closed_plan_.initial_heads,&moving_closed_plan_.root_counts,&moving_closed_plan_.cohort_offsets,&moving_closed_plan_.cohort_policies,&moving_closed_plan_.finish_offsets,&moving_closed_plan_.finish_children};
    for(const auto*v:closed_vectors){const uint64_t count=v->size();append_bytes(&count,sizeof(count));append_bytes(v->data(),v->size()*sizeof(int32_t));}
    const void*closed_pointers[]={closed.next,closed.pending,closed.claimed,closed.ready,closed.initial_next,closed.initial_pending,closed.initial_heads,closed.root_counts,closed.tile_cohort,closed.owner_slots,closed.finish_offsets,closed.finish_children,closed.heads,closed.enqueued,closed.started,closed.completed,closed.exited,closed.abort};
    for(const void*pointer:closed_pointers)append_bytes(&pointer,sizeof(pointer));''')
    call='int error=BeamzEnqueueNativeAsymmetric(stream_.value,launch.program,other_.data(),offset,moving_workspace_);'
    e=once(e,call,'int error=moving_closed_active_?BeamzEnqueueNativeClosedPipeline(stream_.value,launch.program,other_.data(),offset,moving_workspace_,moving_closed_workspace_):BeamzEnqueueNativeAsymmetric(stream_.value,launch.program,other_.data(),offset,moving_workspace_);')
    e=e.replace('"native-asymmetric-moving-v1"','"native-asymmetric-schedule-first-closed-v1"')
    e=once(e,'    result["moving_observation_bytes"]=moving_observation_bytes_;','''    result["moving_closed_pipeline_cpml_uniform_certified"]=moving_closed_cpml_uniform_;
    result["moving_closed_pipeline_active"]=moving_closed_active_;result["moving_closed_pipeline_cohorts"]=moving_closed_plan_.cohort_policies.size();
    result["moving_closed_pipeline_owner_count"]=moving_closed_plan_.owner_count;result["moving_closed_pipeline_internal_owners"]=moving_closed_plan_.internal_ring_owners;
    result["moving_closed_pipeline_export_owners"]=moving_closed_plan_.export_owners;result["moving_closed_pipeline_exported_packets"]=moving_closed_plan_.exported_packets;
    result["moving_closed_pipeline_metadata_bytes"]=moving_closed_plan_.CapacityBytes();result["moving_closed_pipeline_metadata_used_bytes"]=moving_closed_plan_.Bytes();result["moving_closed_pipeline_max_workers"]=moving_closed_workspace_.max_workers;
    result["moving_closed_pipeline_start_edges"]=moving_closed_plan_.start_edges;result["moving_closed_pipeline_finish_edges"]=moving_closed_plan_.finish_edges;
    result["moving_observation_bytes"]=moving_observation_bytes_;''')
    e=once(e,'NB_MODULE(_cuda_native, module) {','NB_MODULE(_cuda_native, module) {\n module.attr("moving_closed_pipeline_compiled")=true;module.attr("moving_closed_pipeline_queue")="unique-node-lifo";')
    if single_kernel:
        e=once(e,'BeamzBuildNativeClosedCohorts(p,4);','BeamzBuildNativeClosedCohorts(p,4,true);')
        e=once(e,'BeamzBuildNativeClosedCohorts(moving_plan_,4);','BeamzBuildNativeClosedCohorts(moving_plan_,4,true);')
        e=e.replace('"native-asymmetric-schedule-first-closed-v1"','"native-asymmetric-global-ready-v1"')
    helpers={}
    for stem in ('cohorts.h','pipeline.h','pipeline_control.cuh','pipeline_kernel.cuh'):
        script=Path('scripts')/('cuda_native_closed_'+stem);name='native_closed_'+stem;content=script.read_bytes()
        if generic_noio and stem=='pipeline_kernel.cuh':
            old='''   case 1:
    if(FixedNoIOCubeEligible<512>(p))status=LaunchClosedPolicy<Depth,1,512>(stream,p,q);
    else if(FixedNoIOCubeEligible<564>(p))status=LaunchClosedPolicy<Depth,1,564>(stream,p,q);
    else status=LaunchClosedPolicy<Depth,1,0>(stream,p,q);
    break;'''
            content=once(content.decode(),old,'   case 1:status=LaunchClosedPolicy<Depth,1,0>(stream,p,q);break;').encode()
        if front_startup and stem=='pipeline_kernel.cuh':
            text=content.decode();old='''  if(Policy!=6&&beamz::cuda::moving::Policy(p.flags[Tile()])!=Policy){if(!threadIdx.x)Abort(2048);}
  else ClosedDispatch<Depth,Policy,CubeN>(p);'''
            new='''  {
   // Hoist the cold all7bootstrap before any native field registers exist.
   // Eligibility already proves every ancestor has STARTED; these waits
   // therefore depend only on resident or completed lower-rank workers.
   const int task=Tile(),oz=(task/(p.ny*p.nx))*BZ,oy=((task/p.nx)%p.ny)*BY,ox=(task%p.nx)*BX;
   const int first=max(1,max(oz-p.phase[0].own[0].z,max(oy-p.phase[0].own[0].y,ox-p.phase[0].own[1].x)));
   const int last=min(Depth,min(oz+BZ-1,min(oy+BY-1,ox+BX-1)));
   if(first>1&&first<=last)WaitParents(p,first-1,true);
  }
  if(!Aborted()){
   if(Policy!=6&&beamz::cuda::moving::Policy(p.flags[Tile()])!=Policy){if(!threadIdx.x)Abort(2048);}
   else ClosedDispatch<Depth,Policy,CubeN>(p);
  }'''
            content=once(text,old,new).encode()
        if single_kernel and stem=='pipeline_kernel.cuh':
            text=content.decode();begin,end,_=block(text,'switch(q.cohort_policies_host[c])')
            replacement='''switch(q.cohort_policies_host[c]){
   case 6:
    if(FixedNoIOCubeEligible<512>(p))status=LaunchClosedPolicy<Depth,6,512>(stream,p,q);
    else if(FixedNoIOCubeEligible<564>(p))status=LaunchClosedPolicy<Depth,6,564>(stream,p,q);
    else status=LaunchClosedPolicy<Depth,6,0>(stream,p,q);
    break;
   default:return cudaErrorInvalidValue;
  }'''
            content=(text[:begin]+replacement+text[end:]).encode()
        if outline_policies and stem=='pipeline_kernel.cuh':
            text=content.decode();marker='template<int Depth,int Policy,int CubeN=0>\n__device__ __forceinline__ void ClosedDispatch('
            text=once(text,marker,'template<int Depth,int Policy,int CubeN>\n__device__ __noinline__ void ClosedLeaf(const Parameters&p);\n'+marker)
            for policy in range(6):text=once(text,f'ClosedDispatch<Depth,{policy},CubeN>(p);return;',f'ClosedLeaf<Depth,{policy},CubeN>(p);return;')
            marker='__global__ void AsymmetricClosedReset('
            text=once(text,marker,'''template<int Depth,int Policy,int CubeN>
__device__ __noinline__ void ClosedLeaf(const Parameters&p){ClosedDispatch<Depth,Policy,CubeN>(p);}
'''+marker)
            content=text.encode()
        (output/'src'/name).write_bytes(content);helpers[name]=hashlib.sha256(content).hexdigest()
    for name,text in [('asymmetric_native.cuh',c),('native_extension.cc',e)]: (output/'src'/name).write_text(text)
    for path in (output/'src').glob('native_plane_pipeline*'):path.unlink()
    proof=dict(schema='beamz.schedule-first-closed-pipeline-compiler.v1',parent_consumer_sha256=expected,consumer_sha256=hashlib.sha256(c.encode()).hexdigest(),extension_sha256=hashlib.sha256(e.encode()).hexdigest(),helpers=helpers,expected_kernels=75 if single_kernel else 81 if generic_noio else 85,generic_noio=generic_noio,front_startup=front_startup,single_kernel=single_kernel,outline_policies=outline_policies,late_startup_neighbors=7,
               body_bindings=intermediate['body_bindings'],numerics='Original native point loops and canonical arithmetic headers retained; only scheduler/owner/ready/coherentpsi bindings change',
               ownership='Schedule-first internal4rank ring plus reserved exportinterval colors; maxold/newowner allocation preserves complete fallback guards',
               scheduling='6static policy worker templates perDepth4/16; unique-node readyLIFO,3START/alloldreadersCOMPLETE eligibility; ALL7late-startup readies then3nearest fullstep readies; no cross-kernel spin',
               program_fact='Captured immutable UniformE certificate admits EVERY Policy1/5 E-update trajectory, including vacuous windows; otherwise originalcompletefallback. NoIO0/512/564 separateAOTworkerentries.',
               graph_identity='Every scheduler array/vector/pointer and worker/timeout/active choice serialized',GPU_executed=False)
    (output.parent/'closed-pipeline-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    (output.parent/'partial-plane-compiler-proof.json').write_text(json.dumps(dict(superseded_by='closed-pipeline-compiler-proof.json',construction_only=True,GPU_executed=False))+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);p.add_argument('--generic-noio',action='store_true');p.add_argument('--front-startup',action='store_true');p.add_argument('--single-kernel',action='store_true');p.add_argument('--outline-policies',action='store_true');a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256,generic_noio=a.generic_noio,front_startup=a.front_startup,single_kernel=a.single_kernel,outline_policies=a.outline_policies)))
