"""Stage paired-Z supernode ownership and phase-level shared-memory handoff."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from scripts.build_cuda_grouped_pipeline import stage as group_stage
from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_trace_phase_cycle import block


WORKER=r'''void AsymmetricClosedStep(const __grid_constant__ Parameters p,
 const __grid_constant__ BeamzNativeClosedPipelineWorkspace q){
 using namespace closed_pipeline;
 if(!LocalThread()){
  auto&cold=const_cast<Cold&>(C());cold.q=q;cold.bad=p.bad;cold.tile=-1;cold.depth=Depth;cold.abort_local=0;cold.parent_ready=0;
  cold.first=Depth+1;cold.last=0;cold.h_ready=cold.e_ready=cold.h_ack=cold.e_ack=0;
 }
 GroupStageProfiles(p,CommonProfiles());__syncthreads();
 if(!q.cpml_uniform){if(!threadIdx.x)Abort(2048);return;}
 for(;;){
  if(!threadIdx.x)Pop();__syncthreads();
  if(Group()==1&&!LocalThread())C().tile=Peer().tile;
  __syncthreads();if(C().tile<0)break;
  const int tile=Tile();
  if(!LocalThread()){
   C().abort_local=0;C().parent_ready=0;C().h_ready=0;C().e_ready=0;C().h_ack=0;C().e_ack=0;
   if(tile<q.original_count){
    const int oz=(tile/(p.ny*p.nx))*BZ,oy=((tile/p.nx)%p.ny)*BY,ox=(tile%p.nx)*BX;
    C().first=max(1,max(oz-p.phase[0].own[0].z,max(oy-p.phase[0].own[0].y,ox-p.phase[0].own[1].x)));
    C().last=min(Depth,min(oz+BZ-1,min(oy+BY-1,ox+BX-1)));
    p.owner_tags[Owner(p,tile)]=(uint64_t(*p.epoch)<<32)|uint32_t(tile);
   }else{C().first=Depth+1;C().last=0;}
  }
  // Both original owner tags and group lifetimes exist before STARTED.
  __syncthreads();if(!threadIdx.x)Started(p);
  if(tile<q.original_count)PhaseRebindingBody<Depth,CubeN,512>(p);
  Join();
  if(!LocalThread()&&tile<q.original_count)Release(q.ready+tile,Depth);
  __syncthreads();if(!threadIdx.x&&!Acquire(q.abort))Complete();__syncthreads();
 }
 __syncthreads();
 if(!threadIdx.x){const int previous=AddAcqRel(q.exited+q.cohort,1);
  if(previous==q.worker_count-1&&!Acquire(q.abort)){
   if(Acquire(q.completed+q.cohort)!=q.cohort_tasks||Acquire(q.started+q.cohort)!=q.cohort_tasks||Acquire(q.enqueued+q.cohort)!=q.cohort_tasks)Abort(512);
  }
 }
}'''


def stage(source,output,expected):
    output=Path(output).resolve();group_stage(source,output,expected);src=output/'src'
    proof=json.loads((output.parent/'closed-pipeline-compiler-proof.json').read_text())
    abi=(src/'native_closed_pipeline.h').read_text();abi=once(abi,' uint64_t timeout_ns=',' int original_count=0,node_plane=0;\n uint64_t timeout_ns=');(src/'native_closed_pipeline.h').write_text(abi)
    (src/'native_paired_cohorts.h').write_bytes(Path(__file__).with_name('cuda_native_paired_cohorts.h').read_bytes())
    control=(src/'native_closed_pipeline_control.cuh').read_text()
    control=once(control,'GroupStride=49504','GroupStride=49536');control=once(control,'SharedBytes==101312','SharedBytes==101376')
    control=once(control,' int tile,depth,abort_local,parent_ready;',' int tile,depth,abort_local,parent_ready;\n int first,last,h_ready,e_ready,h_ack,e_ack;')
    control=once(control,'int Tile(){return C().tile;}','int Tile(){const int plane=C().q.node_plane,s=C().tile;return (2*(s/plane)+Group())*plane+s%plane;}')
    control=once(control,'C().abort_local=1;}','ReleaseBlock(&C().abort_local,1);}')
    a,b,started=block(control,'void Started(');started=once(started,'if(z+1<p.nz)','if(z+1<(p.nz+1)/2)');control=control[:a]+started+control[b:]
    a,b,skipped=block(control,'void Skipped(');skipped=once(skipped,'C().q.ready+C().tile','C().q.ready+Tile()');control=control[:a]+skipped+control[b:]
    a,b,wait=block(control,'void WaitParents(');wait=once(wait,'const int tile=C().tile,','const int tile=Tile(),');wait=once(wait,'const int a=z?Acquire','const int a=(z&&Group()==0)?Acquire');control=control[:a]+wait+control[b:]
    a,b,publish=block(control,'void Publish(');publish=once(publish,'C().q.ready+C().tile','C().q.ready+Tile()');control=control[:a]+publish+control[b:]
    control=once(control,' Release(C().q.ready+C().tile,int32_t(C().depth));\n','')
    (src/'native_closed_pipeline_control.cuh').write_text(control)
    body=(src/'native_phase_rebinding.cuh').read_text()
    # Owner tags and STARTED belong to the supernode's full-CTA prologue.
    body=once(body,' if(!closed_pipeline::LocalThread()){const int owner=closed_pipeline::Owner(p,tile);p.owner_tags[owner]=epoch|uint32_t(tile);}\n closed_pipeline::Join();if(!closed_pipeline::LocalThread())closed_pipeline::Started(p);','')
    a,b,window=block(body,'if(!closed_pipeline::LocalThread()){\n  window[0]=');body=body[:a]+'if(!closed_pipeline::LocalThread()){window[0]=closed_pipeline::C().first;window[1]=closed_pipeline::C().last;}'+body[b:]
    body=once(body,'closed_pipeline::WaitParents(p,prev);','closed_pipeline::WaitParents(p,prev,true);')
    body=once(body,'  closed_pipeline::Join(); // R0:','  if(!closed_pipeline::BeforeHOverwrite(n))return;\n  closed_pipeline::Join(); // R0:')
    body=body.replace('PhaseScatterGhost<Depth,T,','PairedScatterGhost<Depth,T,')
    body=once(body,'closed_pipeline::Join(); // B2: disjoint owned/ghost stores publish the H family.','closed_pipeline::Join();closed_pipeline::HVisible(n); // B2: shared H ready, incoming H readers ACK.')
    body=once(body,'  if(n<last){\n   #pragma unroll','  if(n<last||closed_pipeline::KeepLastE(n)){\n   #pragma unroll')
    body=once(body,'   closed_pipeline::Join(); // B3: E math and ghost-H reloads release the H frame.\n  }','   closed_pipeline::Join(); // B3: E math and ghost-H reloads release the H frame.\n  }\n  if(!closed_pipeline::BeforeEOverwrite(n))return;')
    body=once(body,'if(n<last){const float new_h=PhaseReloadShared(frame+slot);frame[slot]=new_e;state[c][j]=new_h;}','if(n<last||closed_pipeline::KeepLastE(n)){if(n<last)state[c][j]=PhaseReloadShared(frame+slot);frame[slot]=new_e;}')
    body=once(body,'if(n<last){PairedScatterGhost<Depth,T,true>(p,frame,bases,n,oz,oy,ox);closed_pipeline::Join();}','if(n<last){PairedScatterGhost<Depth,T,true>(p,frame,bases,n,oz,oy,ox);closed_pipeline::Join();closed_pipeline::GhostERead(n);}')
    if body.count('if(surface>=0&&n<Depth)')!=2:raise ValueError('Exactly H/E packet export sites required')
    body=body.replace('if(surface>=0&&n<Depth)','if(surface>=0&&n<Depth&&!closed_pipeline::ElideInternal(n,dz,dy,dx))')
    body=once(body,'  closed_pipeline::Publish(n);','  closed_pipeline::Publish(n);closed_pipeline::EVisible(n);')
    # The handoff uses the actual new shared swizzle and native import helpers.
    position=body.index('template<int Depth,int CubeN,int T=512>');body=body[:position]+'#include "native_paired_handoff.cuh"\n'+body[position:]
    (src/'native_phase_rebinding.cuh').write_text(body);(src/'native_paired_handoff.cuh').write_bytes(Path(__file__).with_name('cuda_native_paired_handoff.cuh').read_bytes())
    kernel=(src/'native_closed_pipeline_kernel.cuh').read_text();a,b,_=block(kernel,'void AsymmetricClosedStep(');kernel=kernel[:a]+WORKER+kernel[b:]
    a,b,reset=block(kernel,'void AsymmetricClosedReset(');reset=once(reset,';q.ready[i]=0;',';');reset=once(reset,' if(i<q.cohort_count)',' if(i<q.original_count)q.ready[i]=0;\n if(i<q.cohort_count)');kernel=kernel[:a]+reset+kernel[b:]
    kernel=once(kernel,'closed_pipeline::SharedBytes==101312','closed_pipeline::SharedBytes==101376');kernel=once(kernel,'<<<(q.worker_count+1)/2,1024,','<<<q.worker_count,1024,');(src/'native_closed_pipeline_kernel.cuh').write_text(kernel)
    c=(src/'asymmetric_native.cuh').read_text();c=once(c,'q.tile_count!=w.tile_count','q.original_count!=w.tile_count||q.node_plane!=w.counts[1]*w.counts[2]||q.tile_count!=((w.counts[0]+1)/2)*q.node_plane')
    c=once(c,'AsymmetricClosedReset<<<(q.tile_count+255)/256','AsymmetricClosedReset<<<(std::max(q.tile_count,q.original_count)+255)/256');(src/'asymmetric_native.cuh').write_text(c)
    e=(src/'native_extension.cc').read_text();e=once(e,'#include "native_closed_cohorts.h"','#include "native_closed_cohorts.h"\n#include "native_paired_cohorts.h"');e=e.replace('BeamzBuildNativeClosedCohorts(p,4,true)','BeamzBuildNativePairedCohorts(p)').replace('BeamzBuildNativeClosedCohorts(moving_plan_,4,true)','BeamzBuildNativePairedCohorts(moving_plan_)')
    e=once(e,'BeamzNativeClosedCohorts moving_closed_plan_;','BeamzNativePairedCohorts moving_closed_plan_;')
    e=once(e,'task_bytes=moving_plan_.tile_ids.size()*sizeof(int32_t)','task_bytes=moving_closed_plan_.tile_ids.size()*sizeof(int32_t)')
    e=once(e,'closed.ready=static_cast<int32_t*>(upload(nullptr,task_bytes));','closed.ready=static_cast<int32_t*>(upload(nullptr,moving_plan_.tile_ids.size()*sizeof(int32_t)));')
    e=once(e,'upload(nullptr,task_bytes*8)','upload(nullptr,task_bytes*16)')
    e=once(e,'closed.tile_count=moving_plan_.tile_ids.size();','closed.tile_count=moving_closed_plan_.tile_ids.size();closed.original_count=moving_plan_.tile_ids.size();closed.node_plane=moving_plan_.counts[1]*moving_plan_.counts[2];')
    e=once(e,'closed.max_workers*=2;','')
    e=once(e,'append_bytes(&closed.cpml_uniform,sizeof(int));','append_bytes(&closed.cpml_uniform,sizeof(int));append_bytes(&closed.original_count,sizeof(int));append_bytes(&closed.node_plane,sizeof(int));')
    e=once(e,'module.attr("moving_phase_rebinding_shared_bytes")=101312;','module.attr("moving_phase_rebinding_shared_bytes")=101376;module.attr("moving_paired_Z_handoff")=true;module.attr("moving_paired_packet_epochs")=5;')
    e=once(e,'result["moving_closed_pipeline_active"]=','result["moving_paired_original_tiles"]=moving_closed_workspace_.original_count;result["moving_paired_supernodes"]=moving_closed_workspace_.tile_count;result["moving_paired_packet_epochs"]=5;\n    result["moving_closed_pipeline_active"]=');(src/'native_extension.cc').write_text(e)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    proof.update(paired_handoff=True,paired_packet_epochs=5,phase_group_stride=49536,phase_common_profiles_offset=99072,phase_shared_bytes=101376,front_startup=False,
                 consumer_sha256=sha(src/'asymmetric_native.cuh'),extension_sha256=sha(src/'native_extension.cc'),paired_planner_sha256=sha(src/'native_paired_cohorts.h'),
                 paired_contract='One readyqueue task reserves both adjacentZ tile groups. Both owner tags before STARTED; both groups finish before COMPLETE. 5rank originalpacket ring preserves strict supernode retirement order. Local pureZ H/E frames replace whole-stepZ wait, with per-family reader ACK before alias reuse. Producer-lastE remains live for child. Internal exports elided only for active overlapping consumer; global all7 bootstrap preserved.')
    for name in (*proof['helpers'],'native_closed_pipeline.h','native_paired_cohorts.h','native_paired_handoff.cuh'):proof['helpers'][name]=sha(src/name)
    for name in ('closed-pipeline-compiler-proof.json','phase-rebinding-compiler-proof.json','paired-handoff-compiler-proof.json'):(output.parent/name).write_text(json.dumps(proof,indent=2)+'\n')
    return {k:v for k,v in proof.items() if k not in ('body_bindings','phase_region_bindings')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
