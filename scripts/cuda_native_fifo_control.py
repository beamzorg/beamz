"""Lower ready-task selection to a bounded, non-recycled FIFO publication array."""
from __future__ import annotations
import hashlib
from pathlib import Path

from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_trace_phase_cycle import block


PUSH='''void Push(int tile){
 if(CasAcqRel(C().q.claimed+tile,0,1)!=0){Abort(2048);return;}
 const int ticket=AddAcqRel(C().q.enqueued+C().q.cohort,1);
 if(ticket<0||ticket>=C().q.cohort_tasks){Abort(2048);return;}
 Release(C().q.next+C().q.cohort_offset+ticket,tile);
}'''
POP='''void Pop(){
 C().tile=-1;C().abort_local=0;C().parent_ready=0;RESET_ACK_FLOORS
 if(Acquire(C().q.abort))return;
 int32_t*head=C().q.heads+C().q.cohort;const uint64_t start=Timer();
 for(;;){
  const int ticket=Acquire(head),tail=Acquire(C().q.enqueued+C().q.cohort);
  if(ticket<tail){
   if(ticket<0||ticket>=C().q.cohort_tasks){Abort(2048);return;}
   const int tile=Acquire(C().q.next+C().q.cohort_offset+ticket);
   // A reserved, unpublished ticket belongs to a resident producer executing
   // Push without any intervening wait. Waiting cannot depend on a new CTA.
   if(tile>=0){
    if(tile>=C().q.tile_count||__ldg(C().q.tile_cohort+tile)!=C().q.cohort){Abort(2048);return;}
    if(CasAcqRel(head,ticket,ticket+1)==ticket){C().tile=tile;return;}
   }
  }else{
   if(Acquire(C().q.completed+C().q.cohort)==C().q.cohort_tasks)return;
   if(Acquire(C().q.abort))return;
  }
  if(Timer()-start>C().q.timeout_ns){Abort(256);return;}__nanosleep(32);
 }
}'''


def bind(output,proof):
    src=Path(output)/'src';p=src/'native_closed_pipeline_control.cuh';c=p.read_text()
    for signature,replacement in [('void Push(',PUSH),('void Pop(',POP)]:
        if signature=='void Pop(':replacement=replacement.replace('RESET_ACK_FLOORS','for(int e=0;e<7;++e)C().acked_step[e]=0;' if proof.get('dead_face_discard') else '')
        a,b,_=block(c,signature);c=c[:a]+replacement+c[b:]
    p.write_text(c)
    p=src/'native_closed_cohorts.h';c=p.read_text();a,b,_=block(c,'void InitializeStack(')
    c=c[:a]+'''void InitializeStack(){
  initial_next=initial_queue;initial_heads.assign(cohort_policies.size(),0);
 }'''+c[b:];p.write_text(c)
    p=src/'native_closed_pipeline.h';c=p.read_text();c=once(c,'int cohort=0,cohort_tasks=0,worker_count=0;','int cohort=0,cohort_tasks=0,worker_count=0,cohort_offset=0;');p.write_text(c)
    p=src/'native_closed_pipeline_kernel.cuh';c=p.read_text();c=once(c,'  q.cohort=c;q.cohort_tasks=','  q.cohort=c;q.cohort_offset=q.cohort_offsets_host[c];q.cohort_tasks=');p.write_text(c)
    p=src/'native_extension.cc';c=p.read_text();c=once(c,'module.attr("moving_closed_pipeline_queue")="unique-node-lifo";','module.attr("moving_closed_pipeline_queue")="bounded-monotone-fifo";');p.write_text(c);proof['extension_sha256']=hashlib.sha256(c.encode()).hexdigest()
    proof['queue_fifo']=True;proof['queue_protocol']='Bounded cohort-local monotone head/tail tickets; each task published exactly once with release entry store, acquire entry load before CAS claim; no slot recycling within macro; resident producers never wait between reservation and publication.'
    for name in proof['helpers']:proof['helpers'][name]=hashlib.sha256((src/name).read_bytes()).hexdigest()
