"""Remove the false current-H producer edge from owned H publication.

Every writer publishes/fences owned H before a full-mask warp join. Warp
leaders accumulate a CTA-scope acq_rel arrival chain; thread0 acquires all32
arrivals and releases global H-ready BEFORE waiting for parent H imports.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

try:
    from scripts.patch_cuda_native_persistent import sha,once
    from scripts.patch_cuda_native_persistent_phase_ready import H_PUBLISH
except ModuleNotFoundError:
    from patch_cuda_native_persistent import sha,once
    from patch_cuda_native_persistent_phase_ready import H_PUBLISH


JOIN=r'''// TRACE_H_PUBLICATION_JOIN_BEGIN
__device__ __forceinline__ void ArriveHWriterWarp(){
 const uint32_t address=__cvta_generic_to_shared(const_cast<uint32_t*>(&C().h_arrivals));
 uint32_t previous;asm volatile("atom.acq_rel.cta.shared.add.u32 %0, [%1], 1;":"=r"(previous):"r"(address):"memory");
}
__device__ __forceinline__ uint32_t AcquireHWriterWarps(){
 const uint32_t address=__cvta_generic_to_shared(const_cast<uint32_t*>(&C().h_arrivals));
 uint32_t value;asm volatile("ld.acquire.cta.shared.u32 %0, [%1];":"=r"(value):"r"(address):"memory");return value;
}
__device__ __forceinline__ void JoinHWriterWarpsAndPublish(int step){
 // Called by EVERY lane, after that lane's owned H/psiH/trailH stores and
 // global fence. Warp memory synchronization folds all32 lanes into leader.
 __syncwarp(0xffffffffu);
 if(!(threadIdx.x&31))ArriveHWriterWarp();
 if(!threadIdx.x){
  uint64_t last=Timer();uint32_t previous=0;
  for(;;){
   if(PollAbort())return;
   const uint32_t seen=AcquireHWriterWarps();
   if(seen==32)break;
   if(seen>32){Abort(512);return;}
   if(seen!=previous){previous=seen;last=Timer();}
   if(Timer()-last>C().timeout_ns){Abort(256);return;}__nanosleep(64);
  }
  if(!PollAbort())Release(C().ready+C().owner,uint32_t(2*step-1));
 }
}
// TRACE_H_PUBLICATION_JOIN_END
'''


def publication_join(code):
    if '// TRACE_PHASE_READY_COUNTER:' not in code:raise ValueError('Packed phase-ready parent required')
    if 'if(C().abort_local)return;floor=seen<floor?seen:floor;' not in code:raise ValueError('Fixed startup0 parent required')
    code=once(code,'uint32_t epoch,upstream_floor,import_gate;',
        'uint32_t epoch,upstream_floor,import_gate,h_arrivals;')
    code=once(code,' C().upstream_floor=0;C().import_gate=0;',
        ' C().upstream_floor=0;C().import_gate=0;C().h_arrivals=0;')
    code=once(code,'__device__ __forceinline__ void GateImports(int step){',JOIN+'\n__device__ __forceinline__ void GateImports(int step){')
    code=once(code,H_PUBLISH,'')
    code=once(code,'  GateImports(2*n-1);if(C().abort_local)return;\n','')
    geometry='  const int hz=(-n-1)&(CZ-1),hy=(-n-1)&(CY-1),hx=(-n-1)&31,low=-n-1;'
    publish=H_PUBLISH.replace('// Current owned H is complete; only ghost H imports changed other slots.',
        '// Owned H is complete and publishes BEFORE waiting for any current H ghost.')
    code=once(code,geometry,geometry+'\n'+publish+
        '  JoinHWriterWarpsAndPublish(n);\n  GateImports(2*n-1);if(C().abort_local)return;')
    release='  if(!threadIdx.x)PublishAndWait(2*n-1,2*(Fast?Depth:window[1]));\n'
    if release not in code:release='  if(!threadIdx.x)PublishAndWait(2*n-1,2*window[1]);\n'
    code=once(code,release,'')
    code=once(code,'  if(!threadIdx.x)*phase_step=n+1;',
        '  if(!threadIdx.x){*phase_step=n+1;C().h_arrivals=0;}')
    return code


def patch(source,output):
    source,output=source.resolve(),output.resolve()
    if output.exists() or source==output or source in output.parents or output in source.parents:
        raise ValueError('New disjoint private source required')
    old=(source/'src/native_persistent.cuh').read_text();new=publication_join(old)
    shutil.copytree(source,output);(output/'src/native_persistent.cuh').write_text(new)
    proof={'parent_persistent_consumer_sha256':sha(old.encode()),'persistent_consumer_sha256':sha(new.encode()),
        'patcher_sha256':sha(Path(__file__).read_bytes()),'join_helpers_sha256':sha(JOIN.encode()),
        'workspace_abi_changed':False,'expected_entries':65,
        'false_edge_removed':'owned H(n) publication no longer waits for parent H(n); local H only needs previous E(n-1)',
        'HB':'each writer stores+threadfence -> fullmask syncwarp -> warp leader CTA shared acq_rel RMW chain -> thread0 CTA acquire count32 -> GPU release Hready',
        'reset':'Claim0 initially; reset0 before existingRE, after old arrivals retired by R0; next joins cannot passRE early',
        'abort':'bounded thread0 shared-join watchdog; GateImports still broadcasts abort to all lanes',
        'barriers_unchanged':old.count('__syncthreads()')==new.count('__syncthreads()'),
        'requires':['true phase-DAG oracle','omitted writer-join negative','shared scope/warp HB proof','actual controller proof','resources','all30 exact','sanitizers','paired timing'],'gpu_executed':False}
    (output.parent/'persistent-H-publication-join-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(json.dumps(patch(a.source,a.output)))
