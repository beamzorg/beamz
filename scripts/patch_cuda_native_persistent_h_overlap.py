"""Follow the Yee phase DAG: compute local H before waiting for current trails.

H(n) consumes the E(n-1) ghosts imported in iteration n-1. Current parent
ready(n) is only required when importing H/E(n), after local H computation.
An explicitly typed CTA shared gate broadcasts that acquired fact without
adding a CTA barrier or changing any original field/publication statement.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

try:
    from scripts.patch_cuda_native_persistent import sha,once
except ModuleNotFoundError:
    from patch_cuda_native_persistent import sha,once


GATE=r'''// TRACE_IMPORT_GATE_BEGIN
__device__ __forceinline__ uint32_t AcquireImportGate(){
 const uint32_t address=__cvta_generic_to_shared(const_cast<uint32_t*>(&C().import_gate));
 uint32_t value;asm volatile("ld.acquire.cta.shared.u32 %0, [%1];":"=r"(value):"r"(address):"memory");return value;
}
__device__ __forceinline__ void ReleaseImportGate(uint32_t value){
 const uint32_t address=__cvta_generic_to_shared(const_cast<uint32_t*>(&C().import_gate));
 asm volatile("st.release.cta.shared.u32 [%0], %1;"::"r"(address),"r"(value):"memory");
}
__device__ __forceinline__ void GateImports(int step){
 if(!threadIdx.x){WaitUpstream(step);ReleaseImportGate(uint32_t(step));}
 // Thread0 always releases the gate, including its watchdog-abort path.
 // Every importer acquires the transitive upstream publication dependency.
 while(AcquireImportGate()<uint32_t(step)){}
}
// TRACE_IMPORT_GATE_END
'''


def h_overlap(code,startup_previous=False):
    if '// TRACE_TYPED_GLOBAL_ATOMICS_BEGIN' not in code:raise ValueError('Typed global atomics parent required')
    code=once(code,'uint32_t epoch,upstream_floor;', 'uint32_t epoch,upstream_floor,import_gate;')
    code=once(code,' C().upstream_floor=0;\n const int tile=',
        ' C().upstream_floor=0;C().import_gate=0;\n const int tile=')
    code=once(code,' if(step<last)WaitUpstream(step+1);',
        ' // Current-step dependency is acquired AFTER local H, at GateImports(n).')
    # Keep gate wrappers outside the extracted pure-control unit. No helper
    # uses a GLOBAL wrapper on a shared pointer or a generic address fallback.
    a=code.index('__device__ __forceinline__ void PersistentPairedBody(')
    a=code.rindex('template<',0,a)
    code=code[:a]+GATE+'\n'+code[a:]
    paired='  const int hz=(-n-1)&(CZ-1),hy=(-n-1)&(CY-1),hx=(-n-1)&31,low=-n-1;'
    code=once(code,paired,'  GateImports(n);if(C().abort_local)return;\n'+paired)
    uniform='  const int hz=(-n-1)&7,hy=(-n-1)&15,hx=(-n-1)&31,low=-n-1;'
    if uniform in code:code=once(code,uniform,'  GateImports(n);if(C().abort_local)return;\n'+uniform)
    if startup_previous:
        code=once(code,'if(PollAbort()||uint32_t(step)<=C().upstream_floor)return;',
            'if(PollAbort()||(step>0&&uint32_t(step)<=C().upstream_floor))return;')
        code=code.replace('WaitUpstream(window[0]);','WaitUpstream(window[0]-1);')
        code=code.replace('WaitUpstream(1);','WaitUpstream(0);')
        code=code.replace('// TRACE_IMPORT_GATE_BEGIN','// TRACE_STARTUP_PREVIOUS_VERSION: first1 waits TAG+ready0; late first waits previous E.\n// TRACE_IMPORT_GATE_BEGIN',1)
    return code


def patch(source,output,startup_previous=False):
    source,output=source.resolve(),output.resolve()
    if output.exists() or source==output or source in output.parents or output in source.parents:
        raise ValueError('New disjoint private source required')
    old=(source/'src/native_persistent.cuh').read_text();new=h_overlap(old,startup_previous)
    shutil.copytree(source,output);(output/'src/native_persistent.cuh').write_text(new)
    proof={'parent_persistent_consumer_sha256':sha(old.encode()),'persistent_consumer_sha256':sha(new.encode()),
        'patcher_sha256':sha(Path(__file__).read_bytes()),'shared_gate_helpers_sha256':sha(GATE.encode()),
        'workspace_abi_changed':False,'expected_entries':65,
        'fact':'H(n) uses own old E(n-1), including ghost packets imported after parent ready(n-1); only current packet import requires parent ready(n)',
        'lowering':'move current-step wait from preceding RE to post-local-H import gate; thread0 acquired readiness broadcasts via CTA shared release/acquire',
        'startup':'wait(first-1), with tag validation mandatory at step0; initial A/final B distinct' if startup_previous else 'unchanged conservative wait(first); every late-start packet remains protected',
        'startup_previous':startup_previous,
        'barriers_unchanged':old.count('__syncthreads()')==new.count('__syncthreads()'),
        'ordered_FP_nodes_unchanged':True,'abort':'thread0 releases shared gate even after global watchdog abort; all threads then exit uniformly through existing outer barrier',
        'requires':['phase-DAG RAW/psi ownership proof','typed shared gate HB proof','actual controller proof','resource gate','all30 exact','sanitizers','paired timings'],'gpu_executed':False}
    (output.parent/'persistent-H-overlap-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--startup-previous',action='store_true')
    a=p.parse_args();print(json.dumps(patch(a.source,a.output,a.startup_previous)))
