"""Stage two independent 512-thread named-barrier workers inside one CTA.

Each worker has disjoint field and queue storage. Both share immutable a/b
profiles staged before any worker can leave. This isolation architecture is
the prerequisite for a traced adjacent-tile shared-memory handoff.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from scripts.build_cuda_phase_rebinding import stage as phase_stage
from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_trace_phase_cycle import block


GROUP = r'''
constexpr size_t GroupStride=49504,CommonProfilesOffset=2*GroupStride;
__device__ __forceinline__ int Group(){return int(threadIdx.x)>>9;}
__device__ __forceinline__ int LocalThread(){return int(threadIdx.x)&511;}
__device__ __forceinline__ void Join(){
 const int barrier=Group()+1;asm volatile("bar.sync %0, 512;"::"r"(barrier):"memory");
}
__device__ __forceinline__ float* Frame(){extern __shared__ float frame[];return reinterpret_cast<float*>(reinterpret_cast<char*>(frame)+Group()*GroupStride);}
__device__ __forceinline__ float* CommonProfiles(){extern __shared__ float frame[];return reinterpret_cast<float*>(reinterpret_cast<char*>(frame)+CommonProfilesOffset);}
'''

STAGE = r'''
__device__ __forceinline__ void GroupStageProfiles(const Parameters&p,float*ab){
 if(threadIdx.x<576){const int i=threadIdx.x,slot=i/48,kind=(i/24)&1,packed=i%24;
  const auto&psi=p.phase[slot/6].psi[slot%6];ab[i]=__ldg((kind?psi.b:psi.a)+packed);
 }
}
'''


def stage(source,output,expected):
    output=Path(output).resolve();phase_stage(source,output,expected);src=output/'src'
    proof=json.loads((output.parent/'phase-rebinding-compiler-proof.json').read_text())
    control=(src/'native_closed_pipeline_control.cuh').read_text()
    control=once(control,'namespace closed_pipeline {','namespace closed_pipeline {\n'+GROUP)
    control=once(control,'constexpr size_t SharedBytes=ColdOffset+sizeof(Cold);','constexpr size_t SharedBytes=CommonProfilesOffset+576*sizeof(float);')
    control=once(control,'static_assert(SharedBytes<64512,"Queue state must preserve native64KB carveout");','static_assert(SharedBytes==101312&&SharedBytes<=101376,"Two private groups and immutable a/b profiles must fit SM86");\nstatic_assert(ColdOffset+sizeof(Cold)==GroupStride,"Group cold allocation ABI");')
    control=once(control,'extern __shared__ float frame[];return *reinterpret_cast<volatile Cold*>(reinterpret_cast<char*>(frame)+ColdOffset);','return *reinterpret_cast<volatile Cold*>(reinterpret_cast<char*>(Frame())+ColdOffset);')
    # Only actual controller functions use the local worker identity. Group()
    # and LocalThread() retain hardware threadIdx and no recursive rewriting.
    position=control.index('struct Cold {')
    control=control[:position]+control[position:].replace('threadIdx.x','LocalThread()').replace('__syncthreads();','Join();')
    (src/'native_closed_pipeline_control.cuh').write_text(control)
    body=(src/'native_phase_rebinding.cuh').read_text();body=body.replace('threadIdx.x','closed_pipeline::LocalThread()').replace('__syncthreads();','closed_pipeline::Join();')
    body=once(body,'extern __shared__ float frame[];size_t*bases=', 'float*frame=closed_pipeline::Frame();size_t*bases=')
    body=once(body,'volatile int*window=reinterpret_cast<volatile int*>(bases+8);','volatile int*window=reinterpret_cast<volatile int*>(bases+8);const float*profiles=closed_pipeline::CommonProfiles();')
    for binding in proof['phase_region_bindings']:
        old=binding['emitted'];new=old.replace('PhaseEnvironment<','DoubleSharedEnvironment<').replace('env{p,flags};','env{p,flags,profiles};')
        body=once(body,old,new);binding['emitted']=new
    (src/'native_phase_rebinding.cuh').write_text(body)
    (src/'native_double_environment.cuh').write_bytes(Path(__file__).with_name('cuda_native_double_environment.cuh').read_bytes())
    (src/'native_group_profiles.cuh').write_text(STAGE)
    c=(src/'asymmetric_native.cuh').read_text();c=once(c,'#include "native_phase_environment.cuh"','#include "native_phase_environment.cuh"\n#include "native_double_environment.cuh"\n#include "native_group_profiles.cuh"');(src/'asymmetric_native.cuh').write_text(c)
    kernel=(src/'native_closed_pipeline_kernel.cuh').read_text();a,b,worker=block(kernel,'void AsymmetricClosedStep(')
    worker=worker.replace('threadIdx.x','LocalThread()').replace('__syncthreads();','Join();')
    worker=once(worker,' using namespace closed_pipeline;',' using namespace closed_pipeline;\n GroupStageProfiles(p,CommonProfiles());__syncthreads();\n if(int(blockIdx.x)*2+Group()>=q.worker_count)return;')
    kernel=kernel[:a]+worker+kernel[b:]
    kernel=once(kernel,'__launch_bounds__(512,2)','__launch_bounds__(1024,1)')
    kernel=once(kernel,'closed_pipeline::SharedBytes==49504','closed_pipeline::SharedBytes==101312')
    kernel=once(kernel,'<<<q.worker_count,512,','<<<(q.worker_count+1)/2,1024,')
    (src/'native_closed_pipeline_kernel.cuh').write_text(kernel)
    e=(src/'native_extension.cc').read_text();e=once(e,'module.attr("moving_phase_rebinding_shared_bytes")=49504;','module.attr("moving_phase_rebinding_shared_bytes")=101312;module.attr("moving_grouped_workers")=2;');(src/'native_extension.cc').write_text(e)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    proof.update(phase_grouped_worker=True,phase_group_count=2,phase_group_stride=49504,phase_common_profiles_offset=99008,phase_profile_words=576,phase_shared_bytes=101312,phase_named_barriers=[1,2],phase_hardware_threads=1024,consumer_sha256=sha(src/'asymmetric_native.cuh'),extension_sha256=sha(src/'native_extension.cc'),
                 phase_group_contract='Only one full-CTA barrier, before any group may return. All later joins name barrier1/2 with512participants. Two disjoint49504byte frames, common2304byte immutablea/b profiles, logicalworkeraccounting including odd finalgroup.')
    for name in (*proof['helpers'],'native_double_environment.cuh','native_group_profiles.cuh'):proof['helpers'][name]=sha(src/name)
    for name in ('closed-pipeline-compiler-proof.json','phase-rebinding-compiler-proof.json'):(output.parent/name).write_text(json.dumps(proof,indent=2)+'\n')
    return {k:v for k,v in proof.items() if k not in ('body_bindings','phase_region_bindings')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
