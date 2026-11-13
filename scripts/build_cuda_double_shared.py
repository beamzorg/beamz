"""Lower traced same-family center reads to two-barrier shared in-place phases."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.build_cuda_closed_pipeline import stage as closed_stage
from scripts.build_cuda_partial_plane_pipeline import once
from scripts.build_cuda_phase_rebinding import H_OLD,E_OLD
from scripts.cuda_trace_phase_cycle import block


def stage(source,output,expected):
    output=Path(output).resolve();proof=closed_stage(source,output,expected,front_startup=True,single_kernel=True);src=output/'src';c=(src/'asymmetric_native.cuh').read_text();native=block(c,'void UniformBody(')[2]
    body=re.sub(r'\bSlot\(','PhaseSlot(',Path(__file__).with_name('cuda_native_double_shared.cuh').read_text());position=0;regions=[]
    for phase in range(2):
        marker='{int step;asm volatile("mov.b32 %0,%1;":';a=native.index(marker,position);_,b,region=block(native[a:],marker);position=a+b
        lowered=once(region,H_OLD if phase==0 else E_OLD,'return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];')
        lowered=once(lowered,'h[c][j]=value;' if phase==0 else 'e[c][j]=value;',f'frame[(c+{phase*3})*PAD+Slot(zs,sy,sx)]=value;')
        lowered=re.sub(r'\bSlot\(','PhaseSlot(',lowered)
        generated='\n'.join(('if(!(flags&62u))' if i==0 else 'else if(!(flags&46u))' if i==1 else 'else')+'{'+f'DoubleSharedEnvironment<CubeN,{policy},{uniform}>env{{p,flags,profiles}};'+lowered[1:-1]+'}' for i,(policy,uniform) in enumerate(((2,'false'),(4,'false'),(5,'true'))))
        body=once(body,'// BEAMZ_DOUBLE_'+('H' if phase==0 else 'E')+'_REGION',generated);regions.append(dict(phase=phase,original=region,provider_lowered=lowered,emitted=generated))
    # Reuse only the actual geometry/load primitive helpers, not the register body.
    helpers=re.sub(r'\bSlot\(','PhaseSlot(',Path(__file__).with_name('cuda_native_phase_rebinding.cuh').read_text());helpers=helpers[:helpers.index('template<int Depth,int CubeN,int T=512>')]
    c=once(c,'#include "native_closed_pipeline_kernel.cuh"','#include "native_phase_environment.cuh"\n#include "native_double_environment.cuh"\n#include "native_double_geometry.cuh"\n#include "native_double_shared.cuh"\n#include "native_closed_pipeline_kernel.cuh"');(src/'asymmetric_native.cuh').write_text(c)
    control=(src/'native_closed_pipeline_control.cuh').read_text();control=once(control,'constexpr size_t ColdOffset=62144;','constexpr size_t ColdOffset=100736;');control=once(control,'static_assert(SharedBytes<64512,"Queue state must preserve native64KB carveout");','static_assert(SharedBytes==100960&&SharedBytes<=101376,"Bound shared families plus independent a/b profiles within SM86 block limit");');(src/'native_closed_pipeline_control.cuh').write_text(control)
    kernel=(src/'native_closed_pipeline_kernel.cuh').read_text();kernel=once(kernel,'  else ClosedDispatch<Depth,Policy,CubeN>(p);','  else DoubleSharedBody<Depth,CubeN,1024>(p);')
    kernel=once(kernel,'static_assert(closed_pipeline::ColdOffset>=ResearchSourceSharedBytes<5>,"Queue cold state aliases native profiles");','static_assert(closed_pipeline::ColdOffset>=6*PAD*sizeof(float)+8*sizeof(size_t)+2*sizeof(int)+576*sizeof(float),"Queue state aliases double field/profile storage");')
    anchor='auto status=cudaFuncSetAttribute(AsymmetricClosedStep<Depth,Policy,CubeN>,cudaFuncAttributeMaxDynamicSharedMemorySize,closed_pipeline::SharedBytes);'
    kernel=once(kernel,anchor,'auto status=cudaFuncSetAttribute(AsymmetricClosedStep<Depth,Policy,CubeN>,cudaFuncAttributePreferredSharedMemoryCarveout,100);if(status!=cudaSuccess)return status;\n status=cudaFuncSetAttribute(AsymmetricClosedStep<Depth,Policy,CubeN>,cudaFuncAttributeMaxDynamicSharedMemorySize,closed_pipeline::SharedBytes);');(src/'native_closed_pipeline_kernel.cuh').write_text(kernel)
    for name,value in [('native_double_geometry.cuh',helpers),('native_double_shared.cuh',body),('native_phase_environment.cuh',Path(__file__).with_name('cuda_native_phase_environment.cuh').read_text()),('native_double_environment.cuh',Path(__file__).with_name('cuda_native_double_environment.cuh').read_text())]:(src/name).write_text(value)
    proof.update(double_shared=True,double_shared_bytes=100960,double_barriers_per_step=2,double_region_bindings=regions,consumer_sha256=hashlib.sha256(c.encode()).hexdigest(),double_profile_words=576,
                 double_plan='All6fields in shared; same-family reads are owned centers, opposite-family reads may be neighbors. H phase overwrites H in-place, E overwrites E in-place. Hbarrier and finalwriterfence/CTAjoin suffice; no persistent field registers or permanent ghost scratch.',
                 double_program_fact='Captured allPolicy1/5 UniformE certificate, source20 plane fact, lossless isotropic H, canonical six field staggering; original complete fallback unchanged.')
    for name in (*proof['helpers'],'native_double_geometry.cuh','native_double_shared.cuh','native_phase_environment.cuh','native_double_environment.cuh'):proof['helpers'][name]=hashlib.sha256((src/name).read_bytes()).hexdigest()
    (output.parent/'closed-pipeline-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n');(output.parent/'double-shared-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return {k:v for k,v in proof.items() if k not in ('body_bindings','double_region_bindings')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
