"""Stage traced phase-local registers and direct shared ghost-family imports."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.build_cuda_closed_pipeline import stage as closed_stage
from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_trace_phase_cycle import block


H_OLD='''if(field<3)return h[field][j];const int c=field-3;
      if(!dz&&!ddy&&!ddx)return e[c][j];return frame[c*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];'''
H_NEW='''if(field<3)return h[field][j];const int c=field-3;
      return frame[c*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];'''
E_OLD='''if(field>=3)return e[field-3][j];if(!dz&&!ddy&&!ddx)return h[field][j];return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];'''
E_NEW='''if(field>=3)return e[field-3][j];return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];'''


def stage(source,output,expected):
    output=Path(output).resolve();proof=closed_stage(source,output,expected,front_startup=True,single_kernel=True);src=output/'src'
    c=(src/'asymmetric_native.cuh').read_text();original=block(c,'void UniformBody(')[2]
    template=re.sub(r'\bSlot\(','PhaseSlot(',Path(__file__).with_name('cuda_native_phase_rebinding.cuh').read_text());regions=[];start=0
    for phase in range(2):
        marker='{int step;asm volatile("mov.b32 %0,%1;":';begin=original.index(marker,start);_,end,region=block(original[begin:],marker);start=begin+end
        lowered=once(region,H_OLD,H_NEW) if phase==0 else once(region,E_OLD,E_NEW)
        lowered=re.sub(r'\b(?:h|e)(?=\[)','state',lowered)
        lowered=re.sub(r'\bSlot\(','PhaseSlot(',lowered)
        emitted=[]
        for i,(policy,uniform) in enumerate(((2,'false'),(4,'false'),(5,'true'))):
            condition='if(!(flags&62u))' if i==0 else 'else if(!(flags&46u))' if i==1 else 'else'
            emitted.append(condition+'{'+f'PhaseEnvironment<CubeN,{policy},{uniform}>env{{p,flags}};'+lowered[1:-1]+'}')
        generated='\n'.join(emitted);template=once(template,'// BEAMZ_PHASE_'+('H' if phase==0 else 'E')+'_REGION',generated)
        regions.append(dict(phase=phase,original=region,provider_lowered=lowered,emitted=generated))
    c=once(c,'#include "native_closed_pipeline_kernel.cuh"','#include "native_phase_environment.cuh"\n#include "native_phase_rebinding.cuh"\n#include "native_closed_pipeline_kernel.cuh"')
    (src/'asymmetric_native.cuh').write_text(c)
    control=(src/'native_closed_pipeline_control.cuh').read_text();control=once(control,'constexpr size_t ColdOffset=62144;','constexpr size_t ColdOffset=49280;');(src/'native_closed_pipeline_control.cuh').write_text(control)
    kernel=(src/'native_closed_pipeline_kernel.cuh').read_text();kernel=once(kernel,'__global__ __launch_bounds__(1024,1) void AsymmetricClosedStep','__global__ __launch_bounds__(512,2) void AsymmetricClosedStep')
    kernel=once(kernel,'  else ClosedDispatch<Depth,Policy,CubeN>(p);','  else PhaseRebindingBody<Depth,CubeN,512>(p);')
    kernel=once(kernel,'static_assert(closed_pipeline::ColdOffset>=ResearchSourceSharedBytes<5>,"Queue cold state aliases native profiles");','static_assert(closed_pipeline::ColdOffset>=3*PAD*sizeof(float)+8*sizeof(size_t)+2*sizeof(int),"Queue state aliases phase field/bases/window");\n static_assert(closed_pipeline::SharedBytes==49504,"Bound two-CTA shared footprint");')
    anchor='auto status=cudaFuncSetAttribute(AsymmetricClosedStep<Depth,Policy,CubeN>,cudaFuncAttributeMaxDynamicSharedMemorySize,closed_pipeline::SharedBytes);'
    replacement='auto status=cudaFuncSetAttribute(AsymmetricClosedStep<Depth,Policy,CubeN>,cudaFuncAttributePreferredSharedMemoryCarveout,100);if(status!=cudaSuccess)return status;\n status=cudaFuncSetAttribute(AsymmetricClosedStep<Depth,Policy,CubeN>,cudaFuncAttributeMaxDynamicSharedMemorySize,closed_pipeline::SharedBytes);'
    kernel=once(kernel,anchor,replacement);kernel=once(kernel,'<<<q.worker_count,1024,closed_pipeline::SharedBytes,stream>>>','<<<q.worker_count,512,closed_pipeline::SharedBytes,stream>>>');(src/'native_closed_pipeline_kernel.cuh').write_text(kernel)
    e=(src/'native_extension.cc').read_text();anchor='Check(cudaDeviceGetAttribute(&closed.max_workers,cudaDevAttrMultiProcessorCount,closed_device),"closed pipeline SM count");'
    e=once(e,anchor,anchor+'closed.max_workers*=2;')
    e=once(e,'module.attr("moving_closed_pipeline_compiled")=true;','module.attr("moving_closed_pipeline_compiled")=true;module.attr("moving_phase_rebinding_compiled")=true;module.attr("moving_phase_rebinding_threads")=512;module.attr("moving_phase_rebinding_shared_bytes")=49504;')
    (src/'native_extension.cc').write_text(e)
    env=Path(__file__).with_name('cuda_native_phase_environment.cuh').read_bytes();(src/'native_phase_environment.cuh').write_bytes(env);(src/'native_phase_rebinding.cuh').write_text(template)
    proof.update(phase_rebinding=True,phase_single_register_bank=True,phase_threads=512,phase_resident_target=2,phase_shared_bytes=49504,phase_swizzle=True,consumer_sha256=hashlib.sha256(c.encode()).hexdigest(),extension_sha256=hashlib.sha256(e.encode()).hexdigest(),phase_region_bindings=regions,
                 phase_register_plan='Hprivate during H math / Eshared. Eprivate during E math / Hshared. Capture oldE before R0, explicit volatile Hreload before B3. Direct H/E ghost scatters into final shared family, no LowX scratch.',
                 phase_physics_fact='General Policy1/5 E paths use captured all-trajectory UniformE program certificate; bulk/source/observation optional flag64 retains packed material fallback.',
                 phase_publication='H trail/final stores move to owned-H frame write before E math; disjoint field/version allocations; readiness still after all H/E writes/imports/raw observations and all-writer fence/CTAjoin.')
    for name in (*proof['helpers'],'native_phase_environment.cuh','native_phase_rebinding.cuh'):proof['helpers'][name]=hashlib.sha256((src/name).read_bytes()).hexdigest()
    (output.parent/'closed-pipeline-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    (output.parent/'phase-rebinding-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return {k:v for k,v in proof.items() if k not in ('body_bindings','phase_region_bindings')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
