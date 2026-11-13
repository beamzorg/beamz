"""Map each tile to eight warps with two private Y groups and sixteen slots.

The paired CTA has512threads and one resident block, allowing128registers per
thread instead of64. Spatial coverage and phase ownership stay unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.build_cuda_paired_handoff import stage as paired_stage
from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_trace_phase_cycle import block


MAPPING=r'''PhasePoint PhasePrivateLogical(int thread,int j){
 constexpr int ZGroups=T/(CY*CX)>0?T/(CY*CX):1,LocalZ=CZ/ZGroups,YRows=T/(CY*CX)>0?CY:T/CX;
 return {thread/(CY*CX)+(j%LocalZ)*ZGroups,((thread/CX)&(CY-1))+(j/LocalZ)*YRows,thread&(CX-1)};
}'''


def map_region(region,phase):
    if phase==0:
        region=once(region,'const int dy=(sy-hy)&15,dx=(sx-hx)&31,y=oy+low+dy,x=ox+low+dx;','const int dx=(sx-hx)&31,x=ox+low+dx;')
        region=once(region,'const int zs=zbase+j*ZStep,z=oz+low+((zs-hz)&7);','const auto point=PhasePrivateLogical<T>(closed_pipeline::LocalThread(),j);const int point_y=point.y,zs=point.z,z=oz+low+((zs-hz)&7),dy=(point_y-hy)&15,y=oy+low+dy;')
    else:
        region=once(region,'const int y=oy+elow+((sy-ehy)&15),x=ox+elow+((sx-ehx)&31);','const int x=ox+elow+((sx-ehx)&31);')
        region=once(region,'const int zs=zbase+j*ZStep,z=oz+elow+((zs-ehz)&7);','const auto point=PhasePrivateLogical<T>(closed_pipeline::LocalThread(),j);const int point_y=point.y,zs=point.z,z=oz+elow+((zs-ehz)&7),y=oy+elow+((point_y-ehy)&15);')
    return re.sub(r'\bsy\b','point_y',region)


def stage(source,output,expected):
    output=Path(output).resolve();paired_stage(source,output,expected);src=output/'src';proof=json.loads((output.parent/'paired-handoff-compiler-proof.json').read_text())
    body=(src/'native_phase_rebinding.cuh').read_text()
    for binding in proof['phase_region_bindings']:
        old=binding['emitted'];lowered=map_region(binding['provider_lowered'],binding['phase'])
        generated=old.replace(binding['provider_lowered'][1:-1],lowered[1:-1])
        if generated==old:raise ValueError('Mapped canonical phase region required')
        body=once(body,old,generated);binding['provider_lowered_base']=binding['provider_lowered'];binding['provider_lowered']=lowered;binding['emitted']=generated
    # Only the remaining four storage loops still use the original Z-only map.
    marker='for(int j=0;j<Slots;++j)';position=0;changes=0
    while True:
        try:start=body.index(marker,position)
        except ValueError:break
        _,end,loop=block(body[start:],marker);end+=start
        if 'const int zs=zbase+j*ZStep;' in loop:
            updated=once(loop,'const int zs=zbase+j*ZStep;','const auto point=PhasePrivateLogical<T>(closed_pipeline::LocalThread(),j);const int zs=point.z,point_y=point.y;')
            updated=re.sub(r'\bsy\b','point_y',updated);body=body[:start]+updated+body[end:];end=start+len(updated);changes+=1
        position=end
    if changes!=4:raise ValueError('Exactly init/Hpublish/Hreload/Epublish mappings required:'+str(changes))
    a,b,_=block(body,'PhasePoint PhasePrivateLogical(');body=body[:a]+MAPPING+body[b:]
    body=once(body,'template<int Depth,int CubeN,int T=512>','template<int Depth,int CubeN,int T=256>')
    body=once(body,'T==512,"First prototype preserves native geometry and doubles resident CTA opportunities"','T==256,"Two private Ygroups permit16points per256thread worker"')
    (src/'native_phase_rebinding.cuh').write_text(body)
    control=(src/'native_closed_pipeline_control.cuh').read_text();control=once(control,'int(threadIdx.x)>>9','int(threadIdx.x)>>8');control=once(control,'int(threadIdx.x)&511','int(threadIdx.x)&255');control=once(control,'bar.sync %0, 512;','bar.sync %0, 256;');(src/'native_closed_pipeline_control.cuh').write_text(control)
    profiles=(src/'native_group_profiles.cuh').read_text();profiles=once(profiles,'if(threadIdx.x<576){const int i=threadIdx.x,slot=i/48,kind=(i/24)&1,packed=i%24;','if(threadIdx.x<512)for(int i=threadIdx.x;i<576;i+=512){const int slot=i/48,kind=(i/24)&1,packed=i%24;');(src/'native_group_profiles.cuh').write_text(profiles)
    kernel=(src/'native_closed_pipeline_kernel.cuh').read_text();kernel=once(kernel,'__launch_bounds__(1024,1) void AsymmetricClosedStep','__launch_bounds__(512,1) void AsymmetricClosedStep');kernel=once(kernel,'PhaseRebindingBody<Depth,CubeN,512>(p);','PhaseRebindingBody<Depth,CubeN,256>(p);');kernel=once(kernel,'<<<q.worker_count,1024,','<<<q.worker_count,512,');(src/'native_closed_pipeline_kernel.cuh').write_text(kernel)
    e=(src/'native_extension.cc').read_text();e=once(e,'module.attr("moving_phase_rebinding_threads")=512;','module.attr("moving_phase_rebinding_threads")=256;module.attr("moving_paired_private_Y_groups")=2;module.attr("moving_paired_hardware_threads")=512;');(src/'native_extension.cc').write_text(e)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    proof.update(phase_threads=256,phase_hardware_threads=512,phase_private_Y_groups=2,phase_private_points_per_thread=16,consumer_sha256=sha(src/'asymmetric_native.cuh'),extension_sha256=sha(src/'native_extension.cc'),
                 phase_mapping='thread=(8Yrows,32Xlanes), j=(Ygroup*8+Z); z=j%8,y=thread/32+(j/8)*8,x=thread%32. Every4096 cachepoint owned once. H/E coordinate/providers and four storage loops derive from this same map; field scalar arithmetic unchanged.')
    for name in proof['helpers']:proof['helpers'][name]=sha(src/name)
    for name in ('closed-pipeline-compiler-proof.json','phase-rebinding-compiler-proof.json','paired-handoff-compiler-proof.json'):(output.parent/name).write_text(json.dumps(proof,indent=2)+'\n')
    return {k:v for k,v in proof.items() if k not in ('body_bindings','phase_region_bindings')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
