"""Stage a traced diagonal-warp register cache for certified uniform bulk.

All numerical scalar cells, CPML/source/general bodies and native rank owners
remain frozen. The traced read set determines register/shuffle/shared routes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_space_time_ir import FIELDS,yee_operator
from scripts.cuda_trace_phase_cycle import block


def component_region(phase,component):
    output=FIELDS[3*phase+component]
    update=next(u for u in yee_operator(cpml_axes=()).updates if u.output.field==output)
    reads=[r for r in sorted(update.reads) if r.field in FIELDS]
    lines=['{'];routes=[]
    for i,r in enumerate(reads):
        field=FIELDS.index(r.field);dz,dy,dx=r.offset;name=f'r{i}'
        if phase==0 and not any(r.offset) or phase==1 and field>=3:
            bank='h' if field<3 else 'e';c=field%3;boundary='boundary_'+bank
            lines.append(f' float {name}=DiagonalShuffle(j>0?{bank}[{c}][j-1]:0.f,sx);')
            lines.append(f' if constexpr(T==1024){{if(j==0&&zbase>0&&sy>0&&sx>0){name}={boundary}[{c}*CY*CX+DiagonalBoundarySlot(sy-1,sx-1)];}}')
            route='previous-diagonal'
        elif phase==0 and r.offset==(1,0,0):
            lines.append(f' const float {name}=DiagonalShuffle(e[{field-3}][j],sx);');route='previous-forward-Z'
        else:
            if phase==0:
                value=f'frame[{field-3}*PAD+DiagonalSlot(zs-1+({dz}),sy-1+({dy}),sx-1+({dx}))]';route='previous-shared-XY'
            elif not any(r.offset):value=f'h[{field}][j]';route='current-private-center'
            elif r.offset==(-1,0,0):
                value=f'(j>0?h[{field}][j-1]:(T==1024?boundary_h[{field}*CY*CX+DiagonalBoundarySlot(sy,sx)]:0.f))';route='current-private-or-boundary-Z'
            else:value=f'frame[{field}*PAD+DiagonalSlot(zs+({dz}),sy+({dy}),sx+({dx}))]';route='current-shared-XY'
            name=value
        routes.append(dict(field=field,version=r.version,offset=r.offset,value=name,route=route))
    lines+=[' if(zs>0&&sy>0&&sx>0){','  auto read=[&](int field,int version,int dz,int dy,int dx){']
    for r in routes:
        dz,dy,dx=r['offset'];lines.append(f"   if(field=={r['field']}&&dz=={dz}&&dy=={dy}&&dx=={dx})return {r['value']};")
    lines+=['   return 0.f;','  };',f'  {"h" if phase==0 else "e"}[{component}][j]=env.Valid({3*phase+component},z,y,x)?beamz::cuda::trace_native::AdvanceNativeCell<{phase},{component},false>(env,read,n,z,y,x):0.f;',' }','}']
    return '\n'.join(lines),dict(output=output,phase=phase,component=component,routes=routes)


def stage(source,output,expected,*,observers=False):
    source,output=Path(source).resolve(),Path(output).resolve();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if output.exists() or source==output or source in output.parents or output in source.parents:raise ValueError('New disjoint source tree required')
    if sha(source/'src/asymmetric_native.cuh')!=expected:raise ValueError('Frozen native parent required')
    shutil.copytree(source,output);src=output/'src';c=(src/'asymmetric_native.cuh').read_text();original=c
    body=Path(__file__).with_name('cuda_native_diagonal_logical.cuh').read_text();bindings=[]
    for phase in range(2):
        pieces=[]
        for component in range(3):
            region,binding=component_region(phase,component);pieces.append(region);binding['emitted']=region;bindings.append(binding)
        body=once(body,'// BEAMZ_DIAGONAL_'+('H' if phase==0 else 'E')+'_COMPONENTS','\n'.join(pieces))
    a,b,old=block(c,'void UniformBody(');position=c.rfind('template<',0,a)
    template=c[position:a]
    replacement='''void UniformBody(const Parameters&p,const int32_t*tiles){
 if constexpr(Layout==1&&IsotropicH&&UniformE&&!CachedProfiles&&DIAGONAL_SCOPE)
  DiagonalLogicalBody<Depth,Layout,Fast,IsotropicH,Policy,UniformE,1024>(p,tiles);
 else DiagonalLegacyUniformBody<Depth,Layout,Fast,IsotropicH,Policy,UniformE,ActiveAxes,CachedProfiles>(p,tiles);
}'''
    replacement=replacement.replace('DIAGONAL_SCOPE','((Fast&&Policy==0)||Policy==2)' if observers else '(Fast&&Policy==0)')
    legacy=old.replace('void UniformBody(','void DiagonalLegacyUniformBody(',1)
    c=c[:position]+'#include "native_diagonal_logical.cuh"\n'+template+legacy+'\n'+template+replacement+c[b:]
    # Only kernels which can select the new body need its additional snapshots.
    a,b,launch=block(c,'inline cudaError_t Launch(');new=once(launch,'const auto kernel=Asymmetric<Depth,Layout,Fast,IsotropicH,Policy>;','const auto kernel=Asymmetric<Depth,Layout,Fast,IsotropicH,Policy>;\n constexpr size_t LaunchBytes=(Layout==1&&Fast&&IsotropicH&&Policy==0)?DiagonalSharedBytes<1024>:SharedBytes;')
    new=once(new,'cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes','cudaFuncAttributeMaxDynamicSharedMemorySize,LaunchBytes');new=once(new,'(Fast?1024:512),SharedBytes,stream','(Fast?1024:512),LaunchBytes,stream');c=c[:a]+new+c[b:]
    if observers:
        a,b,observer=block(c,'inline cudaError_t LaunchObservers(');new=once(observer,'cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes','cudaFuncAttributeMaxDynamicSharedMemorySize,DiagonalSharedBytes<1024>');new=once(new,'1024,SharedBytes,stream','1024,DiagonalSharedBytes<1024>,stream');c=c[:a]+new+c[b:]
    (src/'asymmetric_native.cuh').write_text(c);(src/'native_diagonal_logical.cuh').write_text(body)
    e=(src/'native_extension.cc').read_text();e=once(e,'NB_MODULE(_cuda_native, module) {','NB_MODULE(_cuda_native, module) {\n module.attr("moving_diagonal_logical_bulk")=true;module.attr("moving_diagonal_shared_bytes")=61568;');(src/'native_extension.cc').write_text(e)
    proof=dict(schema='beamz.diagonal-logical-native-compiler.v1',diagonal_bulk=True,diagonal_observers=observers,expected_kernels=68,parent_consumer_sha256=expected,consumer_sha256=sha(src/'asymmetric_native.cuh'),extension_sha256=sha(src/'native_extension.cc'),helpers={'native_diagonal_logical.cuh':sha(src/'native_diagonal_logical.cuh')},component_bindings=bindings,legacy_uniform_body=old,uniform_wrapper=replacement,
               shared_bytes=61568,numerics='Frozen full FP32 scalar cell/Environment unchanged. Read provider generated from canonical read sets. Uniform bulk P0Fast only, plus P2observers if explicitly enabled; all CPML/source/material/other policies retain exact frozen bodies.',
               scheduler='Original four rank owners, rank policy streams and native packet/epoch allocation. Four shared read-retirement/publication joins, fullwarp shuffles outside predicates, descending contiguous private Z.',GPU_executed=False)
    (output.parent/'diagonal-logical-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return {k:v for k,v in proof.items() if k not in ('component_bindings','legacy_uniform_body','uniform_wrapper')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);p.add_argument('--observers',action='store_true');a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256,observers=a.observers)))
