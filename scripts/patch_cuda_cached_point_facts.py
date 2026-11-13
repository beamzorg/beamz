"""Cache traced physical-point facts in the frozen fast CUDA schedule.

No field, psi, coefficient or source value is cached by this pass. The point's
support/PML/wall classification is computed when entering an owned cache site
and reused while the same physical coordinate remains resident. Source and
observer effects, shared barriers and trail publications retain their order.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from scripts.cuda_trace_phase_cycle import block
from scripts.cuda_trace_point_facts import CPP

PARENT='a7104fb39695c2a3d8df0bac3db20d647957851e89c655a81125f601742d25c0'


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Unique point-fact binding required:'+old[:100])
    return text.replace(old,new,1)


def environment(text):
    signature='template<int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5),bool UniformE=false,int SourceMask=-1,bool PlanePair=false,int ActiveAxes=-1,bool CachedProfiles=false> struct Environment {'
    signature_new=signature.replace('bool CachedProfiles=false>','bool CachedProfiles=false,bool CachedPointFacts=false>')
    text=once(text,signature,'__device__ __forceinline__ unsigned NativeSupportMask(const Parameters&,int,int,int);\n'+CPP+'\n'+signature_new)
    text=once(text,'const Parameters&p;uint8_t flags;const float*profiles=nullptr;',
              'const Parameters&p;uint8_t flags;const float*profiles=nullptr;uint32_t point_tag=0;')
    old='template<int N,int ActiveAxes=-1>struct FixedCubeCachedEnvironment:'
    text=once(text,old,'template<int N,int ActiveAxes=-1,bool CachedPointFacts=false>struct FixedCubeCachedEnvironment:')
    base='Environment<1,false,true,1,true,-1,false,ActiveAxes,true>'
    if text.count(base)!=2:raise ValueError('Fixed-cube base aliases changed')
    text=text.replace(base,base[:-1]+',CachedPointFacts>')
    for signature in ('struct Environment {','struct FixedCubeCachedEnvironment:'):
        begin,end,body=block(text,signature)
        a,b,method=block(body,'bool Valid(')
        opening=method.index('{')+1
        method=method[:opening]+'\n  if constexpr(CachedPointFacts&&!SupportSafe)return (this->point_tag&(1u<<field))!=0;'+method[opening:]
        body=body[:a]+method+body[b:]
        a,b,method=block(body,'int Packed(');opening=method.index('{')+1
        method=method[:opening]+'\n  if constexpr(CachedPointFacts)return NativePointPacked(this->point_tag,slot);'+method[opening:]
        body=body[:a]+method+body[b:]
        a,b,method=block(body,'bool LocalActive(')
        marker='else if(!(flags&(2u<<axis)))return false;'
        method=once(method,marker,marker+'\n  if constexpr(CachedPointFacts)return NativePointLocalActive(this->point_tag,slot);')
        body=body[:a]+method+body[b:]
        a,b,method=block(body,'float Difference(')
        condition='coordinate==0||coordinate=='+('N' if 'FixedCube' in signature else 'p.physical_n[axis]')
        method=once(method,'if('+condition+')return 0.f;',
                    'if(CachedPointFacts?NativePointElectricWall(this->point_tag,axis):('+condition+'))return 0.f;')
        body=body[:a]+method+body[b:]
        text=text[:begin]+body+text[end:]
    return text


def instrument_body(text,signature):
    begin,end,body=block(text,signature);paired='Paired' in signature
    if paired:
        old='Environment<Layout,Fast,IsotropicH,Policy,UniformE,SourceMask,PlanePair,ActiveAxes,CachedProfiles>env'
    elif 'FixedCube' in signature:old='FixedCubeCachedEnvironment<N,ActiveAxes>env'
    else:old='Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes,CachedProfiles>env'
    enabled='true' if 'FixedCube' in signature else '(Layout==1)'
    body=once(body,old,old.replace('>env',','+enabled+'>env'))
    lines=body.splitlines();indices=[i for i,line in enumerate(lines) if '>env{' in line]
    if len(indices)!=1:raise ValueError('One native environment capture required')
    i=indices[0]
    lines[i+1:i+1]=[' constexpr bool UsePointFacts=(Layout==1)&&(decltype(env)::HasCpml||!decltype(env)::SupportSafe);',
                    ' uint32_t point_facts[UsePointFacts?Slots:1];']
    body='\n'.join(lines)
    loop='for(int n=(Fast?1:window[0]);n<=(Fast?Depth:window[1]);++n){'
    body=once(body,loop,loop+'\n  const bool point_facts_initial=UsePointFacts&&(n==(Fast?1:window[0]));')
    y='ys' if paired else 'sy';index='si' if paired else 'j'
    refresh=f'''if constexpr(UsePointFacts){{
      if(point_facts_initial||((zs-hz)&(CZ-1))==1||(({y}-hy)&(CY-1))==1||((sx-hx)&(CX-1))==1)
       point_facts[{index}]=EncodeNativePointFacts(p,z,y,x);
      env.point_tag=point_facts[{index}];
     }}
     '''
    bind=f'if constexpr(UsePointFacts)env.point_tag=point_facts[{index}];\n     '
    pattern=r'(if\(zs!=hz&&'+y+r'!=hy&&sx!=hx\)\{\s*)(?=const auto read=)'
    matches=list(re.finditer(pattern,body));expected=3 if paired else 1
    if len(matches)!=expected:raise ValueError('Actual H/E point event binding changed:'+signature)
    count=0
    def inject(match):
        nonlocal count
        insertion=refresh if count==0 else bind;count+=1
        return match[1]+insertion
    body=re.sub(pattern,inject,body)
    if not paired:
        pattern=r'(if\(zs!=ehz&&sy!=ehy&&sx!=ehx\)\{\s*)(?=const auto read=)'
        if len(list(re.finditer(pattern,body)))!=1:raise ValueError('Actual E point event changed')
        body=re.sub(pattern,lambda m:m[1]+bind,body)
    return text[:begin]+body+text[end:]


def stage(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or source in output.parents or output in source.parents:raise ValueError('New disjoint research source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if hashlib.sha256(original.encode()).hexdigest()!=PARENT:raise ValueError('Frozen complete temporal control required')
    text=environment(original)
    for signature in ('void UniformBody(','void FixedCubeCachedBody(','void SourcePendingPairedBody('):text=instrument_body(text,signature)
    shutil.copytree(source,output);(output/'src/asymmetric_native.cuh').write_text(text)
    extension=(output/'src/native_extension.cc').read_text()
    extension=once(extension,'NB_MODULE(_cuda_native, module) {','NB_MODULE(_cuda_native, module) {\n module.attr("moving_cached_point_facts")=true;')
    (output/'src/native_extension.cc').write_text(extension)
    proof=dict(schema='beamz.native-point-fact-lifetime-lowering.v1',parent_consumer_sha256=PARENT,
               consumer_sha256=hashlib.sha256(text.encode()).hexdigest(),extension_sha256=hashlib.sha256(extension.encode()).hexdigest(),
               tag_helper_sha256=hashlib.sha256(CPP.encode()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               cache=[8,16,32],extra_private_fact_words='4on1024-thread common paths;8on512-thread fallback; optimized away for safe/no-CPML classes',
               lifetime='Refresh on first owned H update or lower entry surface; same physical point H/E mask and CPML/wall facts reused until exit',
               requires='Actual native predicate equivalence proof, point residency proof, register admission and full coupled GPU suite',
               floating_point_arithmetic_and_effect_order_unchanged=True,enabled_coefficient_layouts=[1],
               block_packed_fallback='Original predicates and schedule; fact-cache code statically disabled',expected_kernels=68,CUDA_executed=False)
    (output.parent/'point-facts-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output)))
