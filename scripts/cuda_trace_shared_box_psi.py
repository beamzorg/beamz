"""Place single-axis CPML histories in registers freed by shared field storage."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,block,sha
from scripts.patch_cuda_traced_lifetime_private_psi import POINT
from scripts.cuda_native_phase_state_fact import generate_guard


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Reviewed anchor changed: '+old[:100])
    return text.replace(old,new,1)


def stage(source,output,phases=3):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:
        raise ValueError('New disjoint copied research source required')
    proof=json.loads((source.parent/'shared-box-proof.json').read_text())
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=proof['consumer_sha256']:raise ValueError('Shared-box source identity changed')
    text=original
    text=once(text,'int physical_n[3];float h_source;', 'int physical_n[3];bool private_psi_independent;float h_source;')
    text=once(text,'out->macro_step=macro;out->h_source=', 'out->macro_step=macro;out->private_psi_independent=w.private_psi_state_independent;out->h_source=')
    marker='template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1,bool CachedProfiles=false>'
    text=once(text,marker,marker[:-1]+',bool CachePsi=false>')
    text=once(text,'template<int Depth,int N,int ActiveAxes=-1>\n__device__ __forceinline__ void FixedCubeCachedBody',
              'template<int Depth,int N,int ActiveAxes=-1,bool CachePsi=false>\n__device__ __forceinline__ void FixedCubeCachedBody')
    for signature in ('void UniformBody(','void FixedCubeCachedBody('):
        begin,end,body=block(text,signature)
        body=once(body,' const int sx=threadIdx.x&31,',
                  ' PrivatePsiHistory<CachePsi,Slots>history;\n const int sx=threadIdx.x&31,')
        for phase,prefix,clock_prefix in ((0,'h',''),(1,'e','e')):
            hz,hy,hx=[clock_prefix+n for n in ('hz','hy','hx')]
            old=f'    if(zs!={hz}&&sy!={hy}&&sx!={hx}){{\n     const auto read='
            new=f'''    if(zs!={hz}&&sy!={hy}&&sx!={hx}){{
     auto point_env=[&] __device__(){{
      if constexpr(CachePsi&&({phases}&{1<<phase})){{
       const int dz=(zs-{hz})&(CZ-1),dy=(sy-{hy})&(CY-1),dx=(sx-{hx})&31;
       return PrivatePsiHistoryPoint<decltype(env)>(env,history.{prefix}a[j],history.{prefix}b[j],
        n==window[0]||dz==1||dy==1||dx==1,n==Depth||dz==BZ||dy==BY||dx==BX);
      }}else return env;
     }}();
     const auto read='''
            body=once(body,old,new)
            for c in range(3):
                old=f'AdvanceNativeCell<{phase},{c},false>(env,read,n,z,y,x)'
                body=once(body,old,old.replace('(env,','(point_env,'))
        text=text[:begin]+body+text[end:]
    text=once(text,marker[:-1]+',bool CachePsi=false>',POINT+'\n'+marker[:-1]+',bool CachePsi=false>')
    # New entries keep the uncached ones available for failed alias admission.
    marker='template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__'
    entries='''template<int Depth,int N=0>
__global__ __launch_bounds__(ResearchThreads,2) void AsymmetricSharedBoxPrivatePsi(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 switch(p.flags[tiles[blockIdx.x]]&14){
'''
    for flag,axis in ((2,1),(4,2),(8,4)):
        entries+=f' case {flag}:if constexpr(N)FixedCubeCachedBody<Depth,N,{axis},true>(p,tiles);else UniformBody<Depth,1,false,true,1,true,{axis},false,true>(p,tiles);return;\n'
    entries+=' default:if constexpr(N)FixedCubeCachedBody<Depth,N>(p,tiles);else UniformBody<Depth,1,false,true,1,true>(p,tiles);return;\n }\n}\n'
    text=once(text,marker,entries+'\n'+marker)
    for n in (512,564):
        anchor=f'  if(FixedNoIOCubeEligible<{n}>(p)){{'
        selected=f'''  if(FixedNoIOCubeEligible<{n}>(p)&&p.private_psi_independent){{
   const auto kernel=AsymmetricSharedBoxPrivatePsi<Depth,{n}>;
   status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes);if(status!=cudaSuccess)return status;
   AsymmetricSharedBoxPrivatePsi<Depth,{n}><<<uniform_count,ResearchThreads,SharedBytes,stream>>>(p,ids);return cudaGetLastError();
  }}
'''
        text=once(text,anchor,selected+anchor)
    anchor='  const auto kernel=AsymmetricNoIOUniform1024<Depth>;'
    selected='''  if(p.private_psi_independent){
   const auto kernel=AsymmetricSharedBoxPrivatePsi<Depth>;
   status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes);if(status!=cudaSuccess)return status;
   AsymmetricSharedBoxPrivatePsi<Depth><<<uniform_count,ResearchThreads,SharedBytes,stream>>>(p,ids);return cudaGetLastError();
  }
'''
    text=once(text,anchor,selected+anchor)
    header=(source/'src/native_moving.h').read_text()
    header=once(header,'  uint32_t source_presence_mask=0;','  uint32_t source_presence_mask=0;\n  bool private_psi_state_independent=false;')
    host=(source/'src/native_extension.cc').read_text()
    host=once(host,'#include "native_moving.h"','#include "native_moving.h"\n#include "runtime_private_psi_state_fact.h"')
    host=once(host,'    BeamzNativeDeepPlan observation;','''    BeamzBuffer cache_state[12]{},cache_profile[36]{};
    for(int family=0;family<2;++family){const auto&v=family?validate.program.e_ab:validate.program.h_ab;
      for(int term=0;term<6;++term){cache_state[6*family+term]=v.inputs[31+term];
        for(int kind=0;kind<3;++kind)cache_profile[18*family+3*term+kind]=v.inputs[13+3*term+kind];}}
    moving_workspace_.private_psi_state_independent=BeamzNativeIndependentPhaseState(cache_state,cache_profile);
    BeamzNativeDeepPlan observation;''')
    host=once(host,'    const bool source_selector=',
              '    append_bytes(&moving_workspace_.private_psi_state_independent,sizeof(bool));\n    const bool source_selector=')
    host=once(host,'    result["moving_source_presence_mask"]=',
              f'    result["moving_shared_box_private_psi_independent"]=moving_workspace_.private_psi_state_independent;result["moving_shared_box_private_psi_phases"]={phases};\n    result["moving_source_presence_mask"]=')
    shutil.copytree(source,output)
    for name,value in [('asymmetric_native.cuh',text),('native_moving.h',header),('native_extension.cc',host),('runtime_private_psi_state_fact.h',generate_guard()['source'])]:
        (output/'src'/name).write_text(value)
    record=dict(schema='beamz.shared-box-private-CPML.v1',phases=phases,owned_zyx=[7,7,31],
                parent_consumer_sha256=sha(original.encode()),consumer_sha256=sha(text.encode()),
                producer_sha256=sha(Path(__file__).read_bytes()),shared_bytes=50408,
                retained_psi_words_per_thread=8*phases.bit_count(),
                first_owned_reload=True,outgoing_and_final_flush=True,distinct_state_guard=True,
                expected_kernels=62,GPU_executed=False)
    (output.parent/'shared-box-private-psi-proof.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phases',type=int,choices=(1,2,3),default=3)
    a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.phases)))
