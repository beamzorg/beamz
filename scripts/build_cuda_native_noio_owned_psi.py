"""Compile complete four-color CPML history into compressed owned shared state.

Existing phase field barriers also retire the old psi readers. Two temporary
psi output colors are reused across phases. No ghost histories/profile cache.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from scripts.build_cuda_native_noio_private_psi import stage as base_stage,region as base_region
from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_trace_phase_cycle import block


def owned_region(phase,original):
    text=base_region(phase,original);prefix='h' if phase==0 else 'e'
    old=f'PrivatePsiHistoryPoint<decltype(env)> point_env(env,history.{prefix}a[j],history.{prefix}b[j],n==first||dz_owned==1||dy_owned==1||dx_owned==1,n==Depth||dz_owned==BZ||dy_owned==BY||dx_owned==BX);'
    new=f'NativeOwnedPsiPoint<decltype(env)> point_env(env,psi_a[j],psi_b[j],psi_cache+{phase*2}*NativeOwnedPsiPoints,NativeOwnedPsiIndex(dz_owned-1,dy_owned-1,dx_owned-1),n==first||dz_owned==1||dy_owned==1||dx_owned==1,n==Depth||dz_owned==BZ||dy_owned==BY||dx_owned==BX);'
    return once(text,old,new)


def stage(source,output,expected):
    base_stage(source,output,expected);output=Path(output).resolve();src=output/'src';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();proof=json.loads((output.parent/'native-noio-private-psi-compiler-proof.json').read_text());c=(src/'asymmetric_native.cuh').read_text();body=(src/'native_noio_private_psi.cuh').read_text()
    for binding in proof['wrapper_bindings']:
        a,b,_=block(c,binding['signature']);start=c.rfind('template<',0,a);c=c[:start]+c[b:];legacy='void NativeNoioLegacyUniformBody(' if 'Uniform' in binding['signature'] else 'void NativeNoioLegacyFixedCubeBody(';c=once(c,legacy,binding['signature'])
    body=once(body,'constexpr size_t NativeNoioPsiSharedBytes=3ULL*PAD*sizeof(float)+8*sizeof(size_t)+2*sizeof(int)+864*sizeof(float);','constexpr size_t NativeNoioPsiSharedBytes=(3ULL*PAD+4ULL*BZ*BY*BX)*sizeof(float)+8*sizeof(size_t)+2*sizeof(int);\nconstexpr size_t NativeNoioOwnedPsiLaunchBytes=(NativeNoioPsiSharedBytes+127)&~size_t(127);')
    body=once(body,'static_assert(NativeNoioPsiSharedBytes<=ProfileCachedSharedBytes,"Retain original NoIO shared allocation and carveout");','static_assert(NativeNoioPsiSharedBytes==101304&&NativeNoioOwnedPsiLaunchBytes==101376,"Full owned CPML state fits exact SM86 dynamic shared limit");')
    body=once(body,'extern __shared__ float frame[];size_t*bases=reinterpret_cast<size_t*>(frame+3*PAD);volatile int*window=reinterpret_cast<volatile int*>(bases+8);float*profiles=frame+3*PAD+8*sizeof(size_t)/sizeof(float)+2;','extern __shared__ float frame[];float*psi_cache=frame+3*PAD;size_t*bases=reinterpret_cast<size_t*>(psi_cache+4*NativeOwnedPsiPoints);volatile int*window=reinterpret_cast<volatile int*>(bases+8);')
    body=once(body,'StageIndependentCpmlProfiles(p,profiles);','// Independent profiles use canonical read-only global pointers.')
    body=once(body,'float state[3][Slots];PrivatePsiHistory<true,Slots>history;','float state[3][Slots],psi_a[Slots],psi_b[Slots];')
    oldenv='auto env=[&] __device__(){if constexpr(CubeN>0)return FixedCubeCachedEnvironment<CubeN,ActiveAxes>{p,flags,profiles};else return Environment<1,false,true,1,true,-1,false,ActiveAxes,true>{p,flags,profiles};}();'
    newenv='auto standard_env=[&] __device__(){if constexpr(CubeN>0)return FixedCubeCachedEnvironment<CubeN,ActiveAxes>{p,flags,nullptr};else return Environment<1,false,true,1,true,-1,false,ActiveAxes,false>{p,flags,nullptr};}();\n NativeOwnedGlobalProfiles<decltype(standard_env)>env{standard_env};'
    body=once(body,oldenv,newenv);bindings=[]
    for binding in proof['region_bindings']:
        phase=binding['phase'];emitted=owned_region(phase,binding['original']);initial='''#pragma unroll
  for(int j=0;j<Slots;++j){psi_a[j]=0.f;psi_b[j]=0.f;}
  ''';body=once(body,binding['emitted'],initial+emitted);bindings.append(dict(**{k:v for k,v in binding.items() if k!='emitted'},emitted=emitted))
    needle='''const int dz=(zs-hz)&7,z=oz+low+dz,y=oy+low+dy,x=ox+low+dx,surface=Surface(dz-1,dy-1,dx-1);
   #pragma unroll'''
    if body.count(needle)!=2:raise ValueError('Both phase publication loops must be matched')
    for phase in range(2):
        publish=f'''const int psi_index=NativeOwnedPsiIndex(dz,dy,dx);psi_cache[{phase*2}*NativeOwnedPsiPoints+psi_index]=psi_a[j];psi_cache[{phase*2+1}*NativeOwnedPsiPoints+psi_index]=psi_b[j];
   #pragma unroll'''
        pos=body.index(needle);body=body[:pos]+body[pos:].replace(needle,needle.replace('#pragma unroll',publish),1)
    body='#include "native_noio_owned_psi.cuh"\n'+body
    entries='''template<int Depth,int CubeN>
__global__ __launch_bounds__(1024,1) void AsymmetricNativeNoioOwnedPsi(const __grid_constant__ Parameters p,const int32_t*tiles){
 switch(p.flags[tiles[blockIdx.x]]&14u){
  case 2:NativeNoioPrivatePsiBody<Depth,CubeN,1>(p,tiles);return;
  case 4:NativeNoioPrivatePsiBody<Depth,CubeN,2>(p,tiles);return;
  case 8:NativeNoioPrivatePsiBody<Depth,CubeN,4>(p,tiles);return;
  default:if constexpr(CubeN>0)FixedCubeCachedBody<Depth,CubeN,-1>(p,tiles);else UniformBody<Depth,1,false,true,1,true,-1,true>(p,tiles);return;
 }
}
template<int Depth,int CubeN>
inline cudaError_t LaunchNativeNoioOwnedPsi(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count){
 const auto kernel=AsymmetricNativeNoioOwnedPsi<Depth,CubeN>;
 auto status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,NativeNoioOwnedPsiLaunchBytes);if(status!=cudaSuccess)return status;
 AsymmetricNativeNoioOwnedPsi<Depth,CubeN><<<count,1024,NativeNoioOwnedPsiLaunchBytes,stream>>>(p,ids);return cudaGetLastError();
}
'''
    anchor='template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\ninline cudaError_t Launch(';c=once(c,anchor,entries+'\n'+anchor)
    a,b,launch=block(c,'inline cudaError_t LaunchNoIO(');replacement=once(launch,'  // Common-shape AOT only; all original capture/admission certificates still apply.','''  if(p.private_psi_independent){
   if(FixedNoIOCubeEligible<512>(p))return LaunchNativeNoioOwnedPsi<Depth,512>(stream,p,ids,uniform_count);
   if(FixedNoIOCubeEligible<564>(p))return LaunchNativeNoioOwnedPsi<Depth,564>(stream,p,ids,uniform_count);
   return LaunchNativeNoioOwnedPsi<Depth,0>(stream,p,ids,uniform_count);
  }
  // Common-shape AOT only; all original capture/admission certificates still apply.''');c=c[:a]+replacement+c[b:]
    (src/'asymmetric_native.cuh').write_text(c);(src/'native_noio_private_psi.cuh').write_text(body);(src/'native_noio_owned_psi.cuh').write_bytes(Path(__file__).with_name('cuda_native_noio_owned_psi.cuh').read_bytes())
    e=(src/'native_extension.cc').read_text();e=once(e,'module.attr("moving_native_noio_private_psi_floats")=16;','module.attr("moving_native_noio_private_psi_floats")=0;module.attr("moving_native_noio_temporary_psi_floats")=8;module.attr("moving_native_noio_owned_psi_words")=13020;');(src/'native_extension.cc').write_text(e)
    proof.update(schema='beamz.native-noio-owned-shared-psi.v1',owned_shared_psi=True,threads=1024,expected_kernels=74,private_field_floats=12,private_psi_floats=0,temporary_psi_floats=8,shared_psi_words=13020,profile_words=0,shared_bytes=101304,dispatch_allocation_bytes=101376,region_bindings=bindings,original_wrapper_bindings=proof['wrapper_bindings'],wrapper_bindings=[],noio_launcher_original=launch,noio_launcher_emitted=replacement,consumer_sha256=sha(src/'asymmetric_native.cuh'),extension_sha256=sha(src/'native_extension.cc'),helpers={name:sha(src/name) for name in (*proof['helpers'],'native_noio_owned_psi.cuh')});(output.parent/'native-noio-private-psi-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return {k:v for k,v in proof.items() if k not in ('region_bindings','wrapper_bindings','original_wrapper_bindings','noio_launcher_original','noio_launcher_emitted')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
