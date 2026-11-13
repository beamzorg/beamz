"""Trace CPML state into three shared planes and one private color.

Keep1024 compute threads and all original68 fallback entries. The six new NoIO
entries use a single phase field bank, three independent shared psi colors and
one private H color. Only288 actual profiles are required on a single-axis path.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from scripts.build_cuda_native_noio_private_psi import stage as base_stage,region as base_region
from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_trace_phase_cycle import block


def shared_region(phase,original):
    text=base_region(phase,original)
    bindings={'history.ha[j]':'private_hpsi[j]','history.hb[j]':'psi_cache[Slot(zs,sy,sx)]','history.ea[j]':'psi_cache[PAD+Slot(zs,sy,sx)]','history.eb[j]':'psi_cache[2*PAD+Slot(zs,sy,sx)]'}
    for before,after in bindings.items():text=text.replace(before,after)
    return text


def stage(source,output,expected):
    base_stage(source,output,expected);output=Path(output).resolve();src=output/'src';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();proof=json.loads((output.parent/'native-noio-private-psi-compiler-proof.json').read_text());c=(src/'asymmetric_native.cuh').read_text();body=(src/'native_noio_private_psi.cuh').read_text()
    for binding in proof['wrapper_bindings']:
        a,b,_=block(c,binding['signature']);start=c.rfind('template<',0,a);c=c[:start]+c[b:];legacy='void NativeNoioLegacyUniformBody(' if 'Uniform' in binding['signature'] else 'void NativeNoioLegacyFixedCubeBody(';c=once(c,legacy,binding['signature'])
    old='constexpr size_t NativeNoioPsiSharedBytes=3ULL*PAD*sizeof(float)+8*sizeof(size_t)+2*sizeof(int)+864*sizeof(float);'
    body=once(body,old,'constexpr size_t NativeNoioPsiSharedBytes=6ULL*PAD*sizeof(float)+8*sizeof(size_t)+2*sizeof(int)+288*sizeof(float);\nconstexpr size_t NativeNoioSharedPsiLaunchBytes=(NativeNoioPsiSharedBytes+127)&~size_t(127);')
    body=once(body,'static_assert(NativeNoioPsiSharedBytes<=ProfileCachedSharedBytes,"Retain original NoIO shared allocation and carveout");','static_assert(NativeNoioSharedPsiLaunchBytes==99584&&NativeNoioSharedPsiLaunchBytes<=101376,"All CPML histories fit SM86 shared limit");')
    body=once(body,'extern __shared__ float frame[];size_t*bases=reinterpret_cast<size_t*>(frame+3*PAD);volatile int*window=reinterpret_cast<volatile int*>(bases+8);float*profiles=frame+3*PAD+8*sizeof(size_t)/sizeof(float)+2;','extern __shared__ float frame[];float*psi_cache=frame+3*PAD;size_t*bases=reinterpret_cast<size_t*>(frame+6*PAD);volatile int*window=reinterpret_cast<volatile int*>(bases+8);float*profiles=frame+6*PAD+8*sizeof(size_t)/sizeof(float)+2;')
    body=once(body,'StageIndependentCpmlProfiles(p,profiles);','NativeNoioCompactStage<ActiveAxes>(p,profiles);')
    body=once(body,'float state[3][Slots];PrivatePsiHistory<true,Slots>history;','float state[3][Slots],private_hpsi[Slots];')
    oldenv='auto env=[&] __device__(){if constexpr(CubeN>0)return FixedCubeCachedEnvironment<CubeN,ActiveAxes>{p,flags,profiles};else return Environment<1,false,true,1,true,-1,false,ActiveAxes,true>{p,flags,profiles};}();'
    newenv='auto standard_env=[&] __device__(){if constexpr(CubeN>0)return FixedCubeCachedEnvironment<CubeN,ActiveAxes>{p,flags,nullptr};else return Environment<1,false,true,1,true,-1,false,ActiveAxes,true>{p,flags,nullptr};}();\n NativeNoioCompactEnvironment<decltype(standard_env),ActiveAxes>env{standard_env,profiles};'
    body=once(body,oldenv,newenv);bindings=[]
    for binding in proof['region_bindings']:
        emitted=shared_region(binding['phase'],binding['original']);body=once(body,binding['emitted'],emitted);bindings.append(dict(**{k:v for k,v in binding.items() if k!='emitted'},emitted=emitted))
    body='#include "native_noio_compact_profiles.cuh"\n'+body
    entries='''template<int Depth,int CubeN>
__global__ __launch_bounds__(1024,1) void AsymmetricNativeNoioSharedPsi(const __grid_constant__ Parameters p,const int32_t*tiles){
 switch(p.flags[tiles[blockIdx.x]]&14u){
  case 2:NativeNoioPrivatePsiBody<Depth,CubeN,1>(p,tiles);return;
  case 4:NativeNoioPrivatePsiBody<Depth,CubeN,2>(p,tiles);return;
  case 8:NativeNoioPrivatePsiBody<Depth,CubeN,4>(p,tiles);return;
  default:if constexpr(CubeN>0)FixedCubeCachedBody<Depth,CubeN,-1>(p,tiles);else UniformBody<Depth,1,false,true,1,true,-1,true>(p,tiles);return;
 }
}
template<int Depth,int CubeN>
inline cudaError_t LaunchNativeNoioSharedPsi(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count){
 const auto kernel=AsymmetricNativeNoioSharedPsi<Depth,CubeN>;
 auto status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,NativeNoioSharedPsiLaunchBytes);if(status!=cudaSuccess)return status;
 AsymmetricNativeNoioSharedPsi<Depth,CubeN><<<count,1024,NativeNoioSharedPsiLaunchBytes,stream>>>(p,ids);return cudaGetLastError();
}
'''
    anchor='template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\ninline cudaError_t Launch(';c=once(c,anchor,entries+'\n'+anchor)
    a,b,launch=block(c,'inline cudaError_t LaunchNoIO(');replacement=once(launch,'  // Common-shape AOT only; all original capture/admission certificates still apply.','''  if(p.private_psi_independent){
   if(FixedNoIOCubeEligible<512>(p))return LaunchNativeNoioSharedPsi<Depth,512>(stream,p,ids,uniform_count);
   if(FixedNoIOCubeEligible<564>(p))return LaunchNativeNoioSharedPsi<Depth,564>(stream,p,ids,uniform_count);
   return LaunchNativeNoioSharedPsi<Depth,0>(stream,p,ids,uniform_count);
  }
  // Common-shape AOT only; all original capture/admission certificates still apply.''');c=c[:a]+replacement+c[b:]
    (src/'asymmetric_native.cuh').write_text(c);(src/'native_noio_private_psi.cuh').write_text(body);(src/'native_noio_compact_profiles.cuh').write_bytes(Path(__file__).with_name('cuda_native_noio_compact_profiles.cuh').read_bytes())
    e=(src/'native_extension.cc').read_text();e=once(e,'module.attr("moving_native_noio_private_psi_floats")=16;','module.attr("moving_native_noio_private_psi_floats")=4;module.attr("moving_native_noio_shared_psi_words")=12288;module.attr("moving_native_noio_compact_profile_words")=288;');(src/'native_extension.cc').write_text(e)
    proof.update(schema='beamz.native-noio-phase-shared-psi.v1',shared_psi=True,threads=1024,expected_kernels=74,private_field_floats=12,private_psi_floats=4,shared_psi_words=12288,profile_words=288,shared_bytes=99528,dispatch_allocation_bytes=99584,region_bindings=bindings,original_wrapper_bindings=proof['wrapper_bindings'],wrapper_bindings=[],noio_launcher_original=launch,noio_launcher_emitted=replacement,consumer_sha256=sha(src/'asymmetric_native.cuh'),extension_sha256=sha(src/'native_extension.cc'),helpers={name:sha(src/name) for name in (*proof['helpers'],'native_noio_compact_profiles.cuh')})
    (output.parent/'native-noio-private-psi-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return {k:v for k,v in proof.items() if k not in ('region_bindings','wrapper_bindings','original_wrapper_bindings','noio_launcher_original','noio_launcher_emitted')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
