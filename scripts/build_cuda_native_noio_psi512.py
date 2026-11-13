"""Spend more registers per thread on the complete native CPML recurrence.

Six separate512-thread entries leave original68 entries and launchers intact.
The original certified NoIO range alone selects them after actual state-range
admission; mixed-axis NoIO keeps direct global psi within the same phase layout.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from scripts.build_cuda_native_noio_private_psi import stage as base_stage,region as base_region
from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_trace_phase_cycle import block


def region512(phase,original):
    region=base_region(phase,original);prefix='h' if phase==0 else 'e'
    point=f'PrivatePsiHistoryPoint<decltype(env)> point_env(env,history.{prefix}a[j],history.{prefix}b[j],n==first||dz_owned==1||dy_owned==1||dx_owned==1,n==Depth||dz_owned==BZ||dy_owned==BY||dx_owned==BX);'
    conditional=f'auto point_env=[&] __device__(){{if constexpr(CachePsi)return PrivatePsiHistoryPoint<decltype(env)>(env,history.{prefix}a[j],history.{prefix}b[j],n==first||dz_owned==1||dy_owned==1||dx_owned==1,n==Depth||dz_owned==BZ||dy_owned==BY||dx_owned==BX);else return env;}}();'
    return once(region,point,conditional)


def stage(source,output,expected):
    base_stage(source,output,expected);output=Path(output).resolve();src=output/'src';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();proof=json.loads((output.parent/'native-noio-private-psi-compiler-proof.json').read_text());c=(src/'asymmetric_native.cuh').read_text();body=(src/'native_noio_private_psi.cuh').read_text()
    # Restore original68 device bodies. This resource change has its own entries.
    for binding in proof['wrapper_bindings']:
        signature=binding['signature'];a,b,_=block(c,signature);start=c.rfind('template<',0,a);template=c[start:a];c=c[:start]+c[b:]
        legacy='void NativeNoioLegacyUniformBody(' if 'Uniform' in signature else 'void NativeNoioLegacyFixedCubeBody('
        c=once(c,legacy,signature)
    body=once(body,'static_assert(ActiveAxes==1||ActiveAxes==2||ActiveAxes==4,"Only single-axis independent local recurrences");','constexpr bool CachePsi=ActiveAxes==1||ActiveAxes==2||ActiveAxes==4;')
    body=once(body,'constexpr int T=1024,Slots=4,ZStep=2;','constexpr int T=512,Slots=8,ZStep=1;')
    bindings=[]
    for binding in proof['region_bindings']:
        emitted=region512(binding['phase'],binding['original']);body=once(body,binding['emitted'],emitted);bindings.append(dict(**{k:v for k,v in binding.items() if k!='emitted'},emitted=emitted))
    entries='''template<int Depth,int CubeN>
__global__ __launch_bounds__(512,1) void AsymmetricNativeNoioPsi512(const __grid_constant__ Parameters p,const int32_t*tiles){
 switch(p.flags[tiles[blockIdx.x]]&14u){
  case 2:NativeNoioPrivatePsiBody<Depth,CubeN,1>(p,tiles);return;
  case 4:NativeNoioPrivatePsiBody<Depth,CubeN,2>(p,tiles);return;
  case 8:NativeNoioPrivatePsiBody<Depth,CubeN,4>(p,tiles);return;
  default:NativeNoioPrivatePsiBody<Depth,CubeN,-1>(p,tiles);return;
 }
}
template<int Depth,int CubeN>
inline cudaError_t LaunchNativeNoioPsi512(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count){
 const auto kernel=AsymmetricNativeNoioPsi512<Depth,CubeN>;
 auto status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,ProfileCachedSharedBytes);if(status!=cudaSuccess)return status;
 AsymmetricNativeNoioPsi512<Depth,CubeN><<<count,512,ProfileCachedSharedBytes,stream>>>(p,ids);return cudaGetLastError();
}
'''
    anchor='template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\ninline cudaError_t Launch('
    c=once(c,anchor,entries+'\n'+anchor)
    a,b,launch=block(c,'inline cudaError_t LaunchNoIO(')
    replacement=once(launch,'  // Common-shape AOT only; all original capture/admission certificates still apply.','''  if(p.private_psi_independent){
   if(FixedNoIOCubeEligible<512>(p))return LaunchNativeNoioPsi512<Depth,512>(stream,p,ids,uniform_count);
   if(FixedNoIOCubeEligible<564>(p))return LaunchNativeNoioPsi512<Depth,564>(stream,p,ids,uniform_count);
   return LaunchNativeNoioPsi512<Depth,0>(stream,p,ids,uniform_count);
  }
  // Common-shape AOT only; all original capture/admission certificates still apply.''');c=c[:a]+replacement+c[b:]
    (src/'asymmetric_native.cuh').write_text(c);(src/'native_noio_private_psi.cuh').write_text(body)
    e=(src/'native_extension.cc').read_text();e=once(e,'module.attr("moving_native_noio_private_psi_floats")=16;','module.attr("moving_native_noio_private_psi_floats")=32;module.attr("moving_native_noio_private_psi_threads")=512;');(src/'native_extension.cc').write_text(e)
    proof.update(schema='beamz.native-noio-phase-private-psi512.v1',threads=512,expected_kernels=74,private_field_floats=24,private_psi_floats=32,region_bindings=bindings,original_wrapper_bindings=proof['wrapper_bindings'],wrapper_bindings=[],noio_launcher_original=launch,noio_launcher_emitted=replacement,consumer_sha256=sha(src/'asymmetric_native.cuh'),extension_sha256=sha(src/'native_extension.cc'),helpers={name:sha(src/name) for name in proof['helpers']})
    (output.parent/'native-noio-private-psi-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return {k:v for k,v in proof.items() if k not in ('region_bindings','wrapper_bindings','original_wrapper_bindings','noio_launcher_original','noio_launcher_emitted')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
