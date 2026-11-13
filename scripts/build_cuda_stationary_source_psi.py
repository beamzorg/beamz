"""Move source/observation finish to phase publication and cache CPML history.

H finish remains before the H publication join and E math; E finish remains
before E publication and the next H math. Arithmetic/source contributor order
inside the original Finish is unchanged. NoIO and all other policies stay frozen.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from scripts.build_cuda_native_noio_stationary_owned_psi import stage as stationary_stage
from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_trace_phase_cycle import block


def source_function(native):
    result=once(native,'void NativeNoioPrivatePsiBody(','void NativeStationarySourcePsiBody(')
    env='NativeOwnedGlobalProfiles<decltype(standard_env)>env{standard_env};'
    result=once(result,env,env+'\n Environment<1,false,true,5,true,20,true,ActiveAxes,false>effects{p,flags,nullptr};')
    result=once(result,'const float old_e=frame[slot],new_h=state[c][j];','const float old_e=frame[slot],new_h=env.Valid(c,z,y,x)?effects.Finish(c,n,z,y,x,state[c][j]):0.f;')
    result=once(result,'const float new_e=state[c][j];','const float new_e=env.Valid(c+3,z,y,x)?effects.Finish(c+3,n,z,y,x,state[c][j]):0.f;')
    return result


def stage(source,output,expected):
    stationary_stage(source,output,expected);output=Path(output).resolve();src=output/'src';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();proof=json.loads((output.parent/'native-noio-private-psi-compiler-proof.json').read_text());c=(src/'asymmetric_native.cuh').read_text();body=(src/'native_noio_private_psi.cuh').read_text()
    # Revert NoIO selection: this A/B isolates source+CPML architecture.
    c=once(c,proof['noio_launcher_emitted'],proof['noio_launcher_original'])
    native=block(body,'void NativeNoioPrivatePsiBody(')[2];source_body='template<int Depth,int CubeN,int ActiveAxes>\n__device__ __forceinline__ '+source_function(native)
    (src/'native_stationary_source_psi.cuh').write_text(source_body+'\n');c=once(c,'#include "native_noio_private_psi.cuh"','#include "native_noio_private_psi.cuh"\n#include "native_stationary_source_psi.cuh"')
    entries='''template<int Depth,int CubeN>
__global__ __launch_bounds__(1024,1) void AsymmetricStationarySourcePsi(const __grid_constant__ Parameters p,const int32_t*ids){
 if(!(p.flags[ids[blockIdx.x]]&64u)){ResearchSourceFacts<Depth,false,true>(p,ids);return;}
 switch(p.flags[ids[blockIdx.x]]&14u){
  case 2:NativeStationarySourcePsiBody<Depth,CubeN,1>(p,ids);return;
  case 4:NativeStationarySourcePsiBody<Depth,CubeN,2>(p,ids);return;
  case 8:NativeStationarySourcePsiBody<Depth,CubeN,4>(p,ids);return;
  default:ResearchSourceFacts<Depth,true,true>(p,ids);return;
 }
}
template<int Depth,int CubeN>
inline cudaError_t LaunchStationarySourcePsi(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count){
 const auto kernel=AsymmetricStationarySourcePsi<Depth,CubeN>;
 auto status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,NativeNoioOwnedPsiLaunchBytes);if(status!=cudaSuccess)return status;
 AsymmetricStationarySourcePsi<Depth,CubeN><<<count,1024,NativeNoioOwnedPsiLaunchBytes,stream>>>(p,ids);return cudaGetLastError();
}
'''
    anchor='template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\ninline cudaError_t Launch(';c=once(c,anchor,entries+'\n'+anchor)
    a,b,launcher=block(c,'inline cudaError_t LaunchSourcePending(')
    hook='''    if constexpr(Policy==5){if(p.private_psi_independent){
     if(FixedNoIOCubeEligible<512>(p))return LaunchStationarySourcePsi<Depth,512>(stream,p,ids,count);
     if(FixedNoIOCubeEligible<564>(p))return LaunchStationarySourcePsi<Depth,564>(stream,p,ids,count);
     return LaunchStationarySourcePsi<Depth,0>(stream,p,ids,count);
    }}
'''
    replacement=once(launcher,'    if(BeamzResearchSourcePlanePairEnabled&&plane_pair){','    if(BeamzResearchSourcePlanePairEnabled&&plane_pair){\n'+hook);c=c[:a]+replacement+c[b:];(src/'asymmetric_native.cuh').write_text(c)
    e=(src/'native_extension.cc').read_text();e=once(e,'module.attr("moving_native_noio_private_psi")=true;','module.attr("moving_native_noio_private_psi")=false;');e=once(e,'module.attr("moving_native_noio_stationary_owned_psi")=true;','module.attr("moving_native_noio_stationary_owned_psi")=false;module.attr("moving_stationary_source_psi")=true;');(src/'native_extension.cc').write_text(e)
    proof.update(schema='beamz.stationary-source-psi-publication-effects.v1',source_only=True,source_function_original=native,source_function_emitted=source_function(native),source_launcher_original=launcher,source_launcher_emitted=replacement,noio_launcher_emitted=proof['noio_launcher_original'],consumer_sha256=sha(src/'asymmetric_native.cuh'),extension_sha256=sha(src/'native_extension.cc'),helpers={name:sha(src/name) for name in (*proof['helpers'],'native_stationary_source_psi.cuh')},source_effect_placement='Original Environment5/20/PlanePairTrue Finish called exactlyonce for each valid owned H/E component during its phase publication; H source before B2/Ecompute, Esourcebefore B4/nextHcompute. Immutable source arrays, contributor round/order unchanged, rawobservationsinsideFinish, DFT after all original macros unchanged. All NoIO dispatch reverted to frozen control; source path only for existingLayout1IsoHpresence20PlanePairfacts andindependentstate, with perCTAUniformE64+singleaxis facts; otherwise exact ResearchSourceFacts fallback.')
    (output.parent/'native-noio-private-psi-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return {k:v for k,v in proof.items() if k not in ('region_bindings','wrapper_bindings','original_wrapper_bindings','noio_launcher_original','noio_launcher_emitted','source_function_original','source_function_emitted','source_launcher_original','source_launcher_emitted')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
