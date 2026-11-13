"""Compare actual fixed-cell physics methods and independent cached profiles."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from scripts.cuda_trace_phase_cycle import block


NOIO=r'''
template<int N,int Axes>static void static_noio(const Parameters&p,uint8_t flags,const float*profiles,const float*frame){
 FixedCubeCachedEnvironment<N,Axes>expected{p,flags,profiles};
 StaticLogicalEnvironment<1,false,true,1,true,-1,false,Axes,true,N>actual{p,flags,frame};
 for(int i=0;i<1024;++i){int z=int(hash32(i*3)%unsigned(N+3))-1,y=int(hash32(i*3+1)%unsigned(N+3))-1,x=int(hash32(i*3+2)%unsigned(N+3))-1;
  for(int f=0;f<6;++f){require(expected.Valid(f,z,y,x)==actual.Valid(f,z,y,x),"AOT NoIO support");
   if(expected.Valid(f,z,y,x))for(int kind=0;kind<2;++kind){require(std::bit_cast<uint32_t>(expected.Coefficient(kind,f,z,y,x))==std::bit_cast<uint32_t>(actual.Coefficient(kind,f,z,y,x)),"AOT NoIO material");++methods;}
  }
  for(int phase=0;phase<2;++phase)for(int axis=0;axis<3;++axis){require(std::bit_cast<uint32_t>(expected.Difference(phase,0,axis,z,y,x,.25f,.5f))==std::bit_cast<uint32_t>(actual.Difference(phase,0,axis,z,y,x,.25f,.5f)),"AOT NoIO derivative");++methods;}
  for(int s=0;s<12;++s){require(expected.LocalActive(s,z,y,x)==actual.LocalActive(s,z,y,x),"AOT NoIO axis activity");++methods;
   if(expected.Valid((s/6)*3+(s%6)/2,z,y,x)&&expected.LocalActive(s,z,y,x))for(int kind=2;kind<5;++kind){require(std::bit_cast<uint32_t>(expected.Coefficient(kind,s,z,y,x))==std::bit_cast<uint32_t>(actual.Coefficient(kind,s,z,y,x)),"AOT NoIO independent profile hole lookup");++methods;}
  }
 }
}
'''


def validate(candidate):
    candidate=Path(candidate).resolve();src=candidate/'source/src';proof=json.loads((candidate/'static-logical-compiler-proof.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(src/'asymmetric_native.cuh')!=proof['consumer_sha256'] or sha(src/'native_static_logical.cuh')!=proof['helpers']['native_static_logical.cuh']:raise ValueError('Actual static compiler binding required')
    c=(src/'asymmetric_native.cuh').read_text();body=(src/'native_static_logical.cuh').read_text();primitive=(src/'yee_primitives.cuh').read_text();parts=[]
    for signature in ('constexpr int CpmlAxis(','float ScaleYeeDifference(','float PackedMaterialSource('):parts.append(('template<class Index>\n' if 'PackedMaterial' in signature else '')+block(primitive,signature)[2])
    environments=[]
    for signature in ('struct Environment {','struct FixedCubeCachedEnvironment:'):
        a=c.index(signature);environments.append(c[c.rfind('template<',0,a):a]+block(c,signature)[2]+';')
    profiles='\n'.join(block(c,s)[2] for s in ('int IndependentProfileCacheIndex(','float ReadIndependentProfileCache('))
    constants='\n'.join(line for line in c.splitlines() if line.startswith('constexpr int BZ=') or line.startswith('constexpr int PAD='))
    helpers=constants+'\n'+'\n'.join(block(body,s)[2] for s in ('int StaticSlot(','bool StaticFieldLive(','int StaticProfileSlot('))
    helpers+='\nstruct CPUThreadIndex{int x=0;};static CPUThreadIndex threadIdx;\ntemplate<int T>\n'+block(body,'void StaticStageProfiles(')[2]
    a=body.index('struct StaticLogicalEnvironment:');newenv=body[body.rfind('template<',0,a):a]+block(body,'struct StaticLogicalEnvironment:')[2]+';'
    methods=Path(__file__).with_name('prove_cuda_effect_regions_host.cc').read_text().replace('// BEAMZ_EFFECT_CANONICAL_PRIMITIVES','\n'.join(parts)).replace('// BEAMZ_EFFECT_ACTUAL_ENVIRONMENTS',profiles+'\n'+'\n'.join(environments)+'\n'+helpers+'\n'+newenv)
    methods=re.sub(r'\bEnvironment<([^<>]+)>',lambda m:'Environment<'+','.join(m[1].split(',')[:-1])+'>' if len(m[1].split(','))==10 else m[0],methods)
    host='''static const float*global_frame;
template<int N,int P,bool U>struct HostPhaseEnvironment:StaticLogicalEnvironment<1,(P==2||P==4),true,P,U,(P>=3?20:-1),(P>=3),-1,(P==5),0>{
 HostPhaseEnvironment(const Parameters&p,uint8_t flags):StaticLogicalEnvironment<1,(P==2||P==4),true,P,U,(P>=3?20:-1),(P>=3),-1,(P==5),0>(p,flags,global_frame){}
};'''
    old='#include "native_phase_environment.cuh"\ntemplate<int N,int P,bool U>using HostPhaseEnvironment=PhaseEnvironment<N,P,U>;'
    if methods.count(old)!=1:raise ValueError('Actual physics harness adapter required')
    methods=methods.replace(old,host)
    marker='template<int CubeN>static void environments(int n){';methods=methods.replace(marker,NOIO+'\n'+marker)
    setup='''std::vector<float>static_frame(6*PAD+2,1.234f);float*cache_frame=static_frame.data()+1;global_frame=cache_frame;
 for(int threads:{512,1024}){
  std::fill(static_frame.begin(),static_frame.end(),1.234f);
  for(int thread=0;thread<threads;++thread){threadIdx.x=thread;if(threads==512)StaticStageProfiles<512>(p,cache_frame);else StaticStageProfiles<1024>(p,cache_frame);}
  require(static_frame.front()==1.234f&&static_frame.back()==1.234f,"Profile staging overrun");
  for(int i=0;i<864;++i){require(std::bit_cast<uint32_t>(cache_frame[StaticProfileSlot(i)])==std::bit_cast<uint32_t>(profiles[i]),"All864 independent profile bitpatterns");++methods;}
 }
 if constexpr(CubeN>0){static_noio<CubeN,1>(p,34,profiles,cache_frame);static_noio<CubeN,2>(p,36,profiles,cache_frame);static_noio<CubeN,4>(p,40,profiles,cache_frame);static_noio<CubeN,-1>(p,46,profiles,cache_frame);}
 '''
    methods=methods.replace('for(unsigned flags=0;flags<128;++flags){',setup+'for(unsigned flags=0;flags<128;++flags){')
    methods=methods.replace('check(boundary,z,y,x);','check(boundary,z,y,x);\n   StaticLogicalEnvironment<1,false,true,5,false,20,true,-1,false,0>uncached{p,uint8_t(flags),global_frame};check(uncached,z,y,x);')
    out=candidate/'host-static-physics-proof';out.mkdir(exist_ok=False);cc=out/'proof.cc';cc.write_text(methods);binary=out/'proof';command=['g++','-std=c++20','-O2','-ffp-contract=off','-fsanitize=address,undefined','-DBEAMZ_PHASE_REGISTER_ENGINE=1','-I'+str(src),str(cc),str(src/'native_moving.cc'),'-o',str(binary)]
    built=subprocess.run(command,capture_output=True,text=True);(out/'build.log').write_text(built.stdout+built.stderr)
    if built.returncode:raise RuntimeError('Actual static physics method compile failed:'+str(out/'build.log'))
    run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=180);(out/'run.log').write_text(run.stdout+run.stderr)
    if run.returncode:raise RuntimeError('Actual static physics methods failed:'+run.stdout+run.stderr)
    result=dict(passed=True,consumer_sha256=proof['consumer_sha256'],helpers=proof['helpers'],actual_physics_methods=json.loads(run.stdout),cached_and_uncached_general=True,original_AOT512_564_NoIO_axes=True,all864_profiles_independent=True,scope='Actual emitted StaticLogicalEnvironment vs original Environment/FixedCube: FP32intrinsic-emulated material, derivative/wall, CPMLprofilelookup, source andrawobservation methods; actual profile staging512/1024 and AOTNoIOaxis facts. CUDA arithmetic/visibility and throughput remain separate.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.candidate)))
