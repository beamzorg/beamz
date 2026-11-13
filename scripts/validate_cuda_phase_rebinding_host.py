"""Validate actual phase providers, canonical region ancestry and physics methods."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_trace_phase_cycle import block


PROVIDERS=r'''
#include <array>
#include <algorithm>
#include <cstdint>
#include <set>
#include <iostream>
#include <stdexcept>
#include "native_closed_pipeline.h"
#define __host__
#define __device__
#define __forceinline__ inline
#define BEAMZ_NATIVE_ASYMMETRIC_SWIZZLE 1
// ACTUAL_DEFINITIONS
static void require(bool v,const char*why){if(!v)throw std::runtime_error(why);}
template<int T>static size_t test(){
 std::set<int>private_slots;for(int thread=0;thread<T;++thread)for(int j=0;j<PAD/T;++j){auto s=PhasePrivateLogical<T>(thread,j);
  require(s.z>=0&&s.z<CZ&&s.y>=0&&s.y<CY&&s.x>=0&&s.x<CX,"Private provider bounds");require(private_slots.insert((s.z*CY+s.y)*CX+s.x).second,"Duplicate private register owner");}
 require(private_slots.size()==PAD,"Incomplete private register coverage");size_t checks=0;
 for(int n=1;n<=32;++n){const int hz=(-n-1)&(CZ-1),hy=(-n-1)&(CY-1),hx=(-n-1)&(CX-1);std::set<int>writes;
  for(int z=0;z<CZ;++z)for(int y=0;y<CY;++y)for(int x=0;x<CX;++x){const int address=Slot(z,y,x);require(address==((z*CY+y)*CX+(x^(y|((z&1)<<4)))),"Shared XOR bank layout");
   if(z!=hz&&y!=hy&&x!=hx)require(writes.insert(address).second,"Owned shared write overlap");}
  std::set<int>ghosts;
  for(int i=0;i<PhaseGhostPoints;++i){auto d=PhaseGhostLogical(i),s=PhaseGhostPhysicalSlot(i,n);
   require(d.z>=0&&d.z<=BZ&&d.y>=0&&d.y<=BY&&d.x>=0&&d.x<=BX&&(!d.z||!d.y||!d.x),"Ghost provider bounds");
   require(ghosts.insert((d.z*CY+d.y)*CX+d.x).second,"Duplicate ghost provider");
   require(s.z==((hz+d.z)&(CZ-1))&&s.y==((hy+d.y)&(CY-1))&&s.x==((hx+d.x)&(CX-1)),"Ghost rotation version");
   require(writes.insert(Slot(s.z,s.y,s.x)).second,"Ghost/owned shared writes overlap");
   const int bits=(!d.z)|((!d.y)<<1)|((!d.x)<<2),source_z=d.z?d.z-1:BZ-1,source_y=d.y?d.y-1:BY-1,source_x=d.x?d.x-1:BX-1;
   require(bits>0&&bits<8&&Surface(source_z,source_y,source_x)>=0&&Surface(source_z,source_y,source_x)<SurfacePoints,"Canonical packet source surface");
   require(-n-1+d.z==-(d.z==0)*BZ-n+source_z&&-n-1+d.y==-(d.y==0)*BY-n+source_y&&-n-1+d.x==-(d.x==0)*BX-n+source_x,"Producer/consumer physical point identity");
   for(int c=0;c<3;++c){const int address=c*PAD+Slot(s.z,s.y,s.x);require(address>=0&&address<3*PAD,"Field frame span");++checks;}
  }
  require(ghosts.size()==PAD-BZ*BY*BX&&writes.size()==PAD,"Complete disjoint shared family");
 }
 return checks;
}
int main(){const size_t checks=test<512>()+test<1024>();
 require(3*PAD*sizeof(float)+8*sizeof(size_t)+2*sizeof(int)<=49280,"Frame/base/window/cold overlap");require(49280+sizeof(Cold)==49504,"Actual twoCTA shared allocation ABI");
 std::cout<<"{\"passed\":true,\"actual_provider_checks\":"<<checks<<",\"actual_shared_bytes\":"<<49280+sizeof(Cold)<<",\"single_bank_slots\":"<<3*PAD<<"}\n";
}
'''


def run_cpp(out,name,code,src,*,moving=False,defines=()):
    folder=out/name;folder.mkdir();source=folder/'proof.cc';source.write_text(code);binary=folder/'proof'
    command=['g++','-std=c++20','-O2','-ffp-contract=off','-fsanitize=address,undefined',*defines,'-I'+str(src),str(source),*([str(src/'native_moving.cc')] if moving else []),'-o',str(binary)]
    built=subprocess.run(command,capture_output=True,text=True);(folder/'build.log').write_text(built.stdout+built.stderr)
    if built.returncode:raise RuntimeError('Phase CPU compile failed:'+str(folder/'build.log'))
    run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=180);(folder/'run.log').write_text(run.stdout+run.stderr)
    if run.returncode:raise RuntimeError('Phase CPU proof failed:'+str(folder/'run.log'))
    return json.loads(run.stdout)


def validate(candidate,model,paired_model=None):
    candidate=Path(candidate).resolve();src=candidate/'source/src';proof=json.loads((candidate/'phase-rebinding-compiler-proof.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if not proof.get('phase_single_register_bank') or sha(src/'asymmetric_native.cuh')!=proof['consumer_sha256']:raise ValueError('Bound single-register-bank compiler required')
    for name,value in proof['helpers'].items():
        if sha(src/name)!=value:raise ValueError('Changed phase helper:'+name)
    model=json.loads(Path(model).read_text())
    if not model.get('passed') or model['source_sha256']!=sha(Path(__file__).with_name('prove_cuda_phase_rebinding.py')) or not all(c.get('single_physical_register_bank') for c in model['cases']) or len(model['negative_controls'])!=4:raise ValueError('Current single-bank canonical version proof required')
    paired=None
    if proof.get('paired_handoff'):
        paired=json.loads(Path(paired_model).read_text()) if paired_model else {}
        if not paired.get('passed') or len(paired.get('negative_controls',{}))!=5 or paired.get('source_sha256')!=sha(Path(__file__).with_name('prove_cuda_paired_phase_handoff.py')) or paired.get('supernode_source_sha256')!=sha(Path(__file__).with_name('cuda_paired_supernodes.py')):raise ValueError('Current paired-phase/5epochownership/ACK/bootstrap canonical model required')
    c=(src/'asymmetric_native.cuh').read_text();body=(src/'native_phase_rebinding.cuh').read_text();native=block(c,'void UniformBody(')[2];position=0
    replacements=[('''if(field<3)return h[field][j];const int c=field-3;
      if(!dz&&!ddy&&!ddx)return e[c][j];return frame[c*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];''','''if(field<3)return h[field][j];const int c=field-3;
      return frame[c*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];'''),
        ('if(field>=3)return e[field-3][j];if(!dz&&!ddy&&!ddx)return h[field][j];return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];','if(field>=3)return e[field-3][j];return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];')]
    for phase,r in enumerate(proof['phase_region_bindings']):
        marker='{int step;asm volatile("mov.b32 %0,%1;":';a=native.index(marker,position);_,b,region=block(native[a:],marker);position=a+b
        if region!=r['original']:raise ValueError('Region is not the actual legacy canonical point loop')
        expected=re.sub(r'\b(?:h|e)(?=\[)','state',once(region,*replacements[phase]))
        expected=re.sub(r'\bSlot\(','PhaseSlot(',expected)
        if proof.get('phase_private_Y_groups')==2:
            from scripts.build_cuda_paired256_handoff import map_region
            if expected!=r['provider_lowered_base']:raise ValueError('Mapped phase must retain canonical storage-provider ancestry')
            expected=map_region(expected,phase)
        if expected!=r['provider_lowered']:raise ValueError('Only field center providers and private bank identifiers may change')
        environment='DoubleSharedEnvironment' if proof.get('phase_grouped_worker') else 'CachedPhaseEnvironment' if proof.get('phase_full_profile_cache') else 'PhaseEnvironment'
        arguments='p,flags,profiles' if proof.get('phase_full_profile_cache') or proof.get('phase_grouped_worker') else 'p,flags'
        generated='\n'.join(('if(!(flags&62u))' if i==0 else 'else if(!(flags&46u))' if i==1 else 'else')+'{'+f'{environment}<CubeN,{policy},{uniform}>env{{{arguments}}};'+expected[1:-1]+'}' for i,(policy,uniform) in enumerate(((2,'false'),(4,'false'),(5,'true'))))
        if generated!=r['emitted'] or body.count(generated)!=1:raise ValueError('Actual emitted phase region binding failed')
    if body.count('float state[3][Slots];')!=1 or re.search(r'float [he]\[3\]\[Slots\]',body):raise ValueError('One physical private field bank required')
    if body.count('ld.volatile.shared.f32')!=1:raise ValueError('Explicit shared reload required for new register definitions')
    out=candidate/'host-phase-rebinding-proof';out.mkdir(exist_ok=False)
    constants='\n'.join(line for line in c.splitlines() if line.startswith('constexpr int BZ=') or line.startswith('constexpr int PAD='))
    helpers=block(c,'int Surface(')[2]+'\n'+block(body,'int PhaseSlot(')[2]
    point=block(body,'struct PhasePoint{')[2]+';'
    functions='\n'.join(('template<int T>\n' if 'Private' in s else '')+block(body,s)[2] for s in ('PhasePoint PhaseGhostLogical(','PhasePoint PhasePrivateLogical(','PhasePoint PhaseGhostPhysicalSlot('))
    cold=block((src/'native_closed_pipeline_control.cuh').read_text(),'struct Cold {')[2]+';'
    provider_code=re.sub(r'\bSlot\(','PhaseSlot(',PROVIDERS).replace('// ACTUAL_DEFINITIONS',constants+'\n'+helpers+'\n'+point+'\nconstexpr int PhaseGhostPoints=PAD-BZ*BY*BX;\n'+functions+'\n'+cold)
    if proof.get('phase_full_profile_cache'):
        provider_code=provider_code.replace('<=49280,','+864*sizeof(float)<=52736,').replace('49280+sizeof(Cold)==49504','52736+sizeof(Cold)==52960').replace('<<49280+sizeof(Cold)','<<52736+sizeof(Cold)')
    if proof.get('paired_handoff'):
        provider_code=provider_code.replace('49280+sizeof(Cold)==49504','49280+sizeof(Cold)==49536')
    if proof.get('phase_private_Y_groups')==2:
        provider_code=provider_code.replace('test<512>()+test<1024>()','test<256>()+test<512>()+test<1024>()')
    providers=run_cpp(out,'actual-providers',provider_code,src)
    primitives=(src/'yee_primitives.cuh').read_text();parts=[]
    for signature in ('constexpr int CpmlAxis(','float ScaleYeeDifference(','float PackedMaterialSource('):parts.append(('template<class Index>\n' if 'PackedMaterial' in signature else '')+block(primitives,signature)[2])
    environment=block(c,'struct Environment {')[2]+';';a=c.index('struct Environment {');environment=c[c.rfind('template<',0,a):a]+environment
    fixed=block(c,'struct FixedCubeCachedEnvironment:')[2]+';';a=c.index('struct FixedCubeCachedEnvironment:');fixed=c[c.rfind('template<',0,a):a]+fixed
    profiles='\n'.join(block(c,s)[2] for s in ('int IndependentProfileCacheIndex(','float ReadIndependentProfileCache('))
    methods=Path(__file__).with_name('prove_cuda_effect_regions_host.cc').read_text().replace('// BEAMZ_EFFECT_CANONICAL_PRIMITIVES','\n'.join(parts)).replace('// BEAMZ_EFFECT_ACTUAL_ENVIRONMENTS',profiles+'\n'+environment+'\n'+fixed)
    if proof.get('phase_full_profile_cache'):
        methods=methods.replace('#include "native_phase_environment.cuh"\ntemplate<int N,int P,bool U>using HostPhaseEnvironment=PhaseEnvironment<N,P,U>;', '#include "native_phase_environment.cuh"\n#include "native_cached_phase_environment.cuh"\nstatic const float*global_profiles;\ntemplate<int N,int P,bool U>struct HostPhaseEnvironment:CachedPhaseEnvironment<N,P,U>{HostPhaseEnvironment(const Parameters&p,uint8_t f):CachedPhaseEnvironment<N,P,U>(p,f,global_profiles){}};')
        methods=methods.replace('for(unsigned flags=0;flags<128;++flags){','global_profiles=profiles;\n for(unsigned flags=0;flags<128;++flags){')
    if proof.get('phase_grouped_worker'):
        methods=methods.replace('#include "native_phase_environment.cuh"\ntemplate<int N,int P,bool U>using HostPhaseEnvironment=PhaseEnvironment<N,P,U>;', '#include "native_phase_environment.cuh"\n#include "native_double_environment.cuh"\nstatic const float*global_ab;\ntemplate<int N,int P,bool U>struct HostPhaseEnvironment:DoubleSharedEnvironment<N,P,U>{HostPhaseEnvironment(const Parameters&p,uint8_t f):DoubleSharedEnvironment<N,P,U>(p,f,global_ab){}};')
        stage='struct CPUThreadIndex{int x=0;};static CPUThreadIndex threadIdx;\n'+block((src/'native_group_profiles.cuh').read_text(),'void GroupStageProfiles(')[2]
        methods=methods.replace('// BEAMZ_EFFECT_ACTUAL_ENVIRONMENTS',stage) if '// BEAMZ_EFFECT_ACTUAL_ENVIRONMENTS' in methods else methods.replace('static uint32_t hash32',stage+'\nstatic uint32_t hash32')
        setup='float guarded_ab[578];guarded_ab[0]=1.234f;guarded_ab[577]=-1.234f;global_ab=guarded_ab+1;\n for(int thread=0;thread<1024;++thread){threadIdx.x=thread;GroupStageProfiles(p,guarded_ab+1);}\n require(guarded_ab[0]==1.234f&&guarded_ab[577]==-1.234f,"Group actual a/b stage overrun");\n for(int i=0;i<576;++i){const int slot=i/48,kind=(i/24)&1,packed=i%24;require(std::bit_cast<uint32_t>(guarded_ab[i+1])==std::bit_cast<uint32_t>(profiles[slot*72+kind*24+packed]),"Group independent a/b binding");++methods;}\n '
        methods=methods.replace('for(unsigned flags=0;flags<128;++flags){',setup+'for(unsigned flags=0;flags<128;++flags){')
    physics=run_cpp(out,'actual-physics-methods',methods,src,moving=True,defines=['-DBEAMZ_PHASE_REGISTER_ENGINE=1'])
    result=dict(passed=True,consumer_sha256=proof['consumer_sha256'],helpers=proof['helpers'],canonical_region_inverse=True,actual_providers=providers,actual_physics_methods=physics,model_sha256=model['source_sha256'],scope='Actual emitted ghost/private/shared providers and physics environment methods, canonical H/E region arithmetic inverse, single-bank version/auxiliary model. CUDA visibility/FP32 full30 parity and performance remain separate.')
    if paired:result['paired_phase_model_sha256']=paired['source_sha256']
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--paired-model',type=Path);a=p.parse_args();print(json.dumps(validate(a.candidate,a.model,a.paired_model)))
