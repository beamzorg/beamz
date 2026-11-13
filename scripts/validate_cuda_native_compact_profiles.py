"""Check actual compact single-axis profiles against864 independent FP32 words."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from scripts.cuda_trace_phase_cycle import block


CODE=r'''
#include <bit>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#define __host__
#define __device__
#define __forceinline__ inline
struct Index{int x;};static Index threadIdx;
template<class T>T __ldg(const T*p){return *p;}
namespace beamz::cuda::yee{
// ACTUAL_AXIS
}
struct Psi{const float*a,*b,*inv;};struct Phase{Psi psi[6];};
struct Parameters{Phase phase[2];int physical_n[3];float base[2][6];};
struct Base{const Parameters&p;
// ACTUAL_PACKED
float Coefficient(int kind,int slot,int,int,int)const{return p.base[kind][slot];}
};
// ACTUAL_COMPACT_HELPER
static size_t checks=0;
static void require(bool condition,const char*why){if(!condition)throw std::runtime_error(why);}
template<int Axes>static void test(int seed){
 Parameters p{};p.physical_n[0]=512;p.physical_n[1]=564;p.physical_n[2]=259;float profiles[864],guarded[290];
 auto pattern=[&](int i){return uint32_t(i+1)*0x9e3779b9u^uint32_t(seed)*0x85ebca6bu;};
 for(int i=0;i<864;++i)profiles[i]=std::bit_cast<float>(pattern(i));
 for(int i=0;i<290;++i)guarded[i]=std::bit_cast<float>(0x7fc12345u);
 for(int f=0;f<6;++f)for(int kind=0;kind<2;++kind)p.base[kind][f]=std::bit_cast<float>(pattern(1000+6*kind+f));
 for(int slot=0;slot<12;++slot)p.phase[slot/6].psi[slot%6]={profiles+slot*72,profiles+slot*72+24,profiles+slot*72+48};
 for(int tid=0;tid<1024;++tid){threadIdx.x=tid;NativeNoioCompactStage<Axes>(p,guarded+1);}
 require(std::bit_cast<uint32_t>(guarded[0])==0x7fc12345u&&std::bit_cast<uint32_t>(guarded[289])==0x7fc12345u,"Compact staging overrun");
 Base base{p};NativeNoioCompactEnvironment<Base,Axes>env{base,guarded+1};std::set<int>slots;
 for(int slot=0;slot<12;++slot)if((1<<beamz::cuda::yee::CpmlAxis(slot%6))==Axes)for(int kind=2;kind<5;++kind)for(int packed=0;packed<24;++packed){
  const int index=NativeNoioCompactIndex<Axes>(slot,kind,packed),family=slot/6,axis=beamz::cuda::yee::CpmlAxis(slot%6);require(index>=0&&index<288&&slots.insert(index).second,"Distinct phase/term/kind/profile words alias");
  int q[3]={17,19,23};q[axis]=packed<12?packed:p.physical_n[axis]+family+packed-24;
  const uint32_t expected=std::bit_cast<uint32_t>(profiles[slot*72+(kind-2)*24+packed]);
  require(std::bit_cast<uint32_t>(guarded[index+1])==expected&&std::bit_cast<uint32_t>(env.Coefficient(kind,slot,q[0],q[1],q[2]))==expected,"Independent profile bits changed/misbound");++checks;
 }
 require(slots.size()==288,"Compact allocation has missing words");
 for(int f=0;f<6;++f)for(int kind=0;kind<2;++kind){require(std::bit_cast<uint32_t>(env.Coefficient(kind,f,17,19,23))==std::bit_cast<uint32_t>(base.Coefficient(kind,f,17,19,23)),"Base material method changed");++checks;}
}
int main(){for(int seed=0;seed<32;++seed){test<1>(seed);test<2>(seed);test<4>(seed);}std::cout<<"{\"passed\":true,\"actual_independent_profile_and_material_checks\":"<<checks<<"}\n";}
'''


def validate(candidate):
    candidate=Path(candidate).resolve();src=candidate/'source/src';proof=json.loads((candidate/'native-noio-private-psi-compiler-proof.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();helper=(src/'native_noio_compact_profiles.cuh').read_text();c=(src/'asymmetric_native.cuh').read_text()
    if sha(src/'native_noio_compact_profiles.cuh')!=proof['helpers']['native_noio_compact_profiles.cuh']:raise ValueError('Actual compact profile helper binding required')
    packed=block(block(c,'struct Environment {')[2],'int Packed(')[2];axis=block((src/'yee_primitives.cuh').read_text(),'constexpr int CpmlAxis(')[2]
    code=CODE.replace('// ACTUAL_AXIS',axis).replace('// ACTUAL_PACKED',packed).replace('// ACTUAL_COMPACT_HELPER',helper);out=candidate/'host-compact-profile-proof';out.mkdir(exist_ok=False);rows=[]
    for name in ('actual','alias-phase-colors','wrong-stage-phase','one-word-overrun'):
        folder=out/name;folder.mkdir();text=code
        if name=='alias-phase-colors':text=text.replace('2*(slot/6)','0*(slot/6)')
        if name=='wrong-stage-phase':text=text.replace('slot=(color/2)*6+','slot=0*(color/2)*6+')
        if name=='one-word-overrun':text=text.replace('threadIdx.x<288','threadIdx.x<289')
        cc=folder/'proof.cc';cc.write_text(text);binary=folder/'proof';built=subprocess.run(['g++','-std=c++20','-O2','-fsanitize=address,undefined',str(cc),'-o',str(binary)],capture_output=True,text=True);(folder/'build.log').write_text(built.stdout+built.stderr)
        if built.returncode:raise RuntimeError('Actual compact profile proof compile failed:'+str(folder/'build.log'))
        run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=30);(folder/'run.log').write_text(run.stdout+run.stderr)
        if (run.returncode!=0 if name=='actual' else run.returncode==0):raise RuntimeError('Unexpected compact profile proof:'+name+run.stdout+run.stderr)
        rows.append(dict(name=name,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr[-1200:]))
    result=dict(passed=True,helper_sha256=proof['helpers']['native_noio_compact_profiles.cuh'],all_active_H_E_term_kind_words_independent=True,actual_host_checks=rows);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.candidate)))
