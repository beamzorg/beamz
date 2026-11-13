"""Check actual bounded profile staging and uncached independent fallbacks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.cuda_trace_phase_cycle import block


def validate(candidate):
    candidate=Path(candidate).resolve();proof=json.loads((candidate/'box-warp-compiler-proof.json').read_text())
    text=(candidate/'source/src/asymmetric_native.cuh').read_text();slots=proof['profile_slots']
    if hashlib.sha256(text.encode()).hexdigest()!=proof['consumer_sha256']:raise ValueError('Final profile consumer binding changed')
    if not 0<slots<12 or proof['cached_shared_bytes']>101376:raise ValueError('Bounded subset profile frame required')
    directory=candidate/'host-profile-subset-proof';directory.mkdir(exist_ok=False)
    helpers='\n'.join(block(text,sig)[2] for sig in ('int IndependentProfileCacheIndex(','float ReadIndependentProfileCache(','void StageIndependentCpmlProfiles('))
    a=text.index('float Coefficient(');_,_,kind_block=block(text[a:],'if(kind>=2)')
    fixed=text.index('struct FixedCubeCachedEnvironment');_,_,fixed_body=block(text[fixed:],'float Coefficient(')
    check=f'if(kind>=2&&index<{slots})return ReadIndependentProfileCache(profiles,index,kind,Packed(index,z,y,x));'
    if check not in fixed_body or 'return Base::Coefficient(kind,index,z,y,x);' not in fixed_body:raise ValueError('Fixed-cube fallback binding changed')
    launch='SourcePendingSharedBytes+(Policy==5?'+str(slots*72)+'ULL*sizeof(float):0)'
    if launch not in text:raise ValueError('Source launch allocates a different profile subset')
    code='''#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#define __host__
#define __device__
#define __forceinline__ inline
struct Dim{int x;};Dim threadIdx,blockDim;
struct Psi{const float*a,*b,*inv;};struct Phase{Psi psi[6];};struct Parameters{Phase phase[2];};
float __ldg(const float*p){return *p;}
'''+helpers+'''
template<bool CachedProfiles>struct Env{Parameters p;float*profiles;
 int Packed(int,int,int,int)const{return 0;}
 float Coefficient(int kind,int index,int,int,int,int packed)const{
'''+kind_block+'''
 return -1.f;
 }};
uint32_t bits(float v){uint32_t b;std::memcpy(&b,&v,4);return b;}
int main(){
 float original[12][3][24];Parameters p{};
 for(int s=0;s<12;++s){for(int k=0;k<3;++k)for(int n=0;n<24;++n){
  const uint32_t b=0x3a000001u+uint32_t((s*3+k)*24+n)*12347u;std::memcpy(&original[s][k][n],&b,4);
 }p.phase[s/6].psi[s%6]={original[s][0],original[s][1],original[s][2]};}
'''+f''' std::vector<float>staged({slots*72}+4,-123.f);blockDim.x={proof['threads']};
 for(threadIdx.x=0;threadIdx.x<blockDim.x;++threadIdx.x)StageIndependentCpmlProfiles(p,staged.data());
 for(int i={slots*72};i<{slots*72}+4;++i)if(staged[i]!=-123.f)return 4;
 Env<true>env{{p,staged.data()}};size_t checks=0;
 for(int s=0;s<12;++s)for(int k=0;k<3;++k)for(int n=0;n<24;++n){{
  if(bits(env.Coefficient(k+2,s,0,0,0,n))!=bits(original[s][k][n]))return 3;++checks;
 }}
 std::cout<<checks<<"\\n";
}}
'''
    # Use the actual extracted kind block, but Packed's input is independently
    # chosen by the harness rather than recomputed by geometry in this proof.
    code=code.replace('const int packed=Packed(index,z,y,x);','')
    # The extracted block uses the original local coordinate names. Bind them
    # even though its cached/global branches use only the supplied packed index.
    code=code.replace('float Coefficient(int kind,int index,int,int,int,int packed)',
                      'float Coefficient(int kind,int index,int z,int y,int x,int packed)')
    rows=[]
    for name in ('actual','cache-guard-overrun'):
        trial=code if name=='actual' else code.replace(f'if(index<{slots})', 'if(index<12)')
        src=directory/(name+'.cc');binary=directory/name;src.write_text(trial)
        subprocess.run(['g++','-std=c++17','-O2','-fsanitize=address,undefined',str(src),'-o',str(binary)],capture_output=True,text=True,check=True)
        r=subprocess.run([str(binary)],capture_output=True,text=True)
        if (r.returncode!=0 if name=='actual' else r.returncode==0):raise RuntimeError(name+r.stdout+r.stderr)
        rows.append(dict(name=name,returncode=r.returncode,stdout=r.stdout,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
    result=dict(passed=True,consumer_sha256=proof['consumer_sha256'],profile_slots=slots,cached_shared_bytes=proof['cached_shared_bytes'],rows=rows,
                scope='Actual staging and coefficient kind block; all864independent FP32 profile words, cache canaries, fixed-cube fallback/source launch binding, ASan/UBSan, overrun negative')
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.candidate)))
