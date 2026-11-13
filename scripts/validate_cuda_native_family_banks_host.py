"""Check actual native bank lowering, exact loop inverse and profile routes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.cuda_trace_phase_cycle import block


def validate(candidate,parent):
    candidate,parent=Path(candidate).resolve(),Path(parent).resolve();src=candidate/'source/src'
    proof=json.loads((candidate/'native-family-banks-compiler-proof.json').read_text());c=(src/'asymmetric_native.cuh').read_text();old=(parent/'src/asymmetric_native.cuh').read_text()
    if hashlib.sha256(c.encode()).hexdigest()!=proof['consumer_sha256'] or hashlib.sha256(old.encode()).hexdigest()!=proof['parent_consumer_sha256']:raise ValueError('Explicit family-bank ancestry changed')
    for edit in proof['bindings']:
        signature=edit['signature'];body=block(c,signature)[2]
        body=once(body,edit['new_phase'],edit['old_phase']);body=once(body,edit['new_retire'],edit['old_retire'])
        body=once(body,'return frame[(3+field)*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];','return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];')
        body=once(body,'bases=reinterpret_cast<size_t*>(frame+6*PAD);','bases=reinterpret_cast<size_t*>(frame+3*PAD);')
        body=once(body,' volatile int*window=phase_step+1;\n\n', ' volatile int*window=phase_step+1;\n float*low_stage=frame+3*PAD+8*sizeof(size_t)/sizeof(float)+3;\n')
        body=once(body,'StagePartialIndependentCpmlProfiles(p,profiles);','StageIndependentCpmlProfiles(p,profiles);')
        body=body.replace('Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes,CachedProfiles,10>env','Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes,CachedProfiles>env')
        if body!=block(old,signature)[2]:raise ValueError('Exact native loop inverse failed:'+signature)
    for filename in ('asymmetric_native_cell_generated.cuh','yee_cell_generated.cuh'):
        if (src/filename).read_bytes()!=(parent/'src'/filename).read_bytes():raise ValueError('Canonical arithmetic header changed')
    for signature in ('void SourcePendingPairedBody(','void Observe(','FieldPacket ImportPacket(','int Packed(','bool LocalActive(','int PsiOffset(','float LoadLocal(','void StoreLocal(','float Difference(','float Finish('):
        if block(c,signature)[2]!=block(old,signature)[2]:raise ValueError('Unrelated math/effect helper changed:'+signature)
    for signature in ('void UniformBody(','void FixedCubeCachedBody('):
        body=block(c,signature)[2]
        if body.count('// F1')!=1 or body.count('// F2')!=1 or 'low_stage' in body:raise ValueError('Two-family phase binding incomplete')
    # Execute the actual12-term and10-term stage helpers and the actual generic
    # and fixed coefficient methods with all12independent term/kind arrays.
    helpers='\n'.join(block(c,s)[2] for s in ('int IndependentProfileCacheIndex(','float ReadIndependentProfileCache(','void StageIndependentCpmlProfiles(','void StagePartialIndependentCpmlProfiles('))
    kind=block(c[c.index('float Coefficient('):],'if(kind>=2)')[2]
    fixed=block(c[c.index('struct FixedCubeCachedEnvironment'):],'float Coefficient(')[2]
    if 'SourcePendingSharedBytes+(Policy==5?864ULL*sizeof(float):0)' not in c:raise ValueError('Original source profile capacity changed')
    directory=candidate/'host-native-family-banks-proof';directory.mkdir(exist_ok=False)
    code='''#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#define __device__
#define __host__
#define __forceinline__ inline
struct Dim{int x;};Dim threadIdx;
struct Psi{const float*a,*b,*inv;};struct Phase{Psi psi[6];};struct Parameters{Phase phase[2];};
float __ldg(const float*p){return *p;}
'''+helpers+'''
template<int ProfileTerms,bool CachedProfiles=true>struct Env{Parameters p;float*profiles;int selected=0;
 int Packed(int,int,int,int)const{return selected;}
 float Coefficient(int kind,int index,int z,int y,int x)const{
'''+kind+'''
 return -1.f;
 }};
struct Fixed:Env<10>{using Base=Env<10>;using Base::profiles;using Base::Packed;
'''+fixed+'''
};
uint32_t bits(float f){uint32_t b;std::memcpy(&b,&f,4);return b;}
int main(){float data[12][3][24];Parameters p{};
 for(int s=0;s<12;++s){for(int k=0;k<3;++k)for(int n=0;n<24;++n){
  const uint32_t b=n?0x3a000001u+uint32_t((s*3+k)*24+n)*12347u:uint32_t((s+k)&1)<<31;std::memcpy(&data[s][k][n],&b,4);
 }p.phase[s/6].psi[s%6]={data[s][0],data[s][1],data[s][2]};}
 std::vector<float>a(724,-123.f),b(868,-123.f);
 for(threadIdx.x=0;threadIdx.x<1024;++threadIdx.x){StagePartialIndependentCpmlProfiles(p,a.data());StageIndependentCpmlProfiles(p,b.data());}
 for(int i=720;i<724;++i)if(a[i]!=-123.f)return 4;for(int i=864;i<868;++i)if(b[i]!=-123.f)return 5;
 Env<10>partial{p,a.data()},global{p,a.data()};Env<12>source{p,b.data()};Fixed fixed;fixed.p=p;fixed.profiles=a.data();size_t checks=0;
 for(int s=0;s<12;++s)for(int k=0;k<3;++k)for(int n=0;n<24;++n){partial.selected=source.selected=fixed.selected=n;
  const uint32_t expected=bits(data[s][k][n]);
  if(bits(partial.Coefficient(k+2,s,0,0,0))!=expected||bits(source.Coefficient(k+2,s,0,0,0))!=expected||bits(fixed.Coefficient(k+2,s,0,0,0))!=expected)return 3;
  checks+=3;
 }
 std::cout<<checks<<"\\n";
}
'''
    rows=[]
    for name in ('actual','wrong-cache-count','partial-stage-overrun'):
        trial=code
        if name=='wrong-cache-count':trial=trial.replace('index<ProfileTerms','index<12')
        if name=='partial-stage-overrun':trial=trial.replace('i<720','i<864')
        cc=directory/(name+'.cc');binary=directory/name;cc.write_text(trial)
        subprocess.run(['g++','-std=c++17','-O2','-fsanitize=address,undefined',str(cc),'-o',str(binary)],capture_output=True,text=True,check=True)
        r=subprocess.run([str(binary)],capture_output=True,text=True)
        if (r.returncode!=0 if name=='actual' else r.returncode==0):raise RuntimeError(name+r.stdout+r.stderr)
        rows.append(dict(name=name,returncode=r.returncode,stdout=r.stdout))
    result=dict(passed=True,consumer_sha256=proof['consumer_sha256'],exact_native_loop_inverse=True,unchanged_effect_helpers=True,rows=rows,
                scope='Actual stage10/stage12, generic/fixed/source profile routes with all864independent FP32 words and signed zeros, canaries, ASan/UBSan andtwo negative controls. Exact loop inverse recovers original math, source, global stores and clock semantics; shared phase race proof and GPU checks separate.')
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Unique family-bank inverse required:'+old[:100])
    return text.replace(old,new,1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.candidate,a.parent)))
