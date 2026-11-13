"""Compare emitted packed fact helpers with the actual native predicates."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.cuda_trace_phase_cycle import block
from scripts.cuda_trace_point_facts import CPP


def validate(source,output):
    source,output=Path(source).resolve(),Path(output).resolve();output.mkdir(exist_ok=False)
    native=(source/'src/asymmetric_native.cuh').read_text()
    if hashlib.sha256(native.encode()).hexdigest()!='a7104fb39695c2a3d8df0bac3db20d647957851e89c655a81125f601742d25c0':raise ValueError('Reviewed frozen native environment required')
    primitives=(source/'src/yee_primitives.cuh').read_text()
    axis=block(primitives,'constexpr int CpmlAxis(')[2]
    methods='\n'.join(block(native,sig)[2] for sig in ('bool Valid(','int Packed(','bool LocalActive(','float Difference('))
    support=block(native,'unsigned NativeSupportMask(')[2]
    code='''#include <cstdint>
#include <iostream>
#include <vector>
#include <set>
#define __device__
#define __host__
#define __forceinline__ inline
float __fsub_rn(float a,float b){return a-b;}
namespace beamz::cuda::yee{
'''+axis+'''
float ScaleYeeDifference(float a,float inv){return a*inv;}
}
struct Field{int z,y,x;};struct Phase{Field own[3];float inv=1.f;};
struct Parameters{Phase phase[2];int physical_n[3];};
'''+support+CPP+'''
struct Reference{const Parameters&p;static constexpr bool SupportSafe=false,HasCpml=true;static constexpr int ActiveAxes=-1;uint8_t flags=14;
'''+methods+'''
};
int main(){size_t checks=0;
 for(int nz:{27,32,512,564}){const int ny=nz+2,nx=nz+4;Parameters p{};p.physical_n[0]=nz;p.physical_n[1]=ny;p.physical_n[2]=nx;
 p.phase[0].own[0]={nz,ny,nx+1};p.phase[0].own[1]={nz,ny+1,nx};p.phase[0].own[2]={nz+1,ny,nx};Reference ref{p};
 std::vector<std::vector<int>>points;
 for(int size:{nz,ny,nx}){std::set<int>s;for(int q=-2;q<15;++q)s.insert(q);for(int q=size-14;q<=size+2;++q)s.insert(q);s.insert(size/2);points.emplace_back(s.begin(),s.end());}
 for(int z:points[0])for(int y:points[1])for(int x:points[2]){const auto tag=EncodeNativePointFacts(p,z,y,x);
  for(int f=0;f<6;++f){const bool valid=ref.Valid(f,z,y,x);if(bool(tag&(1u<<f))!=valid)return 3;if(!valid)continue;
   const int family=f/3,component=f%3;
   for(int term=component*2;term<component*2+2;++term){const int slot=family*6+term,axis=beamz::cuda::yee::CpmlAxis(term);
    if(NativePointLocalActive(tag,slot)!=ref.LocalActive(slot,z,y,x))return 4;
    if(ref.LocalActive(slot,z,y,x)&&NativePointPacked(tag,slot)!=ref.Packed(slot,z,y,x))return 5;
    if(family){const bool wall=ref.Difference(1,0,axis,z,y,x,1.f,0.f)==0.f;if(NativePointElectricWall(tag,axis)!=wall)return 6;}
    ++checks;
   }
  }
 }
 }
 std::cout<<checks<<"\\n";
}
'''
    rows=[]
    for name in ('actual','wrong-high-E-layer','wrong-wall-stagger'):
        trial=code
        if name=='wrong-high-E-layer':trial=trial.replace('slot>=6?13:12','slot>=6?12:12')
        if name=='wrong-wall-stagger':trial=trial.replace('c==0||c==24','c==0||c==23')
        src=output/(name+'.cc');binary=output/name;src.write_text(trial)
        subprocess.run(['g++','-std=c++17','-O2','-fsanitize=address,undefined',str(src),'-o',str(binary)],capture_output=True,text=True,check=True)
        r=subprocess.run([str(binary)],capture_output=True,text=True)
        if (r.returncode!=0 if name=='actual' else r.returncode==0):raise RuntimeError(name+r.stdout+r.stderr)
        rows.append(dict(name=name,returncode=r.returncode,stdout=r.stdout,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
    result=dict(passed=True,rows=rows,reference_consumer_sha256=hashlib.sha256(native.encode()).hexdigest(),helper_sha256=hashlib.sha256(CPP.encode()).hexdigest(),
                scope='Actual emitted tag helpers vs frozen native Valid/LocalActive/Packed/Difference/CpmlAxis methods on valid staggered terms; four clipped/ragged shape families and all boundary coordinates, ASan/UBSan, two negative controls')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.source,a.output)))
