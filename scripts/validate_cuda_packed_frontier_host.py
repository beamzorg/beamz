"""Execute actual generated packed-layout helpers with bit-pattern round trips.

This proves the conversion address map is injective, component-disjoint and
bounded for canonical staggered supports. It does not prove CUDA visibility.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def validate(candidate):
    candidate=Path(candidate).resolve();proof=json.loads((candidate/'physical-frontier-compiler-proof.json').read_text())
    consumer=(candidate/'source/src/asymmetric_native.cuh').read_text()
    if hashlib.sha256(consumer.encode()).hexdigest()!=proof['consumer_sha256']:raise ValueError('Compiled consumer identity changed')
    layout=proof['generated_layout']
    if consumer.count(layout)!=1:raise ValueError('Actual generated layout missing')
    directory=candidate/'host-packed-layout-proof';directory.mkdir(exist_ok=False)
    code='''#include <cstdint>
#include <cstddef>
#include <vector>
#include <iostream>
#define __host__
#define __device__
#define __forceinline__ inline
'''+layout+'''
int main(){size_t checks=0;const int ds[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
 for(int nz:{5,17,32})for(int ny:{7,16,33})for(int nx:{9,31,32,33,63,64}){
  const size_t words=PhysicalFrontierWords(nz,ny,nx);std::vector<uint64_t>bank(words,0);std::vector<std::vector<uint64_t>>native(6);
  size_t token=1;
  for(int f=0;f<6;++f){const int zdim=nz+ds[f][0],ydim=ny+ds[f][1],xdim=nx+ds[f][2];native[f].resize(size_t(zdim)*ydim*xdim);
   for(int z=0;z<zdim;++z)for(int y=0;y<ydim;++y)for(int x=0;x<xdim;++x){
    const int i=PhysicalFrontierIndex(ny,nx,z,y,x)+f*32;
    if(i<0||size_t(i)>=words||bank[i]!=0)return 3;
    // Distinct patterns stand for FP32 bits, including nonzero initial state.
    const uint64_t value=(uint64_t(f+1)<<56)|token++;
    native[f][(size_t(z)*ydim+y)*xdim+x]=value;bank[i]=value;++checks;
   }
  }
  // Independently decode each occupied AoSoA word and compare its physical
  // position/component to the canonical input. This tests inverse layout too.
  for(size_t i=0;i<words;++i){if(!bank[i])continue;
   const size_t block=i/192;const int lane=i%32,f=(i%192)/32;
   const int groups=(nx+1+31)/32,x=int(block%groups)*32+lane,y=int((block/groups)%(ny+1)),z=block/(size_t(groups)*(ny+1));
   const int zdim=nz+ds[f][0],ydim=ny+ds[f][1],xdim=nx+ds[f][2];
   if(z>=zdim||y>=ydim||x>=xdim)return 4;
   if(bank[i]!=native[f][(size_t(z)*ydim+y)*xdim+x])return 5;++checks;
  }
 }
 std::cout<<checks<<"\\n";
}
'''
    rows=[]
    for name in ('actual','wrong-component-offset','wrong-row-groups'):
        trial=code
        if name=='wrong-component-offset':trial=trial.replace('+f*32;','+f*31;')
        if name=='wrong-row-groups':trial=trial.replace('(nx+32)/32','(nx+31)/32')
        src=directory/(name+'.cc');binary=directory/name;src.write_text(trial)
        subprocess.run(['g++','-std=c++17','-O2','-fsanitize=address,undefined',str(src),'-o',str(binary)],capture_output=True,text=True,check=True)
        r=subprocess.run([str(binary)],capture_output=True,text=True)
        if (r.returncode!=0 if name=='actual' else r.returncode==0):raise RuntimeError(name+r.stdout+r.stderr)
        rows.append(dict(name=name,returncode=r.returncode,stdout=r.stdout,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
    result=dict(passed=True,consumer_sha256=proof['consumer_sha256'],rows=rows,
                scope='Actual emitted AoSoA helper bounds, injectivity and inverse bit-pattern conversion;54ragged/warp-boundary shapes, all6staggered fields, ASan/UBSan, two negative controls; no CUDA result')
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.candidate)))
