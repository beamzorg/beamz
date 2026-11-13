"""Execute emitted CUDA frontier address helpers as ordinary host C++.

This complements the symbolic proofs by checking the *actual emitted* compact
Y/Z store/read expressions, every rotating head and both field versions. It
does not establish CUDA visibility or floating-point equation correctness.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def verify(candidate, *, steps=35, label='host-frontier-proof'):
    candidate = Path(candidate).resolve()
    proof = json.loads((candidate/'column-schedule-proof.json').read_text())
    source = candidate/'source/src/asymmetric_native.cuh'
    text = source.read_text()
    if hashlib.sha256(source.read_bytes()).hexdigest() != proof['consumer_sha256']:
        raise ValueError('Staged consumer hash changed')
    if not proof['compact_shared_providers'] or proof['z_groups'] < 2:
        raise ValueError('Host helper verifier admits compact multiple-group layouts')
    begin = text.index('constexpr bool ColumnCacheProfiles=')
    end = text.index('template<int Depth,int N=0,int ActiveAxes=-1,bool Fast=false,bool UniformE=true>')
    helpers = text[begin:end]
    if proof['deduplicated_frontiers']:
        read = 'ColumnReadZValue<Electric>(frame.data(),c,z,y,x,n,delta)'
        negative_helpers = helpers.replace('sourcez=(z+delta)', 'sourcez=(z+delta+1)')
    else:
        read = 'frame[ColumnZOffset+((Electric?ColumnEBase:0)+(c==1))*ColumnZPoints+ColumnZReadIndex(z,y,x,n,delta)]'
        negative_helpers = helpers.replace('((z+delta)', '((z+delta+1)')
    if negative_helpers == helpers:
        raise ValueError('Negative control did not alter a real emitted read')
    prefix = '''#include <vector>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <limits>
#define __device__
#define __forceinline__ inline
constexpr int CZ=8,CY=16,CX=32,BZ=7,BY=15,BX=31,PAD=CZ*CY*CX;
'''
    cache=proof['schedule']['mapping']['cache']
    prefix=prefix.replace('CZ=8,CY=16,CX=32,BZ=7,BY=15,BX=31',
                           f'CZ={cache[0]},CY={cache[1]},CX={cache[2]},BZ={cache[0]-1},BY={cache[1]-1},BX={cache[2]-1}')
    body = r'''
long long checked=0;
float Token(int c,int z,int y,int x,int version){return float(1+((((version*3+c)*CZ+z)*CY+y)*CX+x));}
template<bool Electric>bool Check(int n){
 const int version=Electric?n-1:n;
 std::vector<float> frame(ColumnBaseBytes/sizeof(float),std::numeric_limits<float>::quiet_NaN());
 for(int z=0;z<CZ;++z)for(int y=0;y<CY;++y)for(int x=0;x<CX;++x)for(int c=0;c<3;++c){
  const float value=Token(c,z,y,x,version);
  if(c!=1)ColumnStoreComponent<Electric>(frame.data(),c,z,y,x,version,value);
  if(c!=2)ColumnStoreZComponent<Electric>(frame.data(),c,z,y,x,version,value);
 }
 for(int z=0;z<CZ;++z)for(int y=0;y<CY;++y)for(int x=0;x<CX;++x){
  const int dz=(z+n+1)&7,dy=(y+n+1)&15,dx=(x+n+1)&31;
  if(!dz||!dy||!dx)continue;
  for(int c:{0,2}){
   const float actual=frame[((Electric?ColumnEBase:0)+(c==2))*ColumnPlanePoints+ColumnReadIndex(z,y,x,n)];
   const float expected=Token(c,z,(y+(Electric?1:15))&15,x,version);
   if(actual!=expected){std::fprintf(stderr,"Y mismatch n=%d E=%d zyx=%d,%d,%d c=%d\n",n,Electric,z,y,x,c);return false;}
   ++checked;
  }
  const int delta=Electric?1:-1,sourcez=(z+delta)&7;
  if(sourcez/(CZ/ColumnZGroups)!=z/(CZ/ColumnZGroups))for(int c:{0,1}){
   const float actual=READ_EXPRESSION;
   const float expected=Token(c,sourcez,y,x,version);
   if(actual!=expected){std::fprintf(stderr,"Z mismatch n=%d E=%d zyx=%d,%d,%d c=%d\n",n,Electric,z,y,x,c);return false;}
   ++checked;
  }
 }
 return true;
}
int main(){for(int n=1;n<=NUM_STEPS;++n)if(!Check<true>(n)||!Check<false>(n))return 1;
 std::printf("{\"passed\":true,\"checked_emitted_reads\":%lld}\n",checked);return 0;}
'''.replace('READ_EXPRESSION',read).replace('NUM_STEPS',str(steps)).replace('&7','&(CZ-1)').replace('&15','&(CY-1)').replace('Electric?1:15','Electric?1:CY-1')
    if Path(label).name!=label or label in ('.','..'):
        raise ValueError('Proof label must be a single directory name')
    # Every token is a distinct exactly representable FP32 integer.
    if 1+((steps+1)*3*cache[0]*cache[1]*cache[2]) >= 2**24:
        raise ValueError('Token range exceeds exact FP32 integers')
    directory = candidate/label
    directory.mkdir(exist_ok=False)
    records=[]
    for name, definitions, expected_success in (('actual',helpers,True),('wrong-z-provider',negative_helpers,False)):
        cpp = directory/(name+'.cc')
        exe = directory/name
        cpp.write_text(prefix+definitions+body)
        command = ['g++','-std=c++17','-O2',str(cpp),'-o',str(exe)]
        compiled = subprocess.run(command,capture_output=True,text=True,timeout=60)
        if compiled.returncode:
            raise RuntimeError(compiled.stderr[-8000:])
        executed = subprocess.run([str(exe)],capture_output=True,text=True,timeout=30)
        if (executed.returncode == 0) != expected_success:
            raise ValueError('Unexpected host emitted-helper result: '+name+' '+executed.stderr)
        records.append(dict(name=name,command=command,returncode=executed.returncode,
                            stdout=executed.stdout,stderr=executed.stderr,
                            source_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest()))
    result = dict(schema='beamz.emitted-frontier-host-proof.v1',passed=True,steps=steps,
                  consumer_sha256=proof['consumer_sha256'],records=records,
                  scope='Actual emitted address helpers and negative control; no CUDA visibility claim')
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--steps',type=int,default=35)
    p.add_argument('--label',default='host-frontier-proof')
    a=p.parse_args()
    print(json.dumps(verify(a.candidate,steps=a.steps,label=a.label)))
