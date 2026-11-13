"""Execute the actual generated provider helpers as C++ with unique tokens."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.cuda_trace_tile_topology import TileTopology,field_offsets,verify_topology
from scripts.cuda_trace_tile_bindings import TileReadBinding
from scripts.cuda_stencil_trace import Read


def validate(candidate):
    candidate=Path(candidate).resolve();proof=json.loads((candidate/'tile-compiler-proof.json').read_text())
    cache=tuple(proof['cache']);cz,cy,cx=cache;generated=candidate/'source/src/trace_tile_generated.cuh'
    assert hashlib.sha256(generated.read_bytes()).hexdigest()==proof['generated_sha256']
    helper=generated.read_text().split('constexpr int TraceBulkGroups=')[0]
    directory=candidate/'host-provider-proof';directory.mkdir(exist_ok=False)
    cases=[]
    reads=[]
    for field,offsets in field_offsets().items():
        field_id=['Hx','Hy','Hz','Ex','Ey','Ez'].index(field)
        for delta in sorted(offsets):reads.append('{'+','.join(map(str,(field_id,*delta)))+'}')
    for groups in sorted(set((proof['bulk_groups'],proof['base_groups']))):
        topology=TileTopology(cache,groups);plan=verify_topology(topology)
        body=f'''
#include <cstring>
#include <set>
#include <vector>
#include <iostream>
#define __device__
#define __forceinline__ inline
inline float __int_as_float(unsigned bits){{float f;std::memcpy(&f,&bits,4);return f;}}
{helper}
using S=TraceTileStorage<{groups}>;
constexpr int CZ={cz},CY={cy},CX={cx},PAD=CZ*CY*CX;
float token(int f,int z,int y,int x){{return float(1+f*PAD+(z*CY+y)*CX+x);}}
int owner(int z,int y,int x){{return (z/S::Slots)*CY*CX+y*CX+x;}}
int main(){{
 std::vector<float> frame(S::Words,__int_as_float(0x7fffffff));
 for(int z=0;z<CZ;++z)for(int y=0;y<CY;++y)for(int x=0;x<CX;++x){{
  S::Store<false>(frame.data(),z,y,x,token(0,z,y,x),token(1,z,y,x),token(2,z,y,x));
  S::Store<true>(frame.data(),z,y,x,token(3,z,y,x),token(4,z,y,x),token(5,z,y,x));
 }}
 std::set<float> distinct;
 for(float value:frame){{if(!(value>0))return 2;distinct.insert(value);}}
 if(distinct.size()!=frame.size())return 4;
 const int reads[][4]={{{','.join(reads)}}};size_t checks=0;
 for(auto&r:reads)for(int z=0;z<CZ;++z)for(int y=0;y<CY;++y)for(int x=0;x<CX;++x){{
  const int qz=(z+r[1]+CZ)%CZ,qy=(y+r[2]+CY)%CY,qx=(x+r[3]+CX)%CX;
  if(owner(z,y,x)/32==owner(qz,qy,qx)/32)continue;
  float actual=r[0]<3?S::Read<false>(frame.data(),r[0],qz,qy,qx):S::Read<true>(frame.data(),r[0]-3,qz,qy,qx);
  if(actual!=token(r[0],qz,qy,qx)){{std::cerr<<"provider mismatch";return 3;}}++checks;
 }}
 std::cout<<"{{\\\"passed\\\":true,\\\"checked_shared_reads\\\":"<<checks<<",\\\"shared_words\\\":"<<S::Words<<"}}\\n";
}}
'''
        if proof.get('inline_cell_reads'):
            direct=direct_checks(cache,generated.read_text())
            body=body.replace('float token(int f,','using Storage=S;\nint Slot(int z,int y,int x){return ((z&(CZ-1))*CY+(y&(CY-1)))*CX+(x&(CX-1));}\nfloat token(int f,')
            marker=' std::cout<<'
            at=body.index(marker);body=body[:at]+direct+body[at:]
        for negative in (False,True):
            name=f'g{groups}-'+('wrong-provider' if negative else 'actual')
            source=directory/(name+'.cc');binary=directory/name
            code=body.replace('return frame[','return frame[1+',1) if negative else body
            source.write_text(code)
            subprocess.run(['g++','-std=c++17','-O2',str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
            result=subprocess.run([str(binary)],capture_output=True,text=True)
            if result.returncode!=(3 if negative else 0):raise RuntimeError(name+result.stdout+result.stderr)
            cases.append(dict(name=name,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
                              source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),model=plan))
    result=dict(passed=True,generated_sha256=proof['generated_sha256'],cases=cases,
                scope='Actual emitted shared providers and ranks, all cache points, injective field/location tokens, wrong-provider negative control; no CUDA visibility claim')
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


def direct_checks(cache,emitted):
    """Exercise bound field expressions against independent physical tokens."""
    sections=[]
    offsets=field_offsets()
    for shared_e in (False,True):
        binder=TileReadBinding(cache,shared_e)
        if shared_e:
            sections+=[' frame.resize(S::Words/2+3*PAD);',
                       ' for(int z=0;z<CZ;++z)for(int y=0;y<CY;++y)for(int x=0;x<CX;++x)for(int c=0;c<3;++c)frame[S::Words/2+c*PAD+Slot(z,y,x)]=token(c+3,z,y,x);']
        sections+=[' for(int zs=0;zs<CZ;++zs)for(int sy=0;sy<CY;++sy)for(int sx=0;sx<CX;++sx){',
                   '  constexpr int Slots=S::Slots,Rows=S::Rows;const int j=zs%Slots,zbase=(zs/Slots)*Slots;',
                   '  float h[3][Slots],e[3][Slots];',
                   '  for(int c=0;c<3;++c)for(int k=0;k<Slots;++k){h[c][k]=token(c,zbase+k,sy,sx);e[c][k]=token(c+3,zbase+k,sy,sx);}']
        for phase in (0,1):
            sign=1 if phase==0 else -1;source_base=3 if phase==0 else 0
            sections+=['  {',f'   const int qy=((owner(zs,sy,sx)/32*32+((owner(zs,sy,sx)%32+({sign})*CX+32)%32))%(CY*CX))/CX;',
                       f'   const float wx1=token({source_base+1},zs,sy,(sx+({sign})+CX)%CX),wx2=token({source_base+2},zs,sy,(sx+({sign})+CX)%CX);',
                       f'   const float wy0=token({source_base},zs,qy,sx),wy2=token({source_base+2},zs,qy,sx);']
            output='Hx' if phase==0 else 'Ex'
            source_fields=('Ex','Ey','Ez') if phase==0 else ('Hx','Hy','Hz')
            reads=[Read(f,int(source_base==0),delta) for f in source_fields for delta in sorted(offsets[f])]
            reads += [Read(('H' if phase==0 else 'E')+c,0,(0,0,0)) for c in 'xyz']
            for read in reads:
                expr=binder(output,read)
                if expr not in emitted:raise ValueError('Bound expression not present in actual generated CUDA: '+expr)
                f=['Hx','Hy','Hz','Ex','Ey','Ez'].index(read.field);dz,dy,dx=read.offset
                sections.append(f'   if(({expr})!=token({f},(zs+({dz})+CZ)%CZ,(sy+({dy})+CY)%CY,(sx+({dx})+CX)%CX))return 5;')
            sections+=['  }']
        sections+=[' }']
    return '\n'.join(sections)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True)
    a=p.parse_args();result=validate(a.candidate);print(json.dumps(dict(passed=result['passed'],cases=len(result['cases']))))
