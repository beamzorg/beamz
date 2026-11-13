"""Execute emitted diamond sites and bound reads using injective physical tokens."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.cuda_trace_diamond_geometry import DiamondReadBinding,DiamondGeometry
from scripts.cuda_space_time_ir import FIELDS,yee_operator


def validate(candidate):
    candidate=Path(candidate).resolve();proof=json.loads((candidate/'diamond-compiler-proof.json').read_text())
    header=candidate/'source/src/trace_tile_generated.cuh';text=header.read_text()
    assert hashlib.sha256(header.read_bytes()).hexdigest()==proof['source_sha256']
    helpers=text.split('template<int Depth,int Layout,int Field>')[0]
    directory=candidate/'host-diamond-proof';directory.mkdir(exist_ok=False)
    cz,cy,cx=proof['cache'];geometry=DiamondGeometry(tuple(proof['cache']),proof.get('orientation','xy'),proof.get('storage_gauge','retimed'));binding=DiamondReadBinding(geometry);operator=yee_operator(cpml_axes=(0,1,2))
    source=['''#include <cstdint>
#include <iostream>
#include <vector>
#define __device__
#define __forceinline__ inline
''',f'constexpr int CZ={cz},CY={cy},CX={cx},PAD=CZ*CY*CX;',helpers,
'''uint64_t token(int f,int v,int z,int y,int x){return (uint64_t(v)<<48)|(uint64_t(f)<<45)|(uint64_t(z+8192)<<30)|(uint64_t(y+8192)<<15)|uint64_t(x+8192);}
int main(){
 std::vector<uint64_t>frame(6*DiamondPoints);size_t checks=0;
 const int versions[]={1,2,15,16,17,31,32,33,35};
 for(int bias:{136,137})for(int odd:{0,1})for(int n:versions){
  const int oz=30,op=93,om=odd?0:15;
''']
    for phase in (0,1):
        source.append('  {')
        for field in range(6):
            version='n' if phase==1 and field<3 else 'n-1'
            source.append(f'''   for(int q=0;q<DiamondPoints;++q){{auto s=DiamondPosition<{field}>(q,{version},oz,op,om,bias);frame[{field}*DiamondPoints+q]=token({field},{version},s.z,s.y,s.x);}}''')
        for update in operator.updates:
            output=update.output.field
            if output not in FIELDS or int(output.startswith('E'))!=phase:continue
            field=FIELDS.index(output)
            source+=['   for(int q=0;q<DiamondPoints;++q){',
                     f'    const auto s=DiamondPosition<{field}>(q,n,oz,op,om,bias);if(!s.owned)continue;',
                     ('    const int qz=q/(CY*DiamondHalfX),qp=(q/DiamondHalfX)&(CY-1),m=s.m;' if geometry.orientation=='xy' else '    const int q0=q/(DiamondHalfY*CX),q1=s.m,q2=q&(CX-1);')]
            for read in sorted(update.reads):
                if read.field not in FIELDS:continue
                expression=binding(output,read)
                if expression not in text:raise ValueError('Actual generated diamond operand missing')
                sf=FIELDS.index(read.field);dz,dy,dx=read.offset;version='n' if read.version else 'n-1'
                source.append(f'    if(({expression})!=token({sf},{version},s.z+({dz}),s.y+({dy}),s.x+({dx})))return 3;++checks;')
            source.append('   }')
        source.append('  }')
    source+=[' }',' std::cout<<"{\\\"passed\\\":true,\\\"checked_bound_reads\\\":"<<checks<<"}\\n";','}']
    code='\n'.join(source);rows=[]
    for negative in (False,True):
        name='wrong-stride' if negative else 'actual';src=directory/(name+'.cc');binary=directory/name
        modified=(code.replace('*DiamondHalfX+','*(DiamondHalfX-1)+',1) if geometry.orientation=='xy' else code.replace('*DiamondHalfY+','*(DiamondHalfY-1)+',1)) if negative else code
        src.write_text(modified)
        subprocess.run(['g++','-std=c++17','-O2',str(src),'-o',str(binary)],check=True,capture_output=True,text=True)
        result=subprocess.run([str(binary)],capture_output=True,text=True)
        if result.returncode!=(3 if negative else 0):raise RuntimeError(name+result.stdout+result.stderr)
        rows.append(dict(name=name,returncode=result.returncode,stdout=result.stdout,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
    result=dict(passed=True,generated_sha256=proof['source_sha256'],rows=rows,
                scope='Actual emitted physical site decode and bound field reads; two physical-domain and owner parities, cache wrap versions, wrong-stride control; no CUDA visibility or native planner claim')
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True)
    a=p.parse_args();print(json.dumps(validate(a.candidate)))
