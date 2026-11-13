"""Check emitted source-warp coordinates, providers and typed neighbor operands."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.cuda_trace_source_warp_codegen import SourceWarpReadBinding
from scripts.cuda_trace_source_warp_topology import SourceWarpTopology
from scripts.cuda_space_time_ir import FIELDS,yee_operator


def validate(candidate):
    candidate=Path(candidate).resolve();proof=json.loads((candidate/'source-warp-compiler-proof.json').read_text())
    topology=SourceWarpTopology(x_slots=proof['slots'])
    actual=(candidate/'source/src/trace_source_warp_generated.cuh').read_text()
    if hashlib.sha256(actual.encode()).hexdigest()!=proof['source_sha256']:raise ValueError('Source warp source changed')
    helpers=actual.split('template<int Depth,int Layout,bool Fast')[0]
    source=['''#include <cstdint>
#include <iostream>
#include <vector>
#define __device__
#define __forceinline__ inline
constexpr int CZ=8,CY=16,CX=32,PAD=CZ*CY*CX;
''',helpers,'''uint64_t token(int f,int v,int z,int y,int x){return (uint64_t(v)<<48)|(uint64_t(f)<<45)|(uint64_t(z+8192)<<30)|(uint64_t(y+8192)<<15)|uint64_t(x+8192);}
int origin_z,origin_y,warp_thread;
uint64_t __shfl_sync(unsigned,uint64_t value,int lane){
 const int v=int(value>>48),f=int((value>>45)&7),low=-v-1;
 const int zs=(warp_thread/32/(CX/SourceWarpSlots))*2+(lane>>4),sy=lane&15;
 const int z=origin_z+low+((zs-low)&7),y=origin_y+low+((sy-low)&15),x=int(value&32767)-8192;
 return token(f,v,z,y,x);
}
int main(){
 std::vector<uint64_t>frame(SourceWarpScratchWords);size_t checks=0;
 for(int n:{1,2,7,8,9,15,16,17,31,32,33,35}){
 const int oz=21,oy=45,ox=62;origin_z=oz;origin_y=oy;
''']
    operator=yee_operator(cpml_axes=(0,1,2))
    for phase in (0,1):
        source.append(' {')
        if proof.get('shared_e'):
            for field in range(3,6):
                source.append(f'''  for(int q=0;q<PAD;++q){{const int z=q/(CY*CX),y=(q/CX)&15,x=q&31;const auto s=SourceWarpPosition(z,y,x,n-1,oz,oy,ox);frame[{field-3}*PAD+SourceWarpESlot(z,y,x)]=token({field},n-1,s.z,s.y,s.x);}}''')
        for field,name in enumerate(FIELDS[:3] if proof.get('shared_e') else FIELDS):
            version='n' if phase and field<3 else 'n-1'
            for combined in topology.providers['slots'][name]:
                zl,slot=divmod(combined,topology.slots)
                source.append(f'''  for(int tid=0;tid<SourceWarpThreads;++tid){{if((tid%32/16)!={zl})continue;
   const int zs=(tid/32/(CX/SourceWarpSlots))*2+(tid%32/16),sy=tid%16,sx=(tid/32%(CX/SourceWarpSlots))*SourceWarpSlots+{slot};
   const auto s=SourceWarpPosition(zs,sy,sx,{version},oz,oy,ox);const int index={"3*PAD+" if proof.get("shared_e") else ""}SourceWarpProviderIndex<{field}>(zs,sy,sx);
   if(index<0||index>=SourceWarpScratchWords)return 4;frame[index]=token({field},{version},s.z,s.y,s.x);
  }}''')
        source.append('''  for(int tid=0;tid<SourceWarpThreads;++tid){const int lane=tid&31;
   const int zs=(tid/32/(CX/SourceWarpSlots))*2+(lane>>4),sy=lane&15,xbase=(tid/32%(CX/SourceWarpSlots))*SourceWarpSlots;
   warp_thread=tid&~31;uint64_t h[3][SourceWarpSlots],e[3][SourceWarpSlots];''')
        for slot in range(topology.slots):
            for field in range(6):
                version='n' if phase and field<3 else 'n-1';family='e' if field>=3 else 'h'
                source.append(f'''   {{const auto s=SourceWarpPosition(zs,sy,xbase+{slot},{version},oz,oy,ox);{family}[{field%3}][{slot}]=token({field},{version},s.z,s.y,s.x);}}''')
        for slot in range(topology.slots):
            source.append(f'   {{const int sx=xbase+{slot};const auto s=SourceWarpPosition(zs,sy,sx,n,oz,oy,ox);if(s.owned)'+'{')
            binding=SourceWarpReadBinding(topology,slot,proof.get("shared_e",False));tests=[]
            for update in operator.updates:
                output=update.output.field
                if output not in FIELDS or int(output.startswith('E'))!=phase:continue
                for read in sorted(update.reads):
                    if read.field not in FIELDS:continue
                    expression=binding(output,read)
                    if expression not in actual:raise ValueError('Actual source warp operand missing')
                    sf=FIELDS.index(read.field);version='n' if read.version else 'n-1';dz,dy,dx=read.offset
                    tests.append(f'    if(({expression})!=token({sf},{version},s.z+({dz}),s.y+({dy}),s.x+({dx})))return 3;++checks;')
            for name,(value,target) in binding.shuffles.items():source.append(f'    const uint64_t {name}=__shfl_sync(0xffffffffu,{value},{target});')
            source+=tests+['   }}']
        source+=['  }',' }']
    source+=[' }',' std::cout<<"{\\\"passed\\\":true,\\\"checked_source_reads\\\":"<<checks<<"}\\n";','}']
    directory=candidate/'host-source-warp-proof';directory.mkdir(exist_ok=False)
    code='\n'.join(source);rows=[]
    for name in ('actual','wrong-provider','wrong-z-shuffle'):
        path=directory/(name+'.cc');binary=directory/name
        modified=code.replace('((zs+(-1))&7,(sy+(0))&15,(sx+(0))&31)', '((zs+(-1))&7,((sy+(0))&15)^1,(sx+(0))&31)') if name=='wrong-provider' else code.replace('(lane^16)','lane',1) if name=='wrong-z-shuffle' else code
        if name!='actual' and modified==code:raise ValueError('Negative source transport witness missing')
        path.write_text(modified)
        subprocess.run(['g++','-std=c++17','-O2',str(path),'-o',str(binary)],capture_output=True,text=True,check=True)
        result=subprocess.run([str(binary)],capture_output=True,text=True)
        if result.returncode!=(0 if name=='actual' else 3):raise RuntimeError(name+result.stdout+result.stderr+str(result.returncode))
        rows.append(dict(name=name,returncode=result.returncode,stdout=result.stdout,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result=dict(passed=True,generated_sha256=proof['source_sha256'],rows=rows,
                scope='Actual emitted coordinates/provider ranks and bound register/shared operands; simulated Y/Z shuffles and two negative controls; no CUDA visibility claim')
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True)
    a=p.parse_args();print(json.dumps(validate(a.candidate)))
