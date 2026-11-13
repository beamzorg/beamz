"""Exercise actual private diamond operands with injective physical/version tokens."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess

from scripts.cuda_trace_diamond_private_codegen import PrivateReadBinding,DynamicPrivateReadBinding
from scripts.cuda_trace_diamond_storage import DiamondWarpTopology
from scripts.cuda_trace_diamond_geometry import DiamondGeometry
from scripts.cuda_space_time_ir import FIELDS,yee_operator


def validate(candidate,proof_name='host-private-proof'):
    candidate=Path(candidate).resolve();proof=json.loads((candidate/'diamond-private-proof.json').read_text())
    topology=DiamondWarpTopology(proof['plus'],proof['minus'],DiamondGeometry((16,16,32),'yz',proof.get('storage_gauge','retimed')),proof.get('parity_providers',False),proof.get('shared_field'))
    general=(candidate/'source/src/trace_tile_generated.cuh').read_text()
    actual=(candidate/'source/src/trace_diamond_private_generated.cuh').read_text()
    if hashlib.sha256(actual.encode()).hexdigest()!=proof['source_sha256']:raise ValueError('Private source identity changed')
    helpers=general.split('template<int Depth,int Layout,int Field>')[0]
    private=actual.split('// Begin GPU memory effects.')[0] if proof.get('factorized_cell') else actual.split('template<int Depth,int Layout,class Env>' if proof.get('dynamic_parity_operands') else 'template<int Depth,int Layout,int Parity,class Env>')[0]
    source=['''#include <cstdint>
#include <iostream>
#include <vector>
#define __device__
#define __forceinline__ inline
constexpr int CZ=16,CY=16,CX=32,PAD=CZ*CY*CX;
''',helpers,private,'''uint64_t token(int f,int v,int z,int y,int x){return (uint64_t(v)<<48)|(uint64_t(f)<<45)|(uint64_t(z+8192)<<30)|(uint64_t(y+8192)<<15)|uint64_t(x+8192);}
int shuffle_origin;
uint64_t __shfl_sync(unsigned,uint64_t value,int lane){
 const int version=int(value>>48),low=-version-1;
 const int x=shuffle_origin+low+((lane-low)&(CX-1));
 return (value&~uint64_t(32767))|uint64_t(x+8192);
}
int main(){
 std::vector<uint64_t>frame(DiamondProviderWords);size_t checks=0;
 for(int bias:{136,137})for(int odd:{0,1})for(int n:{1,2,15,16,17,31,32,33,35}){
 const int oz=30,op=93+odd,om=62,parity=(oz+op-bias)&1;shuffle_origin=om;
''']
    operator=yee_operator(cpml_axes=(0,1,2))
    for phase in (0,1):
        source.append(' {')
        if topology.shared_field is not None:
            sf=FIELDS.index(topology.shared_field);sv='n' if phase and sf<3 else 'n-1'
            source.append(f'''   for(int q=0;q<DiamondPoints;++q){{const auto s=DiamondPosition<{sf}>(q,{sv},oz,op,om,bias);frame[DiamondSharedFieldOffset+q]=token({sf},{sv},s.z,s.y,s.x);}}''')
        for field,name in enumerate(FIELDS):
            if name==topology.shared_field:continue
            version='n' if phase and field<3 else 'n-1'
            for slot in topology.providers['slots'][name]:
                store_check=''
                if topology.parity_providers:
                    for parity,layout in enumerate(topology.provider_layouts):
                        if slot not in layout['slots'][name]:continue
                        emitted=f'frame[DiamondProviderIndexTyped<{field},{slot},{parity}>(warp,lane)]='
                        if emitted not in actual:raise ValueError('Actual typed provider store missing')
                        store_check+=f'if(parity=={parity}&&DiamondProviderIndexTyped<{field},{slot},{parity}>(tid>>5,tid&31)!=index)return 7;'
                source.append(f'''  for(int tid=0;tid<DiamondPrivateThreads;++tid){{const int q=DiamondPrivateQ<{slot}>(tid);const auto s=DiamondPosition<{field}>(q,{version},oz,op,om,bias);
   const int index=DiamondProviderIndex<{field}>(q/(DiamondHalfY*CX),(q/CX)&(DiamondHalfY-1),tid&31{",parity" if topology.parity_providers else ""});
   if(index<0){"continue;" if topology.parity_providers else "return 4;"}if(index>=DiamondProviderWords)return 4;{store_check}frame[index]=token({field},{version},s.z,s.y,s.x);
  }}''')
        source.append('  for(int tid=0;tid<DiamondPrivateThreads;++tid){const int lane=tid&31,warp=tid>>5;uint64_t h[3][DiamondPrivateSlots],e[3][DiamondPrivateSlots];')
        for slot in range(topology.slots):
            for field in range(6):
                version='n' if phase and field<3 else 'n-1';family='e' if field>=3 else 'h'
                source.append(f'''   {{const auto s=DiamondPosition<{field}>(DiamondPrivateQ<{slot}>(tid),{version},oz,op,om,bias);{family}[{field%3}][{slot}]=token({field},{version},s.z,s.y,s.x);}}''')
        for slot in range(topology.slots):
            for update in operator.updates:
                output=update.output.field
                if output not in FIELDS or int(output.startswith('E'))!=phase:continue
                field=FIELDS.index(output)
                source.append(f'''   {{const int q=DiamondPrivateQ<{slot}>(tid),q0=q/(DiamondHalfY*CX),q2=q&(CX-1);
    const auto s=DiamondPosition<{field}>(q,n,oz,op,om,bias);const int q1=s.m;''')
                if topology.geometry.storage_gauge=='physical':
                    at=actual.index(f'if(env.Valid({field},z,y,x))')
                    begin=actual.rfind('const bool owned=',0,at)+len('const bool owned=')
                    expression=actual[begin:actual.index(';',begin)]
                    if not re.fullmatch(r'cell\.dz&&cell\.dm&&\(\(cell\.dp\+[01]\)&\(CY-1\)\)',expression):
                        raise ValueError('Actual common-cell ownership expression missing')
                    source.append(f'''    const auto cell=DiamondPosition<0>(q,n,oz,op,om,bias);const bool emitted_owned={expression};
    if(emitted_owned!=s.owned)return 5;
    if(emitted_owned&&(s.z!=cell.z||s.y!=cell.y||s.x!=cell.x))return 6;''')
                source.append('    if(s.owned){')
                for parity in (0,1):
                    binding=DynamicPrivateReadBinding(topology,slot) if proof.get("dynamic_parity_operands") else PrivateReadBinding(topology,slot,parity)
                    tests=[]
                    for read in sorted(update.reads):
                        if read.field not in FIELDS:continue
                        expression=binding(output,read)
                        if expression not in actual:raise ValueError('Actual private operand missing')
                        sf=FIELDS.index(read.field);version='n' if read.version else 'n-1';dz,dy,dx=read.offset
                        tests.append(f'     if(({expression})!=token({sf},{version},s.z+({dz}),s.y+({dy}),s.x+({dx})))return 3;++checks;')
                    source.append(f'    if(parity=={parity})'+'{')
                    for shuffle,value,dx in binding.shuffles.values():
                        source.append(f'     const uint64_t {shuffle}=__shfl_sync(0xffffffffu,{value},(lane+({dx}))&31);')
                    source+=tests+['    }']
                source+=['    }','   }']
        source+=['  }',' }']
    source+=[' }',' std::cout<<"{\\\"passed\\\":true,\\\"checked_private_reads\\\":"<<checks<<"}\\n";','}']
    directory=candidate/proof_name;directory.mkdir(exist_ok=False)
    code='\n'.join(source);rows=[]
    for negative in (False,True):
        name='wrong-provider' if negative else 'actual';path=directory/(name+'.cc');binary=directory/name
        modified=code.replace('frame[DiamondProviderIndex<','frame[1^DiamondProviderIndex<').replace('frame[DiamondProviderIndexTyped<','frame[1^DiamondProviderIndexTyped<') if negative else code
        if negative and modified==code:raise ValueError('Negative transport witness missing')
        path.write_text(modified)
        subprocess.run(['g++','-std=c++17','-O2',str(path),'-o',str(binary)],capture_output=True,text=True,check=True)
        result=subprocess.run([str(binary)],capture_output=True,text=True)
        if result.returncode!=(3 if negative else 0):raise RuntimeError(name+result.stdout+result.stderr+str(result.returncode))
        rows.append(dict(name=name,returncode=result.returncode,stdout=result.stdout,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result=dict(passed=True,generated_sha256=proof['source_sha256'],rows=rows,
                scope='Actual private warp/slot mapping, provider addresses and bound operands; both lattice parities and wraps; simulated X shuffles; no CUDA visibility claim')
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--proof-name',default='host-private-proof')
    a=p.parse_args();print(json.dumps(validate(a.candidate,a.proof_name)))
