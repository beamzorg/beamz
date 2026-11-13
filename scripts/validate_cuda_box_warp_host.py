"""Execute emitted box-warp addressing with versioned physical tokens."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.cuda_space_time_ir import FIELDS,yee_operator
from scripts.cuda_trace_box_warp_storage import BoxWarpStorage,BoxReadBinding


def validate(candidate):
    candidate=Path(candidate).resolve();proof=json.loads((candidate/'box-warp-compiler-proof.json').read_text())
    text=(candidate/'source/src/trace_tile_generated.cuh').read_text()
    if hashlib.sha256(text.encode()).hexdigest()!=proof['source_sha256']:raise ValueError('Generated shader identity changed')
    fields=tuple(proof['shared_field']) if isinstance(proof['shared_field'],list) else proof['shared_field']
    t=BoxWarpStorage(tuple(proof['cache']),proof['private_z'],proof['private_y'],fields,proof.get('profile_slots',12),proof.get('provider_home',False),proof.get('slot_major',False))
    helpers=text.split('template<int Depth,int Layout,bool CachedProfiles,class Env>')[0]
    if proof.get('late_addresses'):
        if text.count('BoxAddressScope(threadIdx.x>>5,n)')!=2*t.slots or text.count('BoxAddressScope(threadIdx.x>>5,first-1)')!=t.slots:
            raise ValueError('Address-region boundary missing at an actual point event')
    if proof.get('volatile_addresses'):
        init=f'if(threadIdx.x<{t.threads//32})reinterpret_cast<volatile int*>(frame+BoxProviderWords+19)[threadIdx.x]=threadIdx.x;'
        if text.count(init)!=1 or proof['shared_bytes']!=t.shared_words*4+76+(t.threads//32)*4:
            raise ValueError('Immutable warp-ID initialization or nonaliasing metadata allocation changed')
    directory=candidate/'host-box-warp-proof';directory.mkdir(exist_ok=False)
    source=['''#include <cstdint>
#include <iostream>
#include <vector>
#define __device__
#define __forceinline__ inline
''',f'constexpr int CZ={t.cache[0]},CY={t.cache[1]},CX={t.cache[2]};',helpers,
'''int Slot(int z,int y,int x){return (((z&(CZ-1))*CY+(y&(CY-1)))*CX+(x&(CX-1)));}
struct Site{int z,y,x;bool owned;};
Site position(int q,int n,int oz,int oy,int ox){const int low=-n-1,zs=q/(CY*CX),ys=(q/CX)&(CY-1),xs=q&(CX-1);
 const int dz=(zs-low)&(CZ-1),dy=(ys-low)&(CY-1),dx=(xs-low)&(CX-1);
 return {oz+low+dz,oy+low+dy,ox+low+dx,bool(dz&&dy&&dx)};
}
uint64_t token(int f,int v,int z,int y,int x){return (uint64_t(v)<<48)|(uint64_t(f)<<45)|(uint64_t(z+8192)<<30)|(uint64_t(y+8192)<<15)|uint64_t(x+8192);}
int xorigin;
uint64_t __shfl_sync(unsigned,uint64_t value,int lane){const int version=value>>48,low=-version-1;
 return (value&~uint64_t(32767))|uint64_t(xorigin+low+((lane-low)&(CX-1))+8192);
}
int main(){size_t checks=0;std::vector<uint64_t>frame(BoxProviderWords);
 for(int n:{1,2,4,15,16,17,31,32,33,35})for(int odd:{0,1}){const int oz=30+odd,oy=90,ox=64+odd;xorigin=ox;
''']
    op=yee_operator(cpml_axes=(0,1,2))
    if proof.get('late_addresses'):
        source.append(f' for(int w=0;w<{t.threads//32};++w){{const auto r=BoxAddressScope(w,n);if(r.warp!=w||r.version!=n)return 8;}}')
    for phase in (0,1):
        source.append(' {')
        for field,name in enumerate(FIELDS):
            version='n' if phase and field<3 else 'n-1'
            if name in t.dense_fields:
                extra=t.dense_fields.index(name)*t.points
                source.append(f'  for(int q=0;q<CZ*CY*CX;++q){{auto s=position(q,{version},oz,oy,ox);frame[BoxDenseOffset+{extra}+q]=token({field},{version},s.z,s.y,s.x);}}')
                continue
            # Seed actual typed store expressions, and check their independent
            # model's affine offset/rank for every warp and lane.
            for slot in t.providers['slots'][name]:
                store=f'frame[BoxProviderIndex<{field},{slot}>(warp,lane)]='
                if store not in text:raise ValueError('Actual typed provider store absent')
                rank=t.providers['slots'][name].index(slot);offset=t.providers['offsets'][name];count=len(t.providers['slots'][name])
                expected=f'{offset}+{rank}*{t.threads}+tid' if t.slot_major else f'{offset}+(warp*{count}+{rank})*32+lane'
                source.append(f'''  for(int tid=0;tid<{t.threads};++tid){{const int warp=tid>>5,lane=tid&31,q=BoxQ<{slot}>(tid);auto s=position(q,{version},oz,oy,ox);
   const int actual=BoxProviderIndex<{field},{slot}>(warp,lane),expected={expected};
   if(actual!=expected||actual<0||actual>=BoxDenseOffset)return 4;frame[actual]=token({field},{version},s.z,s.y,s.x);
  }}''')
        source.append(f'  for(int tid=0;tid<{t.threads};++tid){{const int warp=tid>>5,lane=tid&31;uint64_t h[3][BoxSlots],e[3][BoxSlots];')
        for slot in range(t.slots):
            for field in range(6):
                version='n' if phase and field<3 else 'n-1';family='e' if field>=3 else 'h'
                source.append(f'   {{auto s=position(BoxQ<{slot}>(tid),{version},oz,oy,ox);{family}[{field%3}][{slot}]=token({field},{version},s.z,s.y,s.x);}}')
        for slot in range(t.slots):
            for update in op.updates:
                output=update.output.field
                if output not in FIELDS or int(output.startswith('E'))!=phase:continue
                field=FIELDS.index(output);binding=BoxReadBinding(t,slot);tests=[]
                source.append(f'   {{const int q=BoxQ<{slot}>(tid),zs=q/(CY*CX),ys=(q/CX)&(CY-1);auto s=position(q,n,oz,oy,ox);')
                for read in sorted(update.reads):
                    if read.field not in FIELDS:continue
                    expression=binding(output,read)
                    # The first one-component prototype omitted a literal+0.
                    if expression not in text and expression.replace('BoxDenseOffset+0+','BoxDenseOffset+') in text:
                        expression=expression.replace('BoxDenseOffset+0+','BoxDenseOffset+')
                    if expression not in text:raise ValueError('Actual bound read missing')
                    sf=FIELDS.index(read.field);dz,dy,dx=read.offset;version='n' if read.version else 'n-1'
                    tests.append(f'    if(({expression})!=token({sf},{version},s.z+({dz}),s.y+({dy}),s.x+({dx})))return 3;++checks;')
                for name,(value,dx) in binding.shuffles.items():source.append(f'    const auto {name}=__shfl_sync(0xffffffffu,{value},(lane+({dx}))&31);')
                source+=['    if(s.owned){',*tests,'    }','   }']
        source+=['  }',' }']
    source+=[' }',' std::cout<<checks<<"\\n";','}']
    code='\n'.join(source);rows=[]
    for name in ('actual','wrong-provider-stride','wrong-warp-wrap'):
        changed=code
        if name=='wrong-provider-stride':
            changed=changed.replace('*32+lane;','*31+lane;').replace('provider*32+lane;','provider*31+lane;')
        if name=='wrong-warp-wrap':
            wrap=t.cache[0]//t.private_z-1
            if not wrap:raise ValueError('Negative control requires multiple Z warp groups')
            changed=changed.replace(f'ZDelta)&{wrap}',f'ZDelta)&{wrap-1}')
        src=directory/(name+'.cc');binary=directory/name;src.write_text(changed)
        subprocess.run(['g++','-std=c++17','-O2',str(src),'-o',str(binary)],capture_output=True,text=True,check=True)
        r=subprocess.run([str(binary)],capture_output=True,text=True)
        if (r.returncode!=0 if name=='actual' else r.returncode==0):raise RuntimeError(name+r.stdout+r.stderr)
        rows.append(dict(name=name,returncode=r.returncode,stdout=r.stdout,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
    result=dict(passed=True,generated_sha256=proof['source_sha256'],rows=rows,
                scope='Actual emitted private Q map, typed provider stores and field operands; both physical phases, field homes, cache wraps, full-lane X shuffles, two negative controls; no CUDA visibility claim')
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.candidate)))
