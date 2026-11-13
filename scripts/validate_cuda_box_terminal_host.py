"""Check actual terminal imports and deferred final effects before GPU execution."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.cuda_trace_phase_cycle import block
from scripts.cuda_trace_box_warp_storage import BoxWarpStorage
from scripts.cuda_space_time_ir import FIELDS,yee_operator


def validate(candidate):
    candidate=Path(candidate).resolve()
    proof=json.loads((candidate/'box-warp-compiler-proof.json').read_text())
    consumer=(candidate/'source/src/asymmetric_native.cuh').read_text()
    generated=(candidate/'source/src/trace_tile_generated.cuh').read_text()
    if hashlib.sha256(consumer.encode()).hexdigest()!=proof['consumer_sha256'] or hashlib.sha256(generated.encode()).hexdigest()!=proof['source_sha256']:
        raise ValueError('Actual candidate identity changed')
    if not proof['terminal_epilogue']:raise ValueError('Explicit epilogue candidate required')
    t=BoxWarpStorage(tuple(proof['cache']),proof['private_z'],proof['private_y'],proof['shared_field'],proof['profile_slots'],proof['provider_home'],proof['slot_major'])
    loop=block(generated,'for(int n=first;')[2]
    if 'FinalField' in loop or generated.count('FinalField(p,')!=6*t.slots:
        raise ValueError('Final descriptors escaped the single post-loop scatter')
    epilogue=block(generated,'if(last==Depth)')[2]
    if epilogue.count('FinalField(p,')!=6*t.slots or epilogue.count('if(owned)')!=t.slots:
        raise ValueError('Deferred scatter ownership changed')
    if loop.count('if(surface>=0){float*address=')!=t.slots or loop.count('if(surface>=0&&(Layout!=1||n<Depth)){float*address=')!=t.slots:
        raise ValueError('Terminal H trails or dead E trail guard changed')
    # A family has no same-phase neighbor dependency. Shared primary homes may
    # therefore overwrite their own value before the opposite-family join.
    for update in yee_operator(cpml_axes=(0,1,2)).updates:
        if update.output.field not in FIELDS:continue
        for read in update.reads:
            if read.field in FIELDS and read.field[0]==update.output.field[0]:
                if read.field!=update.output.field or read.offset!=(0,0,0):
                    raise ValueError('In-place family storage has a neighbor hazard')
    surface=block(consumer,'int Surface(')[2]
    mask=block(consumer,'unsigned NativeSupportMask(')[2]
    packet=consumer[consumer.index('struct FieldPacket {'):consumer.index('__global__ void BeginEpoch')]
    if 'FinalField' in packet or 'if(version==Depth)mask&=7u;' not in packet:
        raise ValueError('Terminal import still retains final-field descriptors')
    directory=candidate/'host-box-terminal-proof';directory.mkdir(exist_ok=False)
    cz,cy,cx=t.cache
    code='''#include <cstdint>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>
#define __host__
#define __device__
#define __forceinline__ inline
struct Field{int z,y,x;};struct Phase{Field own[3];};
struct Parameters{Phase phase[2];float*trails;int*bad;};
float __ldcg(const float*p){return *p;}
void atomicOr(int*p,int n){*p|=n;}
float __int_as_float(unsigned x){float f;std::memcpy(&f,&x,4);return f;}
'''+f'constexpr int BZ={cz-1},BY={cy-1},BX={cx-1},SurfacePoints=BZ*BY*BX-(BZ-1)*(BY-1)*(BX-1);\n'+surface+'\n'+mask+'\n'+packet+'''
template<bool Safe,bool Reuse>size_t check(){
 size_t checks=0;constexpr int Depth=16;int bad=0;std::vector<float>trails(8*Depth*6*SurfacePoints);
 for(size_t i=0;i<trails.size();++i)trails[i]=float(i+1);
 size_t bases[8];for(int i=0;i<8;++i)bases[i]=i*Depth*6*SurfacePoints;
 const int ds[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
 for(int N:{17,32,65})for(int origin:{0,13,34,80})for(int v:{1,2,7,15,16})
 for(int dz=0;dz<=BZ;++dz)for(int dy=0;dy<=BY;++dy)for(int dx=0;dx<=BX;++dx){
  if(dz&&dy&&dx)continue;
  Parameters p{};p.trails=trails.data();p.bad=&bad;
  for(int f=0;f<6;++f)p.phase[f/3].own[f%3]={N+ds[f][0],N+ds[f][1],N+ds[f][2]};
  const int z=origin-1+dz-v,y=origin-1+dy-v,x=origin-1+dx-v;
  const int neighbor=(dz==0)|((dy==0)<<1)|((dx==0)<<2);
  const int sz=dz?dz-1:BZ-1,sy=dy?dy-1:BY-1,sx=dx?dx-1:BX-1;
  const int surface=sz==BZ-1?sy*BX+sx:sy==BY-1?BY*BX+sz*BX+sx:BY*BX+(BZ-1)*BX+sz*(BY-1)+sy;
  const auto result=ImportPacket<Depth,Safe,Reuse>(p,bases,v,origin,origin,origin,dz,dy,dx);
  const float actual[6]={result.hx,result.hy,result.hz,result.ex,result.ey,result.ez};
  for(int f=0;f<6;++f){
   const bool valid=Safe||(z>=0&&y>=0&&x>=0&&z<N+ds[f][0]&&y<N+ds[f][1]&&x<N+ds[f][2]);
   const float expected=valid&&!(Reuse&&v==Depth&&f>=3)?trails[bases[neighbor]+((v-1)*6+f)*SurfacePoints+surface]:0.f;
   if(actual[f]!=expected){std::cerr<<N<<","<<origin<<","<<v<<","<<f<<"\\n";return 0;}++checks;
  }
 }
 return bad?0:checks;
}
int main(){const size_t a=check<false,true>(),b=check<false,false>(),c=check<true,true>(),d=check<true,false>();
 if(!a||!b||!c||!d)return 3;std::cout<<a+b+c+d<<"\\n";}
'''
    rows=[]
    for name in ('actual','wrong-terminal-family','wrong-version-row'):
        trial=code
        if name=='wrong-terminal-family':trial=trial.replace('mask&=7u','mask&=63u')
        if name=='wrong-version-row':trial=trial.replace('static_cast<size_t>(version-1)','static_cast<size_t>(version%16)')
        src=directory/(name+'.cc');binary=directory/name;src.write_text(trial)
        subprocess.run(['g++','-std=c++17','-O2','-fsanitize=address,undefined',str(src),'-o',str(binary)],capture_output=True,text=True,check=True)
        r=subprocess.run([str(binary)],capture_output=True,text=True)
        if (r.returncode!=0 if name=='actual' else r.returncode==0):raise RuntimeError(name+r.stdout+r.stderr)
        rows.append(dict(name=name,returncode=r.returncode,stdout=r.stdout,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
    result=dict(passed=True,consumer_sha256=proof['consumer_sha256'],generated_sha256=proof['source_sha256'],rows=rows,
                scope='Actual terminal trail imports over staggered/ragged supports, both layouts and safe/masked paths; deferred scatter and family dependency binding; ASan/UBSan and two negative controls. CUDA timing and inter-CTA visibility separate.')
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.candidate)))
