"""Lower rolled NoIO point loops into named scalar register banks.

Unlike a dynamic CUDA array index, each bank has four explicit scalar members.
Get/Set select these members without exposing an array address. This trades
selection instructions for a smaller repeated point body; real ptxas/local
resources and full-engine timings decide whether the trade is worthwhile.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,PARENT_CONSUMER,block,sha

HELPER=r'''
struct NamedRegisterBank4{
 float q0,q1,q2,q3;
 __device__ __forceinline__ float Get(int j)const{return j==0?q0:j==1?q1:j==2?q2:q3;}
 __device__ __forceinline__ void Set(int j,float value){
  if(j==0)q0=value;if(j==1)q1=value;if(j==2)q2=value;if(j==3)q3=value;
 }
 __device__ __forceinline__ void Rotate1(){const float a=q0;q0=q1;q1=q2;q2=q3;q3=a;}
 __device__ __forceinline__ void Rotate2(){const float a=q0,b=q1;q0=q2;q1=q3;q2=a;q3=b;}
};
template<int T>
__device__ __forceinline__ void RetireNamedRegisterLow(const float*stage,NamedRegisterBank4(&h)[3],NamedRegisterBank4(&e)[3],int n,int sy,int sx){
 const int hz=(-n-1)&7,hy=(-n-1)&15,hx=(-n-1)&31,dy=(sy-hy)&15;
 if(sx==hx&&sy!=hy){
  #pragma unroll
  for(int j=0;j<4;++j){const int zslot=threadIdx.x/512+j*(T/512);if(zslot!=hz){const int dz=(zslot-hz)&7,key=(dz-1)*BY+dy-1;
   #pragma unroll
   for(int c=0;c<3;++c){h[c].Set(j,stage[c*LowXPoints+key]);e[c].Set(j,stage[(c+3)*LowXPoints+key]);}
  }}
 }
}
'''


def lower(body,clock='shared',bank_cycle=0,roll_cold=False):
    old=' float h[3][Slots],e[3][Slots];'
    if body.count(old)!=1:raise ValueError('Private field storage changed')
    body=body.replace(old,' static_assert(T==1024&&Slots==4,"Named bank owner class");\n NamedRegisterBank4 h[3],e[3];')
    # Only the main H/E point loops roll. All publication/import loops retain
    # static bindings and all four original barriers remain.
    for marker in ('  {const int step=*phase_step,hz=','  {const int step=*phase_step,ehz='):
        begin=body.index(marker)
        first=body.index('   #pragma unroll\n   for(int j=0;j<Slots;++j)',begin)
        body=body[:first]+body[first:].replace('   #pragma unroll\n   for(int j=0;j<Slots;++j)',
                                               '   #pragma unroll 1\n   for(int j=0;j<Slots;++j)',1)
    body=re.sub(r'([he])\[([^\]]+)\]\[j\]=([^;]+);',r'\1[\2].Set(j,\3);',body)
    body=re.sub(r'([he])\[([^\]]+)\]\[j\]',r'\1[\2].Get(j)',body)
    body=body.replace('RetireRegisterLow<T,Slots>(low_stage,h,e,n,sy,sx);','RetireNamedRegisterLow<T>(low_stage,h,e,n,sy,sx);')
    if roll_cold:
        flags=body.index(' const uint8_t flags=')
        edits=[]
        for match in re.finditer(r'#pragma unroll\n\s+for\(int j=0;j<Slots;\+\+j\)',body):
            _,_,loop=block(body[match.start():],'for(int j=0;j<Slots;++j)')
            if match.start()<flags or 'ImportPacket<Depth,SupportSafe>' in loop or 'p.trails[' in loop:
                edits.append(match.start())
        for start in reversed(edits):body=body[:start]+body[start:].replace('#pragma unroll\n','#pragma unroll 1\n',1)
    if bank_cycle:
        for marker in ('  {const int step=*phase_step,hz=','  {const int step=*phase_step,ehz='):
            start=body.index(marker);tail=body[start:]
            begin,end,loop=block(tail,'for(int j=0;j<Slots;++j)')
            point=loop[loop.index('{')+1:-1]
            replacement=f'for(int j_base=0;j_base<Slots;j_base+={bank_cycle}){{\n'
            for item in range(bank_cycle):
                cell=re.sub(r'([he])\[([^\]]+)\]\.Set\(j,([^;]+)\);',rf'\1[\2].q{item}=\3;',point)
                cell=re.sub(r'([he])\[([^\]]+)\]\.Get\(j\)',rf'\1[\2].q{item}',cell)
                replacement+=f'{{const int j=j_base+{item};'+cell+'}\n'
            replacement+=f'\n#pragma unroll\nfor(int c=0;c<3;++c){{h[c].Rotate{bank_cycle}();e[c].Rotate{bank_cycle}();}}\n}}'
            tail=tail[:begin]+replacement+tail[end:]
            body=body[:start]+tail
    if clock=='opaque':
        body=body.replace('{const int step=*phase_step,','{int step;asm volatile("mov.b32 %0,%1;":"=r"(step):"r"(n));const int ')
        for s in ('if(!threadIdx.x)*phase_step=1;','if(!threadIdx.x)*phase_step=first;','if(!threadIdx.x)*phase_step=n+1;'):body=body.replace(s,'')
    return body


def stage(source,output,clock='shared',bank_cycle=0,roll_cold=False):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied research source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    parent_sha=sha(original.encode())
    if parent_sha!=PARENT_CONSUMER:
        composition=source.parent/'coalesced-publication-proof.json'
        if not composition.is_file():raise ValueError('Exact current winner or bound coalesced-publication parent required')
        binding=json.loads(composition.read_text())
        if binding['consumer_sha256']!=parent_sha or binding['parent_sha256']!=PARENT_CONSUMER:
            raise ValueError('Coalesced parent identity differs')
    text=original
    begin,end,body=block(text,'void FixedCubeCachedBody(')
    text=text[:begin]+lower(body,clock,bank_cycle,roll_cold)+text[end:]
    begin,end,body=block(text,'void UniformBody(')
    clone=lower(body,clock,bank_cycle,roll_cold).replace('void UniformBody(','void NamedNoIOBody(',1)
    clone='template<int Depth,int ActiveAxes=-1>\n__device__ __forceinline__ '+clone
    clone=clone.replace('void NamedNoIOBody(const Parameters&p,const int32_t*tiles){',
                        'void NamedNoIOBody(const Parameters&p,const int32_t*tiles){\n constexpr int Layout=1,Policy=1;constexpr bool Fast=false,IsotropicH=true,UniformE=true,CachedProfiles=true;')
    marker='template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__'
    if text.count(marker)!=1:raise ValueError('Wrapper insertion changed')
    text=text.replace(marker,clone+'\n'+marker)
    begin,end,wrapper=block(text,'void AsymmetricNoIOUniform1024(')
    for axis in (1,2,4,-1):
        wrapper=wrapper.replace(f'UniformBody<Depth,1,false,true,1,true,{axis},true>',f'NamedNoIOBody<Depth,{axis}>')
    text=text[:begin]+wrapper+text[end:]
    marker='template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1,bool CachedProfiles=false>'
    if text.count(marker)!=1:raise ValueError('Named register declaration placement changed')
    text=text.replace(marker,HELPER+'\n'+marker)
    shutil.copytree(source,output);(output/'src/asymmetric_native.cuh').write_text(text)
    proof=dict(schema='beamz.named-scalar-register-banks.v1',clock=clock,bank_cycle=bank_cycle,roll_cold=roll_cold,
               parent_sha256=parent_sha,consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
               named_words_per_bank=4,banks_per_thread=6,rolled_main_point_loops=2,
               original_generic_body_byte_identical=True,barriers_per_step=4,expected_kernels=66,GPU_executed=False)
    (output.parent/'named-bank-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PARENT);p.add_argument('--output',type=Path,required=True);p.add_argument('--clock',choices=('shared','opaque'),default='shared');p.add_argument('--bank-cycle',type=int,choices=(0,1,2),default=0);p.add_argument('--roll-cold',action='store_true');a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.clock,a.bank_cycle,a.roll_cold)))
