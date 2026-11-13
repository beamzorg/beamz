"""Overlap low-X trail imports with H arithmetic in a copied moving backend.

The trace has distinct low-packet scratch and field-frame regions. Previous
scratch readers retire at B2; H only reads the field frame. Async completion
precedes B1, which retains the original shared-memory publication edge.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT, PARENT, PARENT_CONSUMER, block, sha

HELPER=r'''
__device__ __forceinline__ void CopyTrailWordAsync(float*destination,const float*source,int bytes){
 const unsigned shared=static_cast<unsigned>(__cvta_generic_to_shared(destination));
 asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"::"r"(shared),"l"(source),"r"(bytes):"memory");
}
template<int Depth,bool Fast,int T>
__device__ __forceinline__ void StageRegisterLowAsync(const Parameters&p,float*stage,const size_t*bases,int n,int oz,int oy,int ox){
 for(int i=threadIdx.x;i<LowXPoints;i+=T){
  const int dz=1+i/BY,dy=1+i%BY;
  unsigned mask=63;
  if constexpr(!Fast)mask=NativeSupportMask(p,oz-1+dz-n,oy-1+dy-n,ox-1-n);
  const int surface=Surface(dz-1,dy-1,BX-1);const size_t base=bases[4];
  if(mask&&(surface<0||base==static_cast<size_t>(-1))){
   atomicOr(p.bad,2);const float bad=__int_as_float(0x7fffffff);
   #pragma unroll
   for(int field=0;field<6;++field)stage[field*LowXPoints+i]=(mask&(1u<<field))?bad:0.f;
  }else{
   const float*point=mask?p.trails+base+static_cast<size_t>(n-1)*6*SurfacePoints+surface:InitialField(p,0).data;
   #pragma unroll
   for(int field=0;field<6;++field){
    const bool live=mask&(1u<<field);
    CopyTrailWordAsync(stage+field*LowXPoints+i,live?point+field*SurfacePoints:InitialField(p,0).data,live?4:0);
   }
  }
 }
 asm volatile("cp.async.commit_group;":::"memory");
}
'''


def transform_body(body, early):
    call="  StageRegisterLow<Depth,SupportSafe,T>(p,low_stage,bases,n,oz,oy,ox);"
    if body.count(call)!=1:raise ValueError('Reviewed low-X staging site changed')
    replacement="  StageRegisterLowAsync<Depth,SupportSafe,T>(p,low_stage,bases,n,oz,oy,ox);"
    if early:
        loop="for(int n=(Fast?1:window[0]);n<=(Fast?Depth:window[1]);++n){"
        if body.count(loop)!=1:raise ValueError('Reviewed phase loop changed')
        body=body.replace(call,'').replace(loop,loop+'\n'+replacement)
    else:body=body.replace(call,replacement)
    barrier="  __syncthreads(); // B1 retires all old-E frame readers and publishes low imports."
    if body.count(barrier)!=1:raise ValueError('Reviewed B1 changed')
    return body.replace(barrier,'  asm volatile("cp.async.wait_group 0;":::"memory");\n'+barrier)


def stage(source,output,early=True,scope='both'):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:
        raise ValueError('New disjoint copied research source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=PARENT_CONSUMER:raise ValueError('Exact current winner required')
    text=original
    signatures=['void UniformBody(','void FixedCubeCachedBody('] if scope=='both' else ['void FixedCubeCachedBody(']
    for signature in signatures:
        begin,end,body=block(text,signature)
        text=text[:begin]+transform_body(body,early)+text[end:]
    marker='template<int T,int Slots>\n__device__ __forceinline__ void RetireRegisterLow'
    if text.count(marker)!=1:raise ValueError('Reviewed helper insertion changed')
    text=text.replace(marker,HELPER+'\n'+marker)
    shutil.copytree(source,output)
    (output/'src/asymmetric_native.cuh').write_text(text)
    proof=dict(schema='beamz.traced-async-low-trails.v1',early=early,scope=scope,
               parent_sha256=PARENT_CONSUMER,consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
               async_words_per_interior_tile=6*7*15,
               unique_destination='(field*105+low_packet_index)',
               producer_neighbor=4,unchanged_current_version=True,
               previous_scratch_readers_retired_at_B2=True,
               scratch_disjoint_from_H_frame_and_CPML=True,
               wait_before_original_B1=True,barriers_per_step=4,GPU_executed=False)
    (output.parent/'async-import-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=PARENT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--late',action='store_true')
    p.add_argument('--scope',choices=('both','fixed'),default='both')
    a=p.parse_args();print(json.dumps(stage(a.source,a.output,not a.late,a.scope)))
