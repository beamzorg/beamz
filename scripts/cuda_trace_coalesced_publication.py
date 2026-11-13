"""Transpose the narrow outgoing X face at existing phase barriers.

Its private owners scatter to 3x84 shared words, then84 contiguous threads
publish contiguous trail locations. The same scratch holds H before B3 and E
after B3. B2/B4 publish it; B3/next B1 retire its readers. No barrier is added.
"""
from __future__ import annotations
import argparse,itertools,json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,PARENT_CONSUMER,block,sha
from scripts.cuda_stencil_trace import reads,trace_yee_step

HELPER=r'''
constexpr int OutgoingXPoints=(BZ-1)*(BY-1);
__device__ __forceinline__ float* OutgoingXScratch(float*frame){return frame+SharedBytes/sizeof(float)-3*OutgoingXPoints;}
template<int Family>
__device__ __forceinline__ void PublishCoalescedX(const Parameters&p,const float*stage,const size_t*bases,int n){
 if(threadIdx.x<OutgoingXPoints){
  constexpr int first=BY*BX+(BZ-1)*BX;
  #pragma unroll
  for(int c=0;c<3;++c)p.trails[bases[0]+((n-1)*6+Family*3+c)*SurfacePoints+first+threadIdx.x]=stage[c*OutgoingXPoints+threadIdx.x];
 }
}
'''


def prove():
    owned=tuple(itertools.product(range(1,8),range(1,16),range(1,32)))
    expected={(z,y,x) for z,y,x in owned if x==31 and z<7 and y<15}
    counts=[]
    for t in (512,1024):
        for n in range(1,17):
            heads=((-n-1)&7,(-n-1)&15,(-n-1)&31)
            writers={}
            for tid in range(t):
                sx,sy,base=tid&31,(tid//32)&15,tid//512
                for j in range(4096//t):
                    zs=base+j*(t//512);dz=(zs-heads[0])&7;dy=(sy-heads[1])&15;dx=(sx-heads[2])&31
                    if dx==31 and 0<dz<7 and 0<dy<15:
                        index=(dz-1)*14+dy-1
                        if index in writers:raise AssertionError('Multiple scratch writers')
                        writers[index]=(dz,dy,dx)
            assert set(writers)==set(range(84)) and set(writers.values())==expected
            counts.append(len(writers)*6)
    # Actual stencil reads are retained unchanged; this only changes which
    # thread publishes each exact field/version/surface tuple.
    deps=[(u.output.field,r.field,r.version,r.offset) for u in trace_yee_step() for r in reads(u.value)]
    return dict(owner_versions_checked=len(counts),exact_field_payloads_checked=sum(counts),
                unique_high_X_scratch_words=252,disjoint_high_ZY_and_X_publications=True,
                actual_trace_read_edges=len(deps),
                H_scratch_write='after B1; before B2',H_scratch_read='after B2; before B3',
                E_scratch_write='after B3; before B4',E_scratch_read='after B4; before next B1',
                next_H_scratch_write='after next B1',barriers_added=0)


def stage(source,output,phases=3):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied research source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=PARENT_CONSUMER:raise ValueError('Exact current winner required')
    text=original
    old='6ULL*LowXPoints*sizeof(float);'
    # The first occurrence is the generic SharedBytes allocation; source-only
    # paired storage retains its own unchanged allocation and lowering.
    text=text.replace(old,old[:-1]+'+3ULL*(BZ-1)*(BY-1)*sizeof(float);',1)
    text=text.replace('FixedCubeSharedBytes==55204','FixedCubeSharedBytes==56212')
    text=text.replace('SharedBytes==51748&&ProfileCachedSharedBytes==55204','SharedBytes==52756&&ProfileCachedSharedBytes==56212')
    marker='template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1,bool CachedProfiles=false>'
    if text.count(marker)!=1:raise ValueError('Helper insertion changed')
    text=text.replace(marker,HELPER+'\n'+marker)
    for signature in ('void UniformBody(','void FixedCubeCachedBody('):
        begin,end,body=block(text,signature)
        for phase,family in ((0,'h'),(1,'e')):
            if not phases&(1<<phase):continue
            publication=f'   for(int c=0;c<3;++c)frame[c*PAD+Slot(zs,sy,sx)]={family}[c][j];'
            new=publication+f'''
   const int outgoing_z=(zs-hz)&7;
   if(dx==BX&&dy>0&&dy<BY&&outgoing_z>0&&outgoing_z<BZ){{
    const int outgoing_index=(outgoing_z-1)*(BY-1)+dy-1;float*stage=OutgoingXScratch(frame);
    #pragma unroll
    for(int c=0;c<3;++c)stage[c*OutgoingXPoints+outgoing_index]={family}[c][j];
   }}'''
            if body.count(publication)!=1:raise ValueError('Original field publication changed')
            body=body.replace(publication,new)
            barrier='  __syncthreads(); // B2 publishes all current H values.' if phase==0 else '  __syncthreads(); // B4 publishes E and the next phase counter.'
            if body.count(barrier)!=1:raise ValueError('Original publishing barrier changed')
            body=body.replace(barrier,barrier+f'\n  PublishCoalescedX<{phase}>(p,OutgoingXScratch(frame),bases,n);')
        old='if(surface>=0){p.trails[bases[0]+((n-1)*6+c)*SurfacePoints+surface]=h[c][j];p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=e[c][j];}'
        if body.count(old)!=1:raise ValueError('Original private trail publication changed')
        writes=[]
        for phase,family in ((0,'h'),(1,'e')):
            guard='surface>=0&&(dz==BZ||dy==BY)' if phases&(1<<phase) else 'surface>=0'
            suffix='+3' if phase else ''
            writes.append(f'if({guard})p.trails[bases[0]+((n-1)*6+c{suffix})*SurfacePoints+surface]={family}[c][j];')
        body=body.replace(old,''.join(writes))
        text=text[:begin]+body+text[end:]
    shutil.copytree(source,output);(output/'src/asymmetric_native.cuh').write_text(text)
    record=dict(schema='beamz.traced-X-trail-publication.v1',phases=phases,
                parent_sha256=PARENT_CONSUMER,consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
                shared_bytes=52756,cached_shared_bytes=56212,scratch_bytes=1008,
                proof=prove(),expected_kernels=66,GPU_executed=False,
                caveat='Coalesced warp requests do not imply a measured DRAM traffic reduction')
    (output.parent/'coalesced-publication-proof.json').write_text(json.dumps(record,indent=2)+'\n');return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PARENT);p.add_argument('--output',type=Path,required=True);p.add_argument('--phases',type=int,choices=(1,2,3),default=3);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.phases)))
