"""Choose trail memory layout by the traced publication face.

Wide Z/Y faces retain component-major coalescing. The narrow X face packs six
fields per point and uses three aligned float2 transfers. Payload size, temporal
ownership and synchronization are unchanged; no transpose scratch is needed.
"""
from __future__ import annotations
import argparse,itertools,json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,PARENT_CONSUMER,block,sha

HELPERS=r'''
constexpr int WideSurfacePoints=BY*BX+(BZ-1)*BX;
__device__ __forceinline__ void StoreFacePacket(const Parameters&p,const size_t*bases,int version,int surface,const FieldPacket&v){
 if(surface<0)return;
 float*point=p.trails+bases[0]+static_cast<size_t>(version-1)*6*SurfacePoints;
 if(surface>=WideSurfacePoints){
  float2*packed=reinterpret_cast<float2*>(point+6*WideSurfacePoints+6*(surface-WideSurfacePoints));
  packed[0]=make_float2(v.hx,v.hy);packed[1]=make_float2(v.hz,v.ex);packed[2]=make_float2(v.ey,v.ez);
 }else{
  point[surface]=v.hx;point[WideSurfacePoints+surface]=v.hy;point[2*WideSurfacePoints+surface]=v.hz;
  point[3*WideSurfacePoints+surface]=v.ex;point[4*WideSurfacePoints+surface]=v.ey;point[5*WideSurfacePoints+surface]=v.ez;
 }
}
'''


def index(field,surface):
    wide=651
    return field*wide+surface if surface<wide else 6*wide+6*(surface-wide)+field


def prove(*,wrong_pair=False):
    surface=735;words=6*surface
    image={index(f,s):(f,s) for f,s in itertools.product(range(6),range(surface))}
    assert len(image)==words and set(image)==set(range(words))
    checks=0
    for s in range(651,surface):
        packed=[None]*6
        for pair in range(3):
            for k in range(2):
                field=2*pair+k
                packed[field]=(field^(1 if wrong_pair and pair==1 else 0),s)
        for f in range(6):
            if packed[f]!=(f,s):raise AssertionError('Packet field routing differs')
            checks+=1
    aligned=0
    for owner,depth in itertools.product(range(4),(4,16)):
        for n,s,pair in itertools.product(range(depth),range(651,735),range(3)):
            offset=owner*depth*words+n*words+index(2*pair,s)
            assert offset%2==0;aligned+=1
    return dict(surface_points=surface,wide_surface_points=651,narrow_surface_points=84,
                layout_bijection_words=words,vector_field_checks=checks,aligned_float2_checks=aligned,
                payload_bytes_unchanged=True,all_six_fields_written_for_every_published_packet=True,
                inactive_component_vector_loads_require_nonempty_native_support_mask=True,
                synchronization_unchanged=True)


def stage(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied research source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=PARENT_CONSUMER:raise ValueError('Exact current winner required')
    text=original
    anchor='template<int Depth,bool Fast>\n__device__ __forceinline__ FieldPacket ImportPacket'
    if text.count(anchor)!=1:raise ValueError('Packet import insertion changed')
    text=text.replace(anchor,HELPERS+'\n'+anchor)
    old=''' const float*point=p.trails+base+static_cast<size_t>(version-1)*6*SurfacePoints+surface;
 return {mask&1?__ldcg(point+0*SurfacePoints):0.f,mask&2?__ldcg(point+1*SurfacePoints):0.f,
         mask&4?__ldcg(point+2*SurfacePoints):0.f,mask&8?__ldcg(point+3*SurfacePoints):0.f,
         mask&16?__ldcg(point+4*SurfacePoints):0.f,mask&32?__ldcg(point+5*SurfacePoints):0.f};'''
    new=''' const float*point=p.trails+base+static_cast<size_t>(version-1)*6*SurfacePoints;
 if(surface>=WideSurfacePoints){
  const float2*packed=reinterpret_cast<const float2*>(point+6*WideSurfacePoints+6*(surface-WideSurfacePoints));
  const auto a=__ldcg(packed),b=__ldcg(packed+1),c=__ldcg(packed+2);
  return {mask&1?a.x:0.f,mask&2?a.y:0.f,mask&4?b.x:0.f,mask&8?b.y:0.f,mask&16?c.x:0.f,mask&32?c.y:0.f};
 }
 point+=surface;
 return {mask&1?__ldcg(point+0*WideSurfacePoints):0.f,mask&2?__ldcg(point+1*WideSurfacePoints):0.f,
         mask&4?__ldcg(point+2*WideSurfacePoints):0.f,mask&8?__ldcg(point+3*WideSurfacePoints):0.f,
         mask&16?__ldcg(point+4*WideSurfacePoints):0.f,mask&32?__ldcg(point+5*WideSurfacePoints):0.f};'''
    if text.count(old)!=1:raise ValueError('Reviewed complete import changed')
    text=text.replace(old,new)
    for signature in ('void UniformBody(','void FixedCubeCachedBody('):
        begin,end,body=block(text,signature)
        old='if(surface>=0){p.trails[bases[0]+((n-1)*6+c)*SurfacePoints+surface]=h[c][j];p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=e[c][j];}'
        if body.count(old)!=1:raise ValueError('Original field packet publication changed')
        body=body.replace(old,'')
        anchor='   #pragma unroll\n   for(int c=0;c<3;++c){\n    \n    if(n==Depth)'
        if body.count(anchor)!=1:raise ValueError('Final publication insertion changed')
        body=body.replace(anchor,'   StoreFacePacket(p,bases,n,surface,{h[0][j],h[1][j],h[2][j],e[0][j],e[1][j],e[2][j]});\n'+anchor)
        text=text[:begin]+body+text[end:]
    old='if(surface>=0){p.trails[bases[0]+((n-1)*6+c)*SurfacePoints+surface]=h[c][si];p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=ev;}'
    if text.count(old)!=2:raise ValueError('Paired source packet publications changed')
    text=text.replace(old,'')
    for pair in (0,1):
        anchor=f'''     #pragma unroll
     for(int c=0;c<3;++c){{const float ev=c==0?ex{pair}[gy]:c==1?get_ey(zs,ys,sx,si):ez{pair}[gy];'''
        if text.count(anchor)!=1:raise ValueError('Paired source insertion changed')
        text=text.replace(anchor,f'     StoreFacePacket(p,bases,n,surface,{{h[0][si],h[1][si],h[2][si],ex{pair}[gy],get_ey(zs,ys,sx,si),ez{pair}[gy]}});\n'+anchor)
    if 'p.trails[' in text:raise ValueError('Old scalar layout writer remains')
    shutil.copytree(source,output);(output/'src/asymmetric_native.cuh').write_text(text)
    record=dict(schema='beamz.traced-face-packet-layout.v1',proof=prove(),parent_sha256=PARENT_CONSUMER,
                consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
                shared_allocation_unchanged=True,source_and_monitor_work_unchanged=True,
                ordinary_and_paired_source_writers_migrated=True,expected_kernels=66,GPU_executed=False)
    (output.parent/'face-packet-proof.json').write_text(json.dumps(record,indent=2)+'\n');return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PARENT);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output)))
