"""Specialize traced native support and field strides at tile IO boundaries.

The existing common-shape entry proves canonical staggering and dense native
storage. Runtime data pointers and all coefficient/field bits are retained.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,PARENT_CONSUMER,sha,block
from scripts.cuda_native_geometry_ir import SUPPORT_DELTA


HELPER='''template<int N>__device__ __forceinline__ unsigned ResearchCubeSupport(int z,int y,int x){
 if(z<0||y<0||x<0||z>N||y>N||x>N)return 0;
 const unsigned iz=z<N,iy=y<N,ix=x<N;
 return (iz&&iy)|((iz&&ix)<<1)|((iy&&ix)<<2)|(ix<<3)|(iy<<4)|(iz<<5);
}
template<int N,int Field>__device__ __forceinline__ size_t ResearchCubeOffset(int z,int y,int x){
 constexpr int Y=N+((Field==1||Field==3||Field==5)?1:0);
 constexpr int X=N+((Field==0||Field==4||Field==5)?1:0);
 return (size_t(z)*Y+y)*X+x;
}
template<int N,int Field>__device__ __forceinline__ float ResearchCubeInitial(const Parameters&p,int z,int y,int x){
 return ResearchCubeSupport<N>(z,y,x)&(1u<<Field)?InitialField(p,Field).data[ResearchCubeOffset<N,Field>(z,y,x)]:0.f;
}
'''


def prove():
    checks=0
    # Nonnegative and out-of-support points around every native wall, plus
    # interior and dense-stride endpoints. Ordinary shapes are independent.
    for n in (32,512,564):
        for z in (-2,-1,0,1,n-1,n,n+1,n+2):
            for y in (-2,-1,0,1,n-1,n,n+1,n+2):
                for x in (-2,-1,0,1,n-1,n,n+1,n+2):
                    mask=0 if min(z,y,x)<0 or max(z,y,x)>n else ((z<n and y<n)|((z<n and x<n)<<1)|((y<n and x<n)<<2)|((x<n)<<3)|((y<n)<<4)|((z<n)<<5))
                    for field,d in enumerate(SUPPORT_DELTA):
                        shape=tuple(n+i for i in d)
                        expected=all(0<=v<k for v,k in zip((z,y,x),shape))
                        assert bool(mask&(1<<field))==expected
                        yy=n+int(field in (1,3,5));xx=n+int(field in (0,4,5))
                        assert yy==shape[1] and xx==shape[2]
                        if expected:assert (z*yy+y)*xx+x==(z*shape[1]+y)*shape[2]+x
                        checks+=1
    return dict(component_wall_and_stride_checks=checks,common_shapes=[512,564],canonical_deltas=SUPPORT_DELTA)


def stage(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=PARENT_CONSUMER:raise ValueError('Exact current winner required')
    text=original
    marker='struct FieldPacket {float hx,hy,hz,ex,ey,ez;};'
    assert text.count(marker)==1;text=text.replace(marker,HELPER+'\n'+marker)
    text=text.replace('template<int Depth,bool Fast>\n__device__ __forceinline__ FieldPacket ImportPacket','template<int Depth,bool Fast,int Cube=0>\n__device__ __forceinline__ FieldPacket ImportPacket')
    old='if constexpr(!Fast)mask=NativeSupportMask(p,oz-1+dz-version,oy-1+dy-version,ox-1+dx-version);'
    assert text.count(old)==1
    text=text.replace(old,'if constexpr(!Fast){if constexpr(Cube>0)mask=ResearchCubeSupport<Cube>(oz-1+dz-version,oy-1+dy-version,ox-1+dx-version);else mask=NativeSupportMask(p,oz-1+dz-version,oy-1+dy-version,ox-1+dx-version);}')
    start,end,body=block(text,'void StageRegisterLow(')
    body=body.replace('ImportPacket<Depth,Safe>','ImportPacket<Depth,Safe,Cube>')
    text=text[:start]+body+text[end:]
    old='template<int Depth,bool Safe,int T>\n__device__ __forceinline__ void StageRegisterLow'
    assert text.count(old)==1;text=text.replace(old,'template<int Depth,bool Safe,int T,int Cube=0>\n__device__ __forceinline__ void StageRegisterLow')
    start,end,body=block(text,'void FixedCubeCachedBody(')
    for field in range(6):body=body.replace(f'Initial(p,{field},z,y,x)',f'ResearchCubeInitial<N,{field}>(p,z,y,x)')
    body=body.replace('Initial(p,c,z,y,x)','ResearchCubeInitial<N,c>(p,z,y,x)').replace('Initial(p,c+3,z,y,x)','ResearchCubeInitial<N,c+3>(p,z,y,x)')
    # The dormant Fast startup loop has a runtime c. Retain its original IO;
    # only the live non-Fast path has explicit constant component numbers.
    body=body.replace('ResearchCubeInitial<N,c>(p,z,y,x)','Initial(p,c,z,y,x)').replace('ResearchCubeInitial<N,c+3>(p,z,y,x)','Initial(p,c+3,z,y,x)')
    body=body.replace('ImportPacket<Depth,false>','ImportPacket<Depth,false,N>').replace('ImportPacket<Depth,SupportSafe>','ImportPacket<Depth,SupportSafe,N>').replace('StageRegisterLow<Depth,SupportSafe,T>','StageRegisterLow<Depth,SupportSafe,T,N>')
    old='''if(Contains(hf,z,y,x))hf.data[z*hf.plane+y*hf.row+x]=h[c][j];
     if(Contains(ef,z,y,x))ef.data[z*ef.plane+y*ef.row+x]=e[c][j];'''
    assert body.count(old)==1
    # c is an unrolled ordinary loop variable, so use a small component
    # switch to instantiate compile-time component stride expressions.
    new='''const unsigned support=ResearchCubeSupport<N>(z,y,x);
     const int hny=N+(c==1),hnx=N+(c==0),eny=N+(c!=1),enx=N+(c!=0);
     if(support&(1u<<c))hf.data[(size_t(z)*hny+y)*hnx+x]=h[c][j];
     if(support&(1u<<(c+3)))ef.data[(size_t(z)*eny+y)*enx+x]=e[c][j];'''
    body=body.replace(old,new);text=text[:start]+body+text[end:]
    shutil.copytree(source,output);(output/'src/asymmetric_native.cuh').write_text(text)
    proof=dict(schema='beamz.trace-shape-edges.v1',proof=prove(),parent_sha256=PARENT_CONSUMER,
               consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
               runtime_physics_and_data_pointers_retained=True,expected_kernels=66,GPU_executed=False)
    (output.parent/'shape-edge-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PARENT);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output)))
