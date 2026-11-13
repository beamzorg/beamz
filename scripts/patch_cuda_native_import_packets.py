"""Emit an opt-in canonical-support/six-field trail-import lowering.

The original consumer stays unchanged. Works on either fixed unrolling or the
scalar-queue candidate, so one storage change can be evaluated independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PACKETS = r"""// Canonical staggering was proved by Prepare before this kernel can run.
// Bounds are rectangular; imports outside every support never touch trails.
__device__ __forceinline__ unsigned NativeSupportMask(const Parameters&p,int z,int y,int x){
 const int nz=p.phase[0].own[0].z,ny=p.phase[0].own[0].y,nx=p.phase[0].own[1].x;
 if(z<0||y<0||x<0||z>nz||y>ny||x>nx)return 0;
 const unsigned iz=z<nz,iy=y<ny,ix=x<nx;
 return (iz&&iy)|((iz&&ix)<<1)|((iy&&ix)<<2)|(ix<<3)|(iy<<4)|(iz<<5);
}
struct FieldPacket {float hx,hy,hz,ex,ey,ez;};
template<int Depth,bool Fast>
__device__ __forceinline__ FieldPacket ImportPacket(const Parameters&p,const size_t*bases,int version,
 int oz,int oy,int ox,int dz,int dy,int dx){
 unsigned mask=63;
 if constexpr(!Fast)mask=NativeSupportMask(p,oz-1+dz-version,oy-1+dy-version,ox-1+dx-version);
 if(!mask)return {};
 const int neighbor=(dz==0)|((dy==0)<<1)|((dx==0)<<2);
 const int surface=Surface(dz?dz-1:BZ-1,dy?dy-1:BY-1,dx?dx-1:BX-1);
 const size_t base=bases[neighbor];
 if(!neighbor||surface<0||base==static_cast<size_t>(-1)){
  atomicOr(p.bad,2);const float bad=__int_as_float(0x7fffffff);
  return {mask&1?bad:0.f,mask&2?bad:0.f,mask&4?bad:0.f,mask&8?bad:0.f,mask&16?bad:0.f,mask&32?bad:0.f};
 }
 const float*point=p.trails+base+static_cast<size_t>(version-1)*6*SurfacePoints+surface;
 return {mask&1?point[0*SurfacePoints]:0.f,mask&2?point[1*SurfacePoints]:0.f,
         mask&4?point[2*SurfacePoints]:0.f,mask&8?point[3*SurfacePoints]:0.f,
         mask&16?point[4*SurfacePoints]:0.f,mask&32?point[5*SurfacePoints]:0.f};
}
template<int Depth,bool Fast>
__device__ __forceinline__ void StageLowX(const Parameters&p,float*shared,float*stage,const size_t*bases,int version,int oz,int oy,int ox){
 const int head=WrapZ(-version-1),yh=(-version-1)&15,xh=(-version-1)&31;
 for(int i=threadIdx.x;i<LowXPoints;i+=Threads){const int dz=1+i/BY,dy=1+i%BY;
  int j=head+dz;if(j>=CZ)j-=CZ;const int sy=(yh+dy)&15;
  const auto value=ImportPacket<Depth,Fast>(p,bases,version,oz,oy,ox,dz,dy,0);
  stage[i]=value.hy;stage[LowXPoints+i]=value.ey;
  At(shared,0,j,sy,xh)=value.hx;At(shared,2,j,sy,xh)=value.hz;
  At(shared,3,j,sy,xh)=value.ex;At(shared,5,j,sy,xh)=value.ez;
 }
}
"""

OLD_LOOP = """    #pragma unroll
    for(int c=0;c<3;++c){const float h=Import<Depth>(p,bases,c,n,oz,oy,ox,dz,dy,dx),e=Import<Depth>(p,bases,c+3,n,oz,oy,ox,dz,dy,dx);if(c==1){hy[j]=h;ey[j]=e;}else{At(shared,c,j,sy,sx)=h;At(shared,c+3,j,sy,sx)=e;}}
"""
NEW_LOOP = """    const auto value=ImportPacket<Depth,Fast>(p,bases,n,oz,oy,ox,dz,dy,dx);
    hy[j]=value.hy;ey[j]=value.ey;
    At(shared,0,j,sy,sx)=value.hx;At(shared,2,j,sy,sx)=value.hz;
    At(shared,3,j,sy,sx)=value.ex;At(shared,5,j,sy,sx)=value.ez;
"""


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one packet patch anchor: {old!r}")
    return source.replace(old, new, 1)


def patch(source):
    start = source.index(
        "template<int Depth>\n__device__ __forceinline__ float Import("
    )
    end = source.index("__device__ __forceinline__ void RetireLowX(", start)
    source = source[:start] + PACKETS + source[end:]
    source = once(source, OLD_LOOP, NEW_LOOP)
    return once(source, "StageLowX<Depth>(", "StageLowX<Depth,Fast>(")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input", type=Path, default=ROOT / "scripts/cuda_asymmetric_native.cuh"
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.input.resolve() == a.output.resolve():
        p.error("Input is frozen; output must be a different path")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(patch(a.input.read_text()))
    print(
        json.dumps(
            {
                "input": str(a.input),
                "input_sha256": hashlib.sha256(a.input.read_bytes()).hexdigest(),
                "output": str(a.output),
                "output_sha256": hashlib.sha256(a.output.read_bytes()).hexdigest(),
                "scope": "Import metadata only; all30/race/performance gates required",
            }
        )
    )


if __name__ == "__main__":
    main()
