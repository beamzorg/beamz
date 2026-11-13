"""Copy-only pruning of globally unsupported register-frame tile versions.

Fast tiles retain the original full-depth path. General tiles reconstruct a
late-start cache only from valid lower-neighbor trails, with actual versions.
"""

import argparse
import hashlib
import json
from pathlib import Path


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one time-window anchor: {old!r}")
    return source.replace(old, new, 1)


def patch(source):
    source = once(
        source,
        "sizeof(int)+6ULL*LowXPoints*sizeof(float);",
        "3*sizeof(int)+6ULL*LowXPoints*sizeof(float);",
    )
    source = once(
        source,
        "float*low_stage=frame+3*PAD+8*sizeof(size_t)/sizeof(float)+1;",
        "volatile int*window=phase_step+1;\n float*low_stage=frame+3*PAD+8*sizeof(size_t)/sizeof(float)+3;",
    )
    source = once(
        source,
        "__syncthreads();\n if(threadIdx.x<8)",
        """__syncthreads();
 if constexpr(!Fast){
  if(!threadIdx.x){
   const int mz=p.phase[0].own[0].z+1,my=p.phase[0].own[0].y+1,mx=p.phase[0].own[1].x+1;
   window[0]=max(1,max(oz-mz+1,max(oy-my+1,ox-mx+1)));
   window[1]=min(Depth,min(oz+BZ-1,min(oy+BY-1,ox+BX-1)));
  }
  __syncthreads();
  // The owner tag remains published even when no component can consume trails.
  if(window[0]>window[1])return;
 }
 if(threadIdx.x<8)""",
    )
    begin = source.index(
        " #pragma unroll\n for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+"
    )
    end = source.index(" const uint8_t flags=", begin)
    original = source[begin:end]
    startup = """ // A late start must see the neighbor base table before reading trails.
 if constexpr(!Fast)__syncthreads();
 if constexpr(Fast){
ORIGINAL
 }else{
  const int first=window[0],prev=first-1;
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;
   FieldPacket v{};
   if(first==1){const int z=oz+(zs==7?-1:zs),y=oy+(sy==15?-1:sy),x=ox+(sx==31?-1:sx);
    v={Initial(p,0,z,y,x),Initial(p,1,z,y,x),Initial(p,2,z,y,x),Initial(p,3,z,y,x),Initial(p,4,z,y,x),Initial(p,5,z,y,x)};
   }else{
    const int hz=(-prev-1)&7,hy=(-prev-1)&15,hx=(-prev-1)&31;
    // Previous owned volume is wholly invalid; only the low ghost union may live.
    if(zs==hz||sy==hy||sx==hx)v=ImportPacket<Depth,false>(p,bases,prev,oz,oy,ox,(zs-hz)&7,(sy-hy)&15,(sx-hx)&31);
   }
   h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;
   e[0][j]=v.ex;e[1][j]=v.ey;e[2][j]=v.ez;
   frame[Slot(zs,sy,sx)]=v.ex;frame[PAD+Slot(zs,sy,sx)]=v.ey;frame[2*PAD+Slot(zs,sy,sx)]=v.ez;
  }
  if(!threadIdx.x)*phase_step=first;
  __syncthreads();
 }
""".replace("ORIGINAL", original.rstrip())
    source = source[:begin] + startup + source[end:]
    return once(
        source,
        "for(int n=1;n<=Depth;++n){",
        "for(int n=(Fast?1:window[0]);n<=(Fast?Depth:window[1]);++n){",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("A distinct copied output is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = args.input.read_text()
    args.output.write_text(patch(source))
    print(
        json.dumps(
            {
                "parent_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "consumer_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "scope": "CPU/resource candidate; late-start, all30 and sanitizer gates required",
            }
        )
    )


if __name__ == "__main__":
    main()
