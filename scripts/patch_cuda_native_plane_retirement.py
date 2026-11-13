"""Copy-only native CUDA E-plane retirement storage/schedule candidate.

The traced point operator and immutable v13 parent remain unchanged. H stays in
fixed private slots; old/new E plane vectors bridge shared-frame last uses.
"""

import argparse
import hashlib
import json
from pathlib import Path

PARENT_SHA = "cca7202977dfb765000881b587de287c132bb7a0f14cb05131366f13b09e642c"

BODY = r"""template<int Depth,bool Safe,int Family>
__device__ __forceinline__ void StagePlaneLow(const Parameters&p,float*stage,const size_t*bases,int n,int oz,int oy,int ox){
 for(int i=threadIdx.x;i<LowXPoints;i+=Threads){const int dz=1+i/BY,dy=1+i%BY;
  const auto v=ImportPacket<Depth,Safe>(p,bases,n,oz,oy,ox,dz,dy,0);
  if constexpr(Family==0){stage[i]=v.hx;stage[LowXPoints+i]=v.hy;stage[2*LowXPoints+i]=v.hz;}
  else{stage[i]=v.ex;stage[LowXPoints+i]=v.ey;stage[2*LowXPoints+i]=v.ez;}
 }
}
template<int Slots>
__device__ __forceinline__ void RetirePlaneH(const float*stage,float(&h)[3][Slots],int n,int sy,int sx){
 const int hz=(-n-1)&(CZ-1),hy=(-n-1)&15,hx=(-n-1)&31,dy=(sy-hy)&15;
 if(sx==hx&&sy!=hy){
  #pragma unroll
  for(int j=0;j<Slots;++j)if(j!=hz){const int dz=(j-hz)&(CZ-1),key=(dz-1)*BY+dy-1;
   #pragma unroll
   for(int c=0;c<3;++c)h[c][j]=stage[c*LowXPoints+key];
  }
 }
}
template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE>
__device__ __forceinline__ void UniformBody(const Parameters&p,const int32_t*tiles){
 constexpr int Slots=CZ;
 constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);
 extern __shared__ float frame[];size_t*bases=reinterpret_cast<size_t*>(frame+3*PAD);
 volatile int*phase_step=reinterpret_cast<volatile int*>(bases+8);
 volatile int*window=phase_step+1;
 float*low_stage=frame+3*PAD+8*sizeof(size_t)/sizeof(float)+3;
 const int tile=tiles[blockIdx.x],bz=tile/(p.nx*p.ny),by=(tile/p.nx)%p.ny,bx=tile%p.nx;
 const int oz=bz*BZ,oy=by*BY,ox=bx*BX,rank=bz+by+bx;
 const uint64_t epoch=uint64_t(*p.epoch)<<32;
 if(!threadIdx.x){const int owner=(rank&3)*p.frontier+p.slots[tile];p.owner_tags[owner]=epoch|uint32_t(tile);}
 __syncthreads();
 if constexpr(!Fast){
  if(!threadIdx.x){const int mz=p.phase[0].own[0].z+1,my=p.phase[0].own[0].y+1,mx=p.phase[0].own[1].x+1;
   window[0]=max(1,max(oz-mz+1,max(oy-my+1,ox-mx+1)));
   window[1]=min(Depth,min(oz+BZ-1,min(oy+BY-1,ox+BX-1)));
  }
  __syncthreads();if(window[0]>window[1])return;
 }
 if(threadIdx.x<8){const int bits=threadIdx.x,dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;size_t base=size_t(-1);
  if(bz>=dz&&by>=dy&&bx>=dx){const int producer=tile-dz*p.nx*p.ny-dy*p.nx-dx;const int owner=((rank-dz-dy-dx)&3)*p.frontier+p.slots[producer];
   if(bits&&p.owner_tags[owner]!=(epoch|uint32_t(producer)))atomicOr(p.bad,1);else base=size_t(owner)*Depth*6*SurfacePoints;
  }bases[bits]=base;
 }
 const int sx=threadIdx.x&31,sy=threadIdx.x>>5;
 float h[3][Slots];
 __syncthreads(); // Neighbor bases must precede a possible late-start import.
 const int first=Fast?1:window[0],prev=first-1;
 #pragma unroll
 for(int j=0;j<Slots;++j){FieldPacket v{};
  if(first==1){const int z=oz+(j==CZ-1?-1:j),y=oy+(sy==15?-1:sy),x=ox+(sx==31?-1:sx);
   v={Initial(p,0,z,y,x),Initial(p,1,z,y,x),Initial(p,2,z,y,x),Initial(p,3,z,y,x),Initial(p,4,z,y,x),Initial(p,5,z,y,x)};
  }else{const int hz=(-prev-1)&(CZ-1),hy=(-prev-1)&15,hx=(-prev-1)&31;
   if(j==hz||sy==hy||sx==hx)v=ImportPacket<Depth,false>(p,bases,prev,oz,oy,ox,(j-hz)&(CZ-1),(sy-hy)&15,(sx-hx)&31);
  }
  h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;
  frame[Slot(j,sy,sx)]=v.ex;frame[PAD+Slot(j,sy,sx)]=v.ey;frame[2*PAD+Slot(j,sy,sx)]=v.ez;
 }
 if(!threadIdx.x)*phase_step=first;__syncthreads();
 const uint8_t flags=Policy==0?0:Policy==2?1:Policy==3?16:Policy==4?17:p.flags[tile];
 Environment<Layout,Fast,IsotropicH,Policy,UniformE>env{p,flags};
 #pragma unroll 1
 for(int n=(Fast?1:window[0]);n<=(Fast?Depth:window[1]);++n){
  {const int step=*phase_step,hz=(-step-1)&(CZ-1),hy=(-step-1)&15,hx=(-step-1)&31,low=-step-1;
   const int y=oy+low+((sy-hy)&15),x=ox+low+((sx-hx)&31);
   #pragma unroll
   for(int j=0;j<Slots;++j){const int z=oz+low+((j-hz)&(CZ-1));
    if(j!=hz&&sy!=hy&&sx!=hx){const auto read=[&] __device__(int field,int,int dz,int dy,int dx){
     if(field<3)return h[field][j];return frame[(field-3)*PAD+Slot(j+dz,sy+dy,sx+dx)];
    };
     #pragma unroll
     for(int c=0;c<3;++c){float value=0.f;if(env.Valid(c,z,y,x)){
      if(c==0)value=beamz::cuda::trace_native::AdvanceNativeCell<0,0,false>(env,read,n,z,y,x);
      else if(c==1)value=beamz::cuda::trace_native::AdvanceNativeCell<0,1,false>(env,read,n,z,y,x);
      else value=beamz::cuda::trace_native::AdvanceNativeCell<0,2,false>(env,read,n,z,y,x);
     }h[c][j]=value;}
    }
   }
  }
  {const int hz=(-n-1)&(CZ-1),hy=(-n-1)&15,hx=(-n-1)&31;
   #pragma unroll
   for(int j=0;j<Slots;++j)if(j==hz||sy==hy){const auto v=ImportPacket<Depth,SupportSafe>(p,bases,n,oz,oy,ox,(j-hz)&(CZ-1),(sy-hy)&15,(sx-hx)&31);
    h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;
   }
  }
  StagePlaneLow<Depth,SupportSafe,0>(p,low_stage,bases,n,oz,oy,ox);
  __syncthreads(); // R1: retire ALL old-E H readers; publish low-X H packet.
  RetirePlaneH(low_stage,h,n,sy,sx);
  float last[3],pending[3];
  #pragma unroll
  for(int c=0;c<3;++c){last[c]=frame[c*PAD+Slot(CZ-1,sy,sx)];frame[c*PAD+Slot(CZ-1,sy,sx)]=h[c][CZ-1];}
  __syncthreads(); // R2: publish reserved Hlast; retire every H-stage reader.
  #pragma unroll
  for(int j=0;j<Slots;++j){float current[3];
   #pragma unroll
   for(int c=0;c<3;++c){current[c]=j==CZ-1?last[c]:frame[c*PAD+Slot(j,sy,sx)];if(j!=CZ-1)frame[c*PAD+Slot(j,sy,sx)]=h[c][j];}
   __syncthreads(); // Pj: current H XY publication; old E own-center was saved.
   {const int step=*phase_step,hz=(-step-1)&(CZ-1),hy=(-step-1)&15,hx=(-step-1)&31,low=-step-1;
    const int dy=(sy-hy)&15,dx=(sx-hx)&31,dz=(j-hz)&(CZ-1),z=oz+low+dz,y=oy+low+dy,x=ox+low+dx;
    if(j!=hz&&sy!=hy&&sx!=hx){const auto read=[&] __device__(int field,int,int ddz,int ddy,int ddx){
     if(field>=3)return current[field-3];if(!ddz&&!ddy&&!ddx)return h[field][j];return frame[field*PAD+Slot(j+ddz,sy+ddy,sx+ddx)];
    };
     #pragma unroll
     for(int c=0;c<3;++c){float value=0.f;if(env.Valid(c+3,z,y,x)){
      if(c==0)value=beamz::cuda::trace_native::AdvanceNativeCell<1,0,false>(env,read,n,z,y,x);
      else if(c==1)value=beamz::cuda::trace_native::AdvanceNativeCell<1,1,false>(env,read,n,z,y,x);
      else value=beamz::cuda::trace_native::AdvanceNativeCell<1,2,false>(env,read,n,z,y,x);
     }current[c]=value;}
     const int surface=Surface(dz-1,dy-1,dx-1);
     #pragma unroll
     for(int c=0;c<3;++c){
      if(surface>=0){p.trails[bases[0]+((n-1)*6+c)*SurfacePoints+surface]=h[c][j];p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=current[c];}
      if(n==Depth){const auto&hf=FinalField(p,c);const auto&ef=FinalField(p,c+3);
       if(Contains(hf,z,y,x))hf.data[z*hf.plane+y*hf.row+x]=h[c][j];
       if(Contains(ef,z,y,x))ef.data[z*ef.plane+y*ef.row+x]=current[c];
      }
     }
    }else{
     #pragma unroll
     for(int c=0;c<3;++c)current[c]=0.f; // Current ghosts are restored after all E reads.
    }
   }
   __syncthreads(); // Qj: conservative retirement of current-H E readers.
   #pragma unroll
   for(int c=0;c<3;++c){if(j>=1)frame[c*PAD+Slot(j-1,sy,sx)]=pending[c];pending[c]=current[c];}
  }
  #pragma unroll
  for(int c=0;c<3;++c)frame[c*PAD+Slot(CZ-1,sy,sx)]=pending[c];
  {const int hz=(-n-1)&(CZ-1),hy=(-n-1)&15,hx=(-n-1)&31;
   #pragma unroll
   for(int j=0;j<Slots;++j)if(j==hz||sy==hy){const auto v=ImportPacket<Depth,SupportSafe>(p,bases,n,oz,oy,ox,(j-hz)&(CZ-1),(sy-hy)&15,(sx-hx)&31);
    frame[Slot(j,sy,sx)]=v.ex;frame[PAD+Slot(j,sy,sx)]=v.ey;frame[2*PAD+Slot(j,sy,sx)]=v.ez;
   }
  }
  StagePlaneLow<Depth,SupportSafe,1>(p,low_stage,bases,n,oz,oy,ox);
  __syncthreads(); // R3: publish E low-X packet, all ordinary E writes/ghosts.
  {const int hz=(-n-1)&(CZ-1),hy=(-n-1)&15,hx=(-n-1)&31,dy=(sy-hy)&15;
   if(sx==hx&&sy!=hy){
    #pragma unroll
    for(int j=0;j<Slots;++j)if(j!=hz){const int key=(((j-hz)&(CZ-1))-1)*BY+dy-1;
     #pragma unroll
     for(int c=0;c<3;++c)frame[c*PAD+Slot(j,sy,sx)]=low_stage[c*LowXPoints+key];
    }
   }
  }
  if(!threadIdx.x)*phase_step=n+1;
  __syncthreads(); // R4: complete E frame and phase word for next H update.
 }
}

"""


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one retirement patch anchor: {old!r}")
    return source.replace(old, new, 1)


def patch(source, cached_z=16, reduced_barriers=False, identify=False):
    if hashlib.sha256(source.encode()).hexdigest() != PARENT_SHA:
        raise ValueError("Expected exact frozen v13 consumer")
    source = once(
        source, "BZ=7,BY=15,BX=31,CZ=8", f"BZ={cached_z - 1},BY=15,BX=31,CZ={cached_z}"
    )
    source = once(
        source,
        "3*sizeof(int)+6ULL*LowXPoints*sizeof(float);",
        "3*sizeof(int)+3ULL*LowXPoints*sizeof(float);",
    )
    start = source.index(
        "template<int Depth,bool Safe,int T>\n__device__ __forceinline__ void StageRegisterLow"
    )
    end = source.index(
        "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__",
        start,
    )
    source = source[:start] + BODY + source[end:]
    if reduced_barriers:
        source = once(
            source,
            "__syncthreads(); // R2: publish reserved Hlast; retire every H-stage reader.",
            "// Seed publication/H-stage retirement merge into P0 below.",
        )
        source = once(
            source,
            "__syncthreads(); // Pj: current H XY publication; old E own-center was saved.",
            "if(j!=CZ-1)__syncthreads(); // P0 also publishes seedHlast; Plast is unnecessary.",
        )
        source = once(
            source,
            "__syncthreads(); // Qj: conservative retirement of current-H E readers.",
            "// Previous-H Z readers are same-thread; final readers retire at R3.",
        )
        source = once(
            source,
            "for(int c=0;c<3;++c){if(j>=1)frame[c*PAD+Slot(j-1,sy,sx)]=pending[c];pending[c]=current[c];}",
            "for(int c=0;c<3;++c){if(j==CZ-1)last[c]=current[c];else{if(j>=1)frame[c*PAD+Slot(j-1,sy,sx)]=pending[c];pending[c]=current[c];}}",
        )
        first = source.index(
            "  #pragma unroll\n  for(int c=0;c<3;++c)frame[c*PAD+Slot(CZ-1,sy,sx)]=pending[c];"
        )
        last = source.index("  StagePlaneLow<Depth,SupportSafe,1>", first)
        ghost = source[source.index("  {const int hz=", first) : last]
        source = source[:first] + source[last:]
        source = once(
            source,
            "__syncthreads(); // R3: publish E low-X packet, all ordinary E writes/ghosts.",
            """__syncthreads(); // R3 retires ALL E(last-1)/Elast readers and publishes low-X E.
  #pragma unroll
  for(int c=0;c<3;++c){frame[c*PAD+Slot(CZ-2,sy,sx)]=pending[c];frame[c*PAD+Slot(CZ-1,sy,sx)]=last[c];}
"""
            + ghost,
        )
    if identify:
        source = once(source, "void Asymmetric(", "void AsymmetricRetirePlane(")
        source = once(
            source,
            "const auto kernel=Asymmetric<",
            "const auto kernel=AsymmetricRetirePlane<",
        )
        source = once(
            source,
            " Asymmetric<Depth,Layout,Fast,IsotropicH,Policy><<<",
            " AsymmetricRetirePlane<Depth,Layout,Fast,IsotropicH,Policy><<<",
        )
    source = once(
        source, "__launch_bounds__((Fast?1024:512),1)", "__launch_bounds__(512,1)"
    )
    return once(
        source,
        "<<<count,(Fast?1024:512),SharedBytes,stream>>>",
        "<<<count,Threads,SharedBytes,stream>>>",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cached-z", choices=(8, 16), type=int, default=16)
    parser.add_argument("--reduced-barriers", action="store_true")
    parser.add_argument("--identify", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.source.resolve() == root / "cuda":
        parser.error("Private copied source required")
    consumer = args.source / "src/asymmetric_native.cuh"
    consumer.write_text(
        patch(consumer.read_text(), args.cached_z, args.reduced_barriers, args.identify)
    )
    p = args.source / "src/native_moving.h"
    text = p.read_text()
    text = once(text, "kOwnedZ=7,kOwnedY=15", f"kOwnedZ={args.cached_z - 1},kOwnedY=15")
    text = once(
        text,
        "counts{},owned{7,15,31}",
        f"counts{{}},owned{{{args.cached_z - 1},15,31}}",
    )
    text = once(
        text, "int owned[3]{7,15,31};", f"int owned[3]{{{args.cached_z - 1},15,31}};"
    )
    p.write_text(text)
    if args.identify:
        p = args.source / "src/native_extension.cc"
        text = p.read_text()
        barriers = args.cached_z + 2 if args.reduced_barriers else 2 * args.cached_z + 4
        schedule = "last-use18" if args.reduced_barriers else "conservative"
        text = once(
            text,
            "NB_MODULE(_cuda_native, module) {",
            "NB_MODULE(_cuda_native, module) {\n"
            f'  module.attr("plane_retirement_cached_zyx")=nb::make_tuple({args.cached_z},16,32);\n'
            f'  module.attr("plane_retirement_schedule")="{schedule}";\n'
            f'  module.attr("plane_retirement_barriers_per_step")={barriers};\n'
            '  module.attr("plane_retirement_kernel")="AsymmetricRetirePlane";',
        )
        text = once(
            text,
            'result["moving_depth"]=moving_workspace_.depth;',
            f'result["plane_retirement_cached_zyx"]=nb::make_tuple({args.cached_z},16,32);'
            f'result["plane_retirement_schedule"]="{schedule}";'
            f'result["plane_retirement_barriers_per_step"]={barriers};'
            'result["moving_depth"]=moving_workspace_.depth;',
        )
        p.write_text(text)
    print(
        json.dumps(
            {
                "consumer_sha256": hashlib.sha256(consumer.read_bytes()).hexdigest(),
                "owned_zyx": [args.cached_z - 1, 15, 31],
                "cached_zyx": [args.cached_z, 16, 32],
                "threads": 512,
                "shared_bytes": 3 * args.cached_z * 16 * 32 * 4
                + 76
                + 3 * (args.cached_z - 1) * 15 * 4,
                "barriers_per_step": args.cached_z + 2
                if args.reduced_barriers
                else 2 * args.cached_z + 4,
                "reduced_barriers": args.reduced_barriers,
                "identified_kernel": args.identify,
                "scope": "Resource/correctness candidate only; native arithmetic unchanged",
            }
        )
    )


if __name__ == "__main__":
    main()
