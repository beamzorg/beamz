// Research field-only full-paper cache: warp pairs exchange Z neighbors,
// two shared components plus a narrow Y-component X-edge frame.
#pragma once
#ifndef MOVING_REGISTER_INTERIOR_RESOURCE
#define MOVING_REGISTER_INTERIOR_RESOURCE 0
#endif
namespace moving_register_cache {
constexpr int CZ=16,CY=16,CX=32,PAD=CZ*CY*CX,Slots=16;
constexpr int FrameComponents=2,EdgeFloats=2*CY*CZ;
constexpr size_t SharedBytes=static_cast<size_t>(FrameComponents*PAD+EdgeFloats)*sizeof(float)+8*sizeof(size_t)+sizeof(int);
static_assert(BZ==15&&BY==15&&BX==31&&THREADS==512,"Z-pair cache requires owned15x15x31 and512threads");
static_assert(MOVING_REGISTER_PHASE_METADATA,"Z-pair metadata protocol requires the phase word");
__device__ __forceinline__ int Slot(int z,int y,int x){return ((z&15)*16+(y&15))*32+(x&31);}
__device__ __forceinline__ int ZSlot(int j){return ((threadIdx.x&31)>>4)+2*(j&7);}
__device__ __forceinline__ int YSlot(int j){return (threadIdx.x>>6)+8*(j>>3);}
__device__ __forceinline__ int EdgeSlot(int z,int y,int block){return (z*16+y)*2+block;}
template<int Steps,bool Interior=false>
__device__ __forceinline__ float Incoming(const float*initial,const float*trail,const size_t*bases,
 Geometry g,int version,int field,int oz,int oy,int ox,int dz,int dy,int dx,int*bad){
 const int uz=oz-1+dz,uy=oy-1+dy,ux=ox-1+dx;
 if constexpr(!Interior){
  if(!Valid(field,uz-version,uy-version,ux-version,g.n))return 0.f;
  if(!version)return GlobalAt(initial,field,uz,uy,ux,g.n);
 }
 const int neighbor=(dz==0)|((dy==0)<<1)|((dx==0)<<2);
 const int surface=Surface(dz==0?BZ-1:dz-1,dy==0?BY-1:dy-1,dx==0?BX-1:dx-1);
 const size_t base=bases[neighbor];
 if(!neighbor||surface<0||base==static_cast<size_t>(-1)){atomicAdd(bad,1);return __int_as_float(0x7fffffff);}
 return trail[base+((version-1)*6+field)*S+surface];
}
template<bool Interior>
__device__ __forceinline__ float Initial(const float*input,int c,int z,int y,int x,int n){
 if constexpr(Interior)return input[Offset(c,z,y,x,n)];
 return GlobalAt(input,c,z,y,x,n);
}
}
#include "cuda_moving_register_z_pair_y2_packets.cuh"

template<int Steps,bool Interior=(MOVING_REGISTER_INTERIOR_RESOURCE!=0)> __global__ __launch_bounds__(THREADS,MOVING_MIN_CTAS)
void RegisterCachedMoving(const int*tiles,Geometry g,const float*initial,float*output,float*trail,int*bad){
 using namespace moving_register_cache;
 extern __shared__ float storage[];
 float*frame=storage;float*edge=frame+FrameComponents*PAD;
 size_t*bases=reinterpret_cast<size_t*>(edge+EdgeFloats);
 volatile int*phase_step=reinterpret_cast<volatile int*>(bases+8);
 float*low_stage=storage+SharedBytes/sizeof(float);
 const int tile=tiles[blockIdx.x],bz=tile/(g.nx*g.ny),by=(tile/g.nx)%g.ny,bx=tile%g.nx;
 const int oz=bz*BZ,oy=by*BY,ox=bx*BX;
#if MOVING_TRAIL_RING
 if(!threadIdx.x)g.owner_tags[TrailOwner(g,tile,bz+by+bx)]=tile;
 __syncthreads();
#endif
 if(threadIdx.x<8){const int bits=threadIdx.x,dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;size_t base=static_cast<size_t>(-1);
  if(bz>=dz&&by>=dy&&bx>=dx){const int producer=tile-dz*g.nx*g.ny-dy*g.nx-dx,owner=TrailOwner(g,producer,bz+by+bx-dz-dy-dx);
#if MOVING_TRAIL_RING
   if(bits&&g.owner_tags[owner]!=producer)atomicAdd(bad,1);else
#endif
   base=static_cast<size_t>(owner)*Steps*6*S;
  }bases[bits]=base;
 }
 const int lane=threadIdx.x&31,parity=lane>>4,xblock=(threadIdx.x>>5)&1;
 const int sx=(lane&15)+16*xblock;
 float h[3][Slots],e[3][Slots];
 #pragma unroll
 for(int j=0;j<Slots;++j){const int sy=YSlot(j),zs=ZSlot(j),z=zs==15?-1:zs,y=sy==15?-1:sy,x=sx==31?-1:sx;
  #pragma unroll
  for(int c=0;c<3;++c){h[c][j]=Initial<Interior>(initial,c,oz+z,oy+y,ox+x,g.n);e[c][j]=Initial<Interior>(initial,c+3,oz+z,oy+y,ox+x,g.n);if(c!=1)frame[(c/2)*PAD+Slot(zs,sy,sx)]=e[c][j];}
  if((sx&15)==0)edge[EdgeSlot(zs,sy,xblock)]=e[1][j];
 }
 if(!threadIdx.x)*phase_step=1;__syncthreads();
 #pragma unroll 1
 for(int n=1;n<=Steps;++n){
  {const int phase=*phase_step,hz=(-phase-1)&15,hy=(-phase-1)&15,hx=(-phase-1)&31,low=-phase-1;
   #pragma unroll
   for(int j=0;j<Slots;++j){const int sy=YSlot(j),zs=ZSlot(j);
    // Source parity selects the operand before the cross-parity exchange.
    // All lanes execute both instructions, including ghost/invalid owners.
    const float ey_z=__shfl_xor_sync(0xffffffffu,parity?e[1][j]:e[1][(j&8)|((j+1)&7)],16);
    const float ey_x=__shfl_sync(0xffffffffu,e[1][j],((lane&15)+1)&15,16);
    if(zs!=hz&&sy!=hy&&sx!=hx){
     const auto read=[&] __device__(int field,int,int dz,int dy,int dx){
      if(field<3)return h[field][j];
      const int c=field-3;if(!dz&&!dy&&!dx)return e[c][j];
      if(c==1){if(dz)return ey_z;if(dx)return (sx&15)==15?edge[EdgeSlot(zs,sy,xblock^1)]:ey_x;}
      return frame[(c/2)*PAD+Slot(zs+dz,sy+dy,sx+dx)];
     };
     #pragma unroll
     for(int c=0;c<3;++c){if constexpr(Interior)h[c][j]=Generated<0>(c,read);
      else{const int pz=oz+low+((zs-hz)&15),py=oy+low+((sy-hy)&15),px=ox+low+((sx-hx)&31);h[c][j]=Valid(c,pz,py,px,g.n)?Generated<0>(c,read):0.f;}}

    }
   }
  }
  {const int hz=(-n-1)&15,hy=(-n-1)&15,hx=(-n-1)&31;
   const int dx=(sx-hx)&31;
   #pragma unroll
   for(int j=0;j<Slots;++j){const int sy=YSlot(j),zs=ZSlot(j);if(zs==hz||sy==hy){const int dz=(zs-hz)&15,dy=(sy-hy)&15;
    #pragma unroll
    for(int c=0;c<3;++c){h[c][j]=Incoming<Steps,Interior>(initial,trail,bases,g,n-MOVING_REGISTER_BAD_IMPORT_PHASE,c,oz,oy,ox,dz,dy,dx,bad);e[c][j]=Incoming<Steps,Interior>(initial,trail,bases,g,n-MOVING_REGISTER_BAD_IMPORT_PHASE,c+3,oz,oy,ox,dz,dy,dx,bad);}
   }}
  }
  moving_register_packets::StageLow<Steps,Interior>(low_stage,trail,bases,g,n,oz,oy,ox,bad);
  __syncthreads(); // B1 retires old E frame/edge reads and publishes low stage.
  moving_register_packets::RetireLow(low_stage,h,e,n,0,sx);
  #pragma unroll
  for(int j=0;j<Slots;++j){const int sy=YSlot(j),zs=ZSlot(j);
   #pragma unroll
   for(int c=0;c<3;++c)if(c!=1)frame[(c/2)*PAD+Slot(zs,sy,sx)]=h[c][j];
   if((sx&15)==15)edge[EdgeSlot(zs,sy,xblock)]=h[1][j];
  }
  __syncthreads(); // B2 publishes current H frame and its last-X-column edges.
  {const int phase=*phase_step,hz=(-phase-1)&15,hy=(-phase-1)&15,hx=(-phase-1)&31,low=-phase-1;
   #pragma unroll
   for(int j=0;j<Slots;++j){const int sy=YSlot(j),zs=ZSlot(j);
    const float hy_z=__shfl_xor_sync(0xffffffffu,parity?h[1][(j&8)|((j-1)&7)]:h[1][j],16);
    const float hy_x=__shfl_sync(0xffffffffu,h[1][j],((lane&15)-1)&15,16);
    if(zs!=hz&&sy!=hy&&sx!=hx){
     const auto read=[&] __device__(int field,int,int dz,int dy,int dx){
      if(field>=3)return e[field-3][j];
      if(!dz&&!dy&&!dx)return h[field][j];
      if(field==1){if(dz)return hy_z;if(dx)return (sx&15)==0?edge[EdgeSlot(zs,sy,xblock^1)]:hy_x;}
      return frame[(field/2)*PAD+Slot(zs+dz,sy+dy,sx+dx)];
     };
     #pragma unroll
     for(int c=0;c<3;++c){if constexpr(Interior)e[c][j]=Generated<1>(c,read);
      else{const int pz=oz+low+((zs-hz)&15),py=oy+low+((sy-hy)&15),px=ox+low+((sx-hx)&31);e[c][j]=Valid(c+3,pz,py,px,g.n)?Generated<1>(c,read):0.f;}}

    }
   }
  }
  __syncthreads(); // B3 retires H frame/edge reads before E-frame replacement.
  #pragma unroll
  for(int j=0;j<Slots;++j){const int sy=YSlot(j),zs=ZSlot(j);
   #pragma unroll
   for(int c=0;c<3;++c)if(c!=1)frame[(c/2)*PAD+Slot(zs,sy,sx)]=e[c][j];
   if((sx&15)==0)edge[EdgeSlot(zs,sy,xblock)]=e[1][j];
  }
  if(!threadIdx.x)*phase_step=n+1;__syncthreads(); // B4 publishes old E for next H.
  {const int hz=(-n-1)&15,hy=(-n-1)&15,hx=(-n-1)&31,low=-n-1;
   const int dx=(sx-hx)&31,px=ox+low+dx;
   #pragma unroll
   for(int j=0;j<Slots;++j){const int sy=YSlot(j),zs=ZSlot(j);if(zs!=hz&&sy!=hy&&sx!=hx){const int dy=(sy-hy)&15,py=oy+low+dy,dz=(zs-hz)&15,pz=oz+low+dz,surface=Surface(dz-1,dy-1,dx-1);
    #pragma unroll
    for(int c=0;c<3;++c){if(surface>=0){trail[bases[0]+((n-1)*6+c)*S+surface]=h[c][j];trail[bases[0]+((n-1)*6+c+3)*S+surface]=e[c][j];}
     if(n==Steps){if constexpr(Interior){output[Offset(c,pz,py,px,g.n)]=h[c][j];output[Offset(c+3,pz,py,px,g.n)]=e[c][j];}
      else{if(Valid(c,pz,py,px,g.n))output[Offset(c,pz,py,px,g.n)]=h[c][j];if(Valid(c+3,pz,py,px,g.n))output[Offset(c+3,pz,py,px,g.n)]=e[c][j];}}
    }
   }}
  }
 }
}
