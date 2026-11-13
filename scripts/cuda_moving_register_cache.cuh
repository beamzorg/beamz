// Research split-register cache: retain all six families in fixed GPR slots,
// exchange the opposite family through one shared frame. No CPML/source/DFT.
#pragma once

#ifndef MOVING_REGISTER_WARP_NEIGHBORS
#define MOVING_REGISTER_WARP_NEIGHBORS 0
#endif
#ifndef MOVING_REGISTER_PHASE_METADATA
#define MOVING_REGISTER_PHASE_METADATA 0
#endif
#ifndef MOVING_REGISTER_LOCAL_CENTERS
#define MOVING_REGISTER_LOCAL_CENTERS 0
#endif
#ifndef MOVING_REGISTER_BAD_IMPORT_PHASE
#define MOVING_REGISTER_BAD_IMPORT_PHASE 0
#endif
namespace moving_register_cache {
constexpr int CZ=BZ+1,CY=BY+1,CX=BX+1,PAD=CZ*CY*CX,Slots=PAD/THREADS;
constexpr int FrameComponents=MOVING_REGISTER_WARP_NEIGHBORS?2:3;
constexpr size_t SharedBytes=static_cast<size_t>(FrameComponents)*PAD*sizeof(float)+8*sizeof(size_t)+(MOVING_REGISTER_PHASE_METADATA?sizeof(int):0);
static_assert(BZ==15&&BY==15&&(BX==15||BX==31),"Split register prototype supports owned15x15x15 or15x15x31");
static_assert((Slots==8||Slots==16)&&PAD%THREADS==0&&THREADS%(CX*CY)==0,"Fixed8/16 register slots and whole cache planes required");
static_assert(!MOVING_REGISTER_WARP_NEIGHBORS || (CX==32&&CY==16&&CZ==16&&THREADS==CX*CY&&Slots==16&&MOVING_REGISTER_PHASE_METADATA),"Warp neighbors require cache16x16x32/512threads and phase metadata");
__device__ __forceinline__ int Slot(int z,int y,int x){return ((z&(CZ-1))*CY+(y&(CY-1)))*CX+(x&(CX-1));}
template<int Steps>
__device__ __forceinline__ float Incoming(const float*initial,const float*trail,const size_t*bases,
 Geometry g,int version,int field,int oz,int oy,int ox,int dz,int dy,int dx,int*bad){
 const int uz=oz-1+dz,uy=oy-1+dy,ux=ox-1+dx;
 if(!Valid(field,uz-version,uy-version,ux-version,g.n))return 0.f;
 if(!version)return GlobalAt(initial,field,uz,uy,ux,g.n);
 const int neighbor=(dz==0)|((dy==0)<<1)|((dx==0)<<2);
 const int surface=Surface(dz==0?BZ-1:dz-1,dy==0?BY-1:dy-1,dx==0?BX-1:dx-1);
 const size_t base=bases[neighbor];
 if(!neighbor||surface<0||base==static_cast<size_t>(-1)){atomicAdd(bad,1);return __int_as_float(0x7fffffff);}
 return trail[base+((version-1)*6+field)*S+surface];
}
}

template<int Steps> __global__ __launch_bounds__(THREADS,MOVING_MIN_CTAS) void RegisterCachedMoving(
 const int*tiles,Geometry g,const float*initial,float*output,float*trail,int*bad){
 using namespace moving_register_cache;
 extern __shared__ float storage[];
 float*frame=storage;size_t*bases=reinterpret_cast<size_t*>(storage+FrameComponents*PAD);volatile int*phase_step=reinterpret_cast<volatile int*>(bases+8);
 const int tile=tiles[blockIdx.x],bz=tile/(g.nx*g.ny),by=(tile/g.nx)%g.ny,bx=tile%g.nx;
 const int oz=bz*BZ,oy=by*BY,ox=bx*BX;
#if MOVING_TRAIL_RING
 if(!threadIdx.x)g.owner_tags[TrailOwner(g,tile,bz+by+bx)]=tile;
 __syncthreads();
#endif
 if(threadIdx.x<8){const int bits=threadIdx.x,dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;size_t base=static_cast<size_t>(-1);
  if(bz>=dz&&by>=dy&&bx>=dx){const int producer=tile-dz*g.nx*g.ny-dy*g.nx-dx;const int owner=TrailOwner(g,producer,bz+by+bx-dz-dy-dx);
#if MOVING_TRAIL_RING
   if(bits&&g.owner_tags[owner]!=producer)atomicAdd(bad,1);else
#endif
   base=static_cast<size_t>(owner)*Steps*6*S;
  }bases[bits]=base;
 }
 const int sx=threadIdx.x&(CX-1),sy=(threadIdx.x/CX)&(CY-1);
 float h[3][Slots],e[3][Slots];
 #pragma unroll
 for(int j=0;j<Slots;++j){const int zslot=threadIdx.x/(CX*CY)+j*(THREADS/(CX*CY));const int z=zslot==CZ-1?-1:zslot,y=sy==CY-1?-1:sy,x=sx==CX-1?-1:sx;
  #pragma unroll
  for(int c=0;c<3;++c){h[c][j]=GlobalAt(initial,c,oz+z,oy+y,ox+x,g.n);e[c][j]=GlobalAt(initial,c+3,oz+z,oy+y,ox+x,g.n);if constexpr(MOVING_REGISTER_WARP_NEIGHBORS){if(c!=1)frame[(c/2)*PAD+j*THREADS+threadIdx.x]=e[c][j];}else frame[c*PAD+j*THREADS+threadIdx.x]=e[c][j];}
 }
 if constexpr(MOVING_REGISTER_PHASE_METADATA){if(!threadIdx.x)*phase_step=1;}
 __syncthreads();
 #pragma unroll 1
 for(int n=1;n<=Steps;++n){
  const int hz=(-n-1)&(CZ-1),hy=(-n-1)&(CY-1),hx=(-n-1)&(CX-1),low=-n-1;
  const int dy=(sy-hy)&(CY-1),dx=(sx-hx)&(CX-1),py=oy+low+dy,px=ox+low+dx;
  { const int hphase_n=MOVING_REGISTER_PHASE_METADATA?*phase_step:n;
   const int hphase_hz=(-hphase_n-1)&(CZ-1),hphase_hy=(-hphase_n-1)&(CY-1),hphase_hx=(-hphase_n-1)&(CX-1),hphase_low=-hphase_n-1;
   const int hphase_dy=(sy-hphase_hy)&(CY-1),hphase_dx=(sx-hphase_hx)&(CX-1),hphase_py=oy+hphase_low+hphase_dy,hphase_px=ox+hphase_low+hphase_dx;
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zslot=threadIdx.x/(CX*CY)+j*(THREADS/(CX*CY));const bool owned=zslot!=hphase_hz&&sy!=hphase_hy&&sx!=hphase_hx;const int hphase_pz=oz+hphase_low+((zslot-hphase_hz)&(CZ-1));
#if MOVING_REGISTER_WARP_NEIGHBORS
   // Every lane reaches both shuffles, including physically invalid/halo lanes.
   // CacheXY==THREADS fixes j as Z and makes each warp a complete X row.
   const float x_y=__shfl_sync(0xffffffffu,e[1][j],(threadIdx.x+1)&31);
   const float x_z=__shfl_sync(0xffffffffu,e[2][j],(threadIdx.x+1)&31);
#endif
   if(owned){
    auto read=[&] __device__(int field,int,int dz,int ddy,int ddx){
     if(field<3)return h[field][j];
#if MOVING_REGISTER_WARP_NEIGHBORS
     const int component=field-3;
     if(!dz&&!ddy&&!ddx)return e[component][j];
     if(dz)return e[component][(j+1)&15];
     if(ddx)return component==1?x_y:x_z;
     // Only Ex/Ez have Y derivatives in the canonical Yee graph.
     return frame[(component/2)*PAD+Slot(zslot,sy+ddy,sx)];
#else
#if MOVING_REGISTER_LOCAL_CENTERS
     if(!dz&&!ddy&&!ddx)return e[field-3][j];
#endif
     return frame[(field-3)*PAD+Slot(zslot+dz,sy+ddy,sx+ddx)];
#endif
    };
    #pragma unroll
    for(int c=0;c<3;++c)h[c][j]=Valid(c,hphase_pz,hphase_py,hphase_px,g.n)?Generated<0>(c,read):0.f;
   }
  }
  }
  // Imports modify only evicted low-slot registers. Shared E remains old until
  // every H stencil has completed, including reads of its old high boundary.
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zslot=threadIdx.x/(CX*CY)+j*(THREADS/(CX*CY));if(zslot==hz||sy==hy||sx==hx){const int dz=(zslot-hz)&(CZ-1);
   #pragma unroll
   for(int c=0;c<3;++c){h[c][j]=Incoming<Steps>(initial,trail,bases,g,n-MOVING_REGISTER_BAD_IMPORT_PHASE,c,oz,oy,ox,dz,dy,dx,bad);e[c][j]=Incoming<Steps>(initial,trail,bases,g,n-MOVING_REGISTER_BAD_IMPORT_PHASE,c+3,oz,oy,ox,dz,dy,dx,bad);}
  }}
  __syncthreads();
  #pragma unroll
  for(int j=0;j<Slots;++j){
   #pragma unroll
   for(int c=0;c<3;++c)if constexpr(MOVING_REGISTER_WARP_NEIGHBORS){if(c!=1)frame[(c/2)*PAD+j*THREADS+threadIdx.x]=h[c][j];}else frame[c*PAD+j*THREADS+threadIdx.x]=h[c][j];
  }
  __syncthreads();
  { const int ephase_n=MOVING_REGISTER_PHASE_METADATA?*phase_step:n;
   const int ephase_hz=(-ephase_n-1)&(CZ-1),ephase_hy=(-ephase_n-1)&(CY-1),ephase_hx=(-ephase_n-1)&(CX-1),ephase_low=-ephase_n-1;
   const int ephase_dy=(sy-ephase_hy)&(CY-1),ephase_dx=(sx-ephase_hx)&(CX-1),ephase_py=oy+ephase_low+ephase_dy,ephase_px=ox+ephase_low+ephase_dx;
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zslot=threadIdx.x/(CX*CY)+j*(THREADS/(CX*CY));const bool owned=zslot!=ephase_hz&&sy!=ephase_hy&&sx!=ephase_hx;const int ephase_pz=oz+ephase_low+((zslot-ephase_hz)&(CZ-1));
#if MOVING_REGISTER_WARP_NEIGHBORS
   const float x_y=__shfl_sync(0xffffffffu,h[1][j],(threadIdx.x-1)&31);
   const float x_z=__shfl_sync(0xffffffffu,h[2][j],(threadIdx.x-1)&31);
#endif
   if(owned){
    auto read=[&] __device__(int field,int,int dz,int ddy,int ddx){
     if(field>=3)return e[field-3][j];
#if MOVING_REGISTER_WARP_NEIGHBORS
     if(!dz&&!ddy&&!ddx)return h[field][j];
     if(dz)return h[field][(j-1)&15];
     if(ddx)return field==1?x_y:x_z;
     return frame[(field/2)*PAD+Slot(zslot,sy+ddy,sx)];
#else
#if MOVING_REGISTER_LOCAL_CENTERS
     if(!dz&&!ddy&&!ddx)return h[field][j];
#endif
     return frame[field*PAD+Slot(zslot+dz,sy+ddy,sx+ddx)];
#endif
    };
    #pragma unroll
    for(int c=0;c<3;++c)e[c][j]=Valid(c+3,ephase_pz,ephase_py,ephase_px,g.n)?Generated<1>(c,read):0.f;
   }
  }
  }
  __syncthreads();
  #pragma unroll
  for(int j=0;j<Slots;++j){
   #pragma unroll
   for(int c=0;c<3;++c)if constexpr(MOVING_REGISTER_WARP_NEIGHBORS){if(c!=1)frame[(c/2)*PAD+j*THREADS+threadIdx.x]=e[c][j];}else frame[c*PAD+j*THREADS+threadIdx.x]=e[c][j];
  }
  if constexpr(MOVING_REGISTER_PHASE_METADATA){if(!threadIdx.x)*phase_step=n+1;}
  __syncthreads();
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zslot=threadIdx.x/(CX*CY)+j*(THREADS/(CX*CY));const bool owned=zslot!=hz&&sy!=hy&&sx!=hx;
   if(owned){const int dz=(zslot-hz)&(CZ-1),pz=oz+low+dz,surface=Surface(dz-1,dy-1,dx-1);
    #pragma unroll
    for(int c=0;c<3;++c){if(surface>=0){trail[bases[0]+((n-1)*6+c)*S+surface]=h[c][j];trail[bases[0]+((n-1)*6+c+3)*S+surface]=e[c][j];}
     if(n==Steps){if(Valid(c,pz,py,px,g.n))output[Offset(c,pz,py,px,g.n)]=h[c][j];if(Valid(c+3,pz,py,px,g.n))output[Offset(c+3,pz,py,px,g.n)]=e[c][j];}
    }
   }
  }
  // Publication operands are thread-private registers, so no following shared
  // overwrite can race a pending publication reader.
 }
}
