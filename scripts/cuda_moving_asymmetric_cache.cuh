// Experimental stable asymmetric cache. Shared Hx/Hz/Ex/Ez; private Hy/Ey.
// Every warp is a complete physical X row, every unrolled j is one Z plane.
#pragma once
#ifndef MOVING_ASYMMETRIC_HIGH_X
#define MOVING_ASYMMETRIC_HIGH_X 0
#endif
#ifndef MOVING_ASYMMETRIC_LOW_X
#define MOVING_ASYMMETRIC_LOW_X 0
#endif
namespace moving_asymmetric_cache {
constexpr int CZ=6,CY=16,CX=32,PAD=CZ*CY*CX,Slots=CZ;
constexpr size_t SharedBytes=4ULL*PAD*sizeof(float)+8*sizeof(size_t)+sizeof(int);
static_assert(BZ==5&&BY==15&&BX==31&&THREADS==CX*CY,
              "Asymmetric cache requires owned5x15x31 and512threads");
__device__ __forceinline__ int WrapZ(int x){const int r=x%CZ;return r<0?r+CZ:r;}
__device__ __forceinline__ int Slot(int z,int y,int x){return (WrapZ(z)*CY+(y&15))*CX+(x&31);}
__device__ __forceinline__ float &At(float*shared,int field,int z,int y,int x){
 const int component=field%3;
 const int family=field/3;
 return shared[(2*family+component/2)*PAD+Slot(z,y,x)];
}
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

#if MOVING_ASYMMETRIC_LOW_X
#include "cuda_moving_asymmetric_low_x.cuh"
#endif

template<int Steps> __global__ __launch_bounds__(THREADS,MOVING_MIN_CTAS)
void AsymmetricCachedMoving(const int*tiles,Geometry g,const float*initial,
                            float*output,float*trail,int*bad){
 using namespace moving_asymmetric_cache;
 MOVING_PROFILE_DECLARE;MOVING_PROFILE_BEGIN();
 extern __shared__ float shared[];
 size_t*bases=reinterpret_cast<size_t*>(shared+4*PAD);
 volatile int*phase_step=reinterpret_cast<volatile int*>(bases+8);
#if MOVING_ASYMMETRIC_LOW_X
 float*low_x_stage=shared+moving_asymmetric_cache::SharedBytes/sizeof(float);
#endif
 const int tile=tiles[blockIdx.x],bz=tile/(g.nx*g.ny),by=(tile/g.nx)%g.ny,bx=tile%g.nx;
 const int oz=bz*BZ,oy=by*BY,ox=bx*BX;
#if MOVING_TRAIL_RING
 if(!threadIdx.x)g.owner_tags[TrailOwner(g,tile,bz+by+bx)]=tile;
 __syncthreads();
#endif
 if(threadIdx.x<8){const int bits=threadIdx.x,dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
  size_t base=static_cast<size_t>(-1);
  if(bz>=dz&&by>=dy&&bx>=dx){const int producer=tile-dz*g.nx*g.ny-dy*g.nx-dx;
   const int owner=TrailOwner(g,producer,bz+by+bx-dz-dy-dx);
#if MOVING_TRAIL_RING
   if(bits&&g.owner_tags[owner]!=producer)atomicAdd(bad,1);else
#endif
   base=static_cast<size_t>(owner)*Steps*6*S;
  }bases[bits]=base;
 }
 const int sx=threadIdx.x&31,sy=threadIdx.x/32;
 float hy[Slots],ey[Slots];
 #pragma unroll
 for(int j=0;j<Slots;++j){const int z=j==CZ-1?-1:j,y=sy==15?-1:sy,x=sx==31?-1:sx;
  hy[j]=GlobalAt(initial,1,oz+z,oy+y,ox+x,g.n);
  ey[j]=GlobalAt(initial,4,oz+z,oy+y,ox+x,g.n);
  #pragma unroll
  for(int family=0;family<2;++family){At(shared,family*3,j,sy,sx)=GlobalAt(initial,family*3,oz+z,oy+y,ox+x,g.n);At(shared,family*3+2,j,sy,sx)=GlobalAt(initial,family*3+2,oz+z,oy+y,ox+x,g.n);}
 }
 if(!threadIdx.x)*phase_step=1;
 __syncthreads();
 MOVING_PROFILE_MARK(Init);
 #pragma unroll 1
 for(int n=1;n<=Steps;++n){
  MOVING_PROFILE_MARK(Loop);
  // Independent volatile phase reads bound address liveness across families.
  {const int phase=*phase_step,head=WrapZ(-phase-1),yh=(-phase-1)&15,xh=(-phase-1)&31,low=-phase-1;
   const int py=oy+low+((sy-yh)&15),px=ox+low+((sx-xh)&31);
   MOVING_PROFILE_MARK(HSetup);
   #pragma unroll
   for(int j=0;j<Slots;++j){
    // Full-mask shuffle precedes every owned/physical predicate.
    const float next_ey=__shfl_sync(0xffffffffu,ey[j],(sx+1)&31);
    if(j!=head&&sy!=yh&&sx!=xh){const int pz=oz+low+WrapZ(j-head);
     const auto read=[&] __device__(int field,int,int dz,int dy,int dx){
      if(field<3)return field==1?hy[j]:At(shared,field,j,sy,sx);
      if(field==4){if(dz)return ey[(j+1)%Slots];if(dx)return next_ey;return ey[j];}
      return At(shared,field,j+dz,sy+dy,sx+dx);
     };
     #pragma unroll
     for(int c=0;c<3;++c){const float value=Valid(c,pz,py,px,g.n)?Generated<0>(c,read):0.f;if(c==1)hy[j]=value;else At(shared,c,j,sy,sx)=value;}
    }
   }
  }
  MOVING_PROFILE_MARK(HCompute);
  // B1: old E readers and the preceding version's high-face publishers retire
  // before current low imports recycle those physical cache slots.
  __syncthreads();
  MOVING_PROFILE_MARK(HWait);
  {const int head=WrapZ(-n-1),yh=(-n-1)&15,xh=(-n-1)&31,low=-n-1;
   const int dy=(sy-yh)&15,dx=(sx-xh)&31;
   #pragma unroll
   for(int j=0;j<Slots;++j)if(j==head||sy==yh||(!MOVING_ASYMMETRIC_LOW_X&&sx==xh)){const int dz=WrapZ(j-head);
    #pragma unroll
    for(int c=0;c<3;++c){const float h=Incoming<Steps>(initial,trail,bases,g,n,c,oz,oy,ox,dz,dy,dx,bad);const float e=Incoming<Steps>(initial,trail,bases,g,n,c+3,oz,oy,ox,dz,dy,dx,bad);if(c==1){hy[j]=h;ey[j]=e;}else{At(shared,c,j,sy,sx)=h;At(shared,c+3,j,sy,sx)=e;}}
   }
  }
#if MOVING_ASYMMETRIC_LOW_X
  moving_asymmetric_low_x::Stage(shared,low_x_stage,trail,bases,g,n,oz,oy,ox,bad);
#endif
  // B1 retired H metadata reads. Publish the next-H counter at B2; E
  // decodes its current version as counter-1. This avoids an extra barrier.
  if(!threadIdx.x)*phase_step=n+1;
  MOVING_PROFILE_MARK(Import);
  __syncthreads(); // B2: all H owned/current halo values precede E stencils.
  MOVING_PROFILE_MARK(ImportWait);
#if MOVING_ASYMMETRIC_LOW_X
  moving_asymmetric_low_x::Retire(low_x_stage,hy,ey,n,sy,sx);
#if MOVING_ASYMMETRIC_HIGH_X
  // All lanes participate: retire BOTH shared Hy/Ey stage reads, including
  // otherwise-lazy Ey loads, before same-warp high-X writers reuse their key.
  __syncwarp(0xffffffffu);
#endif
#endif
  {const int phase=*phase_step-1,head=WrapZ(-phase-1),yh=(-phase-1)&15,xh=(-phase-1)&31,low=-phase-1;
   const int py=oy+low+((sy-yh)&15),px=ox+low+((sx-xh)&31);
   MOVING_PROFILE_MARK(ESetup);
   #pragma unroll
   for(int j=0;j<Slots;++j){
    const float previous_hy=__shfl_sync(0xffffffffu,hy[j],(sx-1)&31);
    if(j!=head&&sy!=yh&&sx!=xh){const int pz=oz+low+WrapZ(j-head);
     const auto read=[&] __device__(int field,int,int dz,int dy,int dx){
      if(field>=3)return field==4?ey[j]:At(shared,field,j,sy,sx);
      if(field==1){if(dz)return hy[(j+Slots-1)%Slots];if(dx)return previous_hy;return hy[j];}
      return At(shared,field,j+dz,sy+dy,sx+dx);
     };
     #pragma unroll
     for(int c=0;c<3;++c){const float value=Valid(c+3,pz,py,px,g.n)?Generated<1>(c,read):0.f;if(c==1)ey[j]=value;else At(shared,c+3,j,sy,sx)=value;}
    }
   }
  }
#if MOVING_ASYMMETRIC_HIGH_X
  moving_asymmetric_low_x::StageHigh(low_x_stage,hy,ey,n,sy,sx);
#endif
  MOVING_PROFILE_MARK(ECompute);
  __syncthreads(); // B3: completed E values precede publication / next H.
  MOVING_PROFILE_MARK(EWait);
  {const int head=WrapZ(-n-1),yh=(-n-1)&15,xh=(-n-1)&31,low=-n-1;
   const int dy=(sy-yh)&15,dx=(sx-xh)&31,py=oy+low+dy,px=ox+low+dx;
   #pragma unroll
   for(int j=0;j<Slots;++j)if(j!=head&&sy!=yh&&sx!=xh){const int dz=WrapZ(j-head),pz=oz+low+dz,surface=Surface(dz-1,dy-1,dx-1);
    if(MOVING_ASYMMETRIC_HIGH_X&&dx==BX&&dy<BY&&dz<BZ)continue;
    // Loads themselves are gated. Next H excludes this high-face union; its
    // B1 retires every publication before low-halo recycling.
    if(surface>=0||n==Steps){
     #pragma unroll
     for(int c=0;c<3;++c){const float h=c==1?hy[j]:At(shared,c,j,sy,sx),e=c==1?ey[j]:At(shared,c+3,j,sy,sx);
      if(surface>=0){trail[bases[0]+((n-1)*6+c)*S+surface]=h;trail[bases[0]+((n-1)*6+c+3)*S+surface]=e;}
      if(n==Steps){if(Valid(c,pz,py,px,g.n))output[Offset(c,pz,py,px,g.n)]=h;if(Valid(c+3,pz,py,px,g.n))output[Offset(c+3,pz,py,px,g.n)]=e;}
     }
    }
   }
  }
#if MOVING_ASYMMETRIC_HIGH_X
  moving_asymmetric_low_x::PublishHigh<Steps>(shared,low_x_stage,trail,output,bases,g,n,oz,oy,ox);
#endif
  MOVING_PROFILE_MARK(Publish);
 }
 MOVING_PROFILE_MARK(Loop);MOVING_PROFILE_END(g,tile,bz+by+bx,Steps);
}
