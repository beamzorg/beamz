// Real callback consumer of the traced, streamed-u_z schedule. Field-only.
#pragma once
#include "cuda_streamed_uz_generated.cuh"
namespace moving_streamed_uz {
constexpr int Depth=beamz::cuda::streamed_ir::Depth;
constexpr int SharedBytes=beamz::cuda::streamed_ir::SharedBytes+8*sizeof(size_t);
static_assert(BZ==64&&BY==13&&BX==29&&THREADS==512&&Depth==4,
              "Current generated consumer requires owned64x13x29 andT4");
template<bool Interior> struct Environment {
 const float*initial;const float*trail;float*output;const size_t*bases;int*bad;
 int n,oz,oy,ox;
 long long origin_linear;
 float inv_spacing=1.f;
 __device__ __forceinline__ bool Valid(int field,int z,int y,int x)const{if constexpr(Interior)return true;else return ::Valid(field,z,y,x,n);}
 __device__ __forceinline__ float Coefficient(int kind,int,int,int,int)const{return kind==0?0.99f:0.11f;}
 __device__ __forceinline__ float Finish(int,int,int,int,int,float value)const{return value;}
 __device__ __forceinline__ float Read(int field,int version,int z,int y,int x)const{
  if(!Valid(field,z,y,x))return 0.f;
  if(!version)return initial[Offset(field,z,y,x,n)];
  // The compiler manifest admits only [origin-1,origin+B-1] sources.
  int rz=z+version-oz,ry=y+version-oy,rx=x+version-ox;
  if(rz< -1||ry< -1||rx< -1||rz>=BZ||ry>=BY||rx>=BX){atomicAdd(bad,1);return __int_as_float(0x7fffffff);}
  const int neighbor=(rz<0)|((ry<0)<<1)|((rx<0)<<2);
  rz+=rz<0?BZ:0;ry+=ry<0?BY:0;rx+=rx<0?BX:0;
  const int surface=Surface(rz,ry,rx);const size_t base=bases[neighbor];
  if(!neighbor||surface<0||base==static_cast<size_t>(-1)){atomicAdd(bad,1);return __int_as_float(0x7fffffff);}
  return trail[base+((version-1)*6+field)*S+surface];
 }
 __device__ __forceinline__ void ReadPacket(unsigned requested,int family,int version,int z,int y,int x,float&cx,float&cy,float&cz)const{
  const bool bounded=z>=0&&y>=0&&x>=0&&z<=n&&y<=n&&x<=n;
  const bool iz=z<n,iy=y<n,ix=x<n;
  const unsigned valid=!bounded?0u:family?((ix?1u:0u)|(iy?2u:0u)|(iz?4u:0u)):
   ((iz&&iy?1u:0u)|(iz&&ix?2u:0u)|(iy&&ix?4u:0u));
  const unsigned active=requested&valid;
  size_t point=0;bool usable=true;
  if(active){
   if(!version){const int side=n+1;point=(z*side+y)*side+x;}
   else{int rz=z+version-oz,ry=y+version-oy,rx=x+version-ox;
    if(rz< -1||ry< -1||rx< -1||rz>=BZ||ry>=BY||rx>=BX)usable=false;
    const int neighbor=(rz<0)|((ry<0)<<1)|((rx<0)<<2);
    rz+=rz<0?BZ:0;ry+=ry<0?BY:0;rx+=rx<0?BX:0;
    const int surface=Surface(rz,ry,rx);const size_t base=bases[neighbor];
    usable=usable&&neighbor&&surface>=0&&base!=static_cast<size_t>(-1);
    if(usable)point=base+(version-1)*6*S+surface;
    else atomicAdd(bad,1);
   }
  }
  #pragma unroll
  for(int c=0;c<3;++c)if(requested&(1u<<c)){
   float value=0.f;if(active&(1u<<c)){
    if(!usable)value=__int_as_float(0x7fffffff);
    else if(version)value=trail[point+(family*3+c)*S];
    else {const int side=n+1;value=initial[point+static_cast<size_t>(family*3+c)*side*side*side];}
   }
   if(c==0)cx=value;else if(c==1)cy=value;else cz=value;
  }
 }
 // Relative routes are admitted by the generated one-lower-cell manifest.
 // No division or repeated physical-to-u reconstruction is needed here.
 __device__ __forceinline__ long long Linear(int version,int rz,int ry,int rx)const{
  const int side=n+1;return origin_linear+static_cast<long long>(rz-version)*side*side+static_cast<long long>(ry-version)*side+rx-version;
 }
 __device__ __forceinline__ float ReadRelative(int field,int version,int rz,int ry,int rx)const{
  if constexpr(!Interior){if(!::Valid(field,oz+rz-version,oy+ry-version,ox+rx-version,n))return 0.f;}
  if(!version){const int side=n+1;return initial[Linear(0,rz,ry,rx)+static_cast<long long>(field)*side*side*side];}
  const int neighbor=(rz<0)|((ry<0)<<1)|((rx<0)<<2);
  rz+=rz<0?BZ:0;ry+=ry<0?BY:0;rx+=rx<0?BX:0;
  return trail[bases[neighbor]+((version-1)*6+field)*S+Surface(rz,ry,rx)];
 }
 __device__ __forceinline__ void ReadPacketRelative(unsigned requested,int family,int version,int rz,int ry,int rx,float&cx,float&cy,float&cz)const{
  unsigned valid=7;
  if constexpr(!Interior){const int z=oz+rz-version,y=oy+ry-version,x=ox+rx-version;
   const bool bounded=z>=0&&y>=0&&x>=0&&z<=n&&y<=n&&x<=n;const bool iz=z<n,iy=y<n,ix=x<n;
   valid=!bounded?0u:family?((ix?1u:0u)|(iy?2u:0u)|(iz?4u:0u)):((iz&&iy?1u:0u)|(iz&&ix?2u:0u)|(iy&&ix?4u:0u));
  }
  const unsigned active=requested&valid;long long point=0;
  if(active){if(!version)point=Linear(0,rz,ry,rx);else{const int neighbor=(rz<0)|((ry<0)<<1)|((rx<0)<<2);rz+=rz<0?BZ:0;ry+=ry<0?BY:0;rx+=rx<0?BX:0;point=bases[neighbor]+(version-1)*6*S+Surface(rz,ry,rx);}}
  #pragma unroll
  for(int c=0;c<3;++c)if(requested&(1u<<c)){float value=0.f;if(active&(1u<<c)){if(version)value=trail[point+(family*3+c)*S];else{const int side=n+1;value=initial[point+static_cast<long long>(family*3+c)*side*side*side];}}if(c==0)cx=value;else if(c==1)cy=value;else cz=value;}
 }
 __device__ __forceinline__ void PublishPacketRelative(int version,int rz,int ry,int rx,float hx,float hy,float hz,float ex,float ey,float ez)const{
  const int surface=Surface(rz,ry,rx);if(surface<0&&version!=Depth)return;
  const float values[6]={hx,hy,hz,ex,ey,ez};
  if(surface>=0){const size_t base=bases[0]+(version-1)*6*S+surface;
   #pragma unroll
   for(int field=0;field<6;++field)const_cast<float*>(trail)[base+field*S]=values[field];
  }
  if(version==Depth){const long long point=Linear(version,rz,ry,rx);const int side=n+1;
   #pragma unroll
   for(int field=0;field<6;++field){if constexpr(Interior)output[point+static_cast<long long>(field)*side*side*side]=values[field];else if(::Valid(field,oz+rz-version,oy+ry-version,ox+rx-version,n))output[point+static_cast<long long>(field)*side*side*side]=values[field];}
  }
 }
 __device__ __forceinline__ void Publish(int field,int version,int z,int y,int x,float value)const{
  const int rz=z+version-oz,ry=y+version-oy,rx=x+version-ox;
  if(rz<0||ry<0||rx<0||rz>=BZ||ry>=BY||rx>=BX){atomicAdd(bad,1);return;}
  const int surface=Surface(rz,ry,rx);
  if(surface>=0)const_cast<float*>(trail)[bases[0]+((version-1)*6+field)*S+surface]=value;
  if(version==Depth&&Valid(field,z,y,x))output[Offset(field,z,y,x,n)]=value;
 }
};
}
template<int Steps,bool Interior=false> __global__ __launch_bounds__(THREADS,MOVING_MIN_CTAS)
void StreamedUzMoving(const int*tiles,Geometry g,const float*__restrict__ initial,
 float*__restrict__ output,float*__restrict__ trail,int*__restrict__ bad){
 static_assert(Steps==moving_streamed_uz::Depth);
 __shared__ float exchange[beamz::cuda::streamed_ir::SharedBytes/sizeof(float)];
 __shared__ size_t incoming[8];
 const int tile=tiles[blockIdx.x],bz=tile/(g.nx*g.ny),by=(tile/g.nx)%g.ny,bx=tile%g.nx;
 const int oz=bz*BZ,oy=by*BY,ox=bx*BX;
#if MOVING_TRAIL_RING
 if(!threadIdx.x)g.owner_tags[TrailOwner(g,tile,bz+by+bx)]=tile;
 __syncthreads();
#endif
 if(threadIdx.x<8){const int bits=threadIdx.x,dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;size_t base=static_cast<size_t>(-1);
  if(bz>=dz&&by>=dy&&bx>=dx){const int producer=tile-dz*g.nx*g.ny-dy*g.nx-dx;const int slot=TrailOwner(g,producer,bz+by+bx-dz-dy-dx);
#if MOVING_TRAIL_RING
   if(bits&&g.owner_tags[slot]!=producer)atomicAdd(bad,1);else
#endif
   base=static_cast<size_t>(slot)*Steps*6*S;
  }incoming[bits]=base;
 }
 __syncthreads();
 const int side=g.n+1;const long long origin_linear=(static_cast<long long>(oz)*side+oy)*side+ox;
 moving_streamed_uz::Environment<Interior> env{initial,trail,output,incoming,bad,g.n,oz,oy,ox,origin_linear,1.f};
 beamz::cuda::streamed_ir::RunStreamedTile(env,exchange,oz,oy,ox);
}
