// Research mechanism only: moving-u boxes, real Yee extents, no CPML/source/DFT.
#include <cuda_runtime.h>
#include "yee_cell_generated.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MOVING_BZ
#define MOVING_BZ 4
#endif
#ifndef MOVING_BY
#define MOVING_BY 8
#endif
#ifndef MOVING_BX
#define MOVING_BX 32
#endif
#ifndef MOVING_STREAMED_INTERIOR
#define MOVING_STREAMED_INTERIOR 0
#endif
#ifndef MOVING_STREAMED_UZ
#define MOVING_STREAMED_UZ 0
#endif
#ifndef MOVING_ASYMMETRIC_PIPELINE
#define MOVING_ASYMMETRIC_PIPELINE 0
#endif
#ifndef MOVING_ASYMMETRIC_CACHE
#define MOVING_ASYMMETRIC_CACHE 0
#endif
#ifndef MOVING_REGISTER_CACHE
#define MOVING_REGISTER_CACHE 0
#endif
#ifndef MOVING_REGISTER_LOCAL_CENTERS
#define MOVING_REGISTER_LOCAL_CENTERS 0
#endif
#ifndef MOVING_REGISTER_THREADS
#define MOVING_REGISTER_THREADS 0
#endif
constexpr int BZ=MOVING_BZ, BY=MOVING_BY, BX=MOVING_BX, V=BZ*BY*BX;
static_assert(BZ>0 && BY>0 && BX>0 && V<=(MOVING_STREAMED_UZ?(1<<20):(MOVING_REGISTER_CACHE||MOVING_ASYMMETRIC_CACHE)?8192:1024), "Owned tile exceeds selected cache support");
constexpr int S=V-(BZ-1)*(BY-1)*(BX-1), GUARD=16;
#ifndef MOVING_SPLIT_COMPONENTS
#define MOVING_SPLIT_COMPONENTS 0
#endif
constexpr int OWNED_COMPONENTS=MOVING_SPLIT_COMPONENTS?1:3;
constexpr int THREADS=MOVING_STREAMED_UZ?512:MOVING_ASYMMETRIC_CACHE?512:MOVING_REGISTER_CACHE?(MOVING_REGISTER_THREADS?MOVING_REGISTER_THREADS:(BY+1)*(BX+1)):(MOVING_SPLIT_COMPONENTS?768:256);
#ifndef MOVING_MIN_CTAS
#define MOVING_MIN_CTAS 1
#endif
#ifndef MOVING_RELATIVE_ADDRESSING
#define MOVING_RELATIVE_ADDRESSING 1
#endif
#ifndef MOVING_TRAIL_RING
#define MOVING_TRAIL_RING 0
#endif
#ifndef MOVING_RING_HOIST
#define MOVING_RING_HOIST 0
#endif
static_assert(!MOVING_RING_HOIST || (MOVING_RELATIVE_ADDRESSING && MOVING_TRAIL_RING),
              "Hoisted trail bases require relative addressing and a ring");
#ifndef MOVING_UNSAFE_TWO_EPOCH_NEGATIVE_CONTROL
#define MOVING_UNSAFE_TWO_EPOCH_NEGATIVE_CONTROL 0
#endif
static_assert(MOVING_TRAIL_RING==0 ||
    (MOVING_TRAIL_RING>=4 && (MOVING_TRAIL_RING&(MOVING_TRAIL_RING-1))==0) ||
    (MOVING_TRAIL_RING==2 && MOVING_UNSAFE_TWO_EPOCH_NEGATIVE_CONTROL),
    "Trail epochs must be0 or a power-of-two>=4; two epochs require the explicit unsafe negative-control macro");
#define CU(call) do { const auto status=(call);if(status!=cudaSuccess)throw std::runtime_error(std::string(#call)+": "+cudaGetErrorString(status)); } while(0)


#include "cuda_moving_phase_profile.cuh"
struct Geometry { int n,nx,ny,nz,cube,max_frontier=0;const int *slots=nullptr;int *owner_tags=nullptr;
#if MOVING_PHASE_PROFILE
unsigned long long*phase_profile=nullptr;
#endif
};
__device__ __forceinline__ int TrailOwner(Geometry g,int tile,int rank) {
#if MOVING_TRAIL_RING
  return (rank&(MOVING_TRAIL_RING-1))*g.max_frontier+__ldg(g.slots+tile);
#else
  return tile;
#endif
}
__device__ __forceinline__ float ReadTrail(Geometry g,const float *trail,int steps,int tile,int rank,
    int version,int c,int surface,int *bad) {
  const int owner=TrailOwner(g,tile,rank);
#if MOVING_TRAIL_RING
  if(g.owner_tags[owner]!=tile){atomicAdd(bad,1);return __int_as_float(0x7fffffff);}
#endif
  return trail[((static_cast<size_t>(owner)*steps+version-1)*6+c)*S+surface];
}
__host__ __device__ __forceinline__ bool Valid(int c,int z,int y,int x,int n) {
  const int dz=(c==2||c==3||c==4),dy=(c==1||c==3||c==5),dx=(c==0||c==4||c==5);
  return z>=0&&y>=0&&x>=0&&z<n+dz&&y<n+dy&&x<n+dx;
}
__host__ __device__ __forceinline__ int Offset(int c,int z,int y,int x,int n) {
  const int side=n+1;return c*side*side*side+(z*side+y)*side+x;
}
__device__ __forceinline__ float GlobalAt(const float *p,int c,int z,int y,int x,int n) {
  return Valid(c,z,y,x,n)?p[Offset(c,z,y,x,n)]:0.f;
}
__host__ __device__ __forceinline__ int Surface(int z,int y,int x) {
  if(z==BZ-1)return y*BX+x;
  if(y==BY-1)return BY*BX+z*BX+x;
  if(x==BX-1)return BY*BX+(BZ-1)*BX+z*(BY-1)+y;
  return -1;
}
__device__ __forceinline__ float State(const float *initial,const float *trail,const float *shared,
    Geometry g,int steps,int owner,int version,int c,int uz,int uy,int ux,int *bad) {
  if(!Valid(c,uz-version,uy-version,ux-version,g.n))return 0.f;
  if(version==0)return GlobalAt(initial,c,uz,uy,ux,g.n);
  const int bz=uz/BZ,by=uy/BY,bx=ux/BX;
  const int tile=(bz*g.ny+by)*g.nx+bx;
  const int lz=uz%BZ,ly=uy%BY,lx=ux%BX;
  if(tile==owner)return shared[c*V+(lz*BY+ly)*BX+lx];
  const int oz=owner/(g.nx*g.ny),oy=(owner/g.nx)%g.ny,ox=owner%g.nx;
  const int surface=Surface(lz,ly,lx);
  if(surface<0||bz+by+bx>=oz+oy+ox) { atomicAdd(bad,1);return __int_as_float(0x7fffffff); }
  return ReadTrail(g,trail,steps,tile,bz+by+bx,version,c,surface,bad);
}
__device__ __forceinline__ float Fetch(const float *initial,const float *trail,const float *shared,
    Geometry g,int steps,int owner,int version,int c,int uz,int uy,int ux,int dz,int dy,int dx,int *bad,const size_t *bases) {
#if MOVING_RELATIVE_ADDRESSING
  const int z=uz+dz,y=uy+dy,x=ux+dx;
  if(!Valid(c,z-version,y-version,x-version,g.n))return 0.f;
  if(version==0)return GlobalAt(initial,c,z,y,x,g.n);
  if(dz>0||dy>0||dx>0||dz< -1||dy< -1||dx< -1) {atomicAdd(bad,1);return __int_as_float(0x7fffffff);}
  const int sz=(uz&(BZ-1))+dz,sy=(uy&(BY-1))+dy,sx=(ux&(BX-1))+dx;
  const int tile=owner-(sz<0?g.nx*g.ny:0)-(sy<0?g.nx:0)-(sx<0?1:0);
  const int lz=sz&(BZ-1),ly=sy&(BY-1),lx=sx&(BX-1);
  if(tile==owner)return shared[c*V+(lz*BY+ly)*BX+lx];
  const int surface=Surface(lz,ly,lx);
  if(surface<0||tile<0) {atomicAdd(bad,1);return __int_as_float(0x7fffffff);}
#if MOVING_RING_HOIST
  const int neighbor=(sz<0)|((sy<0)<<1)|((sx<0)<<2);
  const size_t base=bases[neighbor];
  if(base==static_cast<size_t>(-1)){atomicAdd(bad,1);return __int_as_float(0x7fffffff);}
  return trail[base+((version-1)*6+c)*S+surface];
#else
  return ReadTrail(g,trail,steps,tile,(static_cast<unsigned>(z)/BZ)+(static_cast<unsigned>(y)/BY)+(static_cast<unsigned>(x)/BX),version,c,surface,bad);
#endif
#else
  return State(initial,trail,shared,g,steps,owner,version,c,uz+dz,uy+dy,ux+dx,bad);
#endif
}

template<int Phase,typename Reader>
__device__ __forceinline__ float Generated(int c,Reader read) {
  auto coeff=[] __device__ (int kind,int){return kind==0?0.99f:0.11f;};
  if(c==0)return beamz::cuda::generated::AdvanceYeeCell<Phase,0>(1.f,read,coeff);
  if(c==1)return beamz::cuda::generated::AdvanceYeeCell<Phase,1>(1.f,read,coeff);
  return beamz::cuda::generated::AdvanceYeeCell<Phase,2>(1.f,read,coeff);
}

template<int Steps> __global__ __launch_bounds__(THREADS,MOVING_MIN_CTAS) void Moving(
    const int *tiles,Geometry g,const float *initial,float *output,float *trail,int *bad) {
  __shared__ float state[6*V];
  const int tile=tiles[blockIdx.x],bz=tile/(g.nx*g.ny),by=(tile/g.nx)%g.ny,bx=tile%g.nx;
#if MOVING_TRAIL_RING
  if(threadIdx.x==0)g.owner_tags[TrailOwner(g,tile,bz+by+bx)]=tile;
  __syncthreads();
#endif
#if MOVING_RING_HOIST
  __shared__ size_t incoming[8];
  if(threadIdx.x<8){const int bits=threadIdx.x,dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
    size_t base=static_cast<size_t>(-1);
    if(bz>=dz&&by>=dy&&bx>=dx){const int producer=tile-dz*g.nx*g.ny-dy*g.nx-dx;
      const int slot=TrailOwner(g,producer,bz+by+bx-dz-dy-dx);
      if(!bits||g.owner_tags[slot]==producer)base=static_cast<size_t>(slot)*Steps*6*S;
      else atomicAdd(bad,1);}
    incoming[bits]=base;}
  __syncthreads();
#else
  const size_t *incoming=nullptr;
#endif
  const int lane=threadIdx.x%256,first=MOVING_SPLIT_COMPONENTS?threadIdx.x/256:0;
  float old_h[OWNED_COMPONENTS][4],old_e[OWNED_COMPONENTS][4];
  #pragma unroll 1
  for(int n=1;n<=Steps;++n) {
    // All old own-field values must be retired before any H slot is replaced.
    #pragma unroll
    for(int j=0;j<4;++j) {
      const int q=lane+j*256,lz=q/(BY*BX),ly=(q/BX)%BY,lx=q%BX;
      const int uz=bz*BZ+lz,uy=by*BY+ly,ux=bx*BX+lx;
      #pragma unroll
      for(int c=first;c<first+OWNED_COMPONENTS;++c) {
        old_h[c-first][j]=Fetch(initial,trail,state,g,Steps,tile,n-1,c,uz,uy,ux,-1,-1,-1,bad,incoming);
        old_e[c-first][j]=Fetch(initial,trail,state,g,Steps,tile,n-1,3+c,uz,uy,ux,-1,-1,-1,bad,incoming);
      }
    }
    __syncthreads();
    #pragma unroll
    for(int j=0;j<4;++j) {
      const int q=lane+j*256,lz=q/(BY*BX),ly=(q/BX)%BY,lx=q%BX;
      const int uz=bz*BZ+lz,uy=by*BY+ly,ux=bx*BX+lx;
      auto read=[&] __device__ (int c,int,int dz,int dy,int dx) {
        if(c<3)return old_h[c-first][j];
        return Fetch(initial,trail,state,g,Steps,tile,n-1,c,uz,uy,ux,-1+dz,-1+dy,-1+dx,bad,incoming);
      };
      #pragma unroll
      for(int c=first;c<first+OWNED_COMPONENTS;++c)
        state[c*V+q]=Valid(c,uz-n,uy-n,ux-n,g.n)?Generated<0>(c,read):0.f;
    }
    __syncthreads();
    #pragma unroll
    for(int j=0;j<4;++j) {
      const int q=lane+j*256,lz=q/(BY*BX),ly=(q/BX)%BY,lx=q%BX;
      const int uz=bz*BZ+lz,uy=by*BY+ly,ux=bx*BX+lx;
      auto read=[&] __device__ (int c,int,int dz,int dy,int dx) {
        if(c>=3)return old_e[c-3-first][j];
        return Fetch(initial,trail,state,g,Steps,tile,n,c,uz,uy,ux,dz,dy,dx,bad,incoming);
      };
      #pragma unroll
      for(int c=first;c<first+OWNED_COMPONENTS;++c)
        state[(3+c)*V+q]=Valid(3+c,uz-n,uy-n,ux-n,g.n)?Generated<1>(c,read):0.f;
    }
    __syncthreads();
    #pragma unroll
    for(int j=0;j<4;++j) {
      const int q=lane+j*256,lz=q/(BY*BX),ly=(q/BX)%BY,lx=q%BX;
      const int uz=bz*BZ+lz,uy=by*BY+ly,ux=bx*BX+lx,surface=Surface(lz,ly,lx);
      #pragma unroll
      for(int c=0;c<6;++c) {
        if(c%3<first||c%3>=first+OWNED_COMPONENTS)continue;
        if(surface>=0){
#if MOVING_RING_HOIST
          trail[incoming[0]+((n-1)*6+c)*S+surface]=state[c*V+q];
#else
          trail[((static_cast<size_t>(TrailOwner(g,tile,bz+by+bx))*Steps+n-1)*6+c)*S+surface]=state[c*V+q];
#endif
        }
        if(n==Steps&&Valid(c,uz-n,uy-n,ux-n,g.n))output[Offset(c,uz-n,uy-n,ux-n,g.n)]=state[c*V+q];
      }
    }
    // Next iteration's pre-overwrite barrier also protects these trail loads.
  }
}

#ifndef MOVING_FACE_CACHE
#define MOVING_FACE_CACHE 0
#endif
#ifndef MOVING_CIRCULAR_CACHE
#define MOVING_CIRCULAR_CACHE 0
#endif
static_assert((MOVING_FACE_CACHE+MOVING_CIRCULAR_CACHE+MOVING_REGISTER_CACHE+MOVING_ASYMMETRIC_CACHE+MOVING_STREAMED_UZ)<=1, "Cache candidates are mutually exclusive");
#if MOVING_FACE_CACHE || MOVING_CIRCULAR_CACHE
#include "cuda_moving_face_cache.cuh"
#endif
#if MOVING_CIRCULAR_CACHE
#include "cuda_moving_circular_cache.cuh"
#endif
#if MOVING_REGISTER_CACHE
#include "cuda_moving_register_cache.cuh"
#endif
#if MOVING_ASYMMETRIC_CACHE
#include "cuda_moving_asymmetric_cache.cuh"
#endif
#if MOVING_ASYMMETRIC_PIPELINE
#include "cuda_moving_asymmetric_pipeline.cuh"
#endif
#if MOVING_STREAMED_UZ
#include "cuda_moving_streamed_uz.cuh"
#endif

// Independent explicit equations: no generated-cell helper or transformed reads.
__global__ void Plain(const float *old,const float *h_current,float *out,int n,int phase) {
  const int side=n+1,cube=side*side*side;
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=3*cube)return;
  const int c=i/cube,q=i%cube,z=q/(side*side),y=(q/side)%side,x=q%side;
  const int field=phase*3+c;if(!Valid(field,z,y,x,n))return;
  const int components[3][2]={{2,1},{0,2},{1,0}};
  const int axes[3][2]={{1,0},{0,2},{2,1}};
  float d[2];
  #pragma unroll
  for(int t=0;t<2;++t) {
    const int oc=(1-phase)*3+components[c][t],axis=axes[c][t],delta=phase?-1:1;
    const float *other=phase?h_current:old;
    const float center=GlobalAt(other,oc,z,y,x,n);
    const float adjacent=GlobalAt(other,oc,z+(axis==0?delta:0),y+(axis==1?delta:0),x+(axis==2?delta:0),n);
    d[t]=__fmul_rn(phase?__fsub_rn(center,adjacent):__fsub_rn(adjacent,center),1.f);
  }
  out[Offset(field,z,y,x,n)]=__fmaf_rn(phase?0.11f:-0.11f,__fsub_rn(d[0],d[1]),__fmul_rn(0.99f,old[Offset(field,z,y,x,n)]));
}


#ifndef MOVING_TILED_CONTROL
#define MOVING_TILED_CONTROL 0
#endif
#ifndef MOVING_MATCHED_D2_CONTROL
#define MOVING_MATCHED_D2_CONTROL 0
#endif
#if MOVING_MATCHED_D2_CONTROL
#include "cuda_moving_d2_control.cuh"
#endif
#ifndef MOVING_THREE_BANKS
#define MOVING_THREE_BANKS 0
#endif

// Optional stronger spatial control. It uses shared opposite-family samples,
// but still launches H then E separately at every step, like Plain.
template<int Phase> __global__ void TiledPlain(
    const float *old,const float *h_current,float *out,int n) {
  constexpr int TX=32,TY=4,TZ=2,SX=TX+1,SY=TY+1,SZ=TZ+1,SV=SX*SY*SZ;
  __shared__ float opposite[3*SV];
  const int lane=(threadIdx.z*TY+threadIdx.y)*TX+threadIdx.x;
  const int bz=blockIdx.z*TZ,by=blockIdx.y*TY,bx=blockIdx.x*TX;
  for(int i=lane;i<3*SV;i+=TX*TY*TZ){
    const int c=i/SV,q=i%SV,z=q/(SX*SY),y=(q/SX)%SY,x=q%SX;
    opposite[i]=GlobalAt(Phase?h_current:old,(1-Phase)*3+c,
        bz+z-Phase,by+y-Phase,bx+x-Phase,n);
  }
  __syncthreads();
  const int z=bz+threadIdx.z,y=by+threadIdx.y,x=bx+threadIdx.x;
  const int center=((threadIdx.z+Phase)*SY+threadIdx.y+Phase)*SX+threadIdx.x+Phase;
  constexpr int components[3][2]={{2,1},{0,2},{1,0}};
  constexpr int axes[3][2]={{1,0},{0,2},{2,1}},strides[3]={SX*SY,SX,1};
  #pragma unroll
  for(int c=0;c<3;++c){
    const int field=Phase*3+c;
    if(!Valid(field,z,y,x,n))continue;
    float d[2];
    #pragma unroll
    for(int t=0;t<2;++t){
      const int q=components[c][t]*SV+center;
      const float a=opposite[q],b=opposite[q+(Phase?-1:1)*strides[axes[c][t]]];
      d[t]=__fmul_rn(Phase?__fsub_rn(a,b):__fsub_rn(b,a),1.f);
    }
    out[Offset(field,z,y,x,n)]=__fmaf_rn(Phase?0.11f:-0.11f,
        __fsub_rn(d[0],d[1]),__fmul_rn(0.99f,old[Offset(field,z,y,x,n)]));
  }
}

// Same deterministic LCG and exactly represented floats as the original host
// initializer. Initialization is outside every solver timing interval.
__global__ void InitializeFields(float *out,int n,int cube,uint32_t count){
  const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=count)return;
  const int side=n+1,c=i/cube,q=i%cube,z=q/(side*side),y=(q/side)%side,x=q%side;
  uint32_t bits=0xffffffffu;
  if(Valid(c,z,y,x,n)){
    const uint32_t seed=i*1664525u+1013904223u;
    const float value=(static_cast<int>((seed>>8)&4095)-2048)*0x1p-15f;
    bits=__float_as_uint(value);
  }
  reinterpret_cast<uint32_t*>(out)[i]=bits;
}

struct Allocation {
  float *base=nullptr,*data=nullptr;size_t count=0;
  explicit Allocation(size_t count):count(count) {
    CU(cudaMalloc(&base,(count+2*GUARD)*sizeof(float)));data=base+GUARD;set_guards();
  }
  ~Allocation(){if(base)cudaFree(base);}
  void set_guards(){std::array<uint32_t,GUARD>v{};v.fill(0xdeadbeef);CU(cudaMemcpy(base,v.data(),sizeof(v),cudaMemcpyHostToDevice));CU(cudaMemcpy(data+count,v.data(),sizeof(v),cudaMemcpyHostToDevice));}
  bool guards_intact() const {std::array<uint32_t,GUARD>lo{},hi{};CU(cudaMemcpy(lo.data(),base,sizeof(lo),cudaMemcpyDeviceToHost));CU(cudaMemcpy(hi.data(),data+count,sizeof(hi),cudaMemcpyDeviceToHost));return std::all_of(lo.begin(),lo.end(),[](uint32_t x){return x==0xdeadbeef;})&&std::all_of(hi.begin(),hi.end(),[](uint32_t x){return x==0xdeadbeef;});}
};

uint64_t HostAvailableBytes(){
  std::ifstream input("/proc/meminfo");std::string line;
  while(std::getline(input,line))if(line.rfind("MemAvailable:",0)==0){
    std::istringstream fields(line);std::string key,unit;uint64_t value=0;
    fields>>key>>value>>unit;if(unit!="kB")break;return value*1024;
  }
  throw std::runtime_error("Cannot read Linux MemAvailable for host memory admission");
}

size_t CompareSlabs(const float *actual,const float *expected,size_t count){
  constexpr size_t slab=1<<20;
  std::vector<uint32_t>a(std::min(count,slab)),b(a.size());size_t mismatches=0;
  for(size_t offset=0;offset<count;offset+=slab){
    const size_t take=std::min(slab,count-offset);
    CU(cudaMemcpy(a.data(),actual+offset,take*4,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(b.data(),expected+offset,take*4,cudaMemcpyDeviceToHost));
    for(size_t i=0;i<take;++i)if(a[i]!=b[i]){
      if(mismatches<4)fprintf(stderr,"mismatch[%zu] bits %08x != %08x\n",offset+i,a[i],b[i]);
      ++mismatches;
    }
  }
  return mismatches;
}

template<int Steps> int Run(int n,int samples,int order_rotation,bool admission_only) {
  const uint64_t side=static_cast<uint64_t>(n)+1,cube=side*side*side;
  if(cube>static_cast<uint64_t>(INT32_MAX)/6)
    throw std::runtime_error("Six padded field indices exceed INT32_MAX");
  Geometry g{};g.n=n;g.cube=static_cast<int>(cube);
  g.nx=(n+Steps+1+BX-1)/BX;g.ny=(n+Steps+1+BY-1)/BY;g.nz=(n+Steps+1+BZ-1)/BZ;
  const int boxes=g.nx*g.ny*g.nz,rank_count=g.nx+g.ny+g.nz-2;
  const uint64_t host_available=HostAvailableBytes();
  const size_t host_bound=2*(1<<20)*sizeof(uint32_t)+2ULL*boxes*sizeof(int)+(rank_count+1)*sizeof(int)+65536;
  constexpr uint64_t host_reserve=512ULL*1024*1024;
  if(host_available<host_reserve||host_bound>host_available-host_reserve)
    throw std::runtime_error("Host slabs/metadata plus512MiB reserve exceed MemAvailable");
  std::vector<int> ids,offsets{0},slots(boxes);ids.reserve(boxes);offsets.reserve(rank_count+1);
  for(int rank=0;rank<rank_count;++rank){const int begin=ids.size();for(int bz=0;bz<g.nz;++bz)for(int by=0;by<g.ny;++by){const int bx=rank-bz-by;if(bx>=0&&bx<g.nx){const int tile=(bz*g.ny+by)*g.nx+bx;slots[tile]=ids.size()-begin;ids.push_back(tile);}}offsets.push_back(ids.size());g.max_frontier=std::max(g.max_frontier,static_cast<int>(ids.size())-begin);}
#if MOVING_STREAMED_UZ && MOVING_STREAMED_INTERIOR
  std::vector<int> interior_counts(rank_count);
  for(int rank=0;rank<rank_count;++rank){auto begin=ids.begin()+offsets[rank],end=ids.begin()+offsets[rank+1];auto middle=std::stable_partition(begin,end,[&](int tile){const int z=tile/(g.nx*g.ny)*BZ,y=(tile/g.nx)%g.ny*BY,x=tile%g.nx*BX;return !(z>=Steps+1&&y>=Steps+1&&x>=Steps+1&&z+BZ<=n&&y+BY<=n&&x+BX<=n);});interior_counts[rank]=end-middle;for(int i=offsets[rank];i<offsets[rank+1];++i)slots[ids[i]]=i-offsets[rank];}
#endif
  const size_t field_count=6ULL*g.cube,trail_tiles=MOVING_TRAIL_RING?MOVING_TRAIL_RING*g.max_frontier:boxes;
  const size_t trail_count=trail_tiles*Steps*6*S;
  size_t free_bytes=0,total_bytes=0;CU(cudaMemGetInfo(&free_bytes,&total_bytes));
  const size_t metadata_bytes=boxes*(1+(MOVING_TRAIL_RING!=0))*sizeof(int)+(MOVING_TRAIL_RING?trail_tiles*sizeof(int):0)+sizeof(int);
  const size_t profile_bytes=MOVING_PHASE_PROFILE?static_cast<size_t>(boxes)*moving_phase_profile::Stride*sizeof(unsigned long long):0;
  const size_t required_bytes=((MOVING_THREE_BANKS?3:4)*field_count+trail_count)*sizeof(float)+10*GUARD*sizeof(float)+metadata_bytes+profile_bytes+(MOVING_PHASE_PROFILE?2*GUARD*sizeof(float):0);
  const size_t solver_bytes=(2*field_count+trail_count)*sizeof(float)+6*GUARD*sizeof(float)+metadata_bytes;
  constexpr size_t reserve_bytes=256ULL*1024*1024;
  const bool admitted=free_bytes>=reserve_bytes&&required_bytes<=free_bytes-reserve_bytes;
  if(admission_only){
    printf("{\"size\":%d,\"steps\":%d,\"admitted\":%s,\"device_payload_bytes\":%zu,\"device_reserve_bytes\":%zu,\"device_free_bytes\":%zu,\"device_total_bytes\":%zu,\"host_workspace_bound_bytes\":%zu,\"host_mem_available\":%llu,\"boxes\":%d,\"max_frontier\":%d,\"field_bytes\":%zu,\"trail_bytes\":%zu,\"solver_payload_bytes\":%zu,\"harness_field_banks\":%d,\"tile_zyx\":[%d,%d,%d],\"owned_cells_per_box\":%d,\"surface_cells_per_box\":%d}\n",n,Steps,admitted?"true":"false",required_bytes,static_cast<size_t>(reserve_bytes),free_bytes,total_bytes,host_bound,static_cast<unsigned long long>(host_available),boxes,g.max_frontier,field_count*4,trail_count*4,solver_bytes,MOVING_THREE_BANKS?3:4,BZ,BY,BX,V,S);return 0;
  }
  if(!admitted)
    throw std::runtime_error("Moving probe memory admission failed: device payload plus256MiB reserve exceed cudaMemGetInfo");
  Allocation a(field_count),b(MOVING_THREE_BANKS?0:field_count),output(field_count),start(field_count),trail(trail_count);
#if MOVING_PHASE_PROFILE
  Allocation profile_records(profile_bytes/sizeof(float));
  g.phase_profile=reinterpret_cast<unsigned long long*>(profile_records.data);
#endif
  InitializeFields<<<static_cast<int>((field_count+255)/256),256>>>(start.data,n,g.cube,static_cast<uint32_t>(field_count));CU(cudaGetLastError());
  CU(cudaMemset(output.data,0xff,field_count*4));CU(cudaMemset(trail.data,0xff,trail_count*4));
  int *bad=nullptr,*device_tiles=nullptr,*device_slots=nullptr,*owner_tags=nullptr;CU(cudaMalloc(&bad,sizeof(int)));CU(cudaMemset(bad,0,sizeof(int)));
  if(MOVING_TRAIL_RING){CU(cudaMalloc(&device_slots,slots.size()*sizeof(int)));CU(cudaMemcpy(device_slots,slots.data(),slots.size()*sizeof(int),cudaMemcpyHostToDevice));g.slots=device_slots;CU(cudaMalloc(&owner_tags,trail_tiles*sizeof(int)));CU(cudaMemset(owner_tags,0xff,trail_tiles*sizeof(int)));g.owner_tags=owner_tags;}
  CU(cudaMalloc(&device_tiles,ids.size()*sizeof(int)));CU(cudaMemcpy(device_tiles,ids.data(),ids.size()*sizeof(int),cudaMemcpyHostToDevice));
  auto initialize=[&](float*target){InitializeFields<<<static_cast<int>((field_count+255)/256),256>>>(target,n,g.cube,static_cast<uint32_t>(field_count));CU(cudaGetLastError());};
  auto reset_plain=[&](){if(MOVING_MATCHED_D2_CONTROL)initialize(a.data);else CU(cudaMemcpy(a.data,start.data,field_count*4,cudaMemcpyDeviceToDevice));if(!MOVING_THREE_BANKS)CU(cudaMemset(b.data,0xff,field_count*4));};
#if MOVING_ASYMMETRIC_LOW_X
  constexpr size_t dynamic_shared=moving_asymmetric_low_x::SharedBytes;
  CU(cudaFuncSetAttribute(AsymmetricCachedMoving<Steps>,cudaFuncAttributeMaxDynamicSharedMemorySize,static_cast<int>(dynamic_shared)));
#elif MOVING_ASYMMETRIC_PIPELINE
  constexpr size_t dynamic_shared=moving_asymmetric_pipeline::SharedBytes;
  CU(cudaFuncSetAttribute(AsymmetricPipelinedMoving<Steps>,cudaFuncAttributeMaxDynamicSharedMemorySize,static_cast<int>(dynamic_shared)));
#elif MOVING_ASYMMETRIC_CACHE
  constexpr size_t dynamic_shared=moving_asymmetric_cache::SharedBytes;
  CU(cudaFuncSetAttribute(AsymmetricCachedMoving<Steps>,cudaFuncAttributeMaxDynamicSharedMemorySize,static_cast<int>(dynamic_shared)));
#elif MOVING_REGISTER_CACHE
  constexpr size_t dynamic_shared=moving_register_cache::SharedBytes;
  CU(cudaFuncSetAttribute(RegisterCachedMoving<Steps>,cudaFuncAttributeMaxDynamicSharedMemorySize,static_cast<int>(dynamic_shared)));
#else
  constexpr size_t dynamic_shared=0;
#endif
  auto moving=[&](){for(int rank=0;rank<rank_count;++rank){const int count=offsets[rank+1]-offsets[rank];if(count){
#if MOVING_STREAMED_UZ
#if MOVING_STREAMED_INTERIOR
    const int interior=interior_counts[rank],boundary=count-interior;
    if(boundary)StreamedUzMoving<Steps,false><<<boundary,THREADS>>>(device_tiles+offsets[rank],g,start.data,output.data,trail.data,bad);
    if(interior)StreamedUzMoving<Steps,true><<<interior,THREADS>>>(device_tiles+offsets[rank]+boundary,g,start.data,output.data,trail.data,bad);
#else
    StreamedUzMoving<Steps,false><<<count,THREADS>>>(device_tiles+offsets[rank],g,start.data,output.data,trail.data,bad);
#endif
#elif MOVING_ASYMMETRIC_PIPELINE
    AsymmetricPipelinedMoving<Steps><<<count,THREADS,dynamic_shared>>>(device_tiles+offsets[rank],g,start.data,output.data,trail.data,bad);
#elif MOVING_ASYMMETRIC_CACHE
    AsymmetricCachedMoving<Steps><<<count,THREADS,dynamic_shared>>>(device_tiles+offsets[rank],g,start.data,output.data,trail.data,bad);
#elif MOVING_REGISTER_CACHE
    RegisterCachedMoving<Steps><<<count,THREADS,dynamic_shared>>>(device_tiles+offsets[rank],g,start.data,output.data,trail.data,bad);
#elif MOVING_CIRCULAR_CACHE
    CircularCachedMoving<Steps><<<count,THREADS>>>(device_tiles+offsets[rank],g,start.data,output.data,trail.data,bad);
#elif MOVING_FACE_CACHE
    FaceCachedMoving<Steps><<<count,THREADS>>>(device_tiles+offsets[rank],g,start.data,output.data,trail.data,bad);
#else
    Moving<Steps><<<count,THREADS>>>(device_tiles+offsets[rank],g,start.data,output.data,trail.data,bad);
#endif
  }}CU(cudaPeekAtLastError());};
  auto plain=[&](bool tiled,float*single=nullptr){float *in=single?single:a.data,*out=(MOVING_THREE_BANKS||single)?in:b.data;for(int step=0;step<Steps;++step){
#if MOVING_TILED_CONTROL
    if(tiled){const dim3 block(32,4,2),grid((n+32)/32,(n+4)/4,(n+2)/2);TiledPlain<0><<<grid,block>>>(in,in,out,n);TiledPlain<1><<<grid,block>>>(in,out,out,n);}
    else
#endif
    {const int blocks=(3*g.cube+255)/256;Plain<<<blocks,256>>>(in,in,out,n,0);Plain<<<blocks,256>>>(in,out,out,n,1);}
    std::swap(in,out);}CU(cudaPeekAtLastError());return in;};
#if MOVING_MATCHED_D2_CONTROL
  static_assert(MOVING_THREE_BANKS && MOVING_TILED_CONTROL,"Matched D2 control requires three banks plus tiled control");
  auto reset_d2=[&](){initialize(start.data);CU(cudaMemset(output.data,0xff,field_count*4));};
  auto matched_d2=[&](){float*in=start.data,*out=output.data;for(int pair=0;pair<Steps/2;++pair){EnqueueMatchedD2(in,out,n);std::swap(in,out);}return in;};
#endif
  auto guards_ok=[&](){return
#if MOVING_PHASE_PROFILE
profile_records.guards_intact()&&
#endif
a.guards_intact()&&b.guards_intact()&&output.guards_intact()&&start.guards_intact()&&trail.guards_intact();};
  auto validate=[&](const char*stage,const float*actual,const float*expected){
    const size_t mismatches=CompareSlabs(actual,expected,field_count);int bad_reads=0;CU(cudaMemcpy(&bad_reads,bad,4,cudaMemcpyDeviceToHost));
    if(mismatches||bad_reads||!guards_ok())throw std::runtime_error(std::string(stage)+": bitwise/owner/guard validation failed, mismatches="+std::to_string(mismatches)+", bad="+std::to_string(bad_reads));
  };
  reset_plain();float *reference=plain(false);moving();CU(cudaDeviceSynchronize());validate("initial moving",output.data,reference);
#if MOVING_TILED_CONTROL
  if(MOVING_THREE_BANKS){CU(cudaMemcpy(output.data,start.data,field_count*4,cudaMemcpyDeviceToDevice));plain(true,output.data);CU(cudaDeviceSynchronize());validate("initial tiled",output.data,reference);}
  else {reset_plain();float *tiled_result=plain(true);CU(cudaMemcpy(output.data,tiled_result,field_count*4,cudaMemcpyDeviceToDevice));reset_plain();reference=plain(false);CU(cudaDeviceSynchronize());validate("initial tiled",output.data,reference);}
  moving();CU(cudaDeviceSynchronize());validate("restored moving",output.data,reference);
#endif
#if MOVING_MATCHED_D2_CONTROL
  reset_d2();float*d2_result=matched_d2();CU(cudaDeviceSynchronize());validate("initial matched D2",d2_result,reference);
  initialize(start.data);
#endif
  const size_t useful_fields=3ULL*n*n*(n+1)+3ULL*n*(n+1)*(n+1);
  std::vector<float> moving_ms,plain_ms,tiled_ms,d2_ms;std::vector<std::string>timing_orders;
  cudaEvent_t begin,end;CU(cudaEventCreate(&begin));CU(cudaEventCreate(&end));
  const char*names[4]={"plain","tiled","moving","matched_d2"};
  constexpr int four_orders[8][4]={{0,1,2,3},{1,2,3,0},{2,3,0,1},{3,0,1,2},{3,2,1,0},{0,3,2,1},{1,0,3,2},{2,1,0,3}};
  constexpr int permutations[6][3]={{0,1,2},{1,2,0},{2,0,1},{2,1,0},{1,0,2},{0,2,1}};
  for(int sample=0;sample<samples;++sample){
    std::string order;const int rotation=sample+order_rotation;
    for(int position=0;position<(MOVING_MATCHED_D2_CONTROL?4:MOVING_TILED_CONTROL?3:2);++position){
      const int variant=MOVING_MATCHED_D2_CONTROL?four_orders[rotation%8][position]:MOVING_TILED_CONTROL?permutations[rotation%6][position]:((position+(rotation%2))%2?2:0);
      if(!order.empty())order+=",";order+=names[variant];
      if(variant<2)reset_plain();
#if MOVING_MATCHED_D2_CONTROL
      else if(variant==3)reset_d2();else {initialize(start.data);CU(cudaMemset(output.data,0xff,field_count*4));}
#endif
      CU(cudaEventRecord(begin));
#if MOVING_MATCHED_D2_CONTROL
      if(variant==3)matched_d2();else
#endif
      if(variant==2)moving();else plain(variant==1);CU(cudaEventRecord(end));CU(cudaEventSynchronize(end));float elapsed;CU(cudaEventElapsedTime(&elapsed,begin,end));
      (variant==3?d2_ms:variant==2?moving_ms:variant==1?tiled_ms:plain_ms).push_back(elapsed);
    }
    timing_orders.push_back(order);
  }
  // Recompute the independent oracle outside the timed region. All comparison
  // buffers remain slab-sized regardless of domain volume.
#if MOVING_MATCHED_D2_CONTROL
  initialize(start.data);CU(cudaMemset(output.data,0xff,field_count*4));moving();
#endif
  reset_plain();reference=plain(false);CU(cudaDeviceSynchronize());validate("repeat moving",output.data,reference);
#if MOVING_TILED_CONTROL
  if(MOVING_THREE_BANKS){CU(cudaMemcpy(output.data,start.data,field_count*4,cudaMemcpyDeviceToDevice));plain(true,output.data);CU(cudaDeviceSynchronize());validate("repeat tiled",output.data,reference);}
  else {reset_plain();float *repeated_tiled=plain(true);CU(cudaMemcpy(output.data,repeated_tiled,field_count*4,cudaMemcpyDeviceToDevice));reset_plain();reference=plain(false);CU(cudaDeviceSynchronize());validate("repeat tiled",output.data,reference);}
#endif
#if MOVING_MATCHED_D2_CONTROL
  reset_d2();d2_result=matched_d2();reset_plain();reference=plain(false);CU(cudaDeviceSynchronize());validate("repeat matched D2",d2_result,reference);
#endif
#if MOVING_PHASE_PROFILE
  const char*profile_path=std::getenv("MOVING_PROFILE_OUTPUT");
  if(profile_path){std::ofstream file(profile_path,std::ios::binary);if(!file)throw std::runtime_error("Cannot open phase-profile output");std::vector<unsigned char>slab(1<<20);for(size_t offset=0;offset<profile_bytes;offset+=slab.size()){const size_t take=std::min(slab.size(),profile_bytes-offset);CU(cudaMemcpy(slab.data(),reinterpret_cast<unsigned char*>(g.phase_profile)+offset,take,cudaMemcpyDeviceToHost));file.write(reinterpret_cast<char*>(slab.data()),take);}if(!file)throw std::runtime_error("Cannot write phase-profile output");}
#endif
  cudaFuncAttributes attributes{};
#if MOVING_STREAMED_UZ
  auto kernel=StreamedUzMoving<Steps>;
#elif MOVING_ASYMMETRIC_PIPELINE
  auto kernel=AsymmetricPipelinedMoving<Steps>;
#elif MOVING_ASYMMETRIC_CACHE
  auto kernel=AsymmetricCachedMoving<Steps>;
#elif MOVING_REGISTER_CACHE
  auto kernel=RegisterCachedMoving<Steps>;
#elif MOVING_CIRCULAR_CACHE
  auto kernel=CircularCachedMoving<Steps>;
#elif MOVING_FACE_CACHE
  auto kernel=FaceCachedMoving<Steps>;
#else
  auto kernel=Moving<Steps>;
#endif
  CU(cudaFuncGetAttributes(&attributes,kernel));
  int active_blocks=0,device=0,max_threads=0,sm_count=0;
  CU(cudaGetDevice(&device));
  CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,kernel,THREADS,dynamic_shared));
  CU(cudaDeviceGetAttribute(&max_threads,cudaDevAttrMaxThreadsPerMultiProcessor,device));
  CU(cudaDeviceGetAttribute(&sm_count,cudaDevAttrMultiProcessorCount,device));
  printf("{\"scope\":\"field-only moving-box mechanism; no CPML/source/DFT; no production speed claim\",\"size\":%d,\"steps\":%d,\"tile_zyx\":[%d,%d,%d],\"threads\":%d,\"boxes\":%d,\"max_frontier\":%d,\"rank_launches\":%d,\"surface_cells_per_box\":%d,\"trail_bytes\":%zu,\"admitted_payload_bytes\":%zu,\"device_free_bytes_before\":%zu,\"device_total_bytes\":%zu,\"host_mem_available_before\":%llu,\"host_workspace_bound_bytes\":%zu,\"valid_component_updates\":%zu,\"visited_component_slots\":%zu,\"valid_updates_over_useful\":1,\"registers\":%d,\"static_shared_bytes\":%zu,\"local_bytes\":%zu,\"mismatches\":0,\"bad_trail_reads\":0,\"plain_ms\":[",n,Steps,BZ,BY,BX,THREADS,boxes,g.max_frontier,rank_count,S,trail_count*4,required_bytes,free_bytes,total_bytes,static_cast<unsigned long long>(host_available),host_bound,useful_fields*Steps,static_cast<size_t>(boxes)*V*6*Steps,attributes.numRegs,attributes.sharedSizeBytes,attributes.localSizeBytes);
  auto print_values=[](const std::vector<float>&v){for(size_t i=0;i<v.size();++i)printf("%s%.9g",i?",":"",v[i]);};
  print_values(plain_ms);printf("],\"moving_ms\":[");print_values(moving_ms);printf("],\"tiled_ms\":[");print_values(tiled_ms);printf("],\"matched_d2_ms\":[");print_values(d2_ms);printf("],\"timing_orders\":[");for(size_t i=0;i<timing_orders.size();++i)printf("%s\"%s\"",i?",":"",timing_orders[i].c_str());
  printf("],\"repeat_checked\":true,\"repeat_mismatches\":0,\"all_guards_checked\":true,\"trail_guards_checked\":true,\"streamed_readback\":true,\"initialization\":\"GPU index-LCG outside timings\",\"reset_policy\":\"D2D initial-state copy and poison outside timings; moving immutable initial bank\",\"tiled_control\":%s,\"face_cache\":%s,\"circular_cache\":%s,\"harness_field_banks\":%d,\"solver_payload_bytes\":%zu,\"control_in_place\":%s,\"occupancy_max_active_blocks_per_sm\":%d,\"device_max_threads_per_sm\":%d,\"device_sm_count\":%d,\"dynamic_shared_bytes\":%zu,\"register_cache\":%s,\"register_local_centers\":%s",MOVING_TILED_CONTROL?"true":"false",MOVING_FACE_CACHE?"true":"false",MOVING_CIRCULAR_CACHE?"true":"false",MOVING_THREE_BANKS?3:4,solver_bytes,MOVING_THREE_BANKS?"true":"false",active_blocks,max_threads,sm_count,dynamic_shared,MOVING_REGISTER_CACHE?"true":"false",MOVING_REGISTER_LOCAL_CENTERS?"true":"false");
  printf(",\"phase_profile\":%s,\"phase_profile_bytes\":%zu,\"phase_profile_stride_u64\":%d,\"asymmetric_cache\":%s}\n",MOVING_PHASE_PROFILE?"true":"false",profile_bytes,moving_phase_profile::Stride,MOVING_ASYMMETRIC_CACHE?"true":"false");
#if MOVING_MATCHED_D2_CONTROL
  // Secondary metadata follows on stderr so stdout stays one JSON record.
  cudaFuncAttributes d2attr{};CU(cudaFuncGetAttributes(&d2attr,beamz_moving_d2::Kernel<MatchedD2TileX,MatchedD2TileY,MatchedD2Threads>));
  int d2blocks=0;CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&d2blocks,beamz_moving_d2::Kernel<MatchedD2TileX,MatchedD2TileY,MatchedD2Threads>,MatchedD2Threads,MatchedD2SharedBytes));
  fprintf(stderr,"matched_d2 tileXY=%d,%d threads=%d regs=%d shared=%d local=%zu activeCTAs=%d\n",MatchedD2TileX,MatchedD2TileY,MatchedD2Threads,d2attr.numRegs,MatchedD2SharedBytes,d2attr.localSizeBytes,d2blocks);
#endif
  CU(cudaEventDestroy(begin));CU(cudaEventDestroy(end));CU(cudaFree(bad));CU(cudaFree(device_tiles));if(device_slots)CU(cudaFree(device_slots));if(owner_tags)CU(cudaFree(owner_tags));return 0;
}

int main(int argc,char **argv) {
  try {const bool admission=argc>1&&std::strcmp(argv[1],"--admission-only")==0;const int base=admission?2:1;
    const int n=argc>base?std::atoi(argv[base]):33,steps=argc>base+1?std::atoi(argv[base+1]):4,samples=argc>base+2?std::atoi(argv[base+2]):3,rotation=argc>base+3?std::atoi(argv[base+3]):0;
    if(n<3||n>1000||samples<1||samples>21||rotation<0)throw std::runtime_error("size must be3..1000 with six fields fittingint32; samples1..21; nonnegative order rotation");
#if MOVING_STREAMED_UZ
    if(steps!=moving_streamed_uz::Depth)throw std::runtime_error("Steps differ from generated streamed depth");
    return Run<moving_streamed_uz::Depth>(n,samples,rotation,admission);
#else
    if(steps==4)return Run<4>(n,samples,rotation,admission);if(steps==8)return Run<8>(n,samples,rotation,admission);if(steps==16)return Run<16>(n,samples,rotation,admission);if(steps==32)return Run<32>(n,samples,rotation,admission);throw std::runtime_error("steps must be4/8/16/32");
#endif
  }catch(const std::exception &e){fprintf(stderr,"%s\n",e.what());return 1;}
}
