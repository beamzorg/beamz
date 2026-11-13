// Matched two-step rolling control for the field-only moving-box experiment.
// Include after Valid, Offset, GlobalAt, Generated<Phase> and CU are defined.
// Arithmetic/seed/boundaries are the harness's contract, not production PEC.
// The schedule follows production MstepStages: H1/E1/H2 two-plane rings and
// direct E2 output, with a shared E0 input plane. Input/output must be distinct.
#ifndef BEAMZ_CUDA_MOVING_D2_CONTROL_CUH_
#define BEAMZ_CUDA_MOVING_D2_CONTROL_CUH_

#include <cstdint>
#include <limits>
#include <stdexcept>

#ifndef MOVING_D2_TILE_X
#define MOVING_D2_TILE_X 16
#endif
#ifndef MOVING_D2_TILE_Y
#define MOVING_D2_TILE_Y 8
#endif
#ifndef MOVING_D2_THREADS
#define MOVING_D2_THREADS 256
#endif

namespace beamz_moving_d2 {

template<int X,int Y,int K> struct Stage {
  static constexpr int sx=X+3-K, sy=Y+3-K;
  static constexpr int offset=[] {
    int result=0;
    for(int k=0;k<K;++k)result+=6*(X+3-k)*(Y+3-k);
    return result;
  }();
  __device__ static __forceinline__ float &At(
      float *shared,int c,int z,int y,int x) {
    return shared[offset+((c*2+(z&1))*sy+y)*sx+x];
  }
};

template<int X,int Y> struct Layout {
  static constexpr int input_offset=Stage<X,Y,3>::offset;
  static constexpr int input_x=X+4,input_y=Y+4;
  static constexpr int floats=input_offset+3*input_x*input_y;
  static constexpr int shared_bytes=floats*sizeof(float);
  __device__ static __forceinline__ float &Input(float *shared,int c,int y,int x) {
    return shared[input_offset+(c*input_y+y)*input_x+x];
  }
};

template<int X,int Y,int K=0>
__device__ __forceinline__ void AdvanceStages(
    float *shared,const float *input,float *output,int n,int ox,int oy,int wave) {
  using Here=Stage<X,Y,K>;
  constexpr int phase=K&1, shift=phase?1:0;
  const int z=wave-2-K/2;
  if(wave>=K) {
    for(int i=threadIdx.x;i<Here::sx*Here::sy;i+=blockDim.x) {
      const int ix=i%Here::sx,iy=i/Here::sx;
      const int x=ox-2+(K+1)/2+ix,y=oy-2+(K+1)/2+iy;
      #pragma unroll
      for(int c=0;c<3;++c) {
        float next=0.f;
        if(Valid(phase*3+c,z,y,x,n)) {
          float old;
          if constexpr(K==0)old=GlobalAt(input,c,z,y,x,n);
          else if constexpr(K==1)old=Layout<X,Y>::Input(shared,c,iy+1,ix+1);
          else old=Stage<X,Y,K-2>::At(shared,c,z,iy+1,ix+1);
          const auto other=[&](int component,int dz,int dy,int dx) {
            if constexpr(K==0) {
              if(dz==0)return Layout<X,Y>::Input(shared,component,iy+dy,ix+dx);
              return GlobalAt(input,3+component,z+dz,y+dy,x+dx,n);
            } else {
              return Stage<X,Y,K-1>::At(shared,component,z+dz,
                                         iy+shift+dy,ix+shift+dx);
            }
          };
          const auto read=[&](int field,int version,int dz,int dy,int dx) {
            (void)version;
            // Generated phases use relative old/updated versions. K proves
            // their absolute H1/E1/H2/E2 stage and supplies the right bank.
            return field/3==phase?old:other(field%3,dz,dy,dx);
          };
          next=Generated<phase>(c,read);
        }
        if constexpr(K<3) {
          Here::At(shared,c,z,iy,ix)=next;
        } else {
          // Staggered boundary sites differ between H and E. Never put the
          // H2 store under E2's validity condition (e.g. Hx at x==n).
          if(Valid(3+c,z,y,x,n))output[Offset(3+c,z,y,x,n)]=next;
          if(Valid(c,z,y,x,n))
            output[Offset(c,z,y,x,n)]=Stage<X,Y,2>::At(shared,c,z,iy+1,ix+1);
        }
      }
    }
  }
  if constexpr(K<3) {
    __syncthreads();
    AdvanceStages<X,Y,K+1>(shared,input,output,n,ox,oy,wave);
  }
  // No terminal barrier: the following wave only overwrites Input/H1 before
  // its load barrier. E1/H2 remain untouched until every thread passes it.
}

template<int X,int Y,int Threads>
__global__ __launch_bounds__(Threads,4) void Kernel(
    const float *__restrict__ input,float *__restrict__ output,int n) {
  extern __shared__ float shared[];
  const int ox=blockIdx.x*X,oy=blockIdx.y*Y;
  // Wave w loads E0(z=w-2), then computes H1/E1 at that z and H2/E2 at z-1.
  // Three bootstrap/drain waves cover all physical z=0..n terminal sites.
  for(int wave=0;wave<n+4;++wave) {
    const int z=wave-2;
    for(int i=threadIdx.x;i<Layout<X,Y>::input_x*Layout<X,Y>::input_y;i+=blockDim.x) {
      const int ix=i%Layout<X,Y>::input_x,iy=i/Layout<X,Y>::input_x;
      #pragma unroll
      for(int c=0;c<3;++c)
        Layout<X,Y>::Input(shared,c,iy,ix)=GlobalAt(input,3+c,z,oy-2+iy,ox-2+ix,n);
    }
    __syncthreads();
    AdvanceStages<X,Y>(shared,input,output,n,ox,oy,wave);
  }
}

} // namespace beamz_moving_d2

constexpr int MatchedD2TileX=MOVING_D2_TILE_X;
constexpr int MatchedD2TileY=MOVING_D2_TILE_Y;
constexpr int MatchedD2Threads=MOVING_D2_THREADS;
constexpr int MatchedD2SharedBytes=
    beamz_moving_d2::Layout<MatchedD2TileX,MatchedD2TileY>::shared_bytes;
static_assert(MatchedD2TileX>0&&MatchedD2TileY>0);
static_assert(MatchedD2Threads>=32&&MatchedD2Threads<=1024&&MatchedD2Threads%32==0);

inline void EnqueueMatchedD2(const float *input,float *output,int n,
                             cudaStream_t stream=0) {
  if(!input||!output||input==output||n<1)
    throw std::invalid_argument("MatchedD2 requires distinct buffers and positive n");
  const uint64_t side=static_cast<uint64_t>(n)+1;
  if(side>static_cast<uint64_t>(std::numeric_limits<int>::max()/6)/side/side)
    throw std::invalid_argument("MatchedD2 padded field offsets exceed int32");
  if constexpr(MatchedD2SharedBytes>48*1024)
    CU(cudaFuncSetAttribute(
      beamz_moving_d2::Kernel<MatchedD2TileX,MatchedD2TileY,MatchedD2Threads>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,MatchedD2SharedBytes));
  beamz_moving_d2::Kernel<MatchedD2TileX,MatchedD2TileY,MatchedD2Threads>
      <<<dim3((n+1+MatchedD2TileX-1)/MatchedD2TileX,
               (n+1+MatchedD2TileY-1)/MatchedD2TileY),
          MatchedD2Threads,MatchedD2SharedBytes,stream>>>(input,output,n);
  CU(cudaGetLastError());
}

#endif
