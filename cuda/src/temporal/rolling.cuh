// Four-step Yee windows over native staggered field and compact CPML descriptors.
// Include after temporal/kernel.cuh in CUDA translation units.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace beamz::cuda::rolling {

constexpr int Steps = 4, ZPlanes = 32;
struct Box { int lo[3]{}, hi[3]{}; };
struct Tiles {
  Box boxes[6]{};
  int first[7]{}, nx[6]{}, ny[6]{}, count = 0;
  int size() const { return first[count]; }
};
struct TileTrace { unsigned flags; float electric[3]; };
struct Band {
  Tiles core, shell;
  std::array<int,2> source_z{};
  size_t offset = 0;
  int first = 0, end = 0;
};
struct BandReplacement { size_t index; std::vector<Band> pieces; };
struct Plan {
  std::vector<Band> bands;
  size_t mask_bytes = 0;
  std::vector<std::vector<BandReplacement>> windows;

  std::pair<const Band*, size_t> View(size_t index, int elapsed) const {
    if (!windows.empty()) {
      for (const auto& change : windows[size_t(elapsed / Steps) % windows.size()])
        if (change.index == index) return {change.pieces.data(), change.pieces.size()};
    }
    return {&bands[index], 1};
  }
};

// Each piece points at a contiguous interval of the original material/effect
// trace. Split its first Z tile when the causal boundary is not tile-aligned.
inline std::vector<Band> ClipZ(const Band& band, int low, int high) {
  std::vector<Band> pieces;
  for (bool shell : {false, true}) {
    const auto& tiles = shell ? band.shell : band.core;
    const size_t base = band.offset / sizeof(TileTrace) + (shell ? band.core.size() : 0);
    for (int i = 0; i < tiles.count; ++i) {
      const auto& original = tiles.boxes[i];
      const int stop = std::min(high, original.hi[0]);
      for (int z = std::max(low, original.lo[0]); z < stop;) {
        const int offset = z - original.lo[0], remainder = offset % ZPlanes;
        const int end = remainder ? z + std::min(stop - z, ZPlanes - remainder) : stop;
        const size_t first = base + tiles.first[i] +
            size_t(offset / ZPlanes) * tiles.nx[i] * tiles.ny[i];
        if (pieces.empty() || !shell || pieces.back().shell.count == 6 ||
            pieces.back().offset / sizeof(TileTrace) + pieces.back().core.size() +
                pieces.back().shell.size() != first) {
          auto& piece = pieces.emplace_back();
          piece.offset = first * sizeof(TileTrace);
          piece.first = band.first;
          piece.end = band.end;
        }
        auto& group = shell ? pieces.back().shell : pieces.back().core;
        const int k = group.count++;
        group.boxes[k] = original;
        group.boxes[k].lo[0] = z;
        group.boxes[k].hi[0] = end;
        group.nx[k] = tiles.nx[i];
        group.ny[k] = tiles.ny[i];
        group.first[k + 1] = group.first[k] + group.nx[k] * group.ny[k] *
            (1 + (end - z - 1) / ZPlanes);
        z = end;
      }
    }
  }
  return pieces;
}

// Each tile has one immutable record describing its material, observation,
// source and CPML footprint. No intermediate fields or CPML banks are needed.
inline Plan MakePlan(std::array<int,3> physical, int band,
                     std::array<int,3> cut = {-1,-1,-1}, int window = 0, std::array<int,2> source_z = {}) {
  if (band < 32 || band > 512 || band % 32 ||
      std::any_of(physical.begin(), physical.end(), [](int n){ return n < 24 || n == INT32_MAX; }))
    throw std::invalid_argument("rolling requires a CPML12 domain and a band multiple of 32 in [32,512]");
  Box center, domain;
  for (int a=0; a<3; ++a) {
    domain.hi[a]=physical[a]+1;
    center.lo[a]=16;
    center.hi[a]=std::max(16, domain.hi[a]-16);
  }
  Plan plan;
  for (int first=0; first<domain.hi[0];) {
    Band b;
    b.source_z=source_z;
    b.first=first; b.end=first+std::min(band,domain.hi[0]-first); b.offset=plan.mask_bytes;
    auto add=[&](Tiles& tiles, Box box, int y_height) {
      box.lo[0]=std::max(box.lo[0],b.first); box.hi[0]=std::min(box.hi[0],b.end);
      if (box.lo[0]>=box.hi[0] || box.lo[1]>=box.hi[1] || box.lo[2]>=box.hi[2]) return;
      const int i=tiles.count++;
      tiles.boxes[i]=box;
      tiles.nx[i]=1+(box.hi[2]-box.lo[2]-1)/24;
      tiles.ny[i]=1+(box.hi[1]-box.lo[1]-1)/y_height;
      const int nz=1+(box.hi[0]-box.lo[0]-1)/ZPlanes;
      tiles.first[i+1]=tiles.first[i]+tiles.nx[i]*tiles.ny[i]*nz;
    };
    add(b.core,center,24);
    // Mutually exclusive slabs cover corners and edges as well as flat faces.
    Box rest=domain;
    for (int axis=0; axis<3; ++axis) {
      Box low=rest, high=rest;
      low.hi[axis]=center.lo[axis]; high.lo[axis]=center.hi[axis];
      add(b.shell,low,16); add(b.shell,high,16);
      rest.lo[axis]=center.lo[axis]; rest.hi[axis]=center.hi[axis];
    }
    first=b.end; plan.mask_bytes+=sizeof(TileTrace)*(b.core.size()+b.shell.size()); plan.bands.push_back(b);
  }
  if (cut[0] == 0 && (cut[1] >= 0 || cut[2] >= 0)) {
    for (int elapsed = Steps; elapsed <= window; elapsed += Steps) {
      // A Yee step spreads information at most one cell in Z. Retain one
      // extra staggered plane on both sides through the end of the exchange.
      const int halo = window - elapsed + 1;
      const int low = cut[1] < 0 ? 0 : std::max(0, cut[1] - halo);
      const int high = cut[2] < 0 ? domain.hi[0] :
          int(std::min<int64_t>(domain.hi[0], int64_t(cut[2]) + halo));
      auto& changes = plan.windows.emplace_back();
      for (size_t i = 0; i < plan.bands.size(); ++i) {
        const auto& b = plan.bands[i];
        if (low > b.first || high < b.end) changes.push_back({i, ClipZ(b, low, high)});
      }
    }
  }
  return plan;
}

#ifdef __CUDACC__
namespace native = beamz::cuda::moving_native;
using native::Parameters;
constexpr unsigned SourceStartShift=9, SourceWidthShift=15, SourceResidueMask=63;
constexpr unsigned Observation=moving::kMayObserve, Source=moving::kMayHaveSources, Volume=32, ColumnInvariant=64, Mixed=128, PlaneInvariant=256;

struct Tile { Box box; int x, y, z; };
__device__ __forceinline__ Tile Locate(const Tiles& tiles,int id) {
  int k=0; while(k+1<tiles.count && id>=tiles.first[k+1]) ++k;
  const int i=id-tiles.first[k], nx=tiles.nx[k], ny=tiles.ny[k];
  return {tiles.boxes[k],i%nx,i/nx%ny,i/(nx*ny)};
}

template<bool Shell> __device__ void ClassifyEffects(const Parameters& p,
    const Tiles& tiles,TileTrace* mask,int2 source_z) {
  constexpr int Y=Shell?24:32;
  const auto t=Locate(tiles,blockIdx.x);
  const int base[]={t.box.lo[0]+t.z*ZPlanes,t.box.lo[1]+t.y*(Y-2*Steps),t.box.lo[2]+t.x*24};
  const int begin[]={base[0]-Steps,base[1]-Steps,base[2]-Steps};
  const int end[]={min(base[0]+ZPlanes,t.box.hi[0])+Steps,base[1]+Y-Steps,base[2]+32-Steps};
  const int span_size[]={native::BZ,native::BY,native::BX}, counts[]={p.nz,p.ny,p.nx};
  int lo[3],span[3]; unsigned flags=Mixed|(p.volumes?Volume:0);
  #pragma unroll
  for(int a=0;a<3;++a) {
    lo[a]=max(0,(begin[a]+1)/span_size[a]);
    span[a]=max(0,min(counts[a]-1,end[a]/span_size[a])-lo[a]+1);
    if constexpr(Shell) if(begin[a]<12 || end[a]>p.physical_n[a]-12) flags|=2u<<a;
  }
  for(int i=threadIdx.x;i<span[0]*span[1]*span[2];i+=32) {
    int z=lo[0]+i/(span[1]*span[2]),y=lo[1]+i/span[2]%span[1],x=lo[2]+i%span[2];
    const unsigned f=p.flags[(z*p.ny+y)*p.nx+x];
    if(p.scatters || p.pre_e_mask) flags|=f&Source;
    if(!p.observation_tiles && p.observation.key_count) flags|=f&Observation;
  }
  for(const auto& s:p.sources) for(int j=threadIdx.x;j<(s.coincident?min(s.count,1):s.count);j+=32) {
    const int* q=s.starts+3*j;
    if(q[0]<end[0] && int64_t(q[0])+s.nz>begin[0] && q[1]<end[1] && int64_t(q[1])+s.ny>begin[1] &&
       q[2]<end[2] && int64_t(q[2])+s.nx>begin[2]) flags|=Source;
  }
  if(p.observation_tiles) {
    const int z0=max(base[0],p.observation_origin[0]),y0=max(base[1]/8,p.observation_origin[1]),x0=max(base[2]/16,p.observation_origin[2]);
    const int nz=min(min(base[0]+ZPlanes,t.box.hi[0]),p.observation_origin[0]+p.observation_nz)-z0;
    const int ny=min((min(base[1]+Y-2*Steps,t.box.hi[1])+7)/8,p.observation_origin[1]+p.observation_ny)-y0;
    const int nx=min((min(base[2]+24,t.box.hi[2])+15)/16,p.observation_origin[2]+p.observation_nx)-x0;
    for(int i=threadIdx.x;i<max(0,nz)*max(0,ny)*max(0,nx);i+=32) {
      int z=z0+i/(ny*nx)-p.observation_origin[0],y=y0+i/nx%ny-p.observation_origin[1],x=x0+i%nx-p.observation_origin[2];
      if(p.observation_tiles[(z*p.observation_ny+y)*p.observation_nx+x]>=0) flags|=Observation;
    }
  }
  for(int d=16;d;d>>=1) flags|=__shfl_xor_sync(0xffffffffu,flags,d);
  if(!threadIdx.x){
    const int width=source_z.y-source_z.x;
    if((flags&Source) && width>0 && width<=2){
      // Pre-electric injection also reads target+1. Absolute residues stay
      // valid when an exchange window trims the lower edge of a tile.
      const int first=source_z.x-1;
      if(first>=end[0] || source_z.y<=base[0]-2*Steps)flags&=~Source;
      else flags|=((unsigned(first)&SourceResidueMask)<<SourceStartShift)|((width+1)<<SourceWidthShift);
    }
    mask[blockIdx.x].flags=flags;
  }
}

// Prove a material triplet over the complete staggered read footprint. This
// certifies any constant region, including regions different from the scene mode.
template<bool Shell>
__device__ void TraceMaterial(const Parameters& p,const Tiles& tiles,TileTrace* trace) {
  auto& entry=trace[blockIdx.x];
  const auto t=Locate(tiles,blockIdx.x);
  constexpr int yout=Shell?16:24;
  const int origin[]={t.box.lo[0]+t.z*ZPlanes,t.box.lo[1]+t.y*yout,t.box.lo[2]+t.x*24};
  const int low[]={max(0,origin[0]-2*Steps),max(0,origin[1]-Steps),max(0,origin[2]-Steps)};
  const int end[]={min(t.box.hi[0],origin[0]+ZPlanes)+Steps,origin[1]+yout+Steps,origin[2]+24+Steps};
  for(int c=0;c<3;++c){
    const auto& field=native::FinalField(p,c+3);const auto& phase=p.phase[1];
    const int z1=min(end[0],field.z),y1=min(end[1],field.y),x1=min(end[2],field.x);
    const auto value=[&]__device__(int linear){
      const int h=__ldg(reinterpret_cast<const int*>(phase.codes[c])+(linear>>5));
      return __ldg(phase.tables[c]+(h<0?~h:h+(linear&31)));
    };
    if(z1<=low[0]||y1<=low[1]||x1<=low[2])return;
    const float first=value((low[0]*field.y+low[1])*field.x+low[2]);
    bool mismatch=false;
    // Each warp checks the 32-aligned coefficient groups intersecting one row.
    // A repeat header proves the complete group from one value comparison.
    const int nx=x1-low[2],ny=y1-low[1],nr=(z1-low[0])*ny;
    for(int i=threadIdx.x/32;i<nr;i+=blockDim.x/32){
      const int base=((low[0]+i/ny)*field.y+(low[1]+i%ny))*field.x+low[2];
      const int stop=base+nx;
      for(int j=base&~31;j<stop;j+=32){
        const int h=__ldg(reinterpret_cast<const int*>(phase.codes[c])+(j>>5));
        const int lane=threadIdx.x&31;
        if(h<0){if(!lane&&__float_as_uint(__ldg(phase.tables[c]+~h))!=__float_as_uint(first))mismatch=true;}
        else if(j+lane>=base&&j+lane<stop&&__float_as_uint(__ldg(phase.tables[c]+h+lane))!=__float_as_uint(first))mismatch=true;
      }
    }
    if(__syncthreads_or(mismatch))return;
    if(!threadIdx.x)entry.electric[c]=first;
  }
  if(!threadIdx.x)entry.flags&=~Mixed;
}
// Check both symmetries over the complete staggered dependency footprint.
// Each proof is independent and compares the exact IEEE bits of every component.
__device__ void TraceMaterialAxes(const Parameters& p,const Tiles& tiles,TileTrace* entries){
  auto& entry=entries[blockIdx.x];
  if(!(entry.flags&Mixed)||entry.flags&(Source|Observation))return;
  const auto tile=Locate(tiles,blockIdx.x);
  const int y0=tile.box.lo[1]+tile.y*24-Steps,x0=tile.box.lo[2]+tile.x*24-Steps;
  const int start=tile.box.lo[0]+tile.z*ZPlanes,first=start-2*Steps;
  const int end=min(tile.box.hi[0],start+ZPlanes)+Steps;
  bool columns=true,planes=true;
  for(int c=0;c<3&&(columns||planes);++c){
    const auto& field=native::FinalField(p,c+3);const auto& ph=p.phase[1];
    const auto value=[&]__device__(int j){
      const int h=__ldg(reinterpret_cast<const int*>(ph.codes[c])+(j>>5));
      return __float_as_uint(__ldg(ph.tables[c]+(h<0?~h:h+(j&31))));
    };
    bool varying_z=false,varying_xy=false;
    const int pitch=field.x*field.y,anchor=min(y0,p.physical_n[1]-1)*field.x+min(x0,p.physical_n[2]-1);
    for(int t=threadIdx.x;t<1024;t+=blockDim.x){
      const int x=min(x0+t%32,p.physical_n[2]-1),y=min(y0+t/32,p.physical_n[1]-1);
      const int line=y*field.x+x;const unsigned base=value(first*pitch+line);
      for(int z=first;z<end&&((columns&&!varying_z)||(planes&&!varying_xy));++z){
        const unsigned v=value(z*pitch+line);
        varying_z|=(v!=base);if(planes&&!varying_xy)varying_xy=(v!=value(z*pitch+anchor));
      }
    }
    columns&=!__syncthreads_or(varying_z);planes&=!__syncthreads_or(varying_xy);
  }
  if(!threadIdx.x)entry.flags|=(columns?ColumnInvariant:0)|(planes?PlaneInvariant:0);
}
template<bool Shell> __global__ void Classify(const __grid_constant__ Parameters p,
    const __grid_constant__ Tiles tiles,TileTrace* trace,int2 source_z){
  if(threadIdx.x<32)ClassifyEffects<Shell>(p,tiles,trace,source_z);
  __syncthreads();
  TraceMaterial<Shell>(p,tiles,trace);
  if constexpr(!Shell){__syncthreads();TraceMaterialAxes(p,tiles,trace);}
}

struct Three { float v[3]; };
// Each active derivative axis owns two histories per field family. On an
// edge the third axis stays in the ordinary curl and consumes no history slots.
template<int Axes> inline constexpr int ActiveAxes =
    Axes < 0 ? 3 : (Axes & 1) + ((Axes >> 1) & 1) + ((Axes >> 2) & 1);
template<int Axes> struct History { float v[Axes == 0 ? 1 : 4 * ActiveAxes<Axes>]; };

// Adapt the canonical cell trace to register-resident CPML histories. Standard
// CPML profiles come from the existing cache. Native Finish owns all additive
// sources, sparse scatters, DFT publication and volume recorder semantics.
template<int Axes,bool Core,bool Effects,bool Uniform>
struct WindowEnv : native::Environment<native::KernelFacts<2,true>,false,true,5,false,Axes,true> {
  using Base=native::Environment<native::KernelFacts<2,true>,false,true,5,false,Axes,true>;
  using IO=native::Environment<native::KernelFacts<2,true>,true,true,4,false,0,false,false,true>;
  static constexpr bool Packed=false;
  bool common; const float* electric; float* history; int sample_y,sample_x;
  __device__ __forceinline__ static int Key(int slot) {
    if constexpr(Axes<=0) return slot;
    int n=(slot/6)*2*ActiveAxes<Axes>;
    #pragma unroll
    for(int t=0;t<6;++t) if(t<slot%6 && (Axes&(1<<yee::CpmlAxis(t)))) ++n;
    return n;
  }
  __device__ __forceinline__ bool Valid(int field,int z,int y,int x) const {
    return Core || (native::NativeSupportMask<Axes>(this->p,z,y,x)&(1u<<field));
  }
  __device__ __forceinline__ bool LocalActive(int j,int z,int y,int x) const {
    if constexpr(Axes==0) return false;
    return Base::LocalActive(j,z,y,x);
  }
  __device__ __forceinline__ float LoadLocal(int j,int,int,int)const{return history[Key(j)];}
  __device__ __forceinline__ void StoreLocal(int j,int,int,int,float v)const{history[Key(j)]=v;}
  __device__ __forceinline__ float Coefficient(int kind,int idx,int z,int y,int x)const {
    if(kind==1 && idx>=3 && (Uniform||common)) return electric[idx-3];
    return Base::Coefficient(kind,idx,z,Core?sample_y:y,Core?sample_x:x);
  }
  __device__ __forceinline__ float Finish(int f,int time,int z,int y,int x,float value)const {
    if constexpr(Effects) return IO{this->p,this->flags}.Finish(f,time,z,y,x,value);
    return value;
  }
};

template<bool H,int Y,bool Support,class Env>
__device__ __forceinline__ Three Phase(Env& env,Three own,Three f,Three z_neighbor,float* frame,
    int lane,int row,int time,int z,int y,int x) {
  constexpr int D=H?1:-1, family=H?0:1;
  const int tid=row*32+lane, yy=((Y&(Y-1))?min(Y-1,max(0,row+D)):((row+D)&(Y-1)))*32+lane,xx=(lane+D)&31;
  frame[tid]=f.v[0]; frame[Y*32+tid]=f.v[2];
  __syncthreads();
  const float yx=frame[yy],yz=frame[Y*32+yy];
  const float xy=__shfl_sync(0xffffffffu,f.v[1],xx),xz=__shfl_sync(0xffffffffu,f.v[2],xx);
  const auto read=[&]__device__(int field,int,int dz,int dy,int dx) {
    int c=field%3;
    if((field<3)==H) return own.v[c];
    if(dz) return z_neighbor.v[c];
    if(dy) return c==0?yx:yz;
    if(dx) return c==1?xy:xz;
    return f.v[c];
  };
  if constexpr(Env::Packed) {
    // The interior has no CPML, sources or observations. The generated term
    // ordering and shared FP32 primitives are identical to the ordinary curl.
    const auto difference=[&]__device__(yee::CurlTerm term) {
      int c=term.source_component,a=term.derivative_axis;
      float next=read((H?3:0)+c,0,(a==0)*D,(a==1)*D,(a==2)*D);
      return yee::ScaleYeeDifference(H?__fsub_rn(next,f.v[c]):__fsub_rn(f.v[c],next),env.inv[family]);
    };
    Three out;
    #pragma unroll
    for(int c=0;c<3;++c) {
      float curl=__fsub_rn(difference(yee::FirstCurlTerm(c)),difference(yee::SecondCurlTerm(c)));
      out.v[c]=yee::AdvanceYeeField<true>(family,own.v[c],1.f,H?env.h:env.e.v[c],curl);
    }
    return out;
  } else return {{trace_native::AdvanceNativeCell<family,0,Support,true>(env,read,time,z,y,x),
           trace_native::AdvanceNativeCell<family,1,Support,true>(env,read,time,z,y,x),
           trace_native::AdvanceNativeCell<family,2,Support,true>(env,read,time,z,y,x)}};
}

struct VolumeClock {
  const int* step;
  const moving::VolumeRecording* records;
  int count, offset;

  __device__ __forceinline__ unsigned ActiveEffects(unsigned flags) const {
    if(!(flags&Volume) || (flags&Observation)) return flags;
    const int first=*step+offset, last=first+Steps;
    for(int i=0;i<count;++i) {
      const int interval=records[i].interval, frame=first/interval;
      if(frame<last/interval && frame<records[i].capacity) return flags|Observation;
    }
    return flags;
  }
};
// A full-volume recorder touches every tile only in windows that end a frame.
// The device clock follows the resident checkpoint across captured launches.
__host__ __device__ inline VolumeClock MakeVolumeClock(const Parameters& p,int phase) {
  return {p.current_step,p.volumes,p.volume_offsets[6],p.macro_step+phase};
}
struct BulkGeometry {
 VolumeClock clock;
 const float* input[6];float* output[6];
 const float* table[3];const int* code[3];
 float common[3],hscale,hm,em;int nx,ny;
};
inline bool CompactLayout(const Parameters&p,BulkGeometry&g){
 g.nx=p.physical_n[2];g.ny=p.physical_n[1];g.hm=p.phase[0].inv;g.em=p.phase[1].inv;g.hscale=p.h_source;
 for(int c=0;c<6;++c){
  const auto&in=native::InitialField(p,c);const auto&out=native::FinalField(p,c);
  if(in.row!=in.x||in.plane!=in.y*in.x||out.row!=in.row||out.plane!=in.plane)return false;
  g.input[c]=in.data;g.output[c]=out.data;
  if(c<3){g.table[c]=p.phase[1].tables[c];g.code[c]=reinterpret_cast<const int*>(p.phase[1].codes[c]);g.common[c]=p.common_e[c];}
 }
 return true;
}
// Six staggered arrays share four physical offsets. The prepared descriptor
// proves these pitches before the compact path can use them.
struct NativeIndices {
 int i[4];
 __device__ static __forceinline__ NativeIndices at(int z,int y,int x,int ny,int nx){
  const int line=z*ny+y,base=line*nx+x;
  return {{base,base+line,base+z*nx,base+line+z*(nx+1)}};
 }
 template<int C>__device__ __forceinline__ int field()const{
  return i[C==0||C==4?1:C==1||C==3?2:C==2?0:3];
 }
};
struct InteriorCoefficients {
 static constexpr bool Packed=true;
 float h,inv[2]; Three e;
};
template<int C>__device__ __forceinline__ float material(const BulkGeometry&g,const NativeIndices&i){
 const int j=i.template field<C+3>(),h=__ldg(g.code[C]+(j>>5));return __ldg(g.table[C]+(h<0?~h:h+(j&31)));
}

template<int Y,int Axes,bool Core,bool Effects,bool Uniform,bool Planar=false,class Fields>
__device__ __forceinline__ void Window(const Fields&p,Tile t,unsigned flags,int phase,
    float (*frame)[2*Y*32],const float* profiles,float* shared_history,const float* electric) {
  int lane=threadIdx.x&31,row=threadIdx.x>>5;
  int x=t.box.lo[2]+t.x*24-Steps+lane,y=t.box.lo[1]+t.y*(Y-2*Steps)-Steps+row;
  int start=t.box.lo[0]+t.z*ZPlanes,end=min(start+ZPlanes,t.box.hi[0]);
  constexpr bool compact=std::is_same_v<Fields,BulkGeometry>;
  int sx=x,sy=y;
  if constexpr(compact) {sx=min(x,p.nx-1);sy=min(y,p.ny-1);}
  else if constexpr(Core) {sx=min(x,p.physical_n[2]-1);sy=min(y,p.physical_n[1]-1);}

  const bool owner=lane>=Steps&&lane<32-Steps&&row>=Steps&&row<Y-Steps&&x<t.box.hi[2]&&y<t.box.hi[1];
  auto env=[&](){
    if constexpr(compact) return InteriorCoefficients{p.hscale,{p.hm,p.em},{}};
    else return WindowEnv<Axes,Core,Effects,Uniform>{{p,uint8_t(flags&31),profiles},!(flags&Mixed),electric,nullptr,sy,sx};
  }();

  if constexpr(compact&&Uniform){
    if(flags&ColumnInvariant){const auto i=NativeIndices::at(start,sy,sx,p.ny,p.nx);
      env.e={{material<0>(p,i),material<1>(p,i),material<2>(p,i)}};}
    else env.e={{electric[0],electric[1],electric[2]}};
  }
  // The classifier proved equality across this whole footprint on each z plane.
  if constexpr(Planar){
    const int i=threadIdx.x;
    if(i<end-start+3*Steps){
      const auto at=NativeIndices::at(start-2*Steps+i,
          min(t.box.lo[1]+t.y*(Y-2*Steps)-Steps,p.ny-1),
          min(t.box.lo[2]+t.x*24-Steps,p.nx-1),p.ny,p.nx);
      shared_history[3*i]=material<0>(p,at);shared_history[3*i+1]=material<1>(p,at);
      shared_history[3*i+2]=material<2>(p,at);
    }
    __syncthreads();
  }
  Three e[Steps]{},h[Steps+1]{};
  History<Axes> saved[Steps]{};
  // Keep the longest-lived CPML terms in shared memory, indexed by their
  // owning lane. The remaining histories stay in the register pipeline.
  constexpr int spilled=Axes<0?4:2;
  if constexpr(Axes!=0) {
    #pragma unroll
    for(int i=0;i<Steps*spilled;++i) shared_history[i*Y*32+threadIdx.x]=0.f;
  }

  for(int z=start-Steps;z<end+Steps;++z) {
    Three en,hn; History<Axes> next{};
    if constexpr(compact) {
      const auto i=NativeIndices::at(z,sy,sx,p.ny,p.nx);
      hn={{p.input[0][i.template field<0>()],p.input[1][i.template field<1>()],p.input[2][i.template field<2>()]}};
      en={{p.input[3][i.template field<3>()],p.input[4][i.template field<4>()],p.input[5][i.template field<5>()]}};
    } else {
      #pragma unroll
      for(int c=0;c<6;++c) {
        const auto& f=native::InitialField(p,c);
        float val;
        if constexpr(Core) val=__ldcg(f.data+z*f.plane+sy*f.row+sx);
        else val=native::Initial<Axes>(p,c,z,y,x);
        if(c<3) hn.v[c]=val; else en.v[c-3]=val;
      }
    }
    if constexpr(Axes!=0) {
      #pragma unroll
      for(int j=0;j<12;++j) if(env.Valid(j/2,z,y,x)&&env.LocalActive(j,z,y,x))
        next.v[env.Key(j)]=env.Base::LoadLocal(j,z,y,x);
    }
    #pragma unroll
    for(int step=0;step<Steps;++step) {
      const int target=z-step-1,time=phase+step+1;
      History<Axes> state{};
      if constexpr(Axes!=0) {
        #pragma unroll
        for(int j=0;j<int(sizeof(state.v)/sizeof(float));++j)
          state.v[j]=j<spilled?shared_history[(step*spilled+j)*Y*32+threadIdx.x]:saved[step].v[j];
      }
      if constexpr(compact) {
        if constexpr(Planar){ int i=3*(target-start+2*Steps);env.e={{shared_history[i],shared_history[i+1],shared_history[i+2]}};}
        else if constexpr(!Uniform) {const auto i=NativeIndices::at(target,sy,sx,p.ny,p.nx);
          env.e={{material<0>(p,i),material<1>(p,i),material<2>(p,i)}};}
      }else env.history=state.v;
      if constexpr(Effects){
        const unsigned width=(flags>>SourceWidthShift)&3u, first=(flags>>SourceStartShift)&SourceResidueMask;
        const bool current=!width || ((unsigned(target)-first)&SourceResidueMask)<width;
        env.flags=uint8_t((flags&(62^Source))|(current?(flags&Source):0)|((flags&Observation)&&owner&&target>=start&&target<end));
      }
      // The low E plane drives both curls. The high E plane supplies the
      // forward Z difference. Pipeline registers retain the uninjected values.
      auto local_e=e[step], upper_e=en;
      if constexpr(Effects) if(p.pre_e_mask && (env.flags&Source)) {
        typename WindowEnv<Axes,Core,Effects,Uniform>::IO io{p,uint8_t(flags)};
        #pragma unroll
        for(int c=0;c<3;++c) {
          local_e.v[c]=io.PreElectric(3+c,time,target,y,x,local_e.v[c]);
          upper_e.v[c]=io.PreElectric(3+c,time,target+1,y,x,upper_e.v[c]);
        }
      }
      auto hnew=Phase<true,Y,!Core>(env,h[step],local_e,upper_e,frame[0],lane,row,time,target,y,x);
      auto enew=Phase<false,Y,!Core>(env,local_e,hnew,h[step+1],frame[1],lane,row,time,target,y,x);
      e[step]=en; h[step]=hn; en=enew; hn=hnew;
      if constexpr(Axes!=0) {
        #pragma unroll
        for(int j=0;j<int(sizeof(state.v)/sizeof(float));++j) {
          if(j<spilled) shared_history[(step*spilled+j)*Y*32+threadIdx.x]=next.v[j];
          else saved[step].v[j]=next.v[j];
        }
        next=state;
      }
    }
    if(z>=start+Steps && owner) {
      const int target=z-Steps;
      if constexpr(compact) {
        const auto i=NativeIndices::at(target,sy,sx,p.ny,p.nx);
        p.output[0][i.template field<0>()]=hn.v[0]; p.output[1][i.template field<1>()]=hn.v[1]; p.output[2][i.template field<2>()]=hn.v[2];
        p.output[3][i.template field<3>()]=en.v[0]; p.output[4][i.template field<4>()]=en.v[1]; p.output[5][i.template field<5>()]=en.v[2];
      } else {
        #pragma unroll
        for(int c=0;c<6;++c) {
          const auto& f=native::FinalField(p,c);
          if(Core || native::Contains<Axes>(f,target,y,x)) f.data[target*f.plane+y*f.row+x]=c<3?hn.v[c]:en.v[c-3];
        }
      }
      if constexpr(Axes!=0) {
        #pragma unroll
        for(int j=0;j<12;++j) if(env.Valid(j/2,target,y,x)&&env.LocalActive(j,target,y,x))
          env.Base::StoreLocal(j,target,y,x,next.v[env.Key(j)]);
      }
    }
    h[Steps]=hn;
  }
}

template<int Y,bool Core,bool Effects,bool Uniform,bool Multi=false>
__global__ __launch_bounds__(32*Y,1) void Advance(const __grid_constant__ Parameters p,
    const __grid_constant__ Tiles tiles,int phase,const TileTrace* mask) {
  constexpr int split=(Core?24:16)/(Y-2*Steps);
  const int tile=blockIdx.x/split,flags=MakeVolumeClock(p,phase).ActiveEffects(mask[tile].flags),axes=(flags>>1)&7;
  if(bool(flags&(Source|Observation))!=Effects) return;
  if constexpr(Core && !Effects) {if(bool(flags&Mixed)==Uniform) return;}
  if constexpr(!Core) {if(bool(axes&(axes-1))!=Multi) return;}
  const float* electric=mask[tile].electric;
  auto t=Locate(tiles,tile);
  t.y=t.y*split+blockIdx.x%split;
  if(t.y*(Y-2*Steps)>=t.box.hi[1]-t.box.lo[1]) return;
  __shared__ float frame[2][2*Y*32];
  if constexpr(Core) Window<Y,0,true,Effects,Uniform>(p,t,flags,phase,frame,nullptr,nullptr,electric);
  else {
    __shared__ float profiles[864],histories[(Multi?4:2)*Steps*Y*32];
    if(axes) {
      for(int i=threadIdx.x;i<864;i+=32*Y) {
        int u=i/24,slot=u/3,kind=u%3;
        if(axes&(1<<yee::CpmlAxis(slot%6))) {
          const auto& psi=p.phase[slot/6].psi[slot%6];
          profiles[i]=__ldg((kind==0?psi.a:kind==1?psi.b:psi.inv)+i%24);
        }
      }
      __syncthreads();
    }
    if constexpr(Multi) {
      switch(axes) {
        case 3: Window<Y,3,false,Effects,false>(p,t,flags,phase,frame,profiles,histories,electric);break;
        case 5: Window<Y,5,false,Effects,false>(p,t,flags,phase,frame,profiles,histories,electric);break;
        case 6: Window<Y,6,false,Effects,false>(p,t,flags,phase,frame,profiles,histories,electric);break;
        default: Window<Y,-1,false,Effects,false>(p,t,flags,phase,frame,profiles,histories,electric);break;
      }
    }
    else switch(axes) {
      case 0: Window<Y,0,false,Effects,false>(p,t,flags,phase,frame,profiles,histories,electric);break;
      case 1: Window<Y,1,false,Effects,false>(p,t,flags,phase,frame,profiles,histories,electric);break;
      case 2: Window<Y,2,false,Effects,false>(p,t,flags,phase,frame,profiles,histories,electric);break;
      default: Window<Y,4,false,Effects,false>(p,t,flags,phase,frame,profiles,histories,electric);break;
    }
  }
}

inline cudaError_t Prepare(cudaStream_t stream,const Parameters&p,const Band& b,uint8_t* plan_masks) {
  auto* masks=reinterpret_cast<TileTrace*>(plan_masks+b.offset);
  const int2 source_z=make_int2(b.source_z[0],b.source_z[1]);
  if(b.core.size()) Classify<false><<<b.core.size(),128,0,stream>>>(p,b.core,masks,source_z);
  if(b.shell.size()) Classify<true><<<b.shell.size(),128,0,stream>>>(p,b.shell,masks+b.core.size(),source_z);
  return cudaGetLastError();
}

template<bool Uniform>
__global__ __launch_bounds__(1024) void BulkRoll(const __grid_constant__ BulkGeometry g,
    Box box,const TileTrace* mask) {
  const int id=(blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x, flags=g.clock.ActiveEffects(mask[id].flags);
  if((flags&(Source|Observation)) || bool((flags&Mixed)&&!(flags&ColumnInvariant))==Uniform) return;
  __shared__ float frame[2][2*32*32];
  const float* electric=mask[id].electric;
  Tile t{box,int(blockIdx.x),int(blockIdx.y),int(blockIdx.z)};
  if constexpr(!Uniform){
    if(flags&PlaneInvariant){
      __shared__ float profile[3*(ZPlanes+3*Steps)];
      Window<32,0,true,false,false,true>(g,t,flags,0,frame,nullptr,profile,electric);
      return;
    }
  }
  Window<32,0,true,false,Uniform>(g,t,flags,0,frame,nullptr,nullptr,electric);
}

inline cudaError_t Launch(cudaStream_t stream,const Parameters&p,const Band& b,int phase,const uint8_t* plan_masks) {
  const int c=b.core.size(),s=b.shell.size(); const auto* m=reinterpret_cast<const TileTrace*>(plan_masks+b.offset);
  if(c) {
    BulkGeometry geom;
    if(CompactLayout(p,geom)) {
      geom.clock=MakeVolumeClock(p,phase);
      const auto& box=b.core.boxes[0];
      dim3 grid(b.core.nx[0],b.core.ny[0],1+(box.hi[0]-box.lo[0]-1)/ZPlanes);
      BulkRoll<true><<<grid,1024,0,stream>>>(geom,box,m);
      BulkRoll<false><<<grid,1024,0,stream>>>(geom,box,m);
    } else {
      Advance<32,true,false,true><<<c,1024,0,stream>>>(p,b.core,phase,m);
      Advance<32,true,false,false><<<c,1024,0,stream>>>(p,b.core,phase,m);
    }
    Advance<16,true,true,false><<<c*3,512,0,stream>>>(p,b.core,phase,m);
  }
  if(s) {
    m+=c;
    Advance<24,false,false,false><<<s,768,0,stream>>>(p,b.shell,phase,m);
    Advance<16,false,true,false><<<s*2,512,0,stream>>>(p,b.shell,phase,m);
    Advance<16,false,false,false,true><<<s*2,512,0,stream>>>(p,b.shell,phase,m);
    Advance<16,false,true,false,true><<<s*2,512,0,stream>>>(p,b.shell,phase,m);
  }
  return cudaGetLastError();
}
#endif
}  // namespace beamz::cuda::rolling
