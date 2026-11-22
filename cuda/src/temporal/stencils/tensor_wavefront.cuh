// Included in moving_native. Tensor curls and fields share one Yee arithmetic API.
constexpr int TensorZ=beamz::cuda::moving::kTensorOwned[0];
constexpr int TensorY=beamz::cuda::moving::kTensorOwned[1];
constexpr int TensorX=beamz::cuda::moving::kTensorOwned[2];
constexpr int TensorCells=TensorZ*TensorY*TensorX;
constexpr int TensorSurface=beamz::cuda::moving::kTensorTrailPoints;

struct TensorMaterial {
 const float* cross;
 const int32_t* origin;
 const float* widths[3];
 int curl_shape[3][3];
 int node_y,node_x;
 float dt_over_eps;
};

__device__ __forceinline__ int TensorSurfaceIndex(int z,int y,int x){
 if(z>=TensorZ-2)return ((z-(TensorZ-2))*TensorY+y)*TensorX+x;
 if(y>=TensorY-2)return 2*TensorY*TensorX+(2*z+y-(TensorY-2))*TensorX+x;
 if(x>=TensorX-2)return (2*TensorY+2*(TensorZ-2))*TensorX+2*(z*(TensorY-2)+y)+x-(TensorX-2);
 return -1;
}

using TensorEnvironment=Environment<KernelFacts<0,false>,false,false,5,false,-1,false,0,true>;

struct CurlEnvironment : TensorEnvironment {
 __device__ float Coefficient(int kind,int index,int z,int y,int x)const{
  return kind<2?float(kind==1):TensorEnvironment::Coefficient(kind,index,z,y,x);
 }
 __device__ float Finish(int,int,int,int,int,float value)const{return value;}
};

template<int Row,int Column,class ReadCurl>
__device__ __forceinline__ float TensorNodeTerm(const TensorMaterial&m,const int*node,ReadCurl read){
 constexpr int axis=2-Column;
 int low[]{node[0],node[1],node[2]},high[]{node[0],node[1],node[2]};
 const int count=m.curl_shape[Column][axis],n=node[axis];
 low[axis]=max(0,n-1);
 high[axis]=min(count-1,n);
 const float a=read(Column,low);
 float value=a;
 if(n>0&&n<count){
  const float b=read(Column,high);
  if(m.widths[axis]){
   const int index=m.origin[axis]+n;
   const float h=m.widths[axis][index-1];
   const float weight=h/(h+m.widths[axis][index]);
   value=weight*a+(1.f-weight)*b;
  }else value=0.5f*(a+b);
 }
 const int index=((node[0]*m.node_y+node[1])*m.node_x+node[2])*3+Row+Column-1;
 return m.cross[index]*value;
}

template<int Component,class Read>
__device__ __forceinline__ float TensorCorrection(const TensorMaterial&m,int z,int y,int x,Read read){
 int node[]{z-m.origin[0],y-m.origin[1],x-m.origin[2]};
 for(int a=0;a<3;++a)if(node[a]<0||node[a]>=m.curl_shape[Component][a])return 0.f;
 const auto read_curl=[&] __device__(int field,const int*point){
  return read(field,point[0]+m.origin[0],point[1]+m.origin[1],point[2]+m.origin[2]);
 };
 constexpr int first=Component==0?1:0,second=Component==2?1:2;
 const float low=TensorNodeTerm<Component,first>(m,node,read_curl)+TensorNodeTerm<Component,second>(m,node,read_curl);
 ++node[2-Component];
 const float high=TensorNodeTerm<Component,first>(m,node,read_curl)+TensorNodeTerm<Component,second>(m,node,read_curl);
 return m.dt_over_eps*(0.5f*(low+high));
}

// The affine coordinates are q+2*t-1 for H and corrected curls, and q+2*t
// for E. All reads go to the same tile or its lower-rank two-cell faces.
template<int Depth>
__global__ __launch_bounds__(256,1) void TensorTile(
 const __grid_constant__ Parameters p,const __grid_constant__ TensorMaterial material,const int32_t*tiles){
 __shared__ float values[15*TensorCells];
 __shared__ size_t bases[8];
 const int encoded=tiles[blockIdx.x],macro_index=MacroIndex(p,encoded),tile=PhysicalTileID(p,encoded);
 const int tile_z=tile/(p.ny*p.nx),tile_y=(tile/p.nx)%p.ny,tile_x=tile%p.nx;
 const int rank=tile_z+tile_y+tile_x+macro_index*p.macro_level_stride;
 const uint64_t epoch=uint64_t(*p.epoch)<<32;
 const int owner=(rank&3)*p.frontier+p.slots[encoded];
 if(!threadIdx.x)p.owner_tags[owner]=epoch|uint32_t(encoded);
 if(threadIdx.x<8){
  const int bits=threadIdx.x,dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
  size_t base=size_t(-1);
  if(tile_z>=dz&&tile_y>=dy&&tile_x>=dx){
   const int producer=encoded-(dz*p.ny+dy)*p.nx-dx;
   const int prior=((rank-dz-dy-dx)&3)*p.frontier+p.slots[producer];
   if(bits&&p.owner_tags[prior]!=(epoch|uint32_t(producer)))atomicOr(p.bad,1);
   else base=size_t(prior)*(9*Depth-3)*TensorSurface;
  }
  bases[bits]=base;
 }
 __syncthreads();
 TensorEnvironment env{p,p.flags[tile]};
 env.step_offset=macro_index*Depth;
 env.raw_step_offset=(macro_index%p.raw_slots)*Depth;
 CurlEnvironment curl{env};
 const auto slot=[] __device__(int phase,int component,int step,int i){
  return (phase==1?12+component:(step&1)*6+(phase==0?0:3)+component)*TensorCells+i;
 };
 const auto get=[&] __device__(int phase,int component,int step,int z,int y,int x){
  const int field=(phase==0?0:3)+component;
  if(!Contains(FinalField(p,field),z,y,x))return 0.f;
  if(step==0)return Initial(p,field,z,y,x,macro_index);
  const int shift=2*step-(phase!=2);
  const int lz=z+shift-tile_z*TensorZ,ly=y+shift-tile_y*TensorY,lx=x+shift-tile_x*TensorX;
  if(lz<-2||lz>=TensorZ||ly<-2||ly>=TensorY||lx<-2||lx>=TensorX){
   atomicOr(p.bad,16);
   return 0.f;
  }
  const int dz=lz<0,dy=ly<0,dx=lx<0;
  const int zz=lz+dz*TensorZ,yy=ly+dy*TensorY,xx=lx+dx*TensorX;
  const int bits=dz|(dy<<1)|(dx<<2);
  if(!bits)return values[slot(phase,component,step,(zz*TensorY+yy)*TensorX+xx)];
  const int surface=TensorSurfaceIndex(zz,yy,xx);
  if(surface<0||bases[bits]==size_t(-1)){
   atomicOr(p.bad,16);
   return 0.f;
  }
  return p.trails[bases[bits]+((step-1)*9+phase*3+component)*TensorSurface+surface];
 };
 #pragma unroll 1
 for(int step=1;step<=Depth;++step){
  const auto update=[&] __device__(auto stage_constant){
   constexpr int stage=decltype(stage_constant)::value;
   constexpr int family=stage==0?0:3;
   const int shift=2*step-(stage!=2);
   for(int i=threadIdx.x;i<TensorCells;i+=blockDim.x){
    const int iz=i/(TensorY*TensorX),iy=i/TensorX%TensorY,ix=i%TensorX;
    const int z=tile_z*TensorZ+iz-shift,y=tile_y*TensorY+iy-shift,x=tile_x*TensorX+ix-shift;
    const int surface=TensorSurfaceIndex(iz,iy,ix);
    const auto read=[&] __device__(int field,int version,int dz,int dy,int dx){
     const int source_phase=field<3?0:2;
     return stage==1&&field>=3?0.f:get(source_phase,field%3,version,z+dz,y+dy,x+dx);
    };
    const auto component=[&] __device__(auto component_constant){
     constexpr int c=decltype(component_constant)::value,field=family+c;
     if(!env.Valid(field,z,y,x))return;
     float value;
     if constexpr(stage==0)value=AdvanceCell<0,c,false>(env,read,step,z,y,x);
     else if constexpr(stage==1)value=beamz::cuda::trace_native::AdvanceNativeCell<1,c,false>(curl,read,step,z,y,x);
     else{
      const float old=env.PreElectric(field,step,z,y,x,get(2,c,step-1,z,y,x));
      const float native_curl=get(1,c,step,z,y,x);
      value=beamz::cuda::yee::AdvanceYeeField(1,old,env.Coefficient(0,field,z,y,x),env.Coefficient(1,field,z,y,x),native_curl);
      const auto read_curl=[&] __device__(int other,int a,int b,int d){return get(1,other,step,a,b,d);};
      value+=TensorCorrection<c>(material,z,y,x,read_curl);
      value=env.Finish(field,step,z,y,x,value);
     }
     values[slot(stage,c,step,i)]=value;
     if(surface>=0&&(stage!=2||step!=Depth))
      p.trails[bases[0]+((step-1)*9+stage*3+c)*TensorSurface+surface]=value;
     if constexpr(stage!=1)if(step==Depth){
      const auto& out=MacroFinalField(p,field,macro_index);
      out.data[z*out.plane+y*out.row+x]=value;
     }
    };
    component(std::integral_constant<int,0>{});
    component(std::integral_constant<int,1>{});
    component(std::integral_constant<int,2>{});
   }
  };
  update(std::integral_constant<int,0>{});
  __syncthreads();
  update(std::integral_constant<int,1>{});
  __syncthreads();
  update(std::integral_constant<int,2>{});
  __syncthreads();
 }
}

inline cudaError_t LaunchTensorRanks(cudaStream_t stream,const Parameters&p,
 const BeamzNativeMovingWorkspace&w,const BeamzLaunch&e){
 if(w.depth!=4||!e.tensor)return cudaErrorInvalidValue;
 const auto&t=*e.tensor;
 TensorMaterial m{};
 m.cross=static_cast<const float*>(t.cross.data);
 m.origin=static_cast<const int32_t*>(t.origin.data);
 m.dt_over_eps=e.dt_over_eps;
 m.node_y=int(t.cross.dims[1]);m.node_x=int(t.cross.dims[2]);
 for(int axis=0;axis<3;++axis){
  m.widths[axis]=t.widths[axis].rank==0?nullptr:static_cast<const float*>(t.widths[axis].data);
  for(int c=0;c<3;++c)m.curl_shape[c][axis]=int(t.curls[c].dims[axis]);
 }
 for(int rank=0;rank<w.rank_count;++rank){
  const auto*row=w.policy_offsets_host+7*rank;
  if(row[0]==row[6])continue;
  TensorTile<4><<<row[6]-row[0],256,0,stream>>>(p,m,w.tile_ids_device+row[0]);
  auto status=cudaGetLastError();
  if(status!=cudaSuccess)return status;
 }
 return cudaSuccess;
}
