// Fixed logical cells; previous versions shift by(-1,-1,-1), current H does not.
// Three provably unread E normal-high planes hold all864 profile words.
template<int N,int ActiveAxes>struct FixedCubeCachedEnvironment;
__host__ __device__ __forceinline__ int StaticSlot(int z,int y,int x){
 return (z*CY+y)*CX+(x^(y|((z&1)<<4)));
}
__host__ __device__ __forceinline__ bool StaticFieldLive(int field,int z,int y,int x){
 return field<3||(field==3?x!=BX:field==4?y!=BY:z!=BZ);
}
__host__ __device__ __forceinline__ int StaticProfileSlot(int i){
 if(i<CZ*CY)return 3*PAD+StaticSlot(i/CY,i%CY,BX);
 i-=CZ*CY;if(i<CZ*CX)return 4*PAD+StaticSlot(i/CX,BY,i%CX);
 i-=CZ*CX;return 5*PAD+StaticSlot(BZ,i/CX,i%CX);
}
__host__ __device__ __forceinline__ void StaticStoreField(float*frame,int field,int z,int y,int x,float value){
 if(StaticFieldLive(field,z,y,x))frame[field*PAD+StaticSlot(z,y,x)]=value;
}
template<int Phase>__host__ __device__ __forceinline__ int StaticReadSlot(int field,int z,int y,int x,int dz,int dy,int dx){
 const int shift=Phase==0||field>=3?1:0;
 return field*PAD+StaticSlot(z+dz-shift,y+dy-shift,x+dx-shift);
}
template<int T>__device__ __forceinline__ void StaticStageProfiles(const Parameters&p,float*frame){
 for(int i=threadIdx.x;i<864;i+=T){const int profile=i/24,packed=i%24,slot=profile/3,kind=profile%3;const auto&psi=p.phase[slot/6].psi[slot%6];
  frame[StaticProfileSlot(i)]=__ldg((kind==0?psi.a:kind==1?psi.b:psi.inv)+packed);
 }
}

template<int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int SourceMask,bool PlanePair,int Axes,bool Cached,int CubeN>
struct StaticLogicalEnvironment:Environment<Layout,Fast,IsotropicH,Policy,UniformE,SourceMask,PlanePair,Axes,false>{
 using Base=Environment<Layout,Fast,IsotropicH,Policy,UniformE,SourceMask,PlanePair,Axes,false>;
 static_assert(CubeN==0||(Policy==1&&Layout==1&&IsotropicH&&UniformE&&Cached),"Preserve exact original FixedCube admission");
 __device__ __forceinline__ StaticLogicalEnvironment(const Parameters&p,uint8_t flags,const float*frame):Base{p,flags,frame}{}
 __device__ __forceinline__ bool Valid(int f,int z,int y,int x)const{
  if constexpr(CubeN>0){FixedCubeCachedEnvironment<CubeN,Axes>e{this->p,this->flags,nullptr};return e.Valid(f,z,y,x);}else return Base::Valid(f,z,y,x);
 }
 __device__ __forceinline__ int Packed(int s,int z,int y,int x)const{
  if constexpr(CubeN>0){FixedCubeCachedEnvironment<CubeN,Axes>e{this->p,this->flags,nullptr};return e.Packed(s,z,y,x);}else return Base::Packed(s,z,y,x);
 }
 __device__ __forceinline__ bool LocalActive(int s,int z,int y,int x)const{
  if constexpr(CubeN>0){FixedCubeCachedEnvironment<CubeN,Axes>e{this->p,this->flags,nullptr};return e.LocalActive(s,z,y,x);}else return Base::LocalActive(s,z,y,x);
 }
 __device__ __forceinline__ float LoadLocal(int s,int z,int y,int x)const{
  if constexpr(CubeN>0){FixedCubeCachedEnvironment<CubeN,Axes>e{this->p,this->flags,nullptr};return e.LoadLocal(s,z,y,x);}else return Base::LoadLocal(s,z,y,x);
 }
 __device__ __forceinline__ void StoreLocal(int s,int z,int y,int x,float v)const{
  if constexpr(CubeN>0){FixedCubeCachedEnvironment<CubeN,Axes>e{this->p,this->flags,nullptr};e.StoreLocal(s,z,y,x,v);}else Base::StoreLocal(s,z,y,x,v);
 }
 __device__ __forceinline__ float Difference(int phase,int source,int axis,int z,int y,int x,float center,float adjacent)const{
  if constexpr(CubeN>0){FixedCubeCachedEnvironment<CubeN,Axes>e{this->p,this->flags,nullptr};return e.Difference(phase,source,axis,z,y,x,center,adjacent);}else return Base::Difference(phase,source,axis,z,y,x,center,adjacent);
 }
 __device__ __forceinline__ float Coefficient(int kind,int index,int z,int y,int x)const{
  if constexpr(Cached){if(kind>=2)return this->profiles[StaticProfileSlot((index*3+kind-2)*24+Packed(index,z,y,x))];}
  return Base::Coefficient(kind,index,z,y,x);
 }
};

struct StaticPoint{int z,y,x;};
constexpr int StaticGhostPoints=PAD-BZ*BY*BX;
__host__ __device__ __forceinline__ StaticPoint StaticGhostLogical(int i){
 if(i<CY*CX)return {0,i/CX,i%CX};i-=CY*CX;
 if(i<BZ*CX)return {1+i/CX,0,i%CX};i-=BZ*CX;
 return {1+i/BY,1+i%BY,0};
}
template<int Depth,int Layout,int T,int Phase>
__device__ __forceinline__ void StaticScatterGhost(const Parameters&p,float*frame,const size_t*bases,int n,int oz,int oy,int ox){
 for(int i=threadIdx.x;i<StaticGhostPoints;i+=T){const auto d=StaticGhostLogical(i);const auto v=ImportPacket<Depth,false,(Layout==1)>(p,bases,n,oz,oy,ox,d.z,d.y,d.x);
  if constexpr(Phase==0){StaticStoreField(frame,0,d.z,d.y,d.x,v.hx);StaticStoreField(frame,1,d.z,d.y,d.x,v.hy);StaticStoreField(frame,2,d.z,d.y,d.x,v.hz);}
  else{StaticStoreField(frame,3,d.z,d.y,d.x,v.ex);StaticStoreField(frame,4,d.z,d.y,d.x,v.ey);StaticStoreField(frame,5,d.z,d.y,d.x,v.ez);}
 }
}
template<int Depth,int Layout,int T,int Phase>
__device__ __forceinline__ void StaticPublishOwned(const Parameters&p,float*frame,const size_t*bases,const float(&values)[3][PAD/T],int n,int oz,int oy,int ox){
 constexpr int Slots=PAD/T,ZStep=T/(CY*CX);const int sx=threadIdx.x&(CX-1),sy=(threadIdx.x/CX)&(CY-1),zbase=T==512?0:threadIdx.x/(CY*CX);
 #pragma unroll
 for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;if(zs>0&&sy>0&&sx>0){
  const int z=oz-n-1+zs,y=oy-n-1+sy,x=ox-n-1+sx,surface=Surface(zs-1,sy-1,sx-1);
  #pragma unroll
  for(int c=0;c<3;++c){const int f=3*Phase+c;const float value=values[c][j];StaticStoreField(frame,f,zs,sy,sx,value);
   if(surface>=0&&(Layout!=1||n<Depth))p.trails[bases[0]+((n-1)*6+f)*SurfacePoints+surface]=value;
   if(n==Depth){const auto&field=FinalField(p,f);if(Contains(field,z,y,x))field.data[z*field.plane+y*field.row+x]=value;}
  }
 }}
}

template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int Axes,bool Cached,int CubeN,int SourceMask,bool PlanePair,int T>
__device__ __forceinline__ void StaticLogicalBody(const Parameters&p,const int32_t*tiles){
 static_assert(CZ==8&&CY==16&&CX==32&&(T==512||T==1024),"Fixed native logical geometry and fullwarp mappings");
 static_assert(CZ*CY+CZ*CX+CY*CX>=864,"Dead Eplanes must hold every independent profileword");
 constexpr int Slots=PAD/T,ZStep=T/(CY*CX);
 extern __shared__ float frame[];size_t*bases=reinterpret_cast<size_t*>(frame+6*PAD);volatile int*window=reinterpret_cast<volatile int*>(bases+8);
 const int tile=tiles[blockIdx.x],bz=tile/(p.nx*p.ny),by=(tile/p.nx)%p.ny,bx=tile%p.nx;
 const int oz=bz*BZ,oy=by*BY,ox=bx*BX,rank=bz+by+bx;const uint64_t epoch=uint64_t(*p.epoch)<<32;
 if(!threadIdx.x){const int owner=(rank&3)*p.frontier+p.slots[tile];p.owner_tags[owner]=epoch|uint32_t(tile);
  window[0]=Fast?1:max(1,max(oz-p.phase[0].own[0].z,max(oy-p.phase[0].own[0].y,ox-p.phase[0].own[1].x)));
  window[1]=Fast?Depth:min(Depth,min(oz+BZ-1,min(oy+BY-1,ox+BX-1)));
 }
 __syncthreads();const int first=window[0],last=window[1];if(first>last)return;
 if(threadIdx.x<8){const int bits=threadIdx.x,dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;size_t base=size_t(-1);
  if(bz>=dz&&by>=dy&&bx>=dx){const int producer=tile-dz*p.nx*p.ny-dy*p.nx-dx,owner=((rank-dz-dy-dx)&3)*p.frontier+p.slots[producer];
   if(bits&&p.owner_tags[owner]!=(epoch|uint32_t(producer)))atomicOr(p.bad,1);else base=size_t(owner)*Depth*6*SurfacePoints;
  }bases[bits]=base;
 }
 const int sx=threadIdx.x&(CX-1),sy=(threadIdx.x/CX)&(CY-1),zbase=T==512?0:threadIdx.x/(CY*CX);
 const uint8_t flags=Policy==0?0:Policy==2?1:Policy==3?16:Policy==4?17:p.flags[tile];
 __syncthreads();
 #pragma unroll
 for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;FieldPacket v{};
  if(first==1){const int z=oz-first+zs,y=oy-first+sy,x=ox-first+sx;
   v={Initial(p,0,z,y,x),Initial(p,1,z,y,x),Initial(p,2,z,y,x),Initial(p,3,z,y,x),Initial(p,4,z,y,x),Initial(p,5,z,y,x)};
  }else if(zs==0||sy==0||sx==0)v=ImportPacket<Depth,false,(Layout==1)>(p,bases,first-1,oz,oy,ox,zs,sy,sx);
  StaticStoreField(frame,0,zs,sy,sx,v.hx);StaticStoreField(frame,1,zs,sy,sx,v.hy);StaticStoreField(frame,2,zs,sy,sx,v.hz);
  StaticStoreField(frame,3,zs,sy,sx,v.ex);StaticStoreField(frame,4,zs,sy,sx,v.ey);StaticStoreField(frame,5,zs,sy,sx,v.ez);
 }
 if constexpr(Cached){if(flags&14u)StaticStageProfiles<T>(p,frame);}
 __syncthreads();float result[3][Slots];
 StaticLogicalEnvironment<Layout,Fast,IsotropicH,Policy,UniformE,SourceMask,PlanePair,Axes,Cached,CubeN>env{p,flags,frame};
 #pragma unroll 1
 for(int n=first;n<=last;++n){
  // BEAMZ_STATIC_H_REGION
  __syncthreads(); // All old H centers retire before shared H is replaced.
  StaticPublishOwned<Depth,Layout,T,0>(p,frame,bases,result,n,oz,oy,ox);
  StaticScatterGhost<Depth,Layout,T,0>(p,frame,bases,n,oz,oy,ox);
  __syncthreads(); // Current H, including ghosts, is visible to E.
  // BEAMZ_STATIC_E_REGION
  __syncthreads(); // All old E centers retire before shared E is replaced.
  StaticPublishOwned<Depth,Layout,T,1>(p,frame,bases,result,n,oz,oy,ox);
  if(n<last)StaticScatterGhost<Depth,Layout,T,1>(p,frame,bases,n,oz,oy,ox);
  __syncthreads(); // E and immutable profile holes are ready for the next H.
 }
}
