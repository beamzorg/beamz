// XY-diagonal register ownership follows the(-1,-1,-1) temporal shift.
struct DiagonalPoint{int z,y,x;};
template<int T>__host__ __device__ __forceinline__ DiagonalPoint DiagonalComputePoint(int thread,int j){
 constexpr int LocalZ=T==1024?4:8;const int x=thread&31,u=(thread/32)&15;
 return {(T==1024?thread/512:0)*LocalZ+j,(x+u)&15,x};
}
__host__ __device__ __forceinline__ int DiagonalSlot(int z,int y,int x){return (z*CY+y)*CX+((x+2*y+z)&31);}
__host__ __device__ __forceinline__ int DiagonalBoundarySlot(int y,int x){return y*CX+((x+2*y)&31);}
template<int T>constexpr size_t DiagonalSharedBytes=((3ULL*PAD+(T==1024?6ULL*CY*CX:0))*sizeof(float)+8*sizeof(size_t)+2*sizeof(int)+127)&~size_t(127);
static_assert(DiagonalSharedBytes<1024> == 61568&&DiagonalSharedBytes<512> == 49280&&DiagonalSharedBytes<1024><64512,"Retain64KiBcarveout including SM86reserved1KiB");
__device__ __forceinline__ float DiagonalShuffle(float value,int sx){return __shfl_sync(0xffffffffu,value,(sx+31)&31);}
__host__ __device__ __forceinline__ DiagonalPoint DiagonalGhostLogical(int i){
 if(i<CY*CX)return {0,i/CX,i%CX};i-=CY*CX;
 if(i<BZ*CX)return {1+i/CX,0,i%CX};i-=BZ*CX;return {1+i/BY,1+i%BY,0};
}
__host__ __device__ __forceinline__ DiagonalPoint DiagonalSurfaceLogical(int i){
 if(i<BY*BX)return {BZ,1+i/BX,1+i%BX};i-=BY*BX;
 if(i<(BZ-1)*BX)return {1+i/BX,BY,1+i%BX};i-=(BZ-1)*BX;return {1+i/(BY-1),1+i%(BY-1),BX};
}
template<int T>__device__ __forceinline__ void DiagonalStoreFamily(float*frame,float*boundary,int c,int z,int y,int x,float value){
 frame[c*PAD+DiagonalSlot(z,y,x)]=value;
 if constexpr(T==1024){if(z==3)boundary[c*CY*CX+DiagonalBoundarySlot(y,x)]=value;}
}
template<int Depth,int Layout,int T,int Phase>
__device__ __forceinline__ void DiagonalInitializeFamily(const Parameters&p,float*frame,float*boundary,const size_t*bases,int first,int oz,int oy,int ox){
 for(int i=threadIdx.x;i<PAD;i+=T){const int zs=i/(CY*CX),sy=(i/CX)&(CY-1),sx=i&(CX-1);FieldPacket v{};
  if(first==1){const int z=oz-first+zs,y=oy-first+sy,x=ox-first+sx;
   if constexpr(Phase==0)v={Initial(p,0,z,y,x),Initial(p,1,z,y,x),Initial(p,2,z,y,x),0,0,0};
   else v={0,0,0,Initial(p,3,z,y,x),Initial(p,4,z,y,x),Initial(p,5,z,y,x)};
  }else if(zs==0||sy==0||sx==0)v=ImportPacket<Depth,false,(Layout==1)>(p,bases,first-1,oz,oy,ox,zs,sy,sx);
  if constexpr(Phase==0){DiagonalStoreFamily<T>(frame,boundary,0,zs,sy,sx,v.hx);DiagonalStoreFamily<T>(frame,boundary,1,zs,sy,sx,v.hy);DiagonalStoreFamily<T>(frame,boundary,2,zs,sy,sx,v.hz);}
  else{DiagonalStoreFamily<T>(frame,boundary,0,zs,sy,sx,v.ex);DiagonalStoreFamily<T>(frame,boundary,1,zs,sy,sx,v.ey);DiagonalStoreFamily<T>(frame,boundary,2,zs,sy,sx,v.ez);}
 }
}
template<int Depth,int Layout,int T,int Phase>
__device__ __forceinline__ void DiagonalImportGhost(const Parameters&p,float*frame,float*boundary,const size_t*bases,int n,int oz,int oy,int ox){
 for(int i=threadIdx.x;i<PAD-BZ*BY*BX;i+=T){const auto d=DiagonalGhostLogical(i);const auto v=ImportPacket<Depth,false,(Layout==1)>(p,bases,n,oz,oy,ox,d.z,d.y,d.x);
  if constexpr(Phase==0){DiagonalStoreFamily<T>(frame,boundary,0,d.z,d.y,d.x,v.hx);DiagonalStoreFamily<T>(frame,boundary,1,d.z,d.y,d.x,v.hy);DiagonalStoreFamily<T>(frame,boundary,2,d.z,d.y,d.x,v.hz);}
  else{DiagonalStoreFamily<T>(frame,boundary,0,d.z,d.y,d.x,v.ex);DiagonalStoreFamily<T>(frame,boundary,1,d.z,d.y,d.x,v.ey);DiagonalStoreFamily<T>(frame,boundary,2,d.z,d.y,d.x,v.ez);}
 }
}
template<int Depth,int Layout,int T,int Phase>
__device__ __forceinline__ void DiagonalExportFamily(const Parameters&p,const float*frame,const size_t*bases,int n,int oz,int oy,int ox){
 if(Layout!=1||n<Depth){
  for(int i=threadIdx.x;i<SurfacePoints;i+=T){const auto d=DiagonalSurfaceLogical(i);const int slot=DiagonalSlot(d.z,d.y,d.x);
   #pragma unroll
   for(int c=0;c<3;++c)p.trails[bases[0]+((n-1)*6+Phase*3+c)*SurfacePoints+i]=frame[c*PAD+slot];
  }
 }
 if(n==Depth){
  for(int i=threadIdx.x;i<BZ*BY*BX;i+=T){const int zs=1+i/(BY*BX),sy=1+(i/BX)%BY,sx=1+i%BX,z=oz-n-1+zs,y=oy-n-1+sy,x=ox-n-1+sx,slot=DiagonalSlot(zs,sy,sx);
   #pragma unroll
   for(int c=0;c<3;++c){const auto&f=FinalField(p,Phase*3+c);if(Contains(f,z,y,x))f.data[z*f.plane+y*f.row+x]=frame[c*PAD+slot];}
  }
 }
}

template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int T>
__device__ __forceinline__ void DiagonalLogicalBody(const Parameters&p,const int32_t*tiles){
 static_assert(Layout==1&&IsotropicH&&UniformE&&(Policy==0||Policy==2)&&(T==512||T==1024),"First diagonal scope is certified uniform CPML/source-free bulk/observers");
 constexpr int Slots=T==1024?4:8,BoundaryWords=T==1024?6*CY*CX:0;
 extern __shared__ float frame[];float*boundary_h=frame+3*PAD,*boundary_e=boundary_h+3*CY*CX;
 size_t*bases=reinterpret_cast<size_t*>(frame+3*PAD+BoundaryWords);volatile int*window=reinterpret_cast<volatile int*>(bases+8);
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
 if(first>1)__syncthreads();
 const int sx=threadIdx.x&31,u=(threadIdx.x/32)&15,sy=(sx+u)&15,zbase=T==1024?(threadIdx.x/512)*4:0;
 float h[3][Slots],e[3][Slots];
 DiagonalInitializeFamily<Depth,Layout,T,0>(p,frame,boundary_h,bases,first,oz,oy,ox);__syncthreads();
 #pragma unroll
 for(int j=0;j<Slots;++j){const int slot=DiagonalSlot(zbase+j,sy,sx);
  #pragma unroll
  for(int c=0;c<3;++c)h[c][j]=frame[c*PAD+slot];
 }
 __syncthreads();
 DiagonalInitializeFamily<Depth,Layout,T,1>(p,frame,boundary_e,bases,first,oz,oy,ox);__syncthreads();
 #pragma unroll
 for(int j=0;j<Slots;++j){const int slot=DiagonalSlot(zbase+j,sy,sx);
  #pragma unroll
  for(int c=0;c<3;++c)e[c][j]=frame[c*PAD+slot];
 }
 const uint8_t flags=Policy==0?0:1;
 Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,-1,false>env{p,flags,nullptr};
 #pragma unroll 1
 for(int n=first;n<=last;++n){
  #pragma unroll
  for(int j=Slots-1;j>=0;--j){const int zs=zbase+j,z=oz-n-1+zs,y=oy-n-1+sy,x=ox-n-1+sx;
   // BEAMZ_DIAGONAL_H_COMPONENTS
  }
  __syncthreads(); // Retire old E/shared and old H boundary readers, plus prior E IO.
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j;if(zs>0&&sy>0&&sx>0){
   #pragma unroll
   for(int c=0;c<3;++c)DiagonalStoreFamily<T>(frame,boundary_h,c,zs,sy,sx,h[c][j]);
  }}
  DiagonalImportGhost<Depth,Layout,T,0>(p,frame,boundary_h,bases,n,oz,oy,ox);__syncthreads();
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j;if(zs==0||sy==0||sx==0){
   #pragma unroll
   for(int c=0;c<3;++c)h[c][j]=frame[c*PAD+DiagonalSlot(zs,sy,sx)];
  }}
  DiagonalExportFamily<Depth,Layout,T,0>(p,frame,bases,n,oz,oy,ox);
  #pragma unroll
  for(int j=Slots-1;j>=0;--j){const int zs=zbase+j,z=oz-n-1+zs,y=oy-n-1+sy,x=ox-n-1+sx;
   // BEAMZ_DIAGONAL_E_COMPONENTS
  }
  __syncthreads(); // Retire current H/shared, old E boundary and H IO readers.
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j;if(zs>0&&sy>0&&sx>0){
   #pragma unroll
   for(int c=0;c<3;++c)DiagonalStoreFamily<T>(frame,boundary_e,c,zs,sy,sx,e[c][j]);
  }}
  if(n<last)DiagonalImportGhost<Depth,Layout,T,1>(p,frame,boundary_e,bases,n,oz,oy,ox);__syncthreads();
  if(n<last){
   #pragma unroll
   for(int j=0;j<Slots;++j){const int zs=zbase+j;if(zs==0||sy==0||sx==0){
    #pragma unroll
    for(int c=0;c<3;++c)e[c][j]=frame[c*PAD+DiagonalSlot(zs,sy,sx)];
   }}
  }
  DiagonalExportFamily<Depth,Layout,T,1>(p,frame,bases,n,oz,oy,ox);
 }
}
