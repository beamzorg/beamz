// Research source-free packed-material D4 interior. Included after mstep.cuh.
// CPML, sources and observations remain chronological ordinary edge passes.
#pragma once
namespace beamz_rolling_d4 {
template<int X,int Y>struct Tile {
 static constexpr int wx=X+8,wy=Y+8,area=wx*wy,threads=((area+31)/32)*32;
 static constexpr int stage_floats=3*((X+7)*(Y+7)+(X+6)*(Y+6)+(X+5)*(Y+5)+(X+4)*(Y+4)+(X+3)*(Y+3)+(X+2)*(Y+2)+(X+1)*(Y+1));
 static constexpr int shared_bytes=(stage_floats+6*area)*sizeof(float);
};
template<int X,int Y,int K>__device__ __forceinline__ float&Stage(float*shared,int c,int y,int x){
 constexpr int low=(K+1)/2,sx=X+7-K,sy=Y+7-K;
 constexpr int base=[](){int value=0;for(int k=0;k<K;++k)value+=3*(X+7-k)*(Y+7-k);return value;}();
 return shared[base+(c*sy+y-low)*sx+x-low];
}
template<int X,int Y>__device__ __forceinline__ float&Input(float*shared,int c,int z,int y,int x){
 return shared[Tile<X,Y>::stage_floats+((c*2+(z&1))*Tile<X,Y>::wy+y)*Tile<X,Y>::wx+x];
}
template<int X,int Y,int K>__device__ __forceinline__ bool Active(int qx,int qy){
 constexpr int low=(K+1)/2;
 return qx>=low&&qy>=low&&qx<low+X+7-K&&qy<low+Y+7-K;
}
// H_k consumes the retired same-point H_(k-1), current E_(k-1) at z+1,
// and the prior-plane E_(k-1) frame transversely. The old H_k is retired
// before its shared frame is overwritten, for E_k's backward-z dependency.
template<int X,int Y,int K>
__device__ __forceinline__ void H(float*shared,const MstepPackedCoreView&v,
 const MstepHScalars&hc,int qx,int qy,int p,int wave,const int(&h_offset)[3],
 float(&old_h)[3],float(&current_h)[3],float(&old_e)[3],const float(&current_e)[3]){
 if(wave>=K&&Active<X,Y,K>(qx,qy)){
  if constexpr(K>0){
#pragma unroll
   for(int c=0;c<3;++c)old_e[c]=Stage<X,Y,K-1>(shared,c,qy,qx);
  }
#pragma unroll
  for(int c=0;c<3;++c){
   const auto other=[&](int component,int dz,int dy,int dx){
    if constexpr(K==0)return Input<X,Y>(shared,component,p+dz,qy+dy,qx+dx);
    else{
     if(dz)return current_e[component];
     return Stage<X,Y,K-1>(shared,component,qy+dy,qx+dx);
    }
   };
   const float old=K==0?v.input[c].data[h_offset[c]]:old_h[c];
   const float next=MstepCellAdvance<0>(c,old,hc.decay[c],hc.source[c],v.inv,other);
   const float retired=wave>K?Stage<X,Y,K>(shared,c,qy,qx):0.f;
   old_h[c]=retired;current_h[c]=next;
   if constexpr(K==0)Stage<X,Y,0>(shared,c,qy,qx)=next;
  }
 }
 if constexpr(K==0)__syncthreads();
}
template<int X,int Y,int K,int Layout>
__device__ __forceinline__ void E(float*shared,const MstepPackedCoreView&v,
 int qx,int qy,const int(&e_offset)[3],const int(&h_offset)[3],
 const float(&old_h)[3],const float(&current_h)[3],const float(&old_e)[3],float(&current_e)[3],bool owned){
 if(Active<X,Y,K>(qx,qy)){
#pragma unroll
  for(int c=0;c<3;++c){
   const auto other=[&](int component,int dz,int dy,int dx){
    if(dz)return old_h[component];
    if(dy||dx)return Stage<X,Y,K-1>(shared,component,qy+dy,qx+dx);
    return current_h[component];
   };
   const int logical=e_offset[c]-(K/2)*v.input[3+c].plane;
   const float old=K==1?Input<X,Y>(shared,c,0,qy,qx):old_e[c];
   const float source=MstepMaterialSource<Layout>(v,c,logical);
   const float next=MstepCellAdvance<1>(c,old,1.f,source,v.inv,other);
   if constexpr(K==7){
    if(owned){v.output[3+c].data[logical]=next;v.output[c].data[h_offset[c]-3*v.input[c].plane]=current_h[c];}
   }else current_e[c]=next;
  }
 }
}
// Publish the new E_(k-1)/H_k only after all consumers of the prior E frame
// finish. K=6's existing barrier also certifies the asynchronous E0 prefetch.
template<int X,int Y,int K>__device__ __forceinline__ void Publish(float*shared,
 int qx,int qy,int wave,const float(&current_h)[3],const float(&current_e)[3],bool wait_input){
 if(wave>=K)__syncthreads();
#pragma unroll
 for(int c=0;c<3;++c){
  if(wave>=K-1&&Active<X,Y,K-1>(qx,qy))Stage<X,Y,K-1>(shared,c,qy,qx)=current_e[c];
  if(wave>=K&&Active<X,Y,K>(qx,qy))Stage<X,Y,K>(shared,c,qy,qx)=current_h[c];
 }
 if constexpr(K==6)if(wait_input)__pipeline_wait_prior(0);
 __syncthreads();
}
template<int X,int Y,int Layout=0>
__global__ __launch_bounds__(Tile<X,Y>::threads,(X==16&&Y==16)?2:1)
void Kernel(const MstepPackedCoreView v,const int*masks,const int*bounds,int ntx,int segment){
 using T=Tile<X,Y>;
 const int ox=blockIdx.x*X,oy=blockIdx.y*Y;
 const int shift=((oy%16)/8)*2+(ox%32)/16;
 const int mask=masks[(oy/16)*ntx+ox/32]>>shift;
 if(!mask)return;
 const int zb=bounds[0]+blockIdx.z*segment;
 const int zt=segment?min(bounds[1],zb+segment):bounds[1];
 if(zb>=zt)return;
 const int qx=threadIdx.x%T::wx,qy=threadIdx.x/T::wx;
 const bool valid=threadIdx.x<T::area;
 const int x=min(max(ox-4+qx,0),v.min_x-1),y=min(max(oy-4+qy,0),v.min_y-1);
 const int ix=qx-4,iy=qy-4;
 const bool owned=ix>=0&&ix<X&&iy>=0&&iy<Y&&(mask&(1<<((iy>=8?2:0)+(ix>=16?1:0))));
 MstepHScalars hc;int h_offset[3],e_offset[3];
#pragma unroll
 for(int c=0;c<3;++c){hc.decay[c]=*v.h_decay[c];hc.source[c]=*v.h_source[c];h_offset[c]=MstepCoreOffset(v.input[c],zb-4,y,x);e_offset[c]=MstepCoreOffset(v.input[3+c],zb-4,y,x);}
 extern __shared__ float shared[];
 if(valid){
#pragma unroll
  for(int c=0;c<3;++c){Input<X,Y>(shared,c,zb-4,qy,qx)=v.input[3+c].data[e_offset[c]];Input<X,Y>(shared,c,zb-3,qy,qx)=v.input[3+c].data[e_offset[c]+v.input[3+c].plane];}
 }
 __syncthreads();
 const int waves=zt-zb+7;
 for(int wave=0;wave<waves;++wave){
  const int p=zb+wave-4;
  float old_h[3]{},current_h[3]{},old_e[3]{},current_e[3]{};
  H<X,Y,0>(shared,v,hc,qx,qy,p,wave,h_offset,old_h,current_h,old_e,current_e);
  if(wave>=1){
   if(Active<X,Y,1>(qx,qy)){
#pragma unroll
    for(int c=0;c<3;++c){
     const auto other=[&](int component,int dz,int dy,int dx){if(dz)return old_h[component];if(dy||dx)return Stage<X,Y,0>(shared,component,qy+dy,qx+dx);return current_h[component];};
     current_e[c]=MstepCellAdvance<1>(c,Input<X,Y>(shared,c,p,qy,qx),1.f,MstepMaterialSource<Layout>(v,c,e_offset[c]),v.inv,other);
    }
   }
  }
  if(wave+1<waves){if(valid){
#pragma unroll
   for(int c=0;c<3;++c)__pipeline_memcpy_async(&Input<X,Y>(shared,c,p+2,qy,qx),v.input[3+c].data+e_offset[c]+2*v.input[3+c].plane,sizeof(float));
  }__pipeline_commit();}
  H<X,Y,2>(shared,v,hc,qx,qy,p,wave,h_offset,old_h,current_h,old_e,current_e);
  Publish<X,Y,2>(shared,qx,qy,wave,current_h,current_e,false);
  if(wave>=3)E<X,Y,3,Layout>(shared,v,qx,qy,e_offset,h_offset,old_h,current_h,old_e,current_e,owned);
  H<X,Y,4>(shared,v,hc,qx,qy,p,wave,h_offset,old_h,current_h,old_e,current_e);
  Publish<X,Y,4>(shared,qx,qy,wave,current_h,current_e,false);
  if(wave>=5)E<X,Y,5,Layout>(shared,v,qx,qy,e_offset,h_offset,old_h,current_h,old_e,current_e,owned);
  H<X,Y,6>(shared,v,hc,qx,qy,p,wave,h_offset,old_h,current_h,old_e,current_e);
  Publish<X,Y,6>(shared,qx,qy,wave,current_h,current_e,wave+1<waves);
  if(wave>=7)E<X,Y,7,Layout>(shared,v,qx,qy,e_offset,h_offset,old_h,current_h,old_e,current_e,owned);
#pragma unroll
  for(int c=0;c<3;++c){h_offset[c]+=v.input[c].plane;e_offset[c]+=v.input[3+c].plane;}
 }
}
}
