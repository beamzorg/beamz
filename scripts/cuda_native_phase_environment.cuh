// Canonical physics with phase-local storage. Immutable CPML profiles remain
// global so the full shared field frame plus scheduler permits two512CTAs/SM.
template<int CubeN,int Policy,bool UniformE,int Axes=-1>
struct PhaseEnvironment:Environment<1,(Policy==2||Policy==4),true,Policy,UniformE,(Policy>=3?20:-1),(Policy>=3),Axes,false,true>{
 using Base=Environment<1,(Policy==2||Policy==4),true,Policy,UniformE,(Policy>=3?20:-1),(Policy>=3),Axes,false,true>;
 __device__ __forceinline__ PhaseEnvironment(const Parameters&p,uint8_t flags):Base{p,flags,nullptr}{}
 __device__ __forceinline__ bool Valid(int f,int z,int y,int x)const{
  if constexpr(Policy==5&&CubeN>0){FixedCubeCachedEnvironment<CubeN,Axes,true>e{this->p,this->flags,nullptr};return e.Valid(f,z,y,x);}
  else return Base::Valid(f,z,y,x);
 }
 __device__ __forceinline__ bool LocalActive(int s,int z,int y,int x)const{
  if constexpr(Policy==5&&CubeN>0){FixedCubeCachedEnvironment<CubeN,Axes,true>e{this->p,this->flags,nullptr};return e.LocalActive(s,z,y,x);}
  else return Base::LocalActive(s,z,y,x);
 }
 __device__ __forceinline__ float LoadLocal(int s,int z,int y,int x)const{
  if constexpr(Policy==5&&CubeN>0){FixedCubeCachedEnvironment<CubeN,Axes,true>e{this->p,this->flags,nullptr};return e.LoadLocal(s,z,y,x);}
  else return Base::LoadLocal(s,z,y,x);
 }
 __device__ __forceinline__ void StoreLocal(int s,int z,int y,int x,float v)const{
  if constexpr(Policy==5&&CubeN>0){FixedCubeCachedEnvironment<CubeN,Axes,true>e{this->p,this->flags,nullptr};e.StoreLocal(s,z,y,x,v);}
  else Base::StoreLocal(s,z,y,x,v);
 }
 __device__ __forceinline__ float Difference(int phase,int source,int axis,int z,int y,int x,float center,float adjacent)const{
  if constexpr(Policy==5&&CubeN>0){FixedCubeCachedEnvironment<CubeN,Axes,true>e{this->p,this->flags,nullptr};return e.Difference(phase,source,axis,z,y,x,center,adjacent);}
  else return Base::Difference(phase,source,axis,z,y,x,center,adjacent);
 }
 __device__ __forceinline__ float Coefficient(int kind,int index,int z,int y,int x)const{
  if(kind>=2){int packed;
   if constexpr(Policy==5&&CubeN>0){FixedCubeCachedEnvironment<CubeN,Axes,true>e{this->p,this->flags,nullptr};packed=e.Packed(index,z,y,x);}
   else packed=Base::Packed(index,z,y,x);
   const auto&psi=this->p.phase[index/6].psi[index%6];return kind==2?__ldg(psi.a+packed):kind==3?__ldg(psi.b+packed):__ldg(psi.inv+packed);
  }
  if constexpr(!UniformE){if(kind==1&&index>=3&&(this->flags&64u))return this->p.common_e[index-3];}
  return Base::Coefficient(kind,index,z,y,x);
 }
};
