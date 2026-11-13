// Physics effects share one register/shared-memory field layout. Specialize
// only each H/E arithmetic region, rather than six entire temporal bodies.
template<class Base>
struct TracedRuntimeCoefficient:Base{
 __device__ __forceinline__ float Coefficient(int kind,int index,int z,int y,int x)const{
  if(kind==1&&index>=3&&(this->flags&64u))return this->p.common_e[index-3];
  return Base::Coefficient(kind,index,z,y,x);
 }
};
template<int Axes,bool Uniform>struct TracedBoundaryBaseSelector{
 using Type=TracedRuntimeCoefficient<Environment<1,false,true,5,false,20,true,Axes,true,true>>;
};
template<int Axes>struct TracedBoundaryBaseSelector<Axes,true>{using Type=Environment<1,false,true,5,true,20,true,Axes,true,true>;};
template<int Axes,bool Uniform>using TracedBoundaryBase=typename TracedBoundaryBaseSelector<Axes,Uniform>::Type;
template<int CubeN,int ActiveAxes=-1,bool ProgramUniform=false>
struct TracedBoundaryEffects:TracedBoundaryBase<ActiveAxes,ProgramUniform>{
 using Base=TracedBoundaryBase<ActiveAxes,ProgramUniform>;
 __device__ __forceinline__ bool Valid(int f,int z,int y,int x)const{
  if constexpr(CubeN>0){FixedCubeCachedEnvironment<CubeN,ActiveAxes,true>e{this->p,this->flags,this->profiles};return e.Valid(f,z,y,x);}
  else return Base::Valid(f,z,y,x);
 }
 __device__ __forceinline__ bool LocalActive(int s,int z,int y,int x)const{
  if constexpr(CubeN>0){FixedCubeCachedEnvironment<CubeN,ActiveAxes,true>e{this->p,this->flags,this->profiles};return e.LocalActive(s,z,y,x);}
  else return Base::LocalActive(s,z,y,x);
 }
 __device__ __forceinline__ float LoadLocal(int s,int z,int y,int x)const{
  if constexpr(CubeN>0){FixedCubeCachedEnvironment<CubeN,ActiveAxes,true>e{this->p,this->flags,this->profiles};return e.LoadLocal(s,z,y,x);}
  else return Base::LoadLocal(s,z,y,x);
 }
 __device__ __forceinline__ void StoreLocal(int s,int z,int y,int x,float v)const{
  if constexpr(CubeN>0){FixedCubeCachedEnvironment<CubeN,ActiveAxes,true>e{this->p,this->flags,this->profiles};e.StoreLocal(s,z,y,x,v);}
  else Base::StoreLocal(s,z,y,x,v);
 }
 __device__ __forceinline__ float Difference(int phase,int source,int axis,int z,int y,int x,float center,float adjacent)const{
  if constexpr(CubeN>0){FixedCubeCachedEnvironment<CubeN,ActiveAxes,true>e{this->p,this->flags,this->profiles};return e.Difference(phase,source,axis,z,y,x,center,adjacent);}
  else return Base::Difference(phase,source,axis,z,y,x,center,adjacent);
 }
 __device__ __forceinline__ float Coefficient(int kind,int index,int z,int y,int x)const{
  if constexpr(CubeN>0){if(kind>=2){FixedCubeCachedEnvironment<CubeN,ActiveAxes,true>e{this->p,this->flags,this->profiles};return e.Coefficient(kind,index,z,y,x);}}
  return Base::Coefficient(kind,index,z,y,x);
 }
};
