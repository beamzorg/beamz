// Two full shared families need no persistent field registers. Cache independent
// a/b words only; inv remains immutable global to fit SM86's99KB block limit.
template<int CubeN,int Policy,bool UniformE>
struct DoubleSharedEnvironment:PhaseEnvironment<CubeN,Policy,UniformE>{
 using Base=PhaseEnvironment<CubeN,Policy,UniformE>;const float*ab;
 __device__ __forceinline__ DoubleSharedEnvironment(const Parameters&p,uint8_t flags,const float*profiles):Base(p,flags),ab(profiles){}
 __device__ __forceinline__ float Coefficient(int kind,int index,int z,int y,int x)const{
  if(kind==2||kind==3){int packed;
   if constexpr(Policy==5&&CubeN>0){FixedCubeCachedEnvironment<CubeN,-1,true>e{this->p,this->flags,nullptr};packed=e.Packed(index,z,y,x);}
   else packed=Base::Packed(index,z,y,x);
   return ab[(index*2+kind-2)*24+packed];
  }
  return Base::Coefficient(kind,index,z,y,x);
 }
};
