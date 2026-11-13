// All four active single-axis terms remain independent across phase and term.
template<int Axes>__host__ __device__ constexpr int NativeNoioCompactTerm(int color){
 static_assert(Axes==1||Axes==2||Axes==4,"Exactly one traced CPML axis");
 return Axes==1?(color?2:1):Axes==2?(color?5:0):(color?4:3);
}
template<int Axes>__host__ __device__ constexpr int NativeNoioCompactIndex(int slot,int kind,int packed){
 return ((2*(slot/6)+int(slot%6!=NativeNoioCompactTerm<Axes>(0)))*3+(kind-2))*24+packed;
}
template<int Axes>__device__ __forceinline__ void NativeNoioCompactStage(const Parameters&p,float*profiles){
 if(threadIdx.x<288){const int i=threadIdx.x,packed=i%24,profile=i/24,kind=profile%3,color=profile/3,slot=(color/2)*6+NativeNoioCompactTerm<Axes>(color%2);
  const auto&psi=p.phase[slot/6].psi[slot%6];const float*source=kind==0?psi.a:kind==1?psi.b:psi.inv;profiles[i]=__ldg(source+packed);
 }
}
template<class Base,int Axes>struct NativeNoioCompactEnvironment:Base{
 const float*compact;
 __device__ __forceinline__ NativeNoioCompactEnvironment(Base&base,const float*cache):Base(base),compact(cache){}
 __device__ __forceinline__ float Coefficient(int kind,int slot,int z,int y,int x)const{
  if(kind>=2)return compact[NativeNoioCompactIndex<Axes>(slot,kind,Base::Packed(slot,z,y,x))];
  return Base::Coefficient(kind,slot,z,y,x);
 }
};
