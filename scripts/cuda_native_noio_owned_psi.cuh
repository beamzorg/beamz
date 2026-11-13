// Only owned psi histories survive. Incoming low faces reload canonical global
// state; all old shared readers retire before the phase publishes its outputs.
constexpr int NativeOwnedPsiPoints=BZ*BY*BX;
__host__ __device__ __forceinline__ int NativeOwnedPsiIndex(int dz,int dy,int dx){return ((dz-1)*BY+dy-1)*BX+dx-1;}
template<class Base>struct NativeOwnedGlobalProfiles:Base{
 __device__ __forceinline__ NativeOwnedGlobalProfiles(Base&base):Base(base){}
 __device__ __forceinline__ float Coefficient(int kind,int slot,int z,int y,int x)const{
  if(kind>=2){const auto&psi=this->p.phase[slot/6].psi[slot%6];const int packed=Base::Packed(slot,z,y,x);return kind==2?__ldg(psi.a+packed):kind==3?__ldg(psi.b+packed):__ldg(psi.inv+packed);}
  return Base::Coefficient(kind,slot,z,y,x);
 }
};
template<class Base>struct NativeOwnedPsiPoint:Base{
 float&a,&b;const float*previous;int index;bool fresh,flush;
 __device__ __forceinline__ NativeOwnedPsiPoint(Base&base,float&out_a,float&out_b,const float*old,int old_index,bool entering,bool publish):Base(base),a(out_a),b(out_b),previous(old),index(old_index),fresh(entering),flush(publish){}
 __device__ __forceinline__ int Color(int slot)const{const int term=slot%6;return term==2||term==4||term==5;}
 __device__ __forceinline__ float LoadLocal(int slot,int z,int y,int x)const{
  if(fresh)return Base::LoadLocal(slot,z,y,x);
  return previous[Color(slot)*NativeOwnedPsiPoints+index];
 }
 __device__ __forceinline__ void StoreLocal(int slot,int z,int y,int x,float value)const{
  if(Color(slot))b=value;else a=value;
  if(flush)Base::StoreLocal(slot,z,y,x,value);
 }
};
