#include "native_noio_owned_psi.cuh"
template<int N>__host__ __device__ __forceinline__ int NativeStationaryOwnedShift(int n){const int r=n%N;return r?N-r:0;}
__host__ __device__ __forceinline__ int NativeStationaryOwnedIndex(int dz,int dy,int dx,int hz,int hy,int hx){
 int z=dz-1+hz,y=dy-1+hy,x=dx-1+hx;
 if(z>=BZ)z-=BZ;if(y>=BY)y-=BY;if(x>=BX)x-=BX;
 return (z*BY+y)*BX+x;
}
template<class Base>struct NativeStationaryOwnedPsiPoint:Base{
 float*cache;int index;bool fresh,flush;
 __device__ __forceinline__ NativeStationaryOwnedPsiPoint(Base&base,float*values,int slot,bool entering,bool publish):Base(base),cache(values),index(slot),fresh(entering),flush(publish){}
 __device__ __forceinline__ int Color(int slot)const{const int term=slot%6;return term==2||term==4||term==5;}
 __device__ __forceinline__ float LoadLocal(int slot,int z,int y,int x)const{
  if(fresh)return Base::LoadLocal(slot,z,y,x);
  return cache[Color(slot)*NativeOwnedPsiPoints+index];
 }
 __device__ __forceinline__ void StoreLocal(int slot,int z,int y,int x,float value)const{
  cache[Color(slot)*NativeOwnedPsiPoints+index]=value;
  if(flush)Base::StoreLocal(slot,z,y,x,value);
 }
};
