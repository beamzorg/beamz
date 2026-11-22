// Compatibility arithmetic interface, backed by the canonical cell graph.
#pragma once
#include "cell.cuh"
namespace beamz::cuda::generated {
template<int Phase,int Component,class ReadField,class ReadCoefficient>
__device__ __forceinline__ float AdvanceYeeCell(float inv,ReadField read,ReadCoefficient coefficient){
 struct Environment {
  float inv;ReadCoefficient coefficient;
  __device__ bool Valid(int,int,int,int)const{return true;}
  __device__ bool LocalActive(int,int,int,int)const{return false;}
  __device__ float LoadLocal(int,int,int,int)const{return 0.f;}
  __device__ void StoreLocal(int,int,int,int,float)const{}
  __device__ float Coefficient(int kind,int field,int,int,int)const{return coefficient(kind,field%3);}
  __device__ float Difference(int phase,int,int,int,int,int,float center,float adjacent)const{
   return __fmul_rn(phase?__fsub_rn(center,adjacent):__fsub_rn(adjacent,center),inv);
  }
  __device__ float Finish(int,int,int,int,int,float value)const{return value;}
 } env{inv,coefficient};
 return beamz::cuda::trace_native::AdvanceNativeCell<Phase,Component,false>(env,read,1,0,0,0);
}
}
