
#include <cuda_runtime.h>
__global__ void BeamzElectricCoefficients(const float*epsilon,float*out,int n,float dt,float eps0){
 const int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n)out[i]=__fdividef(dt,__fmul_rn(eps0,epsilon[i]));
}
cudaError_t BeamzEnqueueElectricCoefficients(cudaStream_t stream,const float*epsilon,float*out,int n,float dt,float eps0){
 BeamzElectricCoefficients<<<(n+255)/256,256,0,stream>>>(epsilon,out,n,dt,eps0);
 return cudaGetLastError();
}
