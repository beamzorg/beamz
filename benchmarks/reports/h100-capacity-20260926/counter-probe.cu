#include <cuda_runtime.h>
#include <cstdio>
__global__ void probe(float* x) { int i=blockIdx.x*blockDim.x+threadIdx.x; x[i]=x[i]*1.01f+1; }
int main(){ float* x; cudaError_t e=cudaMalloc(&x,1024*1024*sizeof(float)); if(e!=cudaSuccess)return 1; cudaMemset(x,0,1024*1024*sizeof(float)); probe<<<4096,256>>>(x); e=cudaDeviceSynchronize(); printf("cuda status %s\n",cudaGetErrorString(e)); cudaFree(x); return e!=cudaSuccess; }
