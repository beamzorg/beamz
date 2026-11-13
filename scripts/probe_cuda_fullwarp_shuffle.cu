// GPU bit-pattern oracle for the exact admitted complete-warp PTX lowering.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "fullwarp_helper.cuh"

constexpr int Queries=42;
__global__ void CheckFullWarpWords(const uint32_t*input,uint32_t*ptx,uint32_t*sdk,int count){
 const int i=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x&31;
 const float value=__uint_as_float(input[i]);
 for(int q=0;q<Queries;++q){int source;
  if(q<32)source=q;
  else if(q==32)source=(lane+1)&31;
  else if(q==33)source=(lane+31)&31;
  else source=((threadIdx.x/32+q)&1)?((lane+1)&31):((lane+31)&31);
  // Branches are warp-uniform but deliberately differ between adjacent warps.
  // This exercises the same complete-warp contract as private Ey selection.
  float a,b;
  if(q<34||((threadIdx.x/32+q)&1)){
   a=FullWarpIndex32(value,source);b=__shfl_sync(0xffffffffu,value,source,32);
  }else{
   a=FullWarpIndex32(value,source);b=__shfl_sync(0xffffffffu,value,source,32);
  }
  ptx[q*count+i]=__float_as_uint(a);sdk[q*count+i]=__float_as_uint(b);
 }
}
int main(){
 const uint32_t edges[]={0u,0x80000000u,1u,0x80000001u,0x007fffffu,0x00800000u,
  0x7f800000u,0xff800000u,0x7fc00001u,0xffc12345u,0x7f800001u,0xffffffffu,
  0x3f800000u,0xbf800000u,0x7f7fffffu,0xff7fffffu};
 size_t checks=0;
 for(int threads:{256,512,1024}){
  const int n=3*threads;std::vector<uint32_t>input(n),a(n*Queries),b(n*Queries);
  for(int i=0;i<n;++i)input[i]=i%3?edges[i%16]:(uint32_t(i)*1664525u+1013904223u);
  uint32_t *di=nullptr,*da=nullptr,*db=nullptr;
  auto okay=[](cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));return false;}return true;};
  if(!okay(cudaMalloc(&di,n*4))||!okay(cudaMalloc(&da,n*Queries*4))||!okay(cudaMalloc(&db,n*Queries*4)))return 2;
  if(!okay(cudaMemcpy(di,input.data(),n*4,cudaMemcpyHostToDevice)))return 2;
  CheckFullWarpWords<<<3,threads>>>(di,da,db,n);
  if(!okay(cudaGetLastError())||!okay(cudaDeviceSynchronize()))return 2;
  if(!okay(cudaMemcpy(a.data(),da,a.size()*4,cudaMemcpyDeviceToHost))||!okay(cudaMemcpy(b.data(),db,b.size()*4,cudaMemcpyDeviceToHost)))return 2;
  for(int q=0;q<Queries;++q)for(int i=0;i<n;++i){int lane=i&31,source;
   if(q<32)source=q;else if(q==32)source=(lane+1)&31;else if(q==33)source=(lane+31)&31;
   else source=(((i%threads)/32+q)&1)?((lane+1)&31):((lane+31)&31);
   const uint32_t expected=input[(i/32)*32+source];
   if(a[q*n+i]!=expected||b[q*n+i]!=expected){std::fprintf(stderr,"Mismatch block%d q%d i%d expected%08x PTX%08x SDK%08x\n",threads,q,i,expected,a[q*n+i],b[q*n+i]);return 3;}
   checks+=2;
  }
  cudaFree(di);cudaFree(da);cudaFree(db);
 }
 std::printf("{\"pass\":true,\"word_checks\":%zu,\"queries\":42,\"width\":32,\"full_mask\":4294967295,\"blocks_per_size\":3,\"block_sizes\":[256,512,1024],\"patterns\":\"signedzero/subnormal/Inf/NaNpayload/random32bits\"}\n",checks);
}
