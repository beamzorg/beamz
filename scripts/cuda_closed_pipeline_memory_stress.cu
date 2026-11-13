// Standalone GPU visibility/owner-retirement stress using staged actual helpers.
// This is synthetic packet traffic, not an FDTD benchmark or FP32 oracle.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "native_closed_cohorts.h"
#include "native_closed_pipeline.h"

namespace closed_stress {
struct Parameters{int nz,ny,nx;};
#include "native_closed_pipeline_control.cuh"
constexpr int Words=6*735;
__host__ __device__ uint32_t Pattern(int epoch,int tile,int step,int word){return uint32_t(epoch)*0x9e3779b9u^uint32_t(tile+1)*0x85ebca6bu^uint32_t(step)*0xc2b2ae35u^uint32_t(word);}

// BEAMZ_CLOSED_ACTUAL_RESET_INSERT

template<int Depth>
__global__ __launch_bounds__(1024,1) void PacketStress(Parameters p,BeamzNativeClosedPipelineWorkspace q,
 uint32_t*payload,uint64_t*tags,int32_t*bad,int epoch){
 using namespace closed_pipeline;
 if(!threadIdx.x){auto&cold=const_cast<Cold&>(C());cold.q=q;cold.bad=bad;cold.depth=Depth;cold.tile=-1;cold.parent_ready=0;cold.abort_local=0;}
 __syncthreads();
 for(;;){
  if(!threadIdx.x)Pop();__syncthreads();if(Tile()<0)break;
  const int t=Tile(),z=t/(p.ny*p.nx),y=(t/p.nx)%p.ny,x=t%p.nx,owner=Owner(p,t);
  if(!threadIdx.x)tags[owner]=(uint64_t(epoch)<<32)|uint32_t(t);
  __syncthreads();if(!threadIdx.x)Started(p);
  for(int step=1;step<=Depth;++step){
   WaitParents(p,step);if(Aborted())break;
   for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;if(z<dz||y<dy||x<dx)continue;
    const int parent=t-dz*p.ny*p.nx-dy*p.nx-dx,slot=Owner(p,parent);
    if(!threadIdx.x&&__ldcg(tags+slot)!=((uint64_t(epoch)<<32)|uint32_t(parent)))atomicOr(bad,2);
    const uint32_t*source=payload+(size_t(slot)*Depth+step-1)*Words;
    for(int word=threadIdx.x;word<Words;word+=1024)if(__ldcg(source+word)!=Pattern(epoch,parent,step,word))atomicOr(bad,1);
   }
   uint32_t*target=payload+(size_t(owner)*Depth+step-1)*Words;
   for(int word=threadIdx.x;word<Words;word+=1024)target[word]=Pattern(epoch,t,step,word);
   Publish(step);
  }
  __syncthreads();if(!threadIdx.x&&!Aborted())Complete();__syncthreads();
 }
 __syncthreads();
 if(!threadIdx.x){const int before=AddAcqRel(q.exited+q.cohort,1);
  if(before==q.worker_count-1&&!Acquire(q.abort))if(Acquire(q.completed+q.cohort)!=q.cohort_tasks||Acquire(q.started+q.cohort)!=q.cohort_tasks||Acquire(q.enqueued+q.cohort)!=q.cohort_tasks)Abort(512);
 }
}

// Reproduce the clipped-corner publication gap under guaranteed two-CTA
// residency. The producer stays behind a handshake and a30ms delay; the
// consumer must wait for its diagonal, even though all3axial readies are1.
__global__ __launch_bounds__(1024,1) void LateStartupStress(Parameters p,BeamzNativeClosedPipelineWorkspace q,
 uint32_t*payload,int*entered,int*bad,int*observed,int*smids,int corrected){
 using namespace closed_pipeline;
 if(!threadIdx.x){auto&cold=const_cast<Cold&>(C());cold.q=q;cold.bad=bad;cold.depth=4;cold.tile=blockIdx.x?25:12;cold.parent_ready=0;cold.abort_local=0;
  unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));smids[blockIdx.x]=int(sm);}
 __syncthreads();
 if(!blockIdx.x){
  if(!threadIdx.x){const uint64_t begin=Timer();while(!Acquire(entered)){if(Timer()-begin>q.timeout_ns){Abort(256);break;}__nanosleep(32);}
   const uint64_t delay=Timer();while(Timer()-delay<30000000ULL)__nanosleep(32);}
  __syncthreads();
  for(int word=threadIdx.x;word<Words;word+=1024)payload[word]=0xdeadbeefu^uint32_t(word);
  __threadfence();__syncthreads();if(!threadIdx.x)Release(q.ready+12,1);
 }else{
  if(!threadIdx.x)Release(entered,1);
  WaitParents(p,1,corrected!=0);
  if(!threadIdx.x)*observed=Acquire(q.ready+12);
  for(int word=threadIdx.x;word<Words;word+=1024)if(__ldcg(payload+word)!=(0xdeadbeefu^uint32_t(word)))atomicOr(bad,1);
 }
}
} // namespace closed_stress

static void check(cudaError_t status,const char*why){if(status!=cudaSuccess)throw std::runtime_error(std::string(why)+":"+cudaGetErrorString(status));}
struct Allocations{
 std::vector<void*>values;
 ~Allocations(){for(void*p:values)cudaFree(p);}
 void*bytes(size_t n){if(!n)return nullptr;void*p=nullptr;check(cudaMalloc(&p,n),"stress allocate");values.push_back(p);return p;}
 int*ints(size_t n){return static_cast<int*>(bytes(n*sizeof(int)));}
 int*copy(const std::vector<int>&v){int*p=ints(v.size());if(v.size())check(cudaMemcpy(p,v.data(),v.size()*sizeof(int),cudaMemcpyHostToDevice),"stress metadata upload");return p;}
};

static void startup_regression(){
 using namespace closed_stress;int cooperative=0,device=0;check(cudaGetDevice(&device),"startup current device");check(cudaDeviceGetAttribute(&cooperative,cudaDevAttrCooperativeLaunch,device),"startup cooperative support");
 if(!cooperative)throw std::runtime_error("Controlled startup proof needs cooperative launch");
 Allocations memory;BeamzNativeClosedPipelineWorkspace q;std::vector<int>ready(27,1);ready[12]=0;q.ready=memory.copy(ready);q.abort=memory.ints(1);q.tile_count=27;
 auto*payload=static_cast<uint32_t*>(memory.bytes(Words*4));int*entered=memory.ints(1),*bad=memory.ints(1),*observed=memory.ints(1),*smids=memory.ints(2);Parameters p{3,3,3};
 check(cudaFuncSetAttribute(LateStartupStress,cudaFuncAttributeMaxDynamicSharedMemorySize,closed_pipeline::SharedBytes),"startup shared capacity");
 for(int corrected:{0,1}){
  check(cudaMemcpy(q.ready,ready.data(),ready.size()*4,cudaMemcpyHostToDevice),"startup ready reset");check(cudaMemset(q.abort,0,4),"startup abort reset");check(cudaMemset(payload,0,Words*4),"startup payload reset");check(cudaMemset(entered,0,4),"startup handshake reset");check(cudaMemset(bad,0,4),"startup failure reset");
  void*arguments[]={&p,&q,&payload,&entered,&bad,&observed,&smids,&corrected};
  check(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(LateStartupStress),dim3(2),dim3(1024),arguments,closed_pipeline::SharedBytes,nullptr),"cooperative startup launch");check(cudaDeviceSynchronize(),"cooperative startup synchronization");
  int failure=0,seen=0,where[2];check(cudaMemcpy(&failure,bad,4,cudaMemcpyDeviceToHost),"startup result");check(cudaMemcpy(&seen,observed,4,cudaMemcpyDeviceToHost),"startup readiness result");check(cudaMemcpy(where,smids,8,cudaMemcpyDeviceToHost),"startup SM placement");
  if(where[0]==where[1])throw std::runtime_error("Startup proof needs distinct SMs");
  if(corrected?(failure!=0||seen!=1):(failure!=1||seen!=0))throw std::runtime_error("Controlled startup positive/negative failed:"+std::to_string(failure));
 }
}

template<int Depth>static uint64_t run(int n,int workers,int&epoch){
 using namespace closed_stress;
 BeamzBuffer fields[6]{};const int add[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
 for(int c=0;c<6;++c){fields[c].rank=3;fields[c].element_type=kBeamzF32;for(int a=0;a<3;++a)fields[c].dims[a]=n+add[c][a];}
 BeamzSourceGroupLaunch groups[9]{};const int32_t*starts[9]{};int32_t start[6]={12,12,n/3,12,12,n/3};
 for(int g:{5,7}){auto&f=groups[g].coefficients;f.rank=4;f.dims[0]=2;f.dims[1]=f.dims[2]=n-24;f.dims[3]=1;starts[g]=start;}
 const auto native=BeamzBuildNativeMovingPlan(fields,Depth,{{0,0,0},{n/2,n/2,n/2}},groups,starts,12,true);
 const auto plan=BeamzBuildNativeClosedCohorts(native,4);const int count=plan.tile_ids.size(),cohorts=plan.cohort_policies.size();
 Allocations memory;BeamzNativeClosedPipelineWorkspace q;
 q.next=memory.ints(count);q.pending=memory.ints(count);q.claimed=memory.ints(count);q.ready=memory.ints(count);
 q.initial_next=memory.copy(plan.initial_next);q.initial_pending=memory.copy(plan.initial_pending);q.initial_heads=memory.copy(plan.initial_heads);q.root_counts=memory.copy(plan.root_counts);
 q.tile_cohort=memory.copy(plan.tile_cohort);q.owner_slots=memory.copy(plan.owner_slots);q.finish_offsets=memory.copy(plan.finish_offsets);q.finish_children=memory.copy(plan.finish_children);q.finish_count=plan.finish_children.size();
 q.heads=memory.ints(cohorts);q.enqueued=memory.ints(cohorts);q.started=memory.ints(cohorts);q.completed=memory.ints(cohorts);q.exited=memory.ints(cohorts);q.abort=memory.ints(1);
 q.tile_count=count;q.cohort_count=cohorts;q.max_workers=workers;q.cpml_uniform=1;
 const size_t payload_bytes=size_t(plan.owner_count)*Depth*Words*sizeof(uint32_t);
 auto*payload=static_cast<uint32_t*>(memory.bytes(payload_bytes));auto*tags=static_cast<uint64_t*>(memory.bytes(size_t(plan.owner_count)*sizeof(uint64_t)));auto*bad=memory.ints(1);
 check(cudaMemset(payload,0,payload_bytes),"initialize packet payload");check(cudaMemset(tags,0,size_t(plan.owner_count)*sizeof(uint64_t)),"initialize epoch tags");check(cudaMemset(bad,0,4),"initialize failure flag");
 Parameters physics{native.counts[0],native.counts[1],native.counts[2]};uint64_t reads=0;
 for(int t=0;t<count;++t){const int z=t/(physics.ny*physics.nx),y=(t/physics.nx)%physics.ny,x=t%physics.nx;
  reads+=uint64_t((1+(z>0))*(1+(y>0))*(1+(x>0))-1)*Depth*Words;}
 check(cudaFuncSetAttribute(PacketStress<Depth>,cudaFuncAttributeMaxDynamicSharedMemorySize,closed_pipeline::SharedBytes),"stress shared-memory attribute");
 for(int repeat=0;repeat<3;++repeat){++epoch;
  ResetStress<<<(count+255)/256,256>>>(q,bad);check(cudaGetLastError(),"reset stress launch");
  for(int c=0;c<cohorts;++c){q.cohort=c;q.cohort_tasks=plan.cohort_offsets[c+1]-plan.cohort_offsets[c];q.worker_count=std::min(workers,q.cohort_tasks);
   PacketStress<Depth><<<q.worker_count,1024,closed_pipeline::SharedBytes>>>(physics,q,payload,tags,bad,epoch);check(cudaGetLastError(),"packet stress launch");}
  check(cudaDeviceSynchronize(),"packet stress synchronization");int result=0,abort=0;check(cudaMemcpy(&result,bad,4,cudaMemcpyDeviceToHost),"read stress failure");check(cudaMemcpy(&abort,q.abort,4,cudaMemcpyDeviceToHost),"read stress abort");
  if(result||abort)throw std::runtime_error("Packet stress failed:"+std::to_string(result)+" abort:"+std::to_string(abort));
 }
 return reads*3;
}

int main(){
 int device=0,sms=0;check(cudaGetDevice(&device),"stress current device");check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device),"stress SM count");
 startup_regression();int epoch=0,cases=0;uint64_t reads=0;
 for(int n:{64,96})for(int workers:{1,4,sms}){reads+=run<4>(n,workers,epoch);reads+=run<16>(n,workers,epoch);cases+=2;}
 std::cout<<"{\"passed\":true,\"cases\":"<<cases<<",\"epochs\":"<<epoch<<",\"packet_word_reads\":"<<reads<<",\"SMs\":"<<sms<<",\"late_startup_positive_and_negative_passed\":true,\"late_startup_distinct_SM_verified\":true,\"scope\":\"Synthetic4410word versionedpacket visibility/ownerretirement with actual staged schedulefirstplanner and typedLIFO/readyhelpers; controlledall7startup regression; no FDTDspeed or FPclaim\"}\n";
}
