// Actual staged paired controller and H/E ready/ACK protocol, synthetic payload.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "native_paired_cohorts.h"
#include "native_closed_pipeline.h"
namespace paired_stress{
constexpr int BZ=7,BY=15,BX=31,FrameWords=3*4096,FamilyWords=3*735,Words=2*FamilyWords;
struct Parameters{int nz,ny,nx;uint64_t*owner_tags=nullptr;uint32_t*epoch=nullptr;int*bad=nullptr;};
#include "native_closed_pipeline_control.cuh"
// ACTUAL_HANDOFF_OPERATIONS
__host__ __device__ uint32_t Pattern(int epoch,int tile,int step,int word){return uint32_t(epoch)*0x9e3779b9u^uint32_t(tile+1)*0x85ebca6bu^uint32_t(step)*0xc2b2ae35u^uint32_t(word);}
// ACTUAL_RESET

template<bool Electric>__device__ void ReadExternal(const Parameters&p,const uint32_t*payload,int epoch,int n){
 using namespace closed_pipeline;const int t=Tile(),z=t/(p.ny*p.nx),y=(t/p.nx)%p.ny,x=t%p.nx;
 for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
  if(z<dz||y<dy||x<dx||(Group()==1&&bits==1))continue;
  const int parent=t-dz*p.ny*p.nx-dy*p.nx-dx,slot=Owner(p,parent);
  if(!LocalThread()&&__ldcg(p.owner_tags+slot)!=((uint64_t(epoch)<<32)|uint32_t(parent)))atomicOr(p.bad,2);
  const uint32_t*source=payload+(size_t(slot)*C().depth+n-1)*Words+(Electric?FamilyWords:0);
  for(int word=LocalThread();word<FamilyWords;word+=512)if(__ldcg(source+word)!=Pattern(epoch,parent,n,word+(Electric?FrameWords:0)))atomicOr(p.bad,1);
 }
}
template<bool Electric>__device__ void ReadPeer(const Parameters&p,int epoch,int n){
 using namespace closed_pipeline;if(!BeforeGhostImport<Electric>(n))return;
 if(Group()==1&&PeerActive(n)){
  const int parent=Tile()-p.ny*p.nx;const auto*source=reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(Frame())-GroupStride);
  for(int word=LocalThread();word<FrameWords;word+=512)if(source[word]!=Pattern(epoch,parent,n,word+(Electric?FrameWords:0)))atomicOr(p.bad,4);
 }
}
template<int Depth>__device__ void SyntheticBody(const Parameters&p,uint32_t*payload,int epoch){
 using namespace closed_pipeline;auto*frame=reinterpret_cast<uint32_t*>(Frame());const int t=Tile(),owner=Owner(p,t);
 for(int n=1;n<=Depth;++n){
  if(!BeforeHOverwrite(n))return;
  for(int word=LocalThread();word<FrameWords;word+=512)frame[word]=Pattern(epoch,t,n,word);
  WaitParents(p,n);if(Aborted())return;ReadExternal<false>(p,payload,epoch,n);ReadPeer<false>(p,epoch,n);
  for(int word=LocalThread();word<FamilyWords;word+=512)payload[(size_t(owner)*Depth+n-1)*Words+word]=Pattern(epoch,t,n,word);
  Join();HVisible(n);
  if(!BeforeEOverwrite(n))return;Join();
  for(int word=LocalThread();word<FrameWords;word+=512)frame[word]=Pattern(epoch,t,n,FrameWords+word);
  for(int word=LocalThread();word<FamilyWords;word+=512)payload[(size_t(owner)*Depth+n-1)*Words+FamilyWords+word]=Pattern(epoch,t,n,FrameWords+word);
  ReadExternal<true>(p,payload,epoch,n);ReadPeer<true>(p,epoch,n);Join();GhostERead(n);
  Publish(n);EVisible(n);
 }
}
// ACTUAL_RESERVATION_WORKER

// Asymmetric first/last windows exercise producer-last-E retention, a group
// waiting at the final whole-CTA join while its peer still uses named joins,
// and the absence of ACKs before the consumer's first owned step.
__global__ __launch_bounds__(1024,1) void FrameStress(int*bad,int*ready,int*abort,int*last_e_reads){
 using namespace closed_pipeline;BeamzNativeClosedPipelineWorkspace q;q.ready=ready;q.abort=abort;q.original_count=2;q.node_plane=1;
 if(!LocalThread()){auto&c=const_cast<Cold&>(C());c.q=q;c.bad=bad;c.tile=0;c.depth=16;c.abort_local=0;c.parent_ready=0;c.first=Group()?3:1;c.last=Group()?16:14;c.h_ready=c.e_ready=c.h_ack=c.e_ack=0;}
 auto*profiles=reinterpret_cast<uint32_t*>(CommonProfiles());if(threadIdx.x<576)profiles[threadIdx.x]=uint32_t(threadIdx.x)^0xcafe0000u;
 __syncthreads();Parameters p{2,1,1,nullptr,nullptr,bad};auto*frame=reinterpret_cast<uint32_t*>(Frame());const int t=Tile();
 for(int n=C().first;n<=C().last;++n){
  if(!BeforeHOverwrite(n))break;
  for(int word=LocalThread();word<FrameWords;word+=512)frame[word]=Pattern(7,t,n,word);
  ReadPeer<false>(p,7,n);Join();HVisible(n);
  if(!BeforeEOverwrite(n))break;Join();
  for(int word=LocalThread();word<FrameWords;word+=512)frame[word]=Pattern(7,t,n,FrameWords+word);
  if(n<C().last){ReadPeer<true>(p,7,n);Join();GhostERead(n);if(Group()==1&&n==14&&!LocalThread())atomicAdd(last_e_reads,1);}
  for(int i=LocalThread();i<576;i+=512)if(profiles[i]!=(uint32_t(i)^0xcafe0000u))atomicOr(bad,8);
  Publish(n);EVisible(n);
 }
 __syncthreads();
}
// ACTUAL_LATE_STARTUP_WORKER
} // namespace paired_stress

static void check(cudaError_t status,const char*why){if(status!=cudaSuccess)throw std::runtime_error(std::string(why)+":"+cudaGetErrorString(status));}
struct Allocations{
 std::vector<void*>values;~Allocations(){for(void*p:values)cudaFree(p);}
 void*bytes(size_t n){if(!n)return nullptr;void*p=nullptr;check(cudaMalloc(&p,n),"allocate");values.push_back(p);return p;}
 int*ints(size_t n){return static_cast<int*>(bytes(n*4));}
 int*copy(const std::vector<int>&v){int*p=ints(v.size());if(v.size())check(cudaMemcpy(p,v.data(),v.size()*4,cudaMemcpyHostToDevice),"metadata upload");return p;}
};
// CONTROLLED_STARTUP_HOST
static void frame_test(){
 using namespace paired_stress;Allocations memory;int*bad=memory.ints(1),*abort=memory.ints(1),*ready=memory.ints(2),*seen=memory.ints(1);
 check(cudaMemset(bad,0,4),"frame bad reset");check(cudaMemset(abort,0,4),"frame abort reset");check(cudaMemset(ready,0,8),"frame ready reset");check(cudaMemset(seen,0,4),"frame lastE reset");
 check(cudaFuncSetAttribute(FrameStress,cudaFuncAttributeMaxDynamicSharedMemorySize,closed_pipeline::SharedBytes),"frame capacity");
 FrameStress<<<1,1024,closed_pipeline::SharedBytes>>>(bad,ready,abort,seen);check(cudaGetLastError(),"frame launch");check(cudaDeviceSynchronize(),"frame sync");
 int result=0,aborted=0,last=0;check(cudaMemcpy(&result,bad,4,cudaMemcpyDeviceToHost),"frame result");check(cudaMemcpy(&aborted,abort,4,cudaMemcpyDeviceToHost),"frame abort");check(cudaMemcpy(&last,seen,4,cudaMemcpyDeviceToHost),"frame lastE");
 if(result||aborted||last!=1)throw std::runtime_error("H/Eframe ACK/window failure:"+std::to_string(result)+":"+std::to_string(aborted)+":"+std::to_string(last));
}

template<int Depth>static std::pair<uint64_t,uint64_t>run(int n,int workers,int&epoch){
 using namespace paired_stress;BeamzBuffer fields[6]{};const int add[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
 for(int f=0;f<6;++f){fields[f].rank=3;fields[f].element_type=kBeamzF32;for(int a=0;a<3;++a)fields[f].dims[a]=n+add[f][a];}
 BeamzSourceGroupLaunch groups[9]{};const int32_t*starts[9]{};int32_t start[6]={12,12,n/3,12,12,n/3};
 for(int g:{5,7}){auto&f=groups[g].coefficients;f.rank=4;f.dims[0]=2;f.dims[1]=f.dims[2]=n-24;f.dims[3]=1;starts[g]=start;}
 const auto native=BeamzBuildNativeMovingPlan(fields,Depth,{{0,0,0}},groups,starts,12,true);const auto plan=BeamzBuildNativePairedCohorts(native);
 const int count=plan.tile_ids.size(),original=native.tile_ids.size();Allocations memory;BeamzNativeClosedPipelineWorkspace q;
 q.next=memory.ints(count);q.pending=memory.ints(count);q.claimed=memory.ints(count);q.ready=memory.ints(original);
 q.initial_next=memory.copy(plan.initial_next);q.initial_pending=memory.copy(plan.initial_pending);q.initial_heads=memory.copy(plan.initial_heads);q.root_counts=memory.copy(plan.root_counts);
 q.tile_cohort=memory.copy(plan.tile_cohort);q.owner_slots=memory.copy(plan.owner_slots);q.finish_offsets=memory.copy(plan.finish_offsets);q.finish_children=memory.copy(plan.finish_children);q.finish_count=plan.finish_children.size();
 q.heads=memory.ints(1);q.enqueued=memory.ints(1);q.started=memory.ints(1);q.completed=memory.ints(1);q.exited=memory.ints(1);q.abort=memory.ints(1);
 q.tile_count=q.cohort_tasks=count;q.original_count=original;q.node_plane=plan.plane;q.cohort_count=1;q.cpml_uniform=1;q.worker_count=q.max_workers=std::min(workers,count);
 const size_t payload_bytes=size_t(plan.owner_count)*Depth*Words*4;auto*payload=static_cast<uint32_t*>(memory.bytes(payload_bytes));auto*tags=static_cast<uint64_t*>(memory.bytes(size_t(plan.owner_count)*8));auto*clock=static_cast<uint32_t*>(memory.bytes(4));int*bad=memory.ints(1);
 check(cudaMemset(payload,0,payload_bytes),"payload reset");check(cudaMemset(tags,0,size_t(plan.owner_count)*8),"tags reset");check(cudaMemset(bad,0,4),"bad reset");
 Parameters p{native.counts[0],native.counts[1],native.counts[2],tags,clock,bad};uint64_t global_reads=0,shared_reads=0;
 for(int t=0;t<original;++t){const int z=t/(p.ny*p.nx),y=(t/p.nx)%p.ny,x=t%p.nx;
  global_reads+=uint64_t((1+(z>0))*(1+(y>0))*(1+(x>0))-1-(z&1))*Depth*Words;
  shared_reads+=uint64_t(z&1)*Depth*2*FrameWords;
 }
 check(cudaFuncSetAttribute(PacketStress<Depth,6,0>,cudaFuncAttributeMaxDynamicSharedMemorySize,closed_pipeline::SharedBytes),"packet shared capacity");
 for(int repeat=0;repeat<3;++repeat){++epoch;check(cudaMemcpy(clock,&epoch,4,cudaMemcpyHostToDevice),"epoch upload");
  ResetStress<<<(std::max(count,original)+255)/256,256>>>(q,bad);check(cudaGetLastError(),"paired reset launch");
  PacketStress<Depth,6,0><<<q.worker_count,1024,closed_pipeline::SharedBytes>>>(p,q,payload,epoch);check(cudaGetLastError(),"paired packet launch");check(cudaDeviceSynchronize(),"paired packet sync");
  int result=0,aborted=0;check(cudaMemcpy(&result,bad,4,cudaMemcpyDeviceToHost),"packet result");check(cudaMemcpy(&aborted,q.abort,4,cudaMemcpyDeviceToHost),"packet abort");
  if(result||aborted)throw std::runtime_error("Paired packet failure:"+std::to_string(result)+":"+std::to_string(aborted));
 }
 return {global_reads*3,shared_reads*3};
}
int main(){
 int device=0,sms=0;check(cudaGetDevice(&device),"device");check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device),"SM count");startup_regression();frame_test();int epoch=0,cases=0;uint64_t global=0,shared=0;
 for(int n:{64,96})for(int workers:{1,4,sms}){auto a=run<4>(n,workers,epoch),b=run<16>(n,workers,epoch);global+=a.first+b.first;shared+=a.second+b.second;cases+=2;}
 std::cout<<"{\"passed\":true,\"cases\":"<<cases<<",\"epochs\":"<<epoch<<",\"packet_word_reads\":"<<global<<",\"shared_word_reads\":"<<shared<<",\"SMs\":"<<sms<<",\"producer_last_E_and_asymmetric_windows\":true,\"late_startup_positive_and_negative_passed\":true,\"scope\":\"Actual staged paired5rankplanner, fullCTAreservation/START/COMPLETE, nativeoriginaltags/readiness, externalglobalpacket payload andlocal H/EreadyACKsharedframe visibility; asymmetric lastEframe test and controlled all7startup. Synthetic, not FDTD\"}\n";
}
