// Execute the actual emitted body as1024 cooperative host fibers. Every CTA
// barrier waits for all fibers. Canonical operands carry exact version tokens;
// local recurrences use independently tagged physical state/term/version.
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>
#include <ucontext.h>
#define __host__
#define __device__
#define __forceinline__ inline
#define __shared__
using std::min;using std::max;
#ifndef BEAMZ_PSI_FIBER_THREADS
#define BEAMZ_PSI_FIBER_THREADS 1024
#endif
constexpr int FiberThreads=BEAMZ_PSI_FIBER_THREADS;
#ifndef BEAMZ_PSI_FIBER_SHARED_BYTES
#define BEAMZ_PSI_FIBER_SHARED_BYTES 55204
#endif
constexpr int FiberSharedWords=BEAMZ_PSI_FIBER_SHARED_BYTES/4;
#ifndef BEAMZ_PSI_FIBER_SOURCE
#define BEAMZ_PSI_FIBER_SOURCE 0
#endif
constexpr bool FiberSource=BEAMZ_PSI_FIBER_SOURCE;
constexpr int BZ=7,BY=15,BX=31,CZ=8,CY=16,CX=32,PAD=4096,SurfacePoints=735;
constexpr size_t ProfileCachedSharedBytes=55204;
struct Index{int x=0;};static Index threadIdx,blockIdx;
static ucontext_t host_context,fibers[FiberThreads];
static std::vector<char>fiber_stacks(FiberThreads*32768);
static std::array<int,FiberThreads>joins{},finished{};static size_t barrier_checks=0;
static void require(bool value,const char*why){if(!value)throw std::runtime_error(why);}
static void __syncthreads(){const int t=threadIdx.x;++joins[t];if(swapcontext(&fibers[t],&host_context))std::abort();}
template<class T>T __ldg(const T*p){return *p;}
template<class T>T __ldcg(const T*p){return *p;}
static float __int_as_float(int x){return std::bit_cast<float>(x);}
static void atomicOr(int*p,int bits){*p|=bits;}
struct Field{float*data;int z,y,x,row,plane;};
struct Psi{float*input,*output;const float*a,*b,*inv;int row,plane;};
struct Phase{Field own[3],output[3];Psi psi[6];};
struct Parameters{Phase phase[2];const int*slots;const uint8_t*flags;float*trails;uint64_t*owner_tags;const uint32_t*epoch;int*bad;int nx,ny,nz,frontier,physical_n[3];};
static std::array<std::vector<float>,12>psi_state;
static size_t operands=0,recurrences=0,tasks=0,late=0,global_loads=0,global_stores=0;
static size_t finished_effects=0;static std::vector<uint8_t>source_finished;
static constexpr int delta[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}},axes[6]={1,0,0,2,2,1};
static int capacity(const Parameters&p){return (p.physical_n[0]+1)*(p.physical_n[1]+1)*(p.physical_n[2]+1);}
static int point_index(const Parameters&p,int z,int y,int x){return (z*(p.physical_n[1]+1)+y)*(p.physical_n[2]+1)+x;}
static bool supported(const Parameters&p,int f,int z,int y,int x){return z>=0&&y>=0&&x>=0&&z<p.physical_n[0]+delta[f][0]&&y<p.physical_n[1]+delta[f][1]&&x<p.physical_n[2]+delta[f][2];}
static float field_token(const Parameters&p,int f,int n,int z,int y,int x){return supported(p,f,z,y,x)?float(1+(n*6+f)*capacity(p)+point_index(p,z,y,x)):0.f;}
static float psi_token(const Parameters&p,int s,int n,int z,int y,int x){return float(1+(n*12+s)*capacity(p)+point_index(p,z,y,x));}
template<int Layout,bool Fast,bool Iso,int Policy,bool Uniform,int Source,bool Plane,int ActiveAxes,bool Cached>struct Environment{
 const Parameters&p;uint8_t flags;const float*profiles;
 bool Valid(int f,int z,int y,int x)const{return supported(p,f,z,y,x);}
 float LoadLocal(int s,int z,int y,int x)const{++global_loads;return psi_state[s][point_index(p,z,y,x)];}
 void StoreLocal(int s,int z,int y,int x,float value)const{++global_stores;psi_state[s][point_index(p,z,y,x)]=value;}
 float Finish(int f,int n,int z,int y,int x,float value)const{
  if constexpr(Policy==5&&FiberSource){
   require(supported(p,f,z,y,x),"Finish source/observation moved outside its original support guard");
   require(value==-field_token(p,f,n,z,y,x),"Finish sees an already-finished or wrong field version");
   const int key=(n*6+f)*capacity(p)+point_index(p,z,y,x);require(!source_finished.at(key),"Duplicated source/observation finish");source_finished[key]=1;++finished_effects;return -value;
  }else return value;
 }
 static constexpr int active_axes=ActiveAxes;
};
template<int N,int Axes>struct FixedCubeCachedEnvironment:Environment<1,false,true,1,true,-1,false,Axes,true>{
 using Base=Environment<1,false,true,1,true,-1,false,Axes,true>;
 FixedCubeCachedEnvironment(const Parameters&p,uint8_t f,const float*profiles):Base{p,f,profiles}{}
};
struct Read{int phase,output,field,version,dz,dy,dx;};
// CANONICAL_READS
namespace beamz{namespace cuda{namespace trace_native{
template<int Phase,int Component,bool,class Env,class Reader>float AdvanceNativeCell(Env&env,Reader read,int n,int z,int y,int x){
 for(const auto&r:reads)if(r.phase==Phase&&r.output==3*Phase+Component){float actual=read(r.field,n-1+r.version,r.dz,r.dy,r.dx),expected=field_token(env.p,r.field,n-1+r.version,z+r.dz,y+r.dy,x+r.dx);
  if(actual!=expected){std::cerr<<"operand phase"<<Phase<<" field"<<r.field<<" n"<<n<<" at"<<z<<","<<y<<","<<x<<" actual"<<actual<<" expected"<<expected<<"\n";require(false,"Actual native phase field route version mismatch");}++operands;
 }
 for(int i=0;i<2;++i){const int term=2*Component+i,axis=axes[term],coord=axis==0?z:axis==1?y:x,size=env.p.physical_n[axis]+Phase;
  const int active_axes=Env::active_axes>=0?Env::active_axes:int((env.flags>>1)&7u);
  if((active_axes&(1<<axis))&&(coord<12||coord>=size-12)){const int s=6*Phase+term;float old=env.LoadLocal(s,z,y,x);
   if(old!=psi_token(env.p,s,n-1,z,y,x)){std::cerr<<"psi"<<s<<" n"<<n<<" at"<<z<<","<<y<<","<<x<<"\n";require(false,"Private CPML point/term/history version mismatch");}
   env.StoreLocal(s,z,y,x,psi_token(env.p,s,n,z,y,x));++recurrences;
  }
 }
 return (FiberSource?-1.f:1.f)*field_token(env.p,3*Phase+Component,n,z,y,x);
}
}}}
float frame[FiberSharedWords+2];
// ACTUAL_NATIVE_HELPERS
// ACTUAL_PHASE_BODY
static const Parameters*active_p;static const int*active_tiles;static int active_depth,active_axis;
static void enter_fiber(int t){threadIdx.x=t;
 // ACTUAL_BODY_DISPATCH
 finished[t]=1;
}
static void execute(const Parameters&p,const int*tiles,int depth,int axis,std::mt19937&rng){
 active_p=&p;active_tiles=tiles;active_depth=depth;active_axis=axis;joins.fill(0);finished.fill(0);
 for(int t=0;t<FiberThreads;++t){getcontext(&fibers[t]);fibers[t].uc_stack.ss_sp=fiber_stacks.data()+size_t(t)*32768;fibers[t].uc_stack.ss_size=32768;fibers[t].uc_link=&host_context;makecontext(&fibers[t],reinterpret_cast<void(*)()>(enter_fiber),1,t);}
 std::array<int,FiberThreads>order;std::iota(order.begin(),order.end(),0);int generation=0;
 for(;;){std::shuffle(order.begin(),order.end(),rng);for(int t:order){require(!finished[t],"A fiber exited while others still wait at a CTA barrier");threadIdx.x=t;if(swapcontext(&host_context,&fibers[t]))std::abort();}
  const int complete=std::accumulate(finished.begin(),finished.end(),0);
  if(complete){require(complete==FiberThreads,"Divergent return before CTA join");break;}
  ++generation;for(int t=0;t<FiberThreads;++t)require(joins[t]==generation,"Divergent CTA barrier generation");++barrier_checks;
 }
}
static void test(std::array<int,3>n,int depth,int axis,int seed){
 Parameters p{};for(int a=0;a<3;++a)p.physical_n[a]=n[a];p.nz=(n[0]+1+depth+BZ-1)/BZ;p.ny=(n[1]+1+depth+BY-1)/BY;p.nx=(n[2]+1+depth+BX-1)/BX;
 if constexpr(FiberSource)source_finished.assign((depth+1)*6*capacity(p),0);
 std::array<std::vector<float>,6>initial,output;
 for(int f=0;f<6;++f){const int z=n[0]+delta[f][0],y=n[1]+delta[f][1],x=n[2]+delta[f][2];initial[f].resize(size_t(z)*y*x);output[f].assign(initial[f].size(),-3.f);p.phase[f/3].own[f%3]={initial[f].data(),z,y,x,x,y*x};p.phase[f/3].output[f%3]={output[f].data(),z,y,x,x,y*x};
  for(int iz=0;iz<z;++iz)for(int iy=0;iy<y;++iy)for(int ix=0;ix<x;++ix)initial[f][(iz*y+iy)*x+ix]=field_token(p,f,0,iz,iy,ix);
 }
 float profiles[864];for(int i=0;i<864;++i)profiles[i]=float(i+1)*.0001f;
 for(int s=0;s<12;++s){psi_state[s].resize(capacity(p));for(int z=0;z<=n[0];++z)for(int y=0;y<=n[1];++y)for(int x=0;x<=n[2];++x)psi_state[s][point_index(p,z,y,x)]=psi_token(p,s,0,z,y,x);p.phase[s/6].psi[s%6]={nullptr,nullptr,profiles+s*72,profiles+s*72+24,profiles+s*72+48,0,0};}
 const int count=p.nz*p.ny*p.nx;std::vector<std::vector<int>>rank_tiles(p.nz+p.ny+p.nx-2);std::vector<int>slots(count);std::vector<uint8_t>flags(count,uint8_t(32|64|(axis<<1)));
 for(int t=0;t<count;++t){const int z=t/(p.ny*p.nx),y=(t/p.nx)%p.ny,x=t%p.nx;slots[t]=rank_tiles[z+y+x].size();rank_tiles[z+y+x].push_back(t);}
 for(const auto&r:rank_tiles)p.frontier=std::max(p.frontier,int(r.size()));std::vector<float>trails(size_t(4)*p.frontier*depth*6*SurfacePoints,-5.f);std::vector<uint64_t>tags(4*p.frontier);uint32_t epoch=1;int bad=0;p.slots=slots.data();p.flags=flags.data();p.trails=trails.data();p.owner_tags=tags.data();p.epoch=&epoch;p.bad=&bad;
 std::mt19937 rng(seed);for(auto&rank:rank_tiles){std::shuffle(rank.begin(),rank.end(),rng);for(int tile:rank){int id=tile;blockIdx.x=0;std::fill(std::begin(frame),std::end(frame),-7.f);execute(p,&id,depth,axis,rng);require(!bad,"Actual native ImportPacket/owner tag diagnostic");require(frame[FiberSharedWords]==-7.f&&frame[FiberSharedWords+1]==-7.f,"Phase shared/profile footprint overrun");++tasks;}}
 for(int f=0;f<6;++f){const auto&v=p.phase[f/3].output[f%3];for(int z=0;z<v.z;++z)for(int y=0;y<v.y;++y)for(int x=0;x<v.x;++x){require(v.data[(z*v.y+y)*v.x+x]==field_token(p,f,depth,z,y,x),"Actual native body final field incomplete/wrong version");if constexpr(FiberSource)for(int version=1;version<=depth;++version)require(source_finished[(version*6+f)*capacity(p)+point_index(p,z,y,x)]==1,"A valid component/version lost its Finish effect");}}
 for(int s=0;s<12;++s){const int f=(s/6)*3+(s%6)/2,a=axes[s%6];if(!(axis&(1<<a)))continue;for(int z=0;z<n[0]+delta[f][0];++z)for(int y=0;y<n[1]+delta[f][1];++y)for(int x=0;x<n[2]+delta[f][2];++x){const int coord=a==0?z:a==1?y:x,size=n[a]+s/6;if(coord<12||coord>=size-12)require(psi_state[s][point_index(p,z,y,x)]==psi_token(p,s,depth,z,y,x),"Private CPML terminal publication missing");}}
}
int main(){for(int depth:{4,16}){for(int axis:{1,2,4})test({25,29,33},depth,axis,13+axis+depth);if constexpr(FiberThreads==512)for(int axis:{3,7})test({25,29,33},depth,axis,13+axis+depth);}std::cout<<"{\"passed\":true,\"actual_fiber_tasks\":"<<tasks<<",\"canonical_operands\":"<<operands<<",\"physical_CPML_recurrences\":"<<recurrences<<",\"global_psi_loads\":"<<global_loads<<",\"global_psi_stores\":"<<global_stores<<",\"finished_source_effects\":"<<finished_effects<<",\"CTA_barrier_generations\":"<<barrier_checks<<"}\n";}
