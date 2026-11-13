#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "native_moving.h"
#define __host__
#define __device__
#define __forceinline__ inline
template<class T>T __ldg(const T*p){return *p;}
template<class T>T __ldcg(const T*p){return *p;}
float __fadd_rn(float a,float b){return a+b;}
float __fsub_rn(float a,float b){return a-b;}
float __fmul_rn(float a,float b){return a*b;}
float __fmaf_rn(float a,float b,float c){return std::fma(a,b,c);}
using std::min;using std::max;
namespace beamz::cuda::yee {
// BEAMZ_EFFECT_CANONICAL_PRIMITIVES
}
struct Psi{float*input=nullptr,*output=nullptr;const float*a=nullptr,*b=nullptr,*inv=nullptr;int plane=0,row=0;};
struct Phase{Psi psi[6];const float*decay[3]{},*source[3]{};const uint32_t*codes[3]{};const float*tables[3]{};int table_sizes[3]{};float inv=.125f;};
struct Source{const float*coefficients=nullptr,*waveforms=nullptr;const int*starts=nullptr,*current_step=nullptr;int count=0,nz=0,ny=0,nx=0,ntime=0;};
struct Parameters{Phase phase[2];Source sources[6];int physical_n[3]{},macro_step=0;float common_e[3]{},h_source=.25f;};
using Observation=std::array<uint32_t,6>;static std::vector<Observation>observed;
void Observe(const Parameters&,int field,int version,int z,int y,int x,float value){observed.push_back({uint32_t(field),uint32_t(version),uint32_t(z),uint32_t(y),uint32_t(x),std::bit_cast<uint32_t>(value)});}
// BEAMZ_EFFECT_ACTUAL_ENVIRONMENTS
#ifdef BEAMZ_DOUBLE_SHARED_ENGINE
#include "native_phase_environment.cuh"
#include "native_double_environment.cuh"
static const float*global_ab=nullptr;
template<int N,int P,bool U>struct HostPhaseEnvironment:DoubleSharedEnvironment<N,P,U>{
 HostPhaseEnvironment(const Parameters&p,uint8_t f):DoubleSharedEnvironment<N,P,U>(p,f,global_ab){}
};
#elif defined(BEAMZ_PHASE_REGISTER_ENGINE)
#include "native_phase_environment.cuh"
template<int N,int P,bool U>using HostPhaseEnvironment=PhaseEnvironment<N,P,U>;
#else
#include "native_effect_regions.cuh"
#endif
static void require(bool v,const char*why){if(!v)throw std::runtime_error(why);}
static uint32_t hash32(uint32_t i){i^=i>>16;i*=0x7feb352d;i^=i>>15;i*=0x846ca68b;return i^(i>>16);}
static size_t facts=0,methods=0,finish_calls=0;
static void planner_facts(){
 const int add[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
 for(auto n:std::vector<std::array<int,3>>{{25,29,33},{64,72,80},{136,136,136},{512,512,512},{564,564,564},{129,131,133}})for(int depth:{4,16}){
  BeamzBuffer fields[6]{};for(int c=0;c<6;++c){fields[c].rank=3;fields[c].element_type=kBeamzF32;for(int a=0;a<3;++a)fields[c].dims[a]=n[a]+add[c][a];}
  BeamzSourceGroupLaunch groups[9]{};const int32_t*starts[9]{};int32_t start[6]={12,12,n[2]/3,12,12,n[2]/3};
  for(int g:{5,7}){auto&f=groups[g].coefficients;f.rank=4;f.dims[0]=2;f.dims[1]=n[0]-24;f.dims[2]=n[1]-24;f.dims[3]=1;starts[g]=start;}
  auto p=BeamzBuildNativeMovingPlan(fields,depth,{{0,0,0},{n[0]/2,n[1]/2,n[2]/2}},groups,starts,12,true);
  for(size_t t=0;t<p.tile_flags.size();++t){const int q[3]={int(t)/(p.counts[1]*p.counts[2]),(int(t)/p.counts[2])%p.counts[1],int(t)%p.counts[2]};const unsigned flags=p.tile_flags[t];
   for(int a=0;a<3;++a){const int lo=q[a]*p.owned[a]-depth,hi=q[a]*p.owned[a]+p.owned[a]-2;
    if(!(flags&32))require(lo>0&&hi+1<n[a],"Support/wall elision lacks actual planner certificate");
    if(!(flags&(2u<<a)))require(lo>=12&&hi<n[a]-12,"CPML-axis elision lacks actual planner certificate");
    if(!(flags&46u))require(lo>=12&&hi<n[a]-12,"Plane-source transverse guard elision lacks certificate");++facts;
   }
  }
 }
}
template<int CubeN>static void environments(int n){
 Parameters p;for(int a=0;a<3;++a)p.physical_n[a]=n;
 const size_t codewords=(size_t(n+1)*(n+1)*(n+1)+3)/4;
 std::vector<uint32_t>uniform(codewords),mixed(codewords);for(size_t i=0;i<codewords;++i)mixed[i]=hash32(i);
 std::vector<float>sources(size_t(n-24)*(n-24)*2);for(size_t i=0;i<sources.size();++i)sources[i]=float(int(hash32(i)%41)-20)*.001f;
 float table[3][256],profiles[864],waveforms[64];for(int c=0;c<3;++c){for(int i=0;i<256;++i)table[c][i]=float(i+c+1)*.0001f;p.common_e[c]=table[c][0];p.phase[1].tables[c]=table[c];p.phase[1].table_sizes[c]=256;}
 for(int i=0;i<864;++i)profiles[i]=float(i+1)*.00001f;for(int i=0;i<64;++i)waveforms[i]=float(i-5)*.01f;
 int start[6]={12,12,n/3,12,12,n/3},clock=3;
 for(int f:{2,4})p.sources[f]={sources.data(),waveforms,start,&clock,2,n-24,n-24,1,32};
 for(int slot=0;slot<12;++slot){const int family=slot/6,term=slot%6,axis=beamz::cuda::yee::CpmlAxis(term),c=term/2;
  auto&psi=p.phase[family].psi[term];const int ny=axis==1?24:n+(family?(c!=1):(c==1)),nx=axis==2?24:n+(family?(c!=0):(c==0));psi.row=nx;psi.plane=ny*nx;psi.a=profiles+slot*72;psi.b=profiles+slot*72+24;psi.inv=profiles+slot*72+48;}
#ifdef BEAMZ_DOUBLE_SHARED_ENGINE
 float guarded_ab[578];guarded_ab[0]=1.234f;guarded_ab[577]=-1.234f;global_ab=guarded_ab+1;
 for(int thread=0;thread<1024;++thread){threadIdx.x=thread;DoubleStageProfiles<1024>(p,guarded_ab+1);}
 require(guarded_ab[0]==1.234f&&guarded_ab[577]==-1.234f,"Actual a/b staging overrun");
 for(int i=0;i<576;++i){const int slot=i/48,kind=(i/24)&1,packed=i%24;require(std::bit_cast<uint32_t>(guarded_ab[i+1])==std::bit_cast<uint32_t>(profiles[slot*72+kind*24+packed]),"Independent actual a/b stage binding");++methods;}
#endif
 for(unsigned flags=0;flags<128;++flags){for(int c=0;c<3;++c)p.phase[1].codes[c]=(flags&64?uniform:mixed).data();
  Environment<1,false,true,5,false,20,true,-1,true,true>reference{p,uint8_t(flags),profiles};
#if defined(BEAMZ_PHASE_REGISTER_ENGINE)||defined(BEAMZ_DOUBLE_SHARED_ENGINE)
  HostPhaseEnvironment<CubeN,5,true>boundary{p,uint8_t(flags)};
  HostPhaseEnvironment<CubeN,2,false>bulk{p,uint8_t(flags)};
  HostPhaseEnvironment<CubeN,4,false>source{p,uint8_t(flags)};
#else
  TracedBoundaryEffects<CubeN>boundary{{{p,uint8_t(flags),profiles}}};
  TracedRuntimeCoefficient<Environment<1,true,true,2,false,-1,false,-1,false,true>>bulk{{p,uint8_t(flags),nullptr}};
  TracedRuntimeCoefficient<Environment<1,true,true,4,false,20,true,-1,false,true>>source{{p,uint8_t(flags),nullptr}};
#endif
  auto check=[&](auto&candidate,int z,int y,int x){
   for(int f=0;f<6;++f){require(candidate.Valid(f,z,y,x)==reference.Valid(f,z,y,x),"Support specialization");
    if(!reference.Valid(f,z,y,x))continue;
    const auto before=observed.size();float a=reference.Finish(f,3,z,y,x,.314159f);std::vector<Observation>expected(observed.begin()+before,observed.end());observed.resize(before);
    float b=candidate.Finish(f,3,z,y,x,.314159f);std::vector<Observation>actual(observed.begin()+before,observed.end());observed.resize(before);
    require(std::bit_cast<uint32_t>(a)==std::bit_cast<uint32_t>(b)&&actual==expected,"Source/observation specialization");++finish_calls;
    for(int kind=0;kind<2;++kind){float a=reference.Coefficient(kind,f,z,y,x),b=candidate.Coefficient(kind,f,z,y,x);require(std::bit_cast<uint32_t>(a)==std::bit_cast<uint32_t>(b),"Material specialization");++methods;}
   }
   for(int phase=0;phase<2;++phase)for(int axis=0;axis<3;++axis){float a=reference.Difference(phase,0,axis,z,y,x,.25f,.5f),b=candidate.Difference(phase,0,axis,z,y,x,.25f,.5f);require(std::bit_cast<uint32_t>(a)==std::bit_cast<uint32_t>(b),"Derivative/wall specialization");++methods;}
   for(int s=0;s<12;++s){require(candidate.LocalActive(s,z,y,x)==reference.LocalActive(s,z,y,x),"CPML effect specialization");++methods;
    if(reference.Valid((s/6)*3+(s%6)/2,z,y,x)&&reference.LocalActive(s,z,y,x))for(int kind=2;kind<5;++kind){float a=reference.Coefficient(kind,s,z,y,x),b=candidate.Coefficient(kind,s,z,y,x);require(std::bit_cast<uint32_t>(a)==std::bit_cast<uint32_t>(b),"Independent profile specialization");++methods;}
   }
  };
  auto general=[&](int z,int y,int x){
#if defined(BEAMZ_PHASE_REGISTER_ENGINE)||defined(BEAMZ_DOUBLE_SHARED_ENGINE)
   for(int c=0;c<3;++c)p.phase[1].codes[c]=uniform.data();
   check(boundary,z,y,x);
   for(int c=0;c<3;++c)p.phase[1].codes[c]=(flags&64?uniform:mixed).data();
#elif defined(BEAMZ_EFFECT_PROGRAM_UNIFORM_GENERAL)
   for(int c=0;c<3;++c)p.phase[1].codes[c]=uniform.data();
   switch(flags&14u){
    case 2:{TracedBoundaryEffects<CubeN,1,true>e{{p,uint8_t(flags),profiles}};check(e,z,y,x);break;}
    case 4:{TracedBoundaryEffects<CubeN,2,true>e{{p,uint8_t(flags),profiles}};check(e,z,y,x);break;}
    case 8:{TracedBoundaryEffects<CubeN,4,true>e{{p,uint8_t(flags),profiles}};check(e,z,y,x);break;}
    default:{TracedBoundaryEffects<CubeN,-1,true>e{{p,uint8_t(flags),profiles}};check(e,z,y,x);break;}
   }
   for(int c=0;c<3;++c)p.phase[1].codes[c]=(flags&64?uniform:mixed).data();
#else
   check(boundary,z,y,x);
#endif
  };
  for(int i=0;i<1024;++i){int z=int(hash32(i*3)%unsigned(n+3))-1,y=int(hash32(i*3+1)%unsigned(n+3))-1,x=int(hash32(i*3+2)%unsigned(n+3))-1;
   general(z,y,x);
   if(!(flags&46u)){z=12+hash32(i*5)%unsigned(n-24);y=12+hash32(i*5+1)%unsigned(n-24);x=(i%4==0)?n/3:12+hash32(i*5+2)%unsigned(n-24);
#if defined(BEAMZ_PHASE_REGISTER_ENGINE)||defined(BEAMZ_DOUBLE_SHARED_ENGINE)
    if(!(flags&62u))check(bulk,z,y,x);else check(source,z,y,x);
#elif defined(BEAMZ_EFFECT_PROGRAM_UNIFORM_GENERAL)
    if(!(flags&62u)){
     if(flags&64u){Environment<1,true,true,2,true,-1,false,-1,false,true>e{p,uint8_t(flags),nullptr};check(e,z,y,x);}
     else {Environment<1,true,true,2,false,-1,false,-1,false,true>e{p,uint8_t(flags),nullptr};check(e,z,y,x);}
    }else{
     if(flags&64u){Environment<1,true,true,4,true,20,true,-1,false,true>e{p,uint8_t(flags),nullptr};check(e,z,y,x);}
     else {Environment<1,true,true,4,false,20,true,-1,false,true>e{p,uint8_t(flags),nullptr};check(e,z,y,x);}
    }
#else
    if(!(flags&62u))check(bulk,z,y,x);else check(source,z,y,x);
#endif
   }
  }
  for(int z:{-1,0,11,12,13,n-13,n-12,n-1,n,n+1})for(int y:{-1,0,11,12,13,n-13,n-12,n-1,n,n+1})general(z,y,n/3);
 }
}
int main(){planner_facts();environments<0>(64);environments<512>(512);environments<564>(564);
 std::cout<<"{\"passed\":true,\"planner_facts\":"<<facts<<",\"method_bitpattern_checks\":"<<methods<<",\"source_observation_checks\":"<<finish_calls<<",\"scope\":\"Actual emitted Environment/FixedCube/region helpers and canonical primitives; FP32intrinsics emulated; actual planner facts; CUDA execution remains separate\"}\n";}
