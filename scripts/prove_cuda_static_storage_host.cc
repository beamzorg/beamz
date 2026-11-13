// Actual fixed shared-route/profile-allocation helpers with physical FP32patterns.
#include <algorithm>
#include <bit>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>
#define __host__
#define __device__
#define __forceinline__ inline
struct ThreadIndex{int x=0;};static ThreadIndex threadIdx;
template<class T>T __ldg(const T*p){return *p;}
struct Psi{const float*a,*b,*inv;};struct Phase{Psi psi[6];};struct Parameters{Phase phase[2];};
// ACTUAL_CONSTANTS_AND_HELPERS
struct Read{int phase,output,source,version,dz,dy,dx;};
// CANONICAL_READS
static void require(bool v,const char*why){if(!v)throw std::runtime_error(why);}
static uint32_t pattern(int field,int version,int z,int y,int x){uint32_t h=uint32_t(field+1)*0x9e3779b9u^uint32_t(version)*0x85ebca6bu^uint32_t(z+1)*0xc2b2ae35u^uint32_t(y+1)*0x27d4eb2fu^uint32_t(x+1)*0x165667b1u;h^=h>>16;return (h&0x007fffffu)|0x3e800000u;}
static float value(int size,int field,int version,int z,int y,int x){
 const int delta[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
 return z>=0&&y>=0&&x>=0&&z<size+delta[field][0]&&y<size+delta[field][1]&&x<size+delta[field][2]?std::bit_cast<float>(pattern(field,version,z,y,x)):0.f;
}
static size_t profiles_checked=0,reads_checked=0,stores_checked=0;
template<int T>static void test(int size,int oz,int oy,int ox,int n,int phase){
 constexpr uint32_t sentinel=0x42bad123u;std::vector<float>guarded(6*PAD+2,std::bit_cast<float>(sentinel));float*frame=guarded.data()+1;
 float profiles[864];Parameters p{};
 for(int i=0;i<864;++i)profiles[i]=std::bit_cast<float>((uint32_t(i)*73244475u&0x007fffffu)|0x3e800000u);
 for(int slot=0;slot<12;++slot)p.phase[slot/6].psi[slot%6]={profiles+slot*72,profiles+slot*72+24,profiles+slot*72+48};
 std::set<int>allocated;
 for(int i=0;i<864;++i){const int address=StaticProfileSlot(i),f=address/PAD;require(address>=3*PAD&&address<6*PAD&&allocated.insert(address).second,"Profile words alias/outsideEframe");
  bool reserved=false;for(int z=0;z<CZ&&!reserved;++z)for(int y=0;y<CY&&!reserved;++y)for(int x=0;x<CX;++x)if(!StaticFieldLive(f,z,y,x)&&f*PAD+StaticSlot(z,y,x)==address){reserved=true;break;}
  require(reserved,"Profile occupies a live field slot");
 }
 for(int tid=0;tid<T;++tid){threadIdx.x=tid;StaticStageProfiles<T>(p,frame);}
 for(int f=0;f<6;++f)for(int z=0;z<CZ;++z)for(int y=0;y<CY;++y)for(int x=0;x<CX;++x){
  const int version=phase&&f<3?n:n-1;StaticStoreField(frame,f,z,y,x,value(size,f,version,oz-version-1+z,oy-version-1+y,ox-version-1+x));++stores_checked;
 }
 for(int i=0;i<864;++i){require(std::bit_cast<uint32_t>(frame[StaticProfileSlot(i)])==std::bit_cast<uint32_t>(profiles[i]),"Field init/owned/ghost guard clobbers independent profile");++profiles_checked;}
 require(std::bit_cast<uint32_t>(guarded.front())==sentinel&&std::bit_cast<uint32_t>(guarded.back())==sentinel,"Shared frame overrun");
 for(int z=1;z<=BZ;++z)for(int y=1;y<=BY;++y)for(int x=1;x<=BX;++x)for(const auto&r:reads){
  if(r.phase!=phase)continue;const int index=phase?StaticReadSlot<1>(r.source,z,y,x,r.dz,r.dy,r.dx):StaticReadSlot<0>(r.source,z,y,x,r.dz,r.dy,r.dx);
  require(index>=0&&index<6*PAD&&!allocated.count(index),"Canonical field route aliases a profile or escapes frame");
  const int px=ox-n-1+x+r.dx,py=oy-n-1+y+r.dy,pz=oz-n-1+z+r.dz,version=n-1+r.version;
  const auto expected=value(size,r.source,version,pz,py,px);require(std::bit_cast<uint32_t>(frame[index])==std::bit_cast<uint32_t>(expected),"Time-independent canonical field route has wrong field/point/version");++reads_checked;
 }
}
int main(){
 // Exhaustive route and profile-placement sets are translation/time invariant;
 // physical pattern cases add lower/upper native support clipping.
 for(int size:{25,64})for(int origin:{0,7,21})for(int n:{1,4,16,31})for(int phase:{0,1}){test<512>(size,origin,origin+8,origin+16,n,phase);test<1024>(size,origin,origin+8,origin+16,n,phase);}
 std::cout<<"{\"passed\":true,\"actual_field_route_checks\":"<<reads_checked<<",\"independent_profile_checks\":"<<profiles_checked<<",\"guarded_field_stores\":"<<stores_checked<<",\"scope\":\"Actual StaticReadSlot/StaticSlot/StaticStoreField/StaticProfileSlot/StaticStageProfiles512and1024 against canonical readset and independent physical FP32patterns;864distinctcachedprofilebits/guards. No CUDAordering or FDTD FPclaim\"}\n";
}
