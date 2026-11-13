// Actual emitted direct-Z shared provider versus independent physical patterns.
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
struct ThreadIndex{int x;};static ThreadIndex threadIdx;
struct Field{int z,y,x;};struct Phase{Field own[3];};struct Parameters{Phase phase[2];};
struct FieldPacket{float hx,hy,hz,ex,ey,ez;};
static size_t imports=0,checks=0;
static void require(bool v,const char*why){if(!v)throw std::runtime_error(why);}
static uint32_t pattern(int f,int n,int z,int y,int x){uint32_t h=uint32_t(f+1)*0x9e3779b9u^uint32_t(n)*0x85ebca6bu^uint32_t(z+1)*0xc2b2ae35u^uint32_t(y+1)*0x27d4eb2fu^uint32_t(x+1)*0x165667b1u;h^=h>>16;return (h&0x007fffffu)|0x3e800000u;}
// ACTUAL_GEOMETRY
static bool valid(const Parameters&p,int f,int z,int y,int x){
 const int delta[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
 return z>=0&&y>=0&&x>=0&&z<p.phase[0].own[0].z+delta[f][0]&&y<p.phase[0].own[0].y+delta[f][1]&&x<p.phase[0].own[1].x+delta[f][2];
}
static float value(const Parameters&p,int f,int n,int z,int y,int x){return valid(p,f,z,y,x)?std::bit_cast<float>(pattern(f,n,z,y,x)):0.f;}
template<int Depth,bool Fast,bool Reuse>static FieldPacket ImportPacket(const Parameters&p,const size_t*,int n,int oz,int oy,int ox,int dz,int dy,int dx){
 ++imports;const int z=oz-n-1+dz,y=oy-n-1+dy,x=ox-n-1+dx;
 return {value(p,0,n,z,y,x),value(p,1,n,z,y,x),value(p,2,n,z,y,x),value(p,3,n,z,y,x),value(p,4,n,z,y,x),value(p,5,n,z,y,x)};
}
namespace closed_pipeline{
constexpr size_t GroupStride=49536;
static int Group(){return threadIdx.x/512;}
static int LocalThread(){return threadIdx.x%512;}
template<bool Electric>static bool BeforeGhostImport(int){return true;}
}
// ACTUAL_SHARED_IMPORT
template<bool Electric>static void test(const Parameters&p,int oz,int oy,int ox,int n,int group){
 constexpr uint32_t sentinel=0x42bad123u;
 std::vector<float>memory(101376/4,std::bit_cast<float>(sentinel));float*frame=memory.data()+group*closed_pipeline::GroupStride/4;
 std::vector<uint32_t>parent;
 if(group){
  for(int zs=0;zs<CZ;++zs)for(int ys=0;ys<CY;++ys)for(int xs=0;xs<CX;++xs){
   const int dz=(zs+n+1)&(CZ-1),dy=(ys+n+1)&(CY-1),dx=(xs+n+1)&(CX-1);
   const int z=oz-BZ-n-1+dz,y=oy-n-1+dy,x=ox-n-1+dx;
   for(int c=0;c<3;++c)memory[c*PAD+PhaseSlot(zs,ys,xs)]=value(p,c+(Electric?3:0),n,z,y,x);
  }
  for(int i=0;i<3*PAD;++i)parent.push_back(std::bit_cast<uint32_t>(memory[i]));
 }
 const size_t before=imports;size_t bases[8]{};
 for(int tid=0;tid<512;++tid){threadIdx.x=group*512+tid;PairedScatterGhost<16,512,Electric>(p,frame,bases,n,oz,oy,ox);}
 std::set<int>written;
 for(int i=0;i<PhaseGhostPoints;++i){const auto d=PhaseGhostLogical(i),s=PhaseGhostPhysicalSlot(i,n);const int slot=PhaseSlot(s.z,s.y,s.x);
  for(int c=0;c<3;++c){const int index=c*PAD+slot;require(written.insert(index).second,"Ghost write alias");
   const float expected=value(p,c+(Electric?3:0),n,oz-n-1+d.z,oy-n-1+d.y,ox-n-1+d.x);
   require(std::bit_cast<uint32_t>(frame[index])==std::bit_cast<uint32_t>(expected),"Direct Z family source/version/support mismatch");++checks;
  }
 }
 require(imports-before==size_t(PhaseGhostPoints-(group?BY*BX:0)),"Local Z import must bypass only pure-Z global packet values");
 for(int i=0;i<3*PAD;++i)if(!written.count(i))require(std::bit_cast<uint32_t>(frame[i])==sentinel,"Owned shared frame overwritten by import");
 if(group)for(int i=0;i<3*PAD;++i)require(std::bit_cast<uint32_t>(memory[i])==parent[i],"Consumer modified producer shared payload");
 for(int g=0;g<2;++g)for(size_t byte=3*PAD*4;byte<closed_pipeline::GroupStride;byte+=4)require(std::bit_cast<uint32_t>(memory[(g*closed_pipeline::GroupStride+byte)/4])==sentinel,"Queue/window/Cold storage overwritten");
 for(size_t i=2*closed_pipeline::GroupStride/4;i<memory.size();++i)require(std::bit_cast<uint32_t>(memory[i])==sentinel,"Shared profiles overwritten by field imports");
}
int main(){
 for(int nsize:{25,32,64}){Parameters p{};p.phase[0].own[0]={nsize,nsize,nsize+1};p.phase[0].own[1]={nsize,nsize+1,nsize};
  for(int oz:{0,7,14,21,35,63})for(int oy:{0,15,30})for(int ox:{0,31,62})for(int n=1;n<=32;++n)for(int group:{0,1}){
   test<false>(p,oz,oy,ox,n,group);test<true>(p,oz,oy,ox,n,group);
  }
 }
 std::cout<<"{\"passed\":true,\"actual_shared_provider_checks\":"<<checks<<",\"global_packet_provider_calls\":"<<imports<<",\"scope\":\"Actual emitted PairedScatterGhost, PhaseSlot/ghost rotation and NativeSupportMask against independent physical FP32patterns; both families/groups, clipped faces, shared owned/Cold/profile guards. No synchronization or GPU claim\"}\n";
}
