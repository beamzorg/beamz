// The inserted helpers and six component regions are actual emitted CUDA text.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>
#define __host__
#define __device__
#define __forceinline__ inline
struct ThreadIndex{int x=0;};static ThreadIndex threadIdx;
static void require(bool value,const char*why){if(!value)throw std::runtime_error(why);}
constexpr int BZ=7,BY=15,BX=31,CZ=8,CY=16,CX=32,PAD=4096,SurfacePoints=735;
static float token(int f,int v,int z,int y,int x){require(z>=0&&z<8&&y>=0&&y<16&&x>=0&&x<32,"Token outside cache");return float(1+(v*6+f)*PAD+(z*16+y)*32+x);}
static float __shfl_sync(uint32_t mask,float value,int source){
 require(mask==0xffffffffu,"Fullwarp participation required");if(value==0.f)return 0.f;
 int i=int(value)-1,x=i%32,y=(i/32)%16,z=(i/512)%8,f=(i/PAD)%6,v=i/(6*PAD),lane=threadIdx.x%32;
 require(x==lane,"Private source is not in the destination lane before shuffle");
 return token(f,v,z,(y+source-lane+32)%16,source);
}
// ACTUAL_HELPERS
struct Read{int phase,output,field,version,dz,dy,dx;};
// CANONICAL_READS
static size_t operands=0,geometry=0,bank_groups=0;
static int current_z,current_y,current_x,current_n;
struct Environment {bool Valid(int,int,int,int){return true;}};
namespace beamz{namespace cuda{namespace trace_native{
template<int Phase,int Component,bool,class Env,class Reader>float AdvanceNativeCell(Env&,Reader read,int n,int,int,int){
 for(const auto&r:reads)if(r.phase==Phase&&r.output==Phase*3+Component){
  const int shift=Phase==0||r.field>=3?1:0;
  float expected=token(r.field,n-1+r.version,current_z+r.dz-shift,current_y+r.dy-shift,current_x+r.dx-shift);
  require(read(r.field,n-1+r.version,r.dz,r.dy,r.dx)==expected,"Actual generated operand route has wrong point/field/version");++operands;
 }
 return token(Phase*3+Component,n,current_z,current_y,current_x);
}
}}}
template<int T>static void test(int n){
 constexpr int Slots=T==1024?4:8;constexpr int Boundary=T==1024?6*CY*CX:0;
 std::vector<float>shared(3*PAD+Boundary+2,-7.f);float*frame=shared.data()+1,*boundary_h=frame+3*PAD,*boundary_e=boundary_h+3*CY*CX;
 std::vector<float>private_h(T*3*Slots),private_e(T*3*Slots);
 for(int f=0;f<6;++f)for(int z=0;z<8;++z)for(int y=0;y<16;++y)for(int x=0;x<32;++x){
  int tid=(z/Slots)*512+((y-x+32)%16)*32+x,j=z%Slots;const auto d=DiagonalComputePoint<T>(tid,j);
  require(d.z==z&&d.y==y&&d.x==x,"Actual diagonal compute map/inverse failed");++geometry;
  (f<3?private_h:private_e)[(tid*3+f%3)*Slots+j]=token(f,n-1,z,y,x);
  if(f>=3)frame[(f-3)*PAD+DiagonalSlot(z,y,x)]=token(f,n-1,z,y,x);
  if constexpr(T==1024){if(z==3)(f<3?boundary_h:boundary_e)[(f%3)*CY*CX+DiagonalBoundarySlot(y,x)]=token(f,n-1,z,y,x);}
 }
 Environment env;
 for(int phase=0;phase<2;++phase){
  for(int tid=0;tid<T;++tid){threadIdx.x=tid;const int sx=tid&31,u=(tid/32)&15,sy=(sx+u)&15,zbase=T==1024?(tid/512)*4:0;
   float h[3][Slots],e[3][Slots];for(int c=0;c<3;++c)for(int j=0;j<Slots;++j){h[c][j]=private_h[(tid*3+c)*Slots+j];e[c][j]=private_e[(tid*3+c)*Slots+j];}
   for(int j=Slots-1;j>=0;--j){const int zs=zbase+j,z=zs,y=sy,x=sx;current_z=zs;current_y=sy;current_x=sx;
    if(phase==0){
     // ACTUAL_H_REGIONS
    }else{
     // ACTUAL_E_REGIONS
    }
   }
   for(int c=0;c<3;++c)for(int j=0;j<Slots;++j){private_h[(tid*3+c)*Slots+j]=h[c][j];private_e[(tid*3+c)*Slots+j]=e[c][j];}
  }
  // Actual shared publication helper; owned and coalesced ghost writers are
  // disjoint, and jointly complete lowerZ3 snapshots before publication join.
  auto&bank=phase==0?private_h:private_e;float*boundary=phase==0?boundary_h:boundary_e;
  for(int tid=0;tid<T;++tid)for(int j=0;j<Slots;++j){const auto d=DiagonalComputePoint<T>(tid,j);if(d.z&&d.y&&d.x)
   for(int c=0;c<3;++c)DiagonalStoreFamily<T>(frame,boundary,c,d.z,d.y,d.x,bank[(tid*3+c)*Slots+j]);
  }
  std::set<int>ghosts;
  for(int i=0;i<PAD-BZ*BY*BX;++i){const auto d=DiagonalGhostLogical(i);require(!(d.z&&d.y&&d.x)&&ghosts.insert((d.z*16+d.y)*32+d.x).second,"Ghost IO aliases owned storage");
   const int tid=(d.z/Slots)*512+((d.y-d.x+32)%16)*32+d.x,j=d.z%Slots;
   for(int c=0;c<3;++c){const float value=token(phase*3+c,n,d.z,d.y,d.x);DiagonalStoreFamily<T>(frame,boundary,c,d.z,d.y,d.x,value);bank[(tid*3+c)*Slots+j]=value;}
  }
  require(ghosts.size()==PAD-BZ*BY*BX,"Incomplete coalesced ghost IO");
  for(int i=0;i<SurfacePoints;++i){const auto d=DiagonalSurfaceLogical(i);require(Surface(d.z-1,d.y-1,d.x-1)==i,"Native surface inverse mismatch");for(int c=0;c<3;++c)require(frame[c*PAD+DiagonalSlot(d.z,d.y,d.x)]==token(phase*3+c,n,d.z,d.y,d.x),"Coalesced export disagrees with private generated output");}
  require(shared.front()==-7.f&&shared.back()==-7.f,"Shared boundary publication overrun");
 }
 for(int z=0;z<8;++z)for(int u=0;u<16;++u){std::set<int>banks;for(int x=0;x<32;++x)banks.insert(DiagonalSlot(z,(x+u)%16,x)%32);require(banks.size()==32,"Compute warp shared bank conflict");++bank_groups;}
 for(int z=0;z<8;++z)for(int y=0;y<16;++y){std::set<int>banks;for(int x=0;x<32;++x)banks.insert(DiagonalSlot(z,y,x)%32);require(banks.size()==32,"Coalesced IO shared bank conflict");++bank_groups;}
}
int main(){for(int n:{1,4,16,31}){test<512>(n);test<1024>(n);}std::cout<<"{\"passed\":true,\"actual_generated_operands\":"<<operands<<",\"actual_geometry_checks\":"<<geometry<<",\"conflict_free_warp_groups\":"<<bank_groups<<"}\n";}
