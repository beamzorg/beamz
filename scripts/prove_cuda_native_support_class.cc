// Bind the support fact to the actual native planner and rectangular supports.
#include "native_moving.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
int main(){
 using namespace beamz::cuda::moving;
 const int add[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
 size_t cases=0,tiles=0,owned_supports=0,import_supports=0,startup_supports=0,active_cpml=0;
 for(auto dims:std::vector<std::array<int,3>>{{25,25,25},{33,47,61},{64,96,128},{136,136,136},{512,512,512},{568,568,568}})for(int depth:{4,16,32}){
  BeamzBuffer fields[6]{};for(int c=0;c<6;++c){fields[c].rank=3;fields[c].element_type=kBeamzF32;for(int a=0;a<3;++a)fields[c].dims[a]=dims[a]+add[c][a];}
  auto p=BeamzBuildNativeMovingPlan(fields,depth,{},nullptr,nullptr,12,true);
  for(size_t tile=0;tile<p.tile_ids.size();++tile){if(p.tile_flags[tile]&kNativeBoundary)continue;++tiles;
   int q[3]={int(tile/(p.counts[1]*p.counts[2])),int((tile/p.counts[2])%p.counts[1]),int(tile%p.counts[2])};int origin[3];int first=1,last=depth;
   for(int a=0;a<3;++a){origin[a]=q[a]*p.owned[a];assert(origin[a]-depth>0&&origin[a]+p.owned[a]-1<dims[a]);first=std::max(first,origin[a]-p.extent[a]+1);last=std::min(last,origin[a]+p.owned[a]-1);}
   assert(first==1&&last==depth);
   for(int c=0;c<6;++c)for(int a=0;a<3;++a){assert(origin[a]-1>=0&&origin[a]+p.owned[a]-1<fields[c].dims[a]);++startup_supports;}
   for(int n=1;n<=depth;++n)for(int a=0;a<3;++a){
    for(int k=0;k<p.owned[a];++k){int coordinate=origin[a]-n+k;assert(coordinate>0&&coordinate<dims[a]);
     for(int c=0;c<6;++c){assert(coordinate<fields[c].dims[a]);++owned_supports;}
     // Both Hforward/Ebackward reads are support-safe and Ewall is false.
     assert(coordinate-1>=0&&coordinate+1<=dims[a]);
     for(int family=0;family<2;++family){bool active=coordinate<12||coordinate>=dims[a]+family-12;active_cpml+=active;}
    }
    for(int k=0;k<=p.owned[a];++k){int imported=origin[a]-1+k-n;for(int c=0;c<6;++c){assert(imported>=0&&imported<fields[c].dims[a]);++import_supports;}}
   }
  }++cases;
 }
 // The orthogonal support fact MUST NOT erase CPML. This is a constructive
 // counterexample to incorrectly treating certified support as Policy0/Fast.
 assert(active_cpml>0);
 std::cout<<"{\"cases\":"<<cases<<",\"certified_tiles\":"<<tiles<<",\"owned_axis_support_checks\":"<<owned_supports<<",\"import_axis_support_checks\":"<<import_supports<<",\"startup_axis_support_checks\":"<<startup_supports<<",\"active_cpml_guards_that_must_remain\":"<<active_cpml<<",\"pass\":true}\n";
}
