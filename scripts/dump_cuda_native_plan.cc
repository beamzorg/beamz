// CPU-only snapshot of the frozen native planner before material compaction.
#include "native_moving.h"
#include "native_deep.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
static int32_t read(std::istream&in){int32_t x;in.read(reinterpret_cast<char*>(&x),4);if(!in)throw std::runtime_error("Truncated plan packet");return x;}
template<class T>void array(const T&v){std::cout<<'[';bool first=true;for(auto x:v){if(!first)std::cout<<',';first=false;std::cout<<int64_t(x);}std::cout<<']';}
int main(int argc,char**argv){
 if(argc!=2)return 2;std::ifstream in(argv[1],std::ios::binary);if(read(in)!=0x425A504C)return 3;
 const int depth=read(in);BeamzBuffer fields[6]{};
 for(auto&f:fields){f.rank=3;f.element_type=kBeamzF32;for(int a=0;a<3;++a)f.dims[a]=read(in);}
 BeamzSourceGroupLaunch groups[9]{};std::vector<int32_t>storage[9];const int32_t*starts[9]{};
 for(int g=0;g<9;++g){auto&f=groups[g].coefficients;f.rank=4;f.element_type=kBeamzF32;for(int a=0;a<4;++a)f.dims[a]=read(in);
  storage[g].resize(size_t(f.dims[0])*3);for(auto&x:storage[g])x=read(in);starts[g]=storage[g].data();}
 BeamzBuffer indices{};indices.rank=4;indices.element_type=kBeamzS32;size_t count=1;
 for(int a=0;a<4;++a){indices.dims[a]=read(in);count*=indices.dims[a];}
 std::vector<int32_t>packed(count);in.read(reinterpret_cast<char*>(packed.data()),count*4);if(!in)return 4;
 auto observation=BeamzBuildNativeDeepPlan(fields,indices,packed.data());
 std::vector<std::array<int,3>>points;points.reserve(observation.keys.size());int component=0;
 for(int64_t key:observation.keys){while(component<5&&key>=observation.field_base[component+1])++component;
  const auto&f=fields[component];const int64_t local=key-observation.field_base[component];
  points.push_back({int(local/(f.dims[1]*f.dims[2])),int((local/f.dims[2])%f.dims[1]),int(local%f.dims[2])});}
 const auto p=BeamzBuildNativeMovingPlan(fields,depth,points,groups,starts,12,true);
 std::cout<<"{\"schema\":\"beamz.pre-material-native-plan-snapshot.v1\",\"depth\":"<<depth<<",\"epochs\":"<<p.epochs<<",\"max_frontier\":"<<p.max_frontier<<",\"counts\":";array(p.counts);
 std::cout<<",\"owned\":";array(p.owned);std::cout<<",\"extent\":";array(p.extent);std::cout<<",\"tile_ids\":";array(p.tile_ids);
 std::cout<<",\"tile_flags\":";array(p.tile_flags);std::cout<<",\"slots\":";array(p.slots);std::cout<<",\"rank_offsets\":";array(p.rank_offsets);
 std::cout<<",\"policy_offsets\":";array(p.policy_offsets);std::cout<<",\"observation_keys\":"<<observation.keys.size()<<",\"scope\":\"Actual frozen CPU geometry/source/observation policy planner; before GPU material certification/compaction, not exact post-capture slots\"}\n";
}
