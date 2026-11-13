// Independent packet-lifetime/typed-edge audit of the actual paired planner.
#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>
#include "native_paired_cohorts.h"
static void require(bool v,const char*why){if(!v)throw std::runtime_error(why);}
static size_t cases=0,hazards=0,queue_nodes=0;
static void test(const std::array<int,3>&shape,int depth,int compaction){
 BeamzBuffer fields[6]{};const int delta[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
 for(int f=0;f<6;++f){fields[f].rank=3;fields[f].element_type=kBeamzF32;for(int a=0;a<3;++a)fields[f].dims[a]=shape[a]+delta[f][a];}
 BeamzSourceGroupLaunch groups[9]{};const int32_t*starts[9]{};int32_t start[6]={12,12,shape[2]/3,12,12,shape[2]/3};
 for(int g:{5,7}){auto&f=groups[g].coefficients;f.rank=4;f.dims[0]=2;f.dims[1]=shape[0]-24;f.dims[2]=shape[1]-24;f.dims[3]=1;starts[g]=start;}
 auto p=BeamzBuildNativeMovingPlan(fields,depth,{{0,0,0},{shape[0]/2,shape[1]/2,shape[2]/2}},groups,starts,12,true);
 if(compaction){for(int t:p.tile_ids)if((t%5)!=2)p.tile_flags[t]|=64;BeamzCompactNativeMovingUniformNoIO(p,true);if(compaction==2)BeamzCompactNativeMovingUniformObservers(p,true);}
 const auto q=BeamzBuildNativePairedCohorts(p);const int nz=p.counts[0],ny=p.counts[1],nx=p.counts[2],plane=ny*nx,original=p.tile_ids.size(),nodes=((nz+1)/2)*plane;
 require(q.original_count==original&&q.plane==plane&&q.owner_slots.size()==size_t(original)&&q.tile_ids.size()==size_t(nodes)&&q.epochs==5,"Original/supernode geometry or ownership sizes");
 auto super=[&](int t){return (t/plane/2)*plane+t%plane;};
 std::vector<std::set<int>>start_parents(nodes),finish_parents(nodes),start_children(nodes),finish_children(nodes);
 std::map<int,int>last;
 for(int t:p.tile_ids){
  const int z=t/plane,y=(t/nx)%ny,x=t%nx,s=super(t),owner=q.owner_slots[t];
  require(owner>=0&&owner<q.owner_count,"Original tile packet owner coverage");
  require(owner==((z+y+x)%5)*p.max_frontier+p.slots[t],"Five-rank original native packet identity");
  for(int parent:{z?t-plane:-1,y?t-nx:-1,x?t-1:-1})if(parent>=0&&super(parent)!=s)start_parents[s].insert(super(parent));
  auto old=last.find(owner);
  if(old!=last.end()){
   require(q.previous_holder[t]==old->second,"Original predecessor packet");const int previous=old->second,oz=previous/plane,oy=(previous/nx)%ny,ox=previous%nx;
   for(int dz=0;dz<=int(oz+1<nz);++dz)for(int dy=0;dy<=int(oy+1<ny);++dy)for(int dx=0;dx<=int(ox+1<nx);++dx){
    const int reader=previous+dz*plane+dy*nx+dx;require(super(reader)!=s,"Self reader retirement after fusion");finish_parents[s].insert(super(reader));++hazards;
   }
  }else require(q.previous_holder[t]==-1,"Unexpected first-owner predecessor");
  last[owner]=t;
 }
 std::vector<int>pending(nodes),started(nodes),done(nodes),active,ready;
 for(int s=0;s<nodes;++s){
  pending[s]=start_parents[s].size()+finish_parents[s].size();require(pending[s]==q.initial_pending[s],"Typed initial pending count");
  for(int parent:start_parents[s])start_children[parent].insert(s);
  for(int parent:finish_parents[s])finish_children[parent].insert(s);
 }
 for(int parent=0;parent<nodes;++parent){std::set<int>actual(q.finish_children.begin()+q.finish_offsets[parent],q.finish_children.begin()+q.finish_offsets[parent+1]);require(actual==finish_children[parent],"All upper/lower old writer and7reader completion CSR");}
 for(int s=0;s<nodes;++s)if(!pending[s])ready.push_back(s);
 std::set<int>roots;for(int s=q.initial_heads[0];s>=0;s=q.initial_next[s])require(roots.insert(s).second,"Unique initial roots");
 require(roots==std::set<int>(ready.begin(),ready.end())&&roots.size()==size_t(q.root_counts[0]),"Actual initial LIFO root coverage");
 require(q.Bytes()==4*(size_t(7)*nodes+size_t(2)*original+q.finish_children.size()+9),"Exact uploaded metadata accounting");
 require(q.CapacityBytes()==4*(size_t(23)*nodes+size_t(2)*original+9),"Exact reserved metadata accounting");
 std::mt19937 random(13+compaction);size_t completed=0;
 while(completed<size_t(nodes)){
  while(!ready.empty()&&active.size()<5){const int s=ready.back();ready.pop_back();require(!started[s],"Supernode started twice");
   for(int parent:start_parents[s])require(started[parent],"External parent not resident/complete");
   for(int parent:finish_parents[s])require(done[parent],"Live old packet writer/reader");
   started[s]=1;active.push_back(s);
   for(int child:start_children[s]){require(pending[child]>0,"START underflow");if(!--pending[child])ready.push_back(child);}
  }
  require(!active.empty(),"Bounded resident supernode queue deadlock");const int i=random()%active.size(),s=active[i];active.erase(active.begin()+i);done[s]=1;++completed;++queue_nodes;
  for(int child:finish_children[s]){require(pending[child]>0,"COMPLETE underflow");if(!--pending[child])ready.push_back(child);}
 }
 ++cases;
}
int main(){
 for(auto shape:std::vector<std::array<int,3>>{{25,29,33},{64,72,80},{96,96,96},{136,136,136},{512,512,512},{564,564,564}})for(int depth:{4,16})for(int compact:{0,1,2})test(shape,depth,compact);
 std::cout<<"{\"passed\":true,\"cases\":"<<cases<<",\"hazard_checks\":"<<hazards<<",\"bounded_queue_nodes\":"<<queue_nodes<<",\"scope\":\"Actual paired planner/5rankowner identities and16hazard CSR; independent all7reader audit, actualnativeplanner plusmaterialprefixcompactions, bounded5resident supernode eventqueue, exactmetadata allocation. No GPU or FP claim\"}\n";
}
