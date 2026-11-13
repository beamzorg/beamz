#include "cuda_native_cohort_dependencies.h"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>

using Graph=BeamzNativeMovingCohorts;
using Plan=BeamzNativeMovingPlan;
static std::string negative_witness;
static std::vector<std::set<int>> required(const Plan&p,const Graph&g,bool reuse){
 std::vector<std::set<int>> dep(g.offsets.size()-1);
 if(reuse)for(size_t c=0;c<dep.size();++c)dep[c].insert(g.dependencies.begin()+g.offsets[c],g.dependencies.begin()+g.offsets[c+1]);
 else{const int ny=p.counts[1],nx=p.counts[2];
  for(size_t t=0;t<p.tile_ids.size();++t){int z=t/(ny*nx),y=(t/nx)%ny,x=t%nx;
   for(int bits=1;bits<8;++bits){int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;if(z<dz||y<dy||x<dx)continue;int old=t-dz*ny*nx-dy*nx-dx;
    if(g.tile_cohort[t]%6!=g.tile_cohort[old]%6)dep[g.tile_cohort[t]].insert(g.tile_cohort[old]);}
  }
 }
 int last[6]={-1,-1,-1,-1,-1,-1};
 for(size_t c=0;c<dep.size();++c)if(p.policy_offsets[(c/6)*7+c%6]!=p.policy_offsets[(c/6)*7+c%6+1]){if(last[c%6]>=0)dep[c].insert(last[c%6]);last[c%6]=c;}
 return dep;
}
static bool schedule(const Plan&p,const Graph&g,uint32_t seed,bool reuse){
 auto dep=required(p,g,reuse);std::vector<bool> done(dep.size(),false);std::vector<int>owners(4*p.max_frontier,-1);std::mt19937 rng(seed);size_t remaining=g.active_cohorts;
 while(remaining){std::vector<int>ready;
  for(size_t c=0;c<dep.size();++c){if(done[c]||p.policy_offsets[(c/6)*7+c%6]==p.policy_offsets[(c/6)*7+c%6+1])continue;bool okay=true;for(int d:dep[c])okay&=done[d];if(okay)ready.push_back(c);}
  assert(!ready.empty());int c=seed==0?ready.back():ready[rng()%ready.size()],r=c/6,policy=c%6;
  for(int i=p.policy_offsets[r*7+policy];i<p.policy_offsets[r*7+policy+1];++i){int t=p.tile_ids[i];owners[(r&3)*p.max_frontier+p.slots[t]]=t;}
  for(int i=p.policy_offsets[r*7+policy];i<p.policy_offsets[r*7+policy+1];++i){int t=p.tile_ids[i],z=t/(p.counts[1]*p.counts[2]),y=(t/p.counts[2])%p.counts[1],x=t%p.counts[2];
   // The native body publishes its owner tag even for empty windows, then
   // returns before consuming any producer tags. Match that actual guard.
   int first=1,last=p.depth,q[3]={z,y,x};for(int a=0;a<3;++a){int origin=q[a]*p.owned[a];first=std::max(first,origin-p.extent[a]+1);last=std::min(last,origin+p.owned[a]-1);}if(first>last)continue;
   for(int bits=1;bits<8;++bits){int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;if(z<dz||y<dy||x<dx)continue;int old=t-dz*p.counts[1]*p.counts[2]-dy*p.counts[2]-dx,rank=r-dz-dy-dx;
    if(owners[(rank&3)*p.max_frontier+p.slots[old]]!=old){
     if(!reuse&&negative_witness.empty()){int owner=(rank&3)*p.max_frontier+p.slots[old];std::ostringstream w;w<<"{\"seed\":"<<seed<<",\"consumer_cohort\":"<<c<<",\"consumer_tile\":"<<t<<",\"producer_tile\":"<<old<<",\"physical_owner_slot\":"<<owner<<",\"overwriting_tile\":"<<owners[owner]<<"}";negative_witness=w.str();}return false;
    }
   }
  }
  done[c]=true;--remaining;
 }
 return true;
}
int main(){
 using namespace beamz::cuda::moving;
 size_t cases=0,tiles=0,raw=0,reuse=0,edges=0,schedules=0,closure_checks=0,negative=0;
 const int add[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
 std::cout<<"{\"cases\":[";bool first=true;
 for(auto dims:std::vector<std::array<int,3>>{{25,25,25},{33,47,61},{64,96,128},{136,136,136},{512,512,512},{568,568,568},{6000,25,25}})
 for(int depth:{4,16,32})for(int pattern:{0,1}){
  BeamzBuffer f[6]{};for(int c=0;c<6;++c){f[c].rank=3;f[c].element_type=kBeamzF32;for(int a=0;a<3;++a)f[c].dims[a]=dims[a]+add[c][a];}
  BeamzSourceGroupLaunch group[9]{};const int32_t* starts[9]{};int32_t start[6]={dims[0]/2,dims[1]/2,dims[2]/2,dims[0]/2,dims[1]/2,dims[2]/2};group[3].coefficients.rank=4;group[3].coefficients.dims[0]=2;for(int a=1;a<4;++a)group[3].coefficients.dims[a]=1;starts[3]=start;
  auto p=BeamzBuildNativeMovingPlan(f,depth,{{0,0,0},{dims[0]/2,dims[1]/2,dims[2]/2}},group,starts,12,true);
  if(pattern){const uint8_t flags[6]={0,2,1,16,17,3};for(size_t t=0;t<p.tile_flags.size();++t)p.tile_flags[t]=flags[(t*1103515245u+12345u)%6];p.policy_offsets.clear();
   for(size_t r=0;r<p.bulk_counts.size();++r){auto a=p.tile_ids.begin()+p.rank_offsets[r],b=p.tile_ids.begin()+p.rank_offsets[r+1];std::stable_sort(a,b,[&](int a,int b){return Policy(p.tile_flags[a])<Policy(p.tile_flags[b]);});auto it=a;
    for(int k=0;k<6;++k){p.policy_offsets.push_back(it-p.tile_ids.begin());while(it!=b&&Policy(p.tile_flags[*it])==k)++it;}p.policy_offsets.push_back(b-p.tile_ids.begin());for(auto it=a;it!=b;++it)p.slots[*it]=it-a;
   }
  }
  auto g=BeamzBuildNativeMovingCohorts(p);auto deps=required(p,g,true);
  const size_t words=(deps.size()+63)/64;std::vector<std::vector<uint64_t>>anc(deps.size(),std::vector<uint64_t>(words));
  for(size_t c=0;c<deps.size();++c)for(int d:deps[c]){assert(d<int(c));for(size_t w=0;w<words;++w)anc[c][w]|=anc[d][w];anc[c][d/64]|=uint64_t(1)<<(d%64);}
  auto before=[&](int a,int b){++closure_checks;assert(a!=b&&(anc[b][a/64]&(uint64_t(1)<<(a%64))));};
  const int ny=p.counts[1],nx=p.counts[2];std::vector<int>last(4*p.max_frontier,-1);
  for(size_t r=0;r<p.bulk_counts.size();++r)for(int i=p.rank_offsets[r];i<p.rank_offsets[r+1];++i){int t=p.tile_ids[i],z=t/(ny*nx),y=(t/nx)%ny,x=t%nx;
   for(int bits=1;bits<8;++bits){int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;if(z>=dz&&y>=dy&&x>=dx)before(g.tile_cohort[t-dz*ny*nx-dy*nx-dx],g.tile_cohort[t]);}
   int owner=(r&3)*p.max_frontier+p.slots[t],old=last[owner];
   if(old>=0){before(g.tile_cohort[old],g.tile_cohort[t]);int oz=old/(ny*nx),oy=(old/nx)%ny,ox=old%nx;
    for(int bits=1;bits<8;++bits){int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;if(oz+dz<p.counts[0]&&oy+dy<ny&&ox+dx<nx)before(g.tile_cohort[old+dz*ny*nx+dy*nx+dx],g.tile_cohort[t]);}
   }last[owner]=t;
  }
  for(int seed=0;seed<4;++seed){assert(schedule(p,g,seed,true));++schedules;if(!schedule(p,g,seed,false))++negative;}
  assert(g.last_policy_cohort.size()==6);
  if(!first)std::cout<<",";first=false;
  std::cout<<"{\"shape\":["<<dims[0]<<","<<dims[1]<<","<<dims[2]<<"],\"depth\":"<<depth<<",\"synthetic_policy_stress\":"<<pattern<<",\"tiles\":"<<p.tile_ids.size()<<",\"cohorts\":"<<g.active_cohorts<<",\"cross_stream_edges\":"<<g.dependencies.size()<<",\"maximum_dependency_distance\":"<<g.maximum_dependency_distance<<"}";
  ++cases;tiles+=p.tile_ids.size();raw+=g.raw_links;reuse+=g.reuse_links;edges+=g.dependencies.size();
 }
 assert(negative>0);
 // Per-point CPML history has the same affine owner. Between consecutive
 // physical updates each owner axis advances0or1; all nonlocal handoffs are
 // therefore a subset of the seven imported-neighbor edges already proved.
 size_t psi_handoffs=0;for(int d:{4,16,32})for(int z=0;z<65;++z)for(int y=0;y<49;++y)for(int x=0;x<37;++x)for(int n=1;n<=d;++n){int p[3]={z,y,x},owned[3]={kOwnedZ,kOwnedY,kOwnedX},bits=0;for(int a=0;a<3;++a){int old=(p[a]+n-1)/owned[a],next=(p[a]+n)/owned[a];assert(next-old==0||next-old==1);bits|=(next-old)<<a;}assert(bits<8);psi_handoffs+=bool(bits);}
 std::cout<<"],\"case_count\":"<<cases<<",\"tiles\":"<<tiles<<",\"raw_links\":"<<raw<<",\"reuse_links\":"<<reuse<<",\"cross_stream_edges\":"<<edges<<",\"hazard_reachability_checks\":"<<closure_checks<<",\"topological_schedules\":"<<schedules<<",\"missing_reuse_edges_rejected\":"<<negative<<",\"psi_affine_handoffs\":"<<psi_handoffs<<",\"missing_reuse_witness\":"<<negative_witness<<",\"pass\":true}\n";
}
