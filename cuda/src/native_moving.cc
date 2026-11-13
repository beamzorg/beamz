#include "native_moving.h"
#include <algorithm>
#include <limits>
#include <cstring>
#include <cmath>
#include <stdexcept>

BeamzNativeMovingPlan BeamzBuildNativeMovingPlan(
    const BeamzBuffer*fields,int depth,const std::vector<std::array<int,3>>&observed_points,
    const BeamzSourceGroupLaunch*groups,const int32_t*const*host_starts,int thickness,bool partition_policies){
  using namespace beamz::cuda::moving;
  if(depth!=4&&depth!=8&&depth!=16&&depth!=32)throw std::invalid_argument("moving depth must be4/8/16/32");
  if(thickness!=12)throw std::invalid_argument("native moving requires CPML12");
  BeamzNativeMovingPlan p;p.depth=depth;
  int minimum[3]{INT32_MAX,INT32_MAX,INT32_MAX};
  for(int c=0;c<6;++c){
    if(fields[c].rank!=3||fields[c].element_type!=kBeamzF32)throw std::invalid_argument("moving requires six FP32 fields");
    int64_t count=1;
    for(int a=0;a<3;++a){const int64_t n=fields[c].dims[a];
      if(n<1||n>INT32_MAX||count>INT32_MAX/n)throw std::invalid_argument("moving field offsets exceed int32");
      count*=n;p.extent[a]=std::max(p.extent[a],int(n));minimum[a]=std::min(minimum[a],int(n));
    }
  }
  int64_t boxes=1;
  for(int a=0;a<3;++a)if(minimum[a]<=2*thickness)
    throw std::invalid_argument("moving CPML12 requires a positive interior on every axis");
  for(int a=0;a<3;++a){p.counts[a]=(int64_t(p.extent[a])+depth+p.owned[a]-1)/p.owned[a];boxes*=p.counts[a];if(boxes>INT32_MAX)throw std::invalid_argument("moving tile IDs exceed int32");}
  const int nz=p.counts[0],ny=p.counts[1],nx=p.counts[2],ranks=nz+ny+nx-2;
  p.slots.resize(boxes);p.tile_flags.assign(boxes,0);p.tile_ids.reserve(boxes);p.rank_offsets.push_back(0);
  for(int rank=0;rank<ranks;++rank){const int begin=p.tile_ids.size();
    for(int z=0;z<nz;++z)for(int y=0;y<ny;++y){const int x=rank-z-y;if(x<0||x>=nx)continue;
      const int tile=(z*ny+y)*nx+x;p.slots[tile]=p.tile_ids.size()-begin;p.tile_ids.push_back(tile);
      const int origin[]={z*p.owned[0],y*p.owned[1],x*p.owned[2]};int low[3],high[3];uint8_t flags=0;
      for(int a=0;a<3;++a){low[a]=origin[a]-depth;high[a]=origin[a]+p.owned[a]-2;
        if(low[a]<thickness||high[a]>=minimum[a]-thickness)flags|=uint8_t(2<<a);
        // Boundary specialization covers startup high=origin+B-1 and the
        // last imported low=origin-depth-1, beyond owned update points.
        if(low[a]<=0||high[a]+1>=minimum[a])flags|=kNativeBoundary;
      }
      if(groups&&host_starts)for(int g=3;g<9;++g){
        const auto&coeff=groups[g].coefficients;const int32_t*starts=host_starts[g];
        if(coeff.dims[0]&&!starts)throw std::invalid_argument("moving source starts missing");
        for(int s=0;s<coeff.dims[0];++s){bool hit=true;
          for(int a=0;a<3;++a)hit&=low[a]<int64_t(starts[3*s+a])+coeff.dims[a+1]&&high[a]>=starts[3*s+a];
          if(hit)flags|=kMayHaveSources;
        }
      }
      p.tile_flags[tile]=flags;
    }
    p.rank_offsets.push_back(p.tile_ids.size());p.max_frontier=std::max(p.max_frontier,int(p.tile_ids.size())-begin);
  }
  for(const auto&point:observed_points)for(int version=1;version<=depth;++version){int owner[3];
    for(int a=0;a<3;++a){if(point[a]<0||point[a]>=p.extent[a])throw std::invalid_argument("moving observation outside field domain");owner[a]=(point[a]+version)/p.owned[a];}
    const int tile=(owner[0]*ny+owner[1])*nx+owner[2];p.tile_flags[tile]|=kMayObserve;
  }
  // Preserve rank antichains. Both sub-launches finish before the next rank;
  // local slots are recomputed after this operational permutation.
  for(int rank=0;rank<ranks;++rank){
    auto begin=p.tile_ids.begin()+p.rank_offsets[rank],end=p.tile_ids.begin()+p.rank_offsets[rank+1];
    if(partition_policies){
      std::stable_sort(begin,end,[&](int a,int b){return Policy(p.tile_flags[a])<Policy(p.tile_flags[b]);});
      auto cursor=begin;
      for(int policy=0;policy<6;++policy){p.policy_offsets.push_back(cursor-p.tile_ids.begin());while(cursor!=end&&Policy(p.tile_flags[*cursor])==policy)++cursor;}
      p.policy_offsets.push_back(end-p.tile_ids.begin());
      if(cursor!=end)throw std::runtime_error("moving policy partition lost a tile");
    }
    if(!partition_policies)std::stable_partition(begin,end,[&](int tile){return p.tile_flags[tile]==0;});
    // Re-find after the legacy stable partition; policy0 is first in both modes.
    p.bulk_counts.push_back(std::find_if(begin,end,[&](int tile){return p.tile_flags[tile]!=0;})-begin);
    for(auto it=begin;it!=end;++it)p.slots[*it]=it-begin;
  }
  p.surface_points=size_t(p.owned[0])*p.owned[1]*p.owned[2]-size_t(p.owned[0]-1)*(p.owned[1]-1)*(p.owned[2]-1);
  const size_t owners=size_t(p.epochs)*p.max_frontier;
  if(owners>std::numeric_limits<size_t>::max()/depth/6/p.surface_points)throw std::invalid_argument("moving trail allocation overflows");
  p.trail_floats=owners*depth*6*p.surface_points;
  p.metadata_bytes=2*size_t(boxes)*sizeof(int32_t)+size_t(boxes)*sizeof(uint8_t)+owners*sizeof(uint64_t)+sizeof(uint32_t)+sizeof(int32_t);
  if(p.tile_ids.size()!=size_t(boxes)||p.rank_offsets.back()!=boxes)throw std::runtime_error("moving rank coverage mismatch");
  return p;
}

BeamzNativeMovingSparseObservation BeamzBuildNativeMovingSparseObservation(
    const BeamzBuffer*fields,const std::vector<int64_t>&keys){
  BeamzNativeMovingSparseObservation p;int64_t prefix[7]{};int maximum[3]{};
  for(int c=0;c<6;++c){int64_t size=1;
    if(fields[c].rank!=3)throw std::invalid_argument("sparse observations require rank3 fields");
    for(int a=0;a<3;++a){const int64_t n=fields[c].dims[a];if(n<1||n>INT32_MAX||size>INT32_MAX/n)throw std::invalid_argument("sparse field offsets exceed int32");size*=n;maximum[a]=std::max(maximum[a],int(n));}
    prefix[c+1]=prefix[c]+size;
  }
  p.grid={maximum[0],int((int64_t(maximum[1])+7)/8),int((int64_t(maximum[2])+15)/16)};
  int64_t cells=1;for(int n:p.grid){if(cells>INT32_MAX/n)throw std::invalid_argument("sparse coarse offsets exceed int32");cells*=n;}
  if(keys.size()>INT32_MAX)throw std::invalid_argument("sparse raw IDs exceed int32");
  p.coarse.assign(size_t(cells),-1);std::vector<int32_t>occupied;occupied.reserve(keys.size());
  auto locate=[&](int64_t key,int c,int*cell){const auto&f=fields[c];const int64_t local=key-prefix[c],nx=f.dims[2],ny=f.dims[1];const int z=local/(ny*nx),y=(local/nx)%ny,x=local%nx;*cell=(y&7)*16+(x&15);return (z*p.grid[1]+y/8)*p.grid[2]+x/16;};
  int c=0;int64_t previous=-1;
  for(int64_t key:keys){if(key<0||key>=prefix[6]||key<=previous)throw std::invalid_argument("sparse raw keys must be valid unique sorted int64");previous=key;while(c<5&&key>=prefix[c+1])++c;int cell;occupied.push_back(locate(key,c,&cell));}
  std::sort(occupied.begin(),occupied.end());occupied.erase(std::unique(occupied.begin(),occupied.end()),occupied.end());
  if(occupied.size()>std::numeric_limits<size_t>::max()/128/6)throw std::invalid_argument("sparse leaf allocation overflows");
  p.leaf.assign(occupied.size()*128*6,-1);
  for(size_t slot=0;slot<occupied.size();++slot)p.coarse[occupied[slot]]=int32_t(slot);
  c=0;
  for(size_t raw=0;raw<keys.size();++raw){const int64_t key=keys[raw];while(c<5&&key>=prefix[c+1])++c;int cell;const int tile=locate(key,c,&cell);p.leaf[size_t(p.coarse[tile])*128*6+size_t(cell)*6+c]=int32_t(raw);}
  return p;
}

BeamzNativeMovingUniformE BeamzCertifyNativeMovingUniformE(
    const BeamzBuffer*fields,const BeamzBuffer*tables,const BeamzBuffer*codes,
    const void*const*host_tables,const void*const*host_codes,
    const BeamzNativeMovingPlan&plan){
  BeamzNativeMovingUniformE result;result.admitted.assign(plan.tile_flags.size(),0);
  int64_t sizes[3]{};
  for(int c=0;c<3;++c){const auto&f=fields[3+c];const auto&t=tables[c];const auto&code=codes[c];
    if(f.rank!=3||t.rank!=1||t.element_type!=kBeamzF32||t.dims[0]<1||t.dims[0]>256||
       code.rank!=1||code.element_type!=kBeamzS32||!host_tables[c]||!host_codes[c])return result;
    int64_t size=1;for(int a=0;a<3;++a){if(f.dims[a]<1||f.dims[a]>INT32_MAX||size>INT32_MAX/f.dims[a])return result;size*=f.dims[a];}
    if(code.dims[0]!=(size+3)/4)return result;sizes[c]=size;
  }
  std::fill(result.admitted.begin(),result.admitted.end(),1);
  for(int c=0;c<3;++c){const auto&f=fields[3+c];uint32_t lookup[256]{};std::memcpy(lookup,host_tables[c],size_t(tables[c].dims[0])*4);
    const auto*words=static_cast<const uint32_t*>(host_codes[c]);uint64_t counts[256]{};
    for(int64_t i=0;i<sizes[c];++i)++counts[(words[i>>2]>>(8*(i&3)))&255u];
    uint32_t common=0;uint64_t best=0;
    for(int i=0;i<256;++i){uint64_t count=0;for(int j=0;j<256;++j)if(lookup[j]==lookup[i])count+=counts[j];if(count>best||(count==best&&lookup[i]<common)){best=count;common=lookup[i];}}
    std::memcpy(&result.values[c],&common,4);result.bits[c]=common;
    if(!std::isfinite(result.values[c])){std::fill(result.admitted.begin(),result.admitted.end(),0);return result;}
    const size_t sx=size_t(f.dims[2])+1,sy=size_t(f.dims[1])+1,sz=size_t(f.dims[0])+1;
    if(sx>std::numeric_limits<size_t>::max()/sy||sx*sy>std::numeric_limits<size_t>::max()/sz/4)throw std::invalid_argument("uniform-E prefix allocation overflows");
    const size_t plane=sx*sy,cells=plane*sz;result.peak_scratch_bytes=std::max(result.peak_scratch_bytes,cells*4);
    std::vector<int32_t>prefix(cells,0);
    for(int z=1;z<=f.dims[0];++z)for(int y=1;y<=f.dims[1];++y)for(int x=1;x<=f.dims[2];++x){
      const int64_t linear=(int64_t(z-1)*f.dims[1]+y-1)*f.dims[2]+x-1;const unsigned code=(words[linear>>2]>>(8*(linear&3)))&255u;
      const size_t i=size_t(z)*plane+size_t(y)*sx+x;
      const int64_t sum=int64_t(lookup[code]!=common)+prefix[i-plane]+int64_t(prefix[i-sx])+prefix[i-1]-int64_t(prefix[i-plane-sx])-prefix[i-plane-1]-int64_t(prefix[i-sx-1])+prefix[i-plane-sx-1];
      prefix[i]=int32_t(sum);
    }
    for(size_t tile=0;tile<result.admitted.size();++tile){if(!result.admitted[tile])continue;
      const int q[3]={int(tile/(plan.counts[1]*plan.counts[2])),int((tile/plan.counts[2])%plan.counts[1]),int(tile%plan.counts[2])};int lo[3],hi[3];
      for(int a=0;a<3;++a){const int64_t origin=int64_t(q[a])*plan.owned[a];lo[a]=int(std::clamp<int64_t>(origin-plan.depth,0,f.dims[a]));hi[a]=int(std::clamp<int64_t>(origin+plan.owned[a]-1,0,f.dims[a]));}
      int64_t bad=0;for(int corner=0;corner<8;++corner){const int z=(corner&1)?hi[0]:lo[0],y=(corner&2)?hi[1]:lo[1],x=(corner&4)?hi[2]:lo[2];const int ones=(corner&1)+((corner>>1)&1)+((corner>>2)&1);bad+=(ones&1?1:-1)*int64_t(prefix[size_t(z)*plane+size_t(y)*sx+x]);}
      if(bad<0)throw std::runtime_error("uniform-E inclusion/exclusion became negative");if(bad)result.admitted[tile]=0;
    }
  }
  result.supported=true;return result;
}
