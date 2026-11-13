// Exact immutable material facts, independent of byte/block storage format.
// Included by the private native_moving.cc candidate and its CPU oracle tests.
#include <unordered_map>

BeamzNativeMovingUniformE BeamzCertifyNativeMovingUniformE(
    const BeamzBuffer*fields,const BeamzBuffer*tables,const BeamzBuffer*codes,
    const void*const*host_tables,const void*const*host_codes,
    const BeamzNativeMovingPlan&plan){
  BeamzNativeMovingUniformE result;
  result.admitted.assign(plan.tile_flags.size(),0);
  int64_t sizes[3]{};
  for(int c=0;c<3;++c){
    const auto&f=fields[3+c];const auto&t=tables[c];const auto&code=codes[c];
    if(f.rank!=3||t.rank!=1||t.element_type!=kBeamzF32||t.dims[0]<1||
       t.dims[0]>INT32_MAX||code.element_type!=kBeamzS32||
       !host_tables[c]||!host_codes[c])return result;
    int64_t size=1;
    for(int a=0;a<3;++a){
      if(f.dims[a]<1||f.dims[a]>INT32_MAX||size>INT32_MAX/f.dims[a])return result;
      size*=f.dims[a];
    }
    if(code.rank==1){
      if(t.dims[0]>256||code.dims[0]!=(size+3)/4)return result;
    }else if(code.rank==2){
      if(code.dims[0]!=1||code.dims[1]!=(size+31)/32)return result;
      const auto*headers=static_cast<const int32_t*>(host_codes[c]);
      for(int64_t b=0;b<code.dims[1];++b){
        const int64_t h=headers[b],start=h<0?~h:h;
        const int64_t count=h<0?1:std::min<int64_t>(32,size-32*b);
        if(start<0||start>t.dims[0]-count)
          throw std::invalid_argument("uniform-E block header outside coefficient pool");
      }
    }else return result;
    sizes[c]=size;
  }
  std::fill(result.admitted.begin(),result.admitted.end(),1);
  for(int c=0;c<3;++c){
    const auto&f=fields[3+c];
    const auto*pool=static_cast<const uint32_t*>(host_tables[c]);
    const auto*words=static_cast<const uint32_t*>(host_codes[c]);
    const auto*headers=static_cast<const int32_t*>(host_codes[c]);
    uint32_t lookup[256]{};
    if(codes[c].rank==1)std::memcpy(lookup,pool,size_t(tables[c].dims[0])*4);
    const auto value=[&](int64_t i)->uint32_t{
      if(codes[c].rank==1)return lookup[(words[i>>2]>>(8*(i&3)))&255u];
      const int64_t h=headers[i>>5];return pool[h<0?~h:h+(i&31)];
    };
    std::unordered_map<uint32_t,uint64_t>counts;
    if(codes[c].rank==1){
      uint64_t frequency[256]{};
      for(int64_t i=0;i<sizes[c];++i)++frequency[(words[i>>2]>>(8*(i&3)))&255u];
      for(int i=0;i<256;++i)if(frequency[i])counts[lookup[i]]+=frequency[i];
    }else{
      for(int64_t b=0;b<codes[c].dims[1];++b){
        const int64_t h=headers[b],n=std::min<int64_t>(32,sizes[c]-32*b);
        if(h<0)counts[pool[~h]]+=n;
        else for(int64_t lane=0;lane<n;++lane)++counts[pool[h+lane]];
      }
    }
    uint32_t common=0;uint64_t best=0;
    for(const auto&entry:counts)
      if(entry.second>best||(entry.second==best&&entry.first<common)){
        common=entry.first;best=entry.second;
      }
    std::memcpy(&result.values[c],&common,4);result.bits[c]=common;
    if(!std::isfinite(result.values[c])){
      std::fill(result.admitted.begin(),result.admitted.end(),0);return result;
    }
    const size_t sx=size_t(f.dims[2])+1,sy=size_t(f.dims[1])+1,sz=size_t(f.dims[0])+1;
    if(sx>std::numeric_limits<size_t>::max()/sy||
       sx*sy>std::numeric_limits<size_t>::max()/sz/4)
      throw std::invalid_argument("uniform-E prefix allocation overflows");
    const size_t plane=sx*sy,cells=plane*sz;
    result.peak_scratch_bytes=std::max(result.peak_scratch_bytes,cells*4);
    std::vector<int32_t>prefix(cells,0);
    for(int z=1;z<=f.dims[0];++z)for(int y=1;y<=f.dims[1];++y)for(int x=1;x<=f.dims[2];++x){
      const int64_t linear=(int64_t(z-1)*f.dims[1]+y-1)*f.dims[2]+x-1;
      const size_t i=size_t(z)*plane+size_t(y)*sx+x;
      const int64_t sum=int64_t(value(linear)!=common)+prefix[i-plane]+int64_t(prefix[i-sx])+prefix[i-1]
          -int64_t(prefix[i-plane-sx])-prefix[i-plane-1]-int64_t(prefix[i-sx-1])+prefix[i-plane-sx-1];
      prefix[i]=int32_t(sum);
    }
    for(size_t tile=0;tile<result.admitted.size();++tile){
      if(!result.admitted[tile])continue;
      const int q[3]={int(tile/(plan.counts[1]*plan.counts[2])),
          int((tile/plan.counts[2])%plan.counts[1]),int(tile%plan.counts[2])};
      int lo[3],hi[3];
      for(int a=0;a<3;++a){
        const int64_t origin=int64_t(q[a])*plan.owned[a];
        lo[a]=int(std::clamp<int64_t>(origin-plan.depth,0,f.dims[a]));
        hi[a]=int(std::clamp<int64_t>(origin+plan.owned[a]-1,0,f.dims[a]));
      }
      int64_t bad=0;
      for(int corner=0;corner<8;++corner){
        const int z=(corner&1)?hi[0]:lo[0],y=(corner&2)?hi[1]:lo[1],x=(corner&4)?hi[2]:lo[2];
        const int ones=(corner&1)+((corner>>1)&1)+((corner>>2)&1);
        bad+=(ones&1?1:-1)*int64_t(prefix[size_t(z)*plane+size_t(y)*sx+x]);
      }
      if(bad<0)throw std::runtime_error("uniform-E inclusion/exclusion became negative");
      if(bad)result.admitted[tile]=0;
    }
  }
  result.supported=true;return result;
}
