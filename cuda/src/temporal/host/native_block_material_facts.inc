// Exact immutable material facts, independent of byte/block storage format.
// Included by the private native_moving.cc candidate and its CPU oracle tests.

void BeamzValidateMovingBlockHeaders(const int32_t* headers,int64_t cells,int64_t pool_size){
  for(int64_t block=0;block<(cells+31)/32;++block){
    const int64_t h=headers[block],first=h<0?~h:h;
    const int64_t width=h<0?1:std::min<int64_t>(32,cells-32*block);
    if(first>pool_size-width)throw std::invalid_argument("uniform-E block header outside coefficient pool");
  }
}

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
      BeamzValidateMovingBlockHeaders(static_cast<const int32_t*>(host_codes[c]),size,t.dims[0]);
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
    uint32_t common=0;uint64_t best=0;
    {
      // Counts fit31 bits because field offsets are bounded by INT32_MAX.
      // A half-full packed histogram bounds mode selection to64KiB per component.
      constexpr size_t capacity=4096,slots=2*capacity;
      std::vector<uint64_t>histogram(slots,0);size_t distinct=0;
      result.peak_scratch_bytes=std::max(result.peak_scratch_bytes,histogram.capacity()*sizeof(uint64_t));
      const auto add=[&](uint32_t bits,uint64_t count){
        size_t slot=uint32_t(bits*2654435761u)>>19;
        while(histogram[slot]&&uint32_t(histogram[slot])!=bits)slot=(slot+1)&(slots-1);
        if(!histogram[slot]){if(distinct==capacity)return false;histogram[slot]=bits;++distinct;}
        histogram[slot]+=count<<32;return true;
      };
      bool exact=true;
      if(codes[c].rank==1){
        uint64_t frequency[256]{};
        for(int64_t i=0;i<sizes[c];++i)++frequency[(words[i>>2]>>(8*(i&3)))&255u];
        for(int i=0;i<256;++i)if(frequency[i])add(lookup[i],frequency[i]);
      }else for(int64_t b=0;b<codes[c].dims[1]&&exact;++b){
        const int64_t h=headers[b],n=std::min<int64_t>(32,sizes[c]-32*b);
        if(h<0)exact=add(pool[~h],n);
        else for(int64_t lane=0;lane<n;++lane)if(!add(pool[h+lane],1)){exact=false;break;}
      }
      if(exact){
        for(uint64_t entry:histogram){const uint32_t bits=uint32_t(entry);const uint64_t count=entry>>32;
          if(count>best||(count==best&&bits<common)){common=bits;best=count;}
        }
      }else{
        // Weighted majority chooses a deterministic candidate with constant space.
        // It is only a hint: every admitted trajectory is still verified below.
        uint64_t balance=0;
        const auto vote=[&](uint32_t bits,uint64_t count){
          if(bits==common)balance+=count;
          else if(count>=balance){common=bits;balance=count-balance;}
          else balance-=count;
        };
        for(int64_t b=0;b<codes[c].dims[1];++b){
          const int64_t h=headers[b],n=std::min<int64_t>(32,sizes[c]-32*b);
          if(h<0)vote(pool[~h],n);
          else for(int64_t lane=0;lane<n;++lane)vote(pool[h+lane],1);
        }
      }
    } // Histogram storage retires before the spatial verification index.
    std::memcpy(&result.values[c],&common,4);result.bits[c]=common;
    if(!std::isfinite(result.values[c])){
      std::fill(result.admitted.begin(),result.admitted.end(),0);return result;
    }
    // A zero-mismatch component needs no spatial index at all.
    if(best==uint64_t(sizes[c]))continue;
    // An XY mismatch plane determines exactly the same set of tile columns
    // until its bits change. Consume consecutive equal planes as one Z run.
    const int nz=f.dims[0],ny=f.dims[1],nx=f.dims[2];
    const size_t wx=(size_t(nx)+31)/32,pitch=wx+1;
    std::vector<uint32_t> current(size_t(ny)*wx),previous(current.size());
    std::vector<int32_t> prefix((size_t(ny)+1)*pitch);
    const size_t scratch=(2*current.size()+prefix.size())*sizeof(uint32_t);
    result.peak_scratch_bytes=std::max(result.peak_scratch_bytes,scratch);

    const auto exclude=[&](int first_z,int last_z){
      if(std::none_of(previous.begin(),previous.end(),[](uint32_t word){return word!=0;}))return;
      for(int y=0;y<ny;++y){
        int32_t row_sum=0;
        for(size_t bx=0;bx<wx;++bx){
          row_sum+=previous[size_t(y)*wx+bx]!=0;
          prefix[size_t(y+1)*pitch+bx+1]=prefix[size_t(y)*pitch+bx+1]+row_sum;
        }
      }
      // [q*B-depth,q*B+B-1) intersects [first_z,last_z).
      const int z0=int((int64_t(first_z)+1)/plan.owned[0]);
      const int z1=int(std::min<int64_t>(plan.counts[0],1+(int64_t(last_z)+plan.depth-1)/plan.owned[0]));
      for(int qy=0;qy<plan.counts[1];++qy){
        const int y0=int(std::clamp<int64_t>(int64_t(qy)*plan.owned[1]-plan.depth,0,ny));
        const int y1=int(std::min<int64_t>(int64_t(qy+1)*plan.owned[1]-1,ny));
        if(y0>=y1)continue;
        for(int qx=0;qx<plan.counts[2];++qx){
          const int x0=int(std::clamp<int64_t>(int64_t(qx)*plan.owned[2]-plan.depth,0,nx));
          const int x1=int(std::min<int64_t>(int64_t(qx+1)*plan.owned[2]-1,nx));
          if(x0>=x1)continue;
          const size_t left=size_t(x0)/32,right=(size_t(x1)+31)/32;
          const auto count=[&](size_t a,size_t b){
            return prefix[size_t(y1)*pitch+b]-prefix[size_t(y1)*pitch+a]
                 -prefix[size_t(y0)*pitch+b]+prefix[size_t(y0)*pitch+a];
          };
          if(!count(left,right))continue;
          // Whole interior words are an exact witness. At the two partial
          // words only the occupied bits inside the physical trajectory count.
          bool bad=right>left+2 && count(left+1,right-1);
          const uint32_t lo=~uint32_t(0)<<(x0&31);
          const uint32_t hi=~uint32_t(0)>>(31-((x1-1)&31));
          for(int y=y0;y<y1&&!bad;++y){
            const auto* row=previous.data()+size_t(y)*wx;
            bad=right==left+1?bool(row[left]&lo&hi):bool((row[left]&lo)|(row[right-1]&hi));
          }
          if(bad)for(int qz=z0;qz<z1;++qz)
            result.admitted[(size_t(qz)*plan.counts[1]+qy)*plan.counts[2]+qx]=0;
        }
      }
    };
    int first_z=0;
    for(int z=0;z<nz;++z){
      for(int y=0;y<f.dims[1];++y)for(size_t bx=0;bx<wx;++bx){
        const int start=int(bx*32),width=std::min(32,int(f.dims[2])-start);
        const int64_t base=(int64_t(z)*f.dims[1]+y)*f.dims[2]+start;
        uint32_t mask=0;
        for(int lane=0;lane<width;){
          const int64_t linear=base+lane;
          if(codes[c].rank==2&&headers[linear>>5]<0){
            const int count=std::min(width-lane,32-int(linear&31));
            if(pool[~headers[linear>>5]]!=common){
              const uint32_t bits=count==32?~uint32_t(0):(uint32_t(1)<<count)-1;
              mask|=bits<<lane;
            }
            lane+=count;
          }else{if(value(linear)!=common)mask|=uint32_t(1)<<lane;++lane;}
        }
        current[size_t(y)*wx+bx]=mask;
      }
      if(z && current==previous)continue;
      if(z)exclude(first_z,z);
      first_z=z;previous.swap(current);
    }
    exclude(first_z,nz);
  }
  result.supported=true;return result;
}
