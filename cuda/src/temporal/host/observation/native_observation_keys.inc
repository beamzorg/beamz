#include <cstring>
// Moving sparse observation needs host keys to compile rows and classify tiles,
// but the CUDA consumer reads only row descriptors, raw values and remapped DFT
// indices. Do not materialize a dense publication mask or upload unused keys.
template<class Function>
inline void BeamzForPackedObservationRows(const BeamzBuffer&indices,
 const int32_t*rows,const int32_t*counts,int monitors,Function&&visit){
 const int neighbors=BeamzDftNeighbors(indices);
 for(int m=0;m<monitors;++m)for(int c=0;c<6;++c){
  const int row=rows[6*m+c];
  if(row>=0)visit((c+3)%6,size_t(row)*neighbors,size_t(counts[5*m+1])*neighbors);
 }
}

inline BeamzNativeDeepPlan BeamzBuildNativeMovingKeyPlan(const BeamzBuffer*fields,
 const BeamzBuffer&indices,const int32_t*host_indices,const int32_t*counts=nullptr,int monitors=0){
 if(indices.element_type!=kBeamzS32||
    (indices.rank!=2&&(indices.rank!=4||indices.dims[1]!=6)))
  throw std::invalid_argument("moving observations require int32 gather indices");
 BeamzNativeDeepPlan plan;int64_t sizes[6]{},prefix=0;
 for(int c=0;c<6;++c){if(fields[c].rank!=3)throw std::invalid_argument("moving observation fields must have rank3");sizes[c]=1;
  for(int axis=0;axis<3;++axis){const int64_t n=fields[c].dims[axis];if(n<1||n>INT32_MAX||sizes[c]>INT32_MAX/n)throw std::invalid_argument("moving field offsets exceed int32");sizes[c]*=n;plan.mask_shape[axis]=std::max(plan.mask_shape[axis],int(n));}
  plan.field_base[c]=prefix;prefix+=sizes[c];
 }
 int64_t count=1;for(int a=0;a<4;++a){if(indices.dims[a]<0||(indices.dims[a]&&count>INT32_MAX/indices.dims[a]))throw std::invalid_argument("moving observation records exceed int32");count*=indices.dims[a];}
 if(count&&!host_indices)throw std::invalid_argument("moving observation indices missing");
 const int64_t stride=indices.dims[2]*indices.dims[3];
 // Build the sorted key set one field at a time. Plane interpolation often
 // supplies the same physical sample through several neighboring rows.
 using Row=std::pair<size_t,size_t>;
 std::array<std::vector<Row>,6> rows;
 if(indices.rank==2){
  BeamzForPackedObservationRows(indices,host_indices,counts,monitors,
   [&](int field,size_t start,size_t width){rows[field].emplace_back(start,width);});
 }else if(stride){
  for(int m=0;m<indices.dims[0];++m)for(int c=0;c<6;++c)
   rows[(c+3)%6].emplace_back(size_t(6*m+c)*stride,stride);
 }
 for(int field=0;field<6;++field){
  int first=INT32_MAX,last=-1;
  size_t samples=0;
  for(const auto& [start,width]:rows[field]){
   samples+=width;
   for(size_t i=start;i<start+width;++i){
    const int x=host_indices[i];
    if(x>=0&&x<sizes[field]){first=std::min(first,x);last=std::max(last,x);}
   }
  }
  if(last<first)continue;
  const size_t words=size_t(last-first)/64+1;
  // A bitmap is bounded by the space needed to sort the sparse input.
  const bool bitmap=words<=samples;
  std::vector<uint64_t> present(bitmap?words:0);
  const auto begin=plan.keys.size();
  for(const auto& [start,width]:rows[field]){
   for(size_t i=start;i<start+width;++i){
    const int x=host_indices[i];
    if(x<first||x>last)continue;
    if(bitmap){const unsigned offset=unsigned(x-first);present[offset/64]|=uint64_t{1}<<(offset%64);}
    else if(plan.keys.size()==begin||plan.keys.back()!=plan.field_base[field]+x)
     plan.keys.push_back(plan.field_base[field]+x);
   }
  }
  if(bitmap){
   for(size_t word=0;word<words;++word)
    for(uint64_t bits=present[word];bits;bits&=bits-1)
     plan.keys.push_back(plan.field_base[field]+first+int64_t(word*64)+__builtin_ctzll(bits));
  }else{
   const auto start=plan.keys.begin()+begin;
   std::sort(start,plan.keys.end());
   plan.keys.erase(std::unique(start,plan.keys.end()),plan.keys.end());
  }
 }
 if(plan.keys.size()>INT32_MAX)throw std::invalid_argument("moving unique observations exceed int32");
 return plan;
}
// Eight-corner interpolation reads a whole point at once. Its raw IDs need
// only enough bits to address the observed field samples, plus a missing ID.
inline size_t BeamzMovingIndexWidth(int neighbors,size_t keys) {
 if(neighbors!=8)return 4;
 return keys<=0xffff ? 2 : keys<=0xffffff ? 3 : 4;
}


inline void BeamzRemapNativeMovingRows(const BeamzBuffer*fields,const BeamzBuffer&indices,
 const int32_t*host_indices,BeamzNativeDeepPlan&plan,const BeamzNativeRowObservation&rows,
 const std::vector<int32_t>&offsets={},const int32_t*counts=nullptr,int packed_monitors=0,size_t width_bytes=4){
 if(width_bytes!=4&&((width_bytes!=2&&width_bytes!=3)||BeamzDftNeighbors(indices)!=8||
     width_bytes<BeamzMovingIndexWidth(8,plan.keys.size())))
  throw std::invalid_argument("invalid moving gather index width");
 // Eight-corner points always start at an eight-index boundary. Separate
 // low words and high bytes so the GPU can read each whole point contiguously.
 const auto allocate=[&](size_t count){
  plan.remapped_indices.assign(count*width_bytes/sizeof(int32_t),-1);
 };
 const auto remap=[&](int native,size_t source,size_t destination,size_t width){
  const auto&field=fields[native];
  const int nx=int(field.dims[2]),ny=int(field.dims[1]);
  const int64_t size=int64_t(nx)*ny*field.dims[0];
  for(size_t i=0;i<width;++i){const int logical=host_indices[source+i];if(logical<0||logical>=size)continue;
   const int physical_row=logical/nx,z=physical_row/ny,y=physical_row%ny,x=logical%nx;
   const int raw=BeamzMovingRawOffset(rows,native,z,y,x);
   if(raw<0||size_t(raw)>=plan.keys.size()||plan.keys[raw]!=plan.field_base[native]+logical)throw std::logic_error("moving row remap differs from canonical sorted key");
   const size_t index=destination+i;
   if(width_bytes==4)plan.remapped_indices[index]=raw;
   else{
    const size_t lane=index%8;
    auto* point=reinterpret_cast<unsigned char*>(plan.remapped_indices.data())+(index-lane)*width_bytes;
    const auto low=static_cast<uint16_t>(raw);
    std::memcpy(point+2*lane,&low,sizeof(low));
    if(width_bytes==3)point[16+lane]=static_cast<unsigned char>(unsigned(raw)>>16);
   }
  }
 };
 if(indices.rank==2){
  allocate(size_t(indices.dims[0])*indices.dims[1]);
  if(width_bytes==4)std::copy_n(host_indices,size_t(packed_monitors)*6,plan.remapped_indices.begin());
  BeamzForPackedObservationRows(indices,host_indices,counts,packed_monitors,
   [&](int native,size_t start,size_t width){remap(native,start,start,width);});
  return;
 }
 const size_t monitors=size_t(indices.dims[0]),stride=size_t(indices.dims[2])*size_t(indices.dims[3]);
 if(!offsets.empty()&&(offsets.size()!=monitors+1||offsets.front()!=0))throw std::invalid_argument("moving compact remap offsets invalid");
 if(!offsets.empty())for(size_t m=0;m<monitors;++m){
  const int64_t width=int64_t(offsets[m+1])-offsets[m];
  if(width<0||width%6||size_t(width/6)>stride||(width&&width/6%indices.dims[3]))throw std::invalid_argument("moving compact remap width invalid");
 }
 const size_t count=offsets.empty()?monitors*6*stride:size_t(offsets.back());
 if(count&&(!host_indices||stride<1))throw std::invalid_argument("moving row remap input invalid");
 allocate(count);
 for(size_t m=0;m<monitors;++m){const size_t width=offsets.empty()?stride:size_t(offsets[m+1]-offsets[m])/6;
  for(int c=0;c<6;++c){
   const size_t source=(6*m+c)*stride,destination=offsets.empty()?source:size_t(offsets[m])+c*width;
   remap((c+3)%6,source,destination,width);
  }
 }
}
