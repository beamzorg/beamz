// Exact rank encoding of sorted raw observation keys. Within a field and one
// physical row, observed X positions in a16-cell block have consecutive raw IDs.
// One64-bit record holds the first raw ID and16presence bits; empty rows are0.
struct BeamzNativeRowObservation {
 std::vector<int32_t>coarse;
 std::vector<uint64_t>leaf;
 std::array<int,3>grid{},origin{};
 size_t device_bytes=0;
};
inline int BeamzMovingRawOffset(const BeamzNativeRowObservation& rows,
                               int component,int z,int y,int x) {
 const int iz=z-rows.origin[0],iy=(y>>3)-rows.origin[1],ix=(x>>4)-rows.origin[2];
 if(unsigned(iz)>=unsigned(rows.grid[0])||unsigned(iy)>=unsigned(rows.grid[1])||unsigned(ix)>=unsigned(rows.grid[2]))return -1;
 const int encoded=rows.coarse[(iz*rows.grid[1]+iy)*rows.grid[2]+ix];
 if(encoded<0||!(encoded&(1u<<component)))return -1;
 const uint64_t row=rows.leaf[size_t(unsigned(encoded)>>6)*48+component*8+(y&7)];
 const unsigned mask=unsigned(row&65535),bit=1u<<(x&15);
 return mask&bit?int((row>>16)+unsigned(__builtin_popcount(mask&(bit-1)))):-1;
}
template<bool Materialize=true>
inline BeamzNativeRowObservation BeamzBuildNativeMovingRowObservation(
 const BeamzBuffer*fields,const std::vector<int64_t>&keys){
 BeamzNativeRowObservation plan;int64_t prefix[7]{};int maximum[3]{};
 for(int c=0;c<6;++c){int64_t size=1;
  if(fields[c].rank!=3)throw std::invalid_argument("row observations require rank3 fields");
  for(int a=0;a<3;++a){const int64_t n=fields[c].dims[a];if(n<1||n>INT32_MAX||size>INT32_MAX/n)throw std::invalid_argument("row field offsets exceed int32");size*=n;maximum[a]=std::max(maximum[a],int(n));}
  prefix[c+1]=prefix[c]+size;
 }
 plan.grid={maximum[0],int((int64_t(maximum[1])+7)/8),int((int64_t(maximum[2])+15)/16)};
 int64_t cells=1;for(int n:plan.grid){if(cells>INT32_MAX/n)throw std::invalid_argument("row coarse offsets exceed int32");cells*=n;}
 if(keys.size()>INT32_MAX)throw std::invalid_argument("row raw IDs exceed int32");
 // Keep the row directory only over the observed Yee support. Validate raw
 // keys before translating them into field coordinates for the bounding pass.
 if(cells>=4096&&!keys.empty()){
  std::array<int,3> lo=plan.grid,hi{};int field=0;
  for(int64_t key:keys){
   if(key<0||key>=prefix[6]||key<prefix[field])throw std::invalid_argument("invalid row observation key");
   while(field<5&&key>=prefix[field+1])++field;
   const int local=int(key-prefix[field]),nx=fields[field].dims[2],ny=fields[field].dims[1];
   const int q[]={local/(nx*ny),(local/nx%ny)>>3,(local%nx)>>4};
   for(int a=0;a<3;++a){lo[a]=std::min(lo[a],q[a]);hi[a]=std::max(hi[a],q[a]);}
  }
  cells=1;
  for(int a=0;a<3;++a){plan.origin[a]=lo[a];plan.grid[a]=1+hi[a]-lo[a];cells*=plan.grid[a];}
 }
 // Admission only needs the number of occupied tiles. One bit per tile
 // suffices to count exactly the same leaf records execution will upload.
 std::vector<uint64_t> occupied;
 if constexpr(Materialize)plan.coarse.assign(size_t(cells),-1);
 else occupied.resize((size_t(cells)+63)/64);
 size_t slots=0;
 int component=0;
 int64_t previous=-1;
 for(size_t raw=0;raw<keys.size();){
  const int64_t key=keys[raw];
  if(key<0||key>=prefix[6]||key<=previous)throw std::invalid_argument("row raw keys must be valid unique sorted int64");
  while(component<5&&key>=prefix[component+1])++component;
  const auto&field=fields[component];
  const int local=int(key-prefix[component]),nx=int(field.dims[2]),ny=int(field.dims[1]);
  const int physical_row=local/nx,z=physical_row/ny,y=physical_row%ny,x=local%nx;
  const int tile=((z-plan.origin[0])*plan.grid[1]+(y>>3)-plan.origin[1])*plan.grid[2]+(x>>4)-plan.origin[2];
  // Consecutive X samples share one mask, including in a partial final tile.
  const int available=std::min(16-(x&15),nx-x);
  const size_t limit=std::min(keys.size()-raw,size_t(available));
  size_t run=1;
  while(run<limit&&keys[raw+run]==key+int64_t(run))++run;
  if constexpr(Materialize){
   int32_t& encoded=plan.coarse[tile];
   if(encoded<0){
    if(slots>size_t(INT32_MAX>>6))throw std::invalid_argument("row observation slot overflows");
    encoded=int32_t(slots++<<6);
    plan.leaf.resize(slots*48,0);
   }
   encoded|=1u<<component;
   uint64_t& row=plan.leaf[size_t(unsigned(encoded)>>6)*48+component*8+(y&7)];
   const unsigned mask=unsigned(row&65535u),bits=((1u<<run)-1)<<(x&15);
   if(mask){
    if((mask&bits)||raw!=(row>>16)+unsigned(__builtin_popcount(mask)))throw std::logic_error("observation row rank is not contiguous");
   }else row=uint64_t(raw)<<16;
   row|=bits;
  }else{
   auto& word=occupied[unsigned(tile)/64];
   const uint64_t bit=uint64_t{1}<<(tile&63);
   if(!(word&bit)){word|=bit;++slots;}
  }
  previous=key+int64_t(run)-1;
  raw+=run;
 }
 if(slots>size_t(INT32_MAX>>6)+1)throw std::invalid_argument("row observation slot overflows");
 plan.device_bytes=size_t(cells)*sizeof(int32_t)+slots*48*sizeof(uint64_t);
 return plan;
}
