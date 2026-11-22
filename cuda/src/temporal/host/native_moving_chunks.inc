// RunMoving exposes only its final state. Sources use absolute integer steps;
// DFT uses origin + absolute elapsed step, and clocks are rebuilt from origin.
// Consequently a caller's progress cuts need not force scalar-kernel tails.
// Align full batches to the physical depth within the supplied staging capacity.
inline std::vector<int> BeamzAlignNativeMovingChunks(const std::vector<int>&requested,int depth,int capacity=256){
 if(requested.empty())throw std::invalid_argument("moving schedule cannot be empty");
 if(!beamz::cuda::moving::ValidDepth(depth))throw std::invalid_argument("temporal depth must be a multiple of four from 4 to 32");
 int64_t total=0;
 for(int count:requested){
  if(count<1)throw std::invalid_argument("moving chunks must be positive");
  total+=count;if(total>INT32_MAX)throw std::invalid_argument("moving total exceeds int32");
 }
 const int batch=(capacity/depth)*depth;
 if(batch<=0)throw std::invalid_argument("moving requires staging for a complete temporal block");
 std::vector<int>result;result.reserve(size_t((total+batch-1)/batch));
 while(total>=batch){result.push_back(batch);total-=batch;}
 if(total)result.push_back(int(total));
 return result;
}
