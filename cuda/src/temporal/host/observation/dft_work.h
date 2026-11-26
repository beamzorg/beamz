#pragma once
#include <cstdint>

namespace beamz::cuda::temporal {
inline constexpr int kDftPointsPerBlock=128;
inline constexpr int kDftFrequencyBatch=8;

struct DftWork {
 int point_blocks;
 uint64_t blocks;
};

// Host admission and CUDA dispatch must enumerate the same monitor rectangles.
#ifdef __CUDACC__
__host__ __device__
#endif
inline DftWork DftWorkFor(int frequencies,int points){
 if(frequencies<=0||points<=0)return {};
 const int point_blocks=1+(points-1)/kDftPointsPerBlock;
 return {point_blocks,uint64_t(point_blocks)*(1+(frequencies-1)/kDftFrequencyBatch)};
}
}
