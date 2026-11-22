#ifndef BEAMZ_CUDA_NATIVE_DEEP_H_
#define BEAMZ_CUDA_NATIVE_DEEP_H_

#include <array>
#include <vector>
#include "deep_temporal_observation.cuh"

struct BeamzNativeDeepPlan {
  std::vector<int64_t> keys;
  std::vector<int32_t> remapped_indices;
  std::vector<uint32_t> publication_words;
  std::array<int64_t, 6> field_base{};
  std::array<int, 3> mask_shape{};
};

BeamzNativeDeepPlan BeamzBuildNativeDeepPlan(const BeamzBuffer* fields,
                                            const BeamzBuffer& indices,
                                            const int32_t* host_indices);

#endif
