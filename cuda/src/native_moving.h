// Private research ABI for the complete asymmetric point-loop runner.
// No installed component or public FFI target changes.
#ifndef BEAMZ_CUDA_NATIVE_MOVING_H_
#define BEAMZ_CUDA_NATIVE_MOVING_H_
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "deep_temporal_observation.cuh"

namespace beamz::cuda::moving {
inline constexpr int kOwnedZ=5,kOwnedY=15,kOwnedX=31,kRankEpochs=4;
inline constexpr uint8_t kMayObserve=1,kCpmlZ=2,kCpmlY=4,kCpmlX=8,
                         kMayHaveSources=16,kNativeBoundary=32;
inline constexpr uint32_t kLosslessIsotropicH=1,kBytePackedE=2,kBlockPackedE=4;
inline constexpr int Policy(uint8_t flags){
  if(!flags)return 0;
  if(!(flags&(kMayObserve|kMayHaveSources)))return 1;
  if(flags==kMayObserve)return 2;
  if(flags==kMayHaveSources)return 3;
  if(flags==(kMayObserve|kMayHaveSources))return 4;
  return 5;
}
}

struct BeamzNativeMovingWorkspace {
  const int32_t* tile_ids_device=nullptr;     // rank-ordered tile IDs
  const int32_t* rank_offsets_host=nullptr;   // persistent HOST prefix offsets
  const int32_t* policy_offsets_host=nullptr; // optional[rank,7] six policy ranges
  const int32_t* bulk_counts_host=nullptr;    // first source/PML/monitor-free IDs per rank
  const int32_t* slots_device=nullptr;        // rank-local slot per tile ID
  const uint8_t* tile_flags_device=nullptr;   // observed/source/CPML trajectory facts
  float* trails=nullptr;                     // [owner,version,field,surface]
  uint64_t* owner_tags=nullptr;               // (private epoch<<32)|producer tile
  uint32_t* epoch=nullptr;                    // monotone, never restored with physics
  int32_t* bad=nullptr;                       // sticky OR flag; reset once/invocation
  int counts[3]{};                           // tile counts Z,Y,X
  int max_frontier=0,rank_count=0,tile_count=0,depth=0,epochs=4;
  int owned[3]{5,15,31};
  size_t trail_floats=0;
  const int32_t* observation_tiles=nullptr;
  const int32_t* observation_leaf=nullptr;
  int observation_grid[3]{};
  float h_decay[3]{},h_source[3]{};           // captured with initial device upload
  void* policy_streams[6]{};
  void* policy_fork_event=nullptr;
  void* policy_done_events[6]{};
  bool parallel_policies=false;
  uint32_t data_flags=0;
  beamz::cuda::deep::Observation observation{}; // values[depth,key_count]
};

struct BeamzNativeMovingPlan {
  std::vector<int32_t> tile_ids,rank_offsets,slots;
  std::vector<int32_t> bulk_counts,policy_offsets;
  std::vector<uint8_t> tile_flags;
  std::array<int,3> counts{},owned{5,15,31},extent{};
  int depth=0,epochs=4,max_frontier=0;
  size_t surface_points=0,trail_floats=0,metadata_bytes=0;
};

struct BeamzNativeMovingSparseObservation {
  std::vector<int32_t>coarse,leaf;
  std::array<int,3>grid{};
};
BeamzNativeMovingSparseObservation BeamzBuildNativeMovingSparseObservation(
    const BeamzBuffer*fields,const std::vector<int64_t>&keys);

struct BeamzNativeMovingUniformE {
  std::vector<uint8_t>admitted;
  std::array<float,3>values{};
  std::array<uint32_t,3>bits{};
  size_t peak_scratch_bytes=0;
  bool supported=false;
};
BeamzNativeMovingUniformE BeamzCertifyNativeMovingUniformE(
    const BeamzBuffer*fields,const BeamzBuffer*tables,const BeamzBuffer*codes,
    const void*const*host_tables,const void*const*host_codes,
    const BeamzNativeMovingPlan&plan);

// CPU-only builder. observed_points are physical Z,Y,X of unique raw keys.
// Source starts must be immutable snapshots captured with their GPU upload.
BeamzNativeMovingPlan BeamzBuildNativeMovingPlan(
    const BeamzBuffer*fields,int depth,const std::vector<std::array<int,3>>&observed_points,
    const BeamzSourceGroupLaunch*groups,const int32_t*const*host_starts,int thickness,
    bool partition_policies=false);

// CUDA consumer owns the private epoch increment and ordered rank launches.
// It reads immutable full fields from program, updates packed psi IN PLACE,
// applies point-local post-H/post-E sources, publishes raw staged observations,
// and writes complete final_fields after workspace.depth physical updates.
// pre-E sources are rejected by the initial native admission contract.
int BeamzEnqueueNativeAsymmetric(void*stream,const BeamzProgramLaunch&program,
    const BeamzBuffer*final_fields,int macro_step,const BeamzNativeMovingWorkspace&workspace);

#endif
