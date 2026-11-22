// Native moving-wavefront execution and storage contracts.
#ifndef BEAMZ_CUDA_NATIVE_MOVING_H_
#define BEAMZ_CUDA_NATIVE_MOVING_H_
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include "deep_temporal_observation.cuh"

namespace beamz::cuda::moving {
inline constexpr int kOwnedZ=7,kOwnedY=15,kOwnedX=31,kRankEpochs=4;
inline constexpr int kSlabMacros=16;
inline constexpr bool ValidDepth(int depth) {
  return depth >= 4 && depth <= 32 && depth % 4 == 0;
}
inline constexpr int kTrailSurfacePoints=kOwnedY*(kOwnedX+1)+(kOwnedZ-1)*(kOwnedX+1)+(kOwnedZ-1)*(kOwnedY+1);
// Single-axis neighbors retire after one rank. Edges and the shared corner
// can survive for three more ranks. Pack their four generations into rows.
inline constexpr size_t FaceLines(const std::array<int,3>& n) {
  return size_t(n[1])*n[2]+size_t(n[0])*n[2]+size_t(n[0])*n[1];
}
#if defined(__CUDACC__)
__host__ __device__
#endif
inline constexpr size_t FaceLine(int nz,int ny,int nx,int z,int y,int x,int macro,int axis) {
  const size_t zy=size_t(nz)*ny,zx=size_t(nz)*nx,yx=size_t(ny)*nx;
  return macro*(zy+zx+yx)+(axis==0?size_t(y)*nx+x:axis==1?yx+size_t(z)*nx+x:yx+zx+size_t(z)*ny+y);
}
inline constexpr int kEdgePitch=64;
inline constexpr int kEdgeOwnersPerRow=kTrailSurfacePoints/kEdgePitch;
inline constexpr size_t TrailOwners(size_t frontier, bool tensor) {
  if (tensor) return 4 * frontier;
  return frontier + (4 * frontier + kEdgeOwnersPerRow - 1) / kEdgeOwnersPerRow;
}
inline constexpr std::array<int,3> kTensorOwned{6,8,16};
inline constexpr int kTensorTrailPoints=6*8*16-4*6*14;
inline constexpr uint8_t kMayObserve=1,kCpmlZ=2,kCpmlY=4,kCpmlX=8,
                         kMayHaveSources=16,kNativeBoundary=32;
inline constexpr uint32_t kLosslessIsotropicH=1,kBytePackedE=2,kBlockPackedE=4;
inline constexpr int kCpmlAxes[6]{1,0,0,2,2,1};
struct CpmlWidth { int low=0, high=0; };
using CpmlWidths = std::array<CpmlWidth,12>;
struct SourceCell { std::array<int,3> point; bool pre_e; };
struct Scatter {
  const int32_t* indices;
  const int32_t* ends;
  const float* coefficients;
  const float* waveform;
  int count, samples;
};
struct VolumeRecording {
  float* frames;
  int points, interval, capacity;
  float sign;
};
inline constexpr int Policy(uint8_t flags){
  flags&=63;
  if(!flags)return 0;
  if(!(flags&(kMayObserve|kMayHaveSources)))return 1;
  if(flags==kMayObserve)return 2;
  if(flags==kMayHaveSources)return 3;
  if(flags==(kMayObserve|kMayHaveSources))return 4;
  return 5;
}
}

namespace beamz::cuda::rolling { struct Plan; }

struct BeamzNativeMovingWorkspace {
  const beamz::cuda::rolling::Plan* rolling=nullptr;
  uint8_t* rolling_masks=nullptr;
  int rolling_band=0;
  const int32_t* tile_ids_device=nullptr;     // rank-ordered tile IDs
  const int32_t* rank_offsets_host=nullptr;   // persistent HOST prefix offsets
  const int32_t* policy_offsets_host=nullptr; // optional[rank,7] six policy ranges
  const int32_t* uniform_observer_counts_host=nullptr; // certified policy2 prefix per rank (HOST)
  const int32_t* uniform_noio_counts_host=nullptr; // certified policy1 prefix per rank (HOST)
  const int32_t* bulk_counts_host=nullptr;    // first source/PML/monitor-free IDs per rank
  const int32_t* slots_device=nullptr;        // rank-local slot per tile ID
  const uint8_t* tile_flags_device=nullptr;   // observed/source/CPML trajectory facts
  float* trails=nullptr;                     // [owner,version,field,surface]
  uint64_t* owner_tags=nullptr;               // (private epoch<<32)|producer tile
  uint32_t* epoch=nullptr;                    // monotone, never restored with physics
  int32_t* bad=nullptr;                       // sticky OR flag; reset once/invocation
  int counts[3]{};                           // tile counts Z,Y,X
  int max_frontier=0,rank_count=0,tile_count=0,depth=0,epochs=4;
  int owned[3]{7,15,31};
  size_t trail_floats=0;
  const int32_t* observation_tiles=nullptr;
  const int32_t* observation_leaf=nullptr;
  int observation_grid[3]{},observation_origin[3]{};
  float common_e[3]{};
  float h_decay[3]{},h_source[3]{};           // captured with initial device upload
  void* policy_streams[6]{};
  void* policy_fork_event=nullptr;
  void* policy_done_events[6]{};
  const int32_t*slab_tile_ids_device=nullptr;
  const int32_t*slab_policy_offsets_host=nullptr;
  const int32_t*slab_noio_counts_host=nullptr,*slab_observer_counts_host=nullptr;
  const int32_t*slab_slots_device=nullptr;
  int slab_level_count=0,slab_max_frontier=0,slab_macro_stride=0;
  int raw_slots=3;
  bool parallel_policies=false;
  uint32_t data_flags=0;
  uint32_t source_presence_mask=0;
  beamz::cuda::moving::CpmlWidths cpml_widths{};
  const beamz::cuda::moving::Scatter* scatters=nullptr;
  std::array<int,10> scatter_offsets{};
  const beamz::cuda::moving::VolumeRecording* volumes=nullptr;
  std::array<int,7> volume_offsets{};
  beamz::cuda::deep::Observation observation{}; // values[depth,key_count]
};

struct BeamzNativeMovingPlan {
  std::vector<int32_t> tile_ids,rank_offsets,slots;
  std::vector<int32_t> bulk_counts,policy_offsets,uniform_noio_counts,uniform_observer_counts;
  std::vector<uint8_t> tile_flags;
  std::array<int,3> counts{},owned{7,15,31},extent{};
  int depth=0,epochs=4,max_frontier=0,skew=1;
  size_t surface_points=0,trail_floats=0,metadata_bytes=0;
};

struct BeamzNativeMovingUniformE {
  std::vector<uint8_t>admitted;
  std::array<float,3>values{};
  std::array<uint32_t,3>bits{};
  size_t peak_scratch_bytes=0;
  bool supported=false;
};
void BeamzValidateMovingBlockHeaders(const int32_t* headers,int64_t cells,int64_t pool_size);
BeamzNativeMovingUniformE BeamzCertifyNativeMovingUniformE(
    const BeamzBuffer*fields,const BeamzBuffer*tables,const BeamzBuffer*codes,
    const void*const*host_tables,const void*const*host_codes,
    const BeamzNativeMovingPlan&plan);

// CPU-only builder. observed_points are physical Z,Y,X of unique raw keys.
// Source starts must be immutable snapshots captured with their GPU upload.
BeamzNativeMovingPlan BeamzBuildNativeMovingPlan(
    const BeamzBuffer*fields,int depth,const std::vector<std::array<int,3>>&observed_points,
    const BeamzSourceGroupLaunch*groups,const int32_t*const*host_starts,
    const beamz::cuda::moving::CpmlWidths& widths,
    bool partition_policies=false,
    const std::vector<beamz::cuda::moving::SourceCell>& sparse_sources={},
    bool record_volume=false, bool tensor=false,
    const std::vector<int64_t>& observation_keys={}, unsigned transparent_faces=0);

// Historical host proofs pass a symmetric width.
inline BeamzNativeMovingPlan BeamzBuildNativeMovingPlan(
    const BeamzBuffer* fields,int depth,const std::vector<std::array<int,3>>& points,
    const BeamzSourceGroupLaunch* groups,const int32_t*const* starts,int thickness,
    bool partition_policies=false) {
  beamz::cuda::moving::CpmlWidths widths;
  widths.fill({thickness,thickness});
  return BeamzBuildNativeMovingPlan(fields,depth,points,groups,starts,widths,partition_policies);
}

// Private host-only stable partition after immutable material certification.
void BeamzCompactNativeMovingUniformTiles(BeamzNativeMovingPlan& plan, bool enabled);

// Counts are immutable validated descriptor facts, not coefficient values.
inline uint32_t BeamzNativeMovingSourcePresenceMask(const BeamzSourceGroupLaunch*groups,int count){
 if(!groups||count!=9)throw std::invalid_argument("moving source presence requires nine groups");
 uint32_t mask=0;
 for(int field=0;field<6;++field){const auto sources=groups[3+field].coefficients.dims[0];
  if(sources<0)throw std::invalid_argument("negative source count");
  if(sources>0)mask|=uint32_t(1u<<field);
 }
 return mask;
}
// CUDA consumer owns the private epoch increment and ordered rank launches.
// It reads immutable full fields from program, updates packed psi IN PLACE,
// applies point-local post-H/post-E sources, publishes raw staged observations,
// and writes complete final_fields after workspace.depth physical updates.
// pre-E sources are rejected by the initial native admission contract.
int BeamzPrepareNativeRolling(void*stream,const BeamzProgramLaunch&program,
    const BeamzBuffer*final_fields,const BeamzNativeMovingWorkspace&workspace);
int BeamzEnqueueNativeAsymmetric(void*stream,const BeamzProgramLaunch&program,
    const BeamzBuffer*final_fields,int macro_step,const BeamzNativeMovingWorkspace&workspace);

int BeamzEnqueueNativeMacroSlab(void*stream,const BeamzProgramLaunch&program,
 const BeamzBuffer*final_fields,int macro_step,const BeamzNativeMovingWorkspace&workspace,int level_begin,int level_end,bool begin_epoch);

#endif
