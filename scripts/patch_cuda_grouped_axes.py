#!/usr/bin/env python3
"""Lower immutable native edge queues into conservative CPML-axis buckets.

This patches copied CUDA sources only. Original host arrays remain untouched;
run_grouped uploads a stable permutation to its private device queues once.
run_schedule controls use a separate fresh session with original queue order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DECLARATION = r"""#pragma once
#include <cuda_runtime_api.h>
#include "launch.h"
cudaError_t BeamzEnqueueMstepGroupedEdge(cudaStream_t stream,
 const BeamzLaunch&phase,const BeamzBuffer&cells,const int32_t*offsets,int ny,int nx);
"""
SHADER = r"""
#if defined(BEAMZ_CUDA_NATIVE_GROUPED_AXES) && BEAMZ_CUDA_NATIVE_GROUPED_AXES
// One exact data-class-selected kernel per nonempty queue bucket. Included
// axes retain component-specific staggered slab checks; omitted axes provably
// have no valid component in CPML at any queued coordinate.
template<int Phase,int Axes,int Layout>
__global__ void MstepGroupedFullEdge(
 const __grid_constant__ BeamzLaunch launch,
 const __grid_constant__ BeamzBuffer cells,int ny,int nx,MstepIndexDecode decode){
 const int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=cells.dims[0])return;
 const int row=static_cast<const int32_t*>(cells.data)[i];
 const int z=MstepQuotient(static_cast<uint32_t>(row),decode.plane);
 const int plane_row=row-z*ny*nx;
 const int y=MstepQuotient(static_cast<uint32_t>(plane_row),decode.x);
 const int x=plane_row-y*nx;
 UpdateComponent<Phase,0,true,0,false,true,kBeamzF32,Phase==1,false,false,Axes,Layout>(launch,z,y,x);
 UpdateComponent<Phase,1,true,0,false,true,kBeamzF32,Phase==1,false,false,Axes,Layout>(launch,z,y,x);
 UpdateComponent<Phase,2,true,0,false,true,kBeamzF32,Phase==1,false,false,Axes,Layout>(launch,z,y,x);
}
template<int Phase,int Layout>
cudaError_t LaunchMstepGroupedAxes(cudaStream_t stream,const BeamzLaunch&phase,
 const BeamzBuffer&cells,const int32_t*offsets,int ny,int nx){
 const auto decode=MakeMstepIndexDecode(ny,nx);
#define BEAMZ_GROUP_AXIS(M) if(offsets[M+1]>offsets[M]){ \
 auto part=cells;part.data=static_cast<int32_t*>(cells.data)+offsets[M]; \
 part.dims[0]=offsets[M+1]-offsets[M]; \
 MstepGroupedFullEdge<Phase,M,Layout><<<(part.dims[0]+255)/256,256,0,stream>>>(phase,part,ny,nx,decode); \
 if(auto error=cudaPeekAtLastError();error!=cudaSuccess)return error; }
 BEAMZ_GROUP_AXIS(0);BEAMZ_GROUP_AXIS(1);BEAMZ_GROUP_AXIS(2);BEAMZ_GROUP_AXIS(3);
 BEAMZ_GROUP_AXIS(4);BEAMZ_GROUP_AXIS(5);BEAMZ_GROUP_AXIS(6);BEAMZ_GROUP_AXIS(7);
#undef BEAMZ_GROUP_AXIS
 return cudaSuccess;
}
#endif
"""
LAUNCHER = r"""
#if defined(BEAMZ_CUDA_NATIVE_GROUPED_AXES) && BEAMZ_CUDA_NATIVE_GROUPED_AXES
#include "native_grouped.h"
cudaError_t BeamzEnqueueMstepGroupedEdge(cudaStream_t stream,
 const BeamzLaunch&phase,const BeamzBuffer&cells,const int32_t*offsets,int ny,int nx){
 if(!offsets||phase.uniform_cpml_thickness<=0||phase.metallic_edges||phase.nterms!=6)
  return BeamzEnqueueMstepEdgeFull(stream,phase,cells,ny,nx);
 for(int t=0;t<6;++t)if(phase.inputs[31+t].element_type!=kBeamzF32||phase.outputs[3+t].element_type!=kBeamzF32)
  return BeamzEnqueueMstepEdgeFull(stream,phase,cells,ny,nx);
 if(phase.phase==0){
  for(int i=6;i<12;++i)if(phase.inputs[i].rank!=0||phase.inputs[i].element_type!=kBeamzF32)
   return BeamzEnqueueMstepEdgeFull(stream,phase,cells,ny,nx);
  return LaunchMstepGroupedAxes<0,-1>(stream,phase,cells,offsets,ny,nx);
 }
 const int rank=phase.inputs[9].rank;
 for(int c=0;c<3;++c)if(phase.inputs[9+c].element_type!=kBeamzS32||phase.inputs[9+c].rank!=rank||phase.inputs[6+c].rank!=1||phase.inputs[6+c].element_type!=kBeamzF32)
  return BeamzEnqueueMstepEdgeFull(stream,phase,cells,ny,nx);
 if(rank==1)return LaunchMstepGroupedAxes<1,1>(stream,phase,cells,offsets,ny,nx);
 if(rank==2)return LaunchMstepGroupedAxes<1,2>(stream,phase,cells,offsets,ny,nx);
 return BeamzEnqueueMstepEdgeFull(stream,phase,cells,ny,nx);
}
#endif
"""
PREPARE = r"""
#if BEAMZ_CUDA_NATIVE_GROUPED_AXES
  void PrepareGrouped(bool grouped) {
    const int requested=grouped?1:0;
    if(grouped_mode_>=0&&grouped_mode_!=requested)
      throw std::invalid_argument("queue order is fixed per session; use a fresh session for grouped/control");
    if(!grouped){grouped_mode_=0;return;}
    if(grouped_ready_)return;
    if(!mstep_||!attributes_.cpml_enabled||attributes_.metric_kind!=0||!BEAMZ_CUDA_MSTEP_FULL_INTERMEDIATES)
      throw std::invalid_argument("grouped axes requires uniform CPML full-intermediate Mstep");
    BindOutputs();BeamzNativeLaunchStorage prepared;
    BeamzNativePrepare(attributes_,inputs_.data(),outputs_.data(),source_offset_,monitor_offset_,workspace_count_,true,&prepared);
    const auto&h=prepared.mstep.program.h_ab;const auto&e=prepared.mstep.program.e_ab;
    const int thickness=h.uniform_cpml_thickness;
    if(thickness<=0||thickness!=e.uniform_cpml_thickness)
      throw std::invalid_argument("grouped axes requires symmetric uniform positive CPML");
    int minimum[3],extent[3]{};
    for(int a=0;a<3;++a){minimum[a]=INT32_MAX;
      for(int c=0;c<6;++c){minimum[a]=std::min(minimum[a],int(base_inputs_[c].dims[a]));extent[a]=std::max(extent[a],int(base_inputs_[c].dims[a]));}
    }
    const int64_t volume=int64_t(extent[0])*extent[1]*extent[2];
    if(volume>INT32_MAX)throw std::invalid_argument("grouped queue dimensions exceed int32");
    auto axis_mask=[&](int32_t row){
      const int coordinates[]={row/(extent[1]*extent[2]),(row/extent[2])%extent[1],row%extent[2]};int mask=0;
      for(int a=0;a<3;++a)if(coordinates[a]<thickness||coordinates[a]>=minimum[a]-thickness)mask|=1<<a;
      return mask;
    };
    // Validate both queues before any operational device mutation.
    for(int q=0;q<2;++q){const int index=132+q;const auto*rows=static_cast<const int32_t*>(host_[index].data);
      for(int64_t i=0;i<base_inputs_[index].dims[0];++i)
        if(rows[i]<0||rows[i]>=volume||(i&&rows[i]<=rows[i-1]))
          throw std::invalid_argument("grouped queues require sorted unique valid physical rows");
    }
    for(int q=0;q<2;++q){const int index=132+q;const int count=int(base_inputs_[index].dims[0]);
      const auto*rows=static_cast<const int32_t*>(host_[index].data);int32_t counts[8]{};
      for(int i=0;i<count;++i)++counts[axis_mask(rows[i])];
      auto&offsets=grouped_offsets_[q];offsets[0]=0;
      for(int mask=0;mask<8;++mask)offsets[mask+1]=offsets[mask]+counts[mask];
      auto cursors=offsets;std::vector<int32_t>ordered(count);
      for(int i=0;i<count;++i){const int mask=axis_mask(rows[i]);ordered[cursors[mask]++]=rows[i];}
      for(int mask=0;mask<8;++mask)if(cursors[mask]!=offsets[mask+1])throw std::runtime_error("grouped queue cardinality failed");
      if(count)Check(cudaMemcpy(base_inputs_[index].data,ordered.data(),host_[index].bytes,cudaMemcpyHostToDevice),"upload private grouped queue permutation");
    }
    grouped_mode_=1;grouped_ready_=true;
  }
#endif
"""
MEMBERS = r"""
#if BEAMZ_CUDA_NATIVE_GROUPED_AXES
  int grouped_mode_=-1;
  bool grouped_ready_=false;
  std::array<std::array<int32_t,9>,2>grouped_offsets_{};
#endif
"""


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Patch site count{text.count(old)}: {old[:90]!r}")
    return text.replace(old, new, 1)


def patch(source):
    if source.resolve() == (Path(__file__).resolve().parents[1] / "cuda").resolve():
        raise ValueError("Copied candidate sources required")
    src = source / "src"
    if (src / "native_grouped.h").exists():
        raise ValueError("Already patched")
    (src / "native_grouped.h").write_text(DECLARATION)
    p = src / "launch.h"
    s = p.read_text()
    s = once(
        s,
        "  BeamzBuffer edge_cells;\n};",
        "  BeamzBuffer edge_cells;\n  const int32_t* first_axis_offsets=nullptr;\n  const int32_t* edge_axis_offsets=nullptr;\n};",
    )
    p.write_text(s)
    p = source / "CMakeLists.txt"
    s = p.read_text()
    s += '\noption(BEAMZ_CUDA_NATIVE_GROUPED_AXES "Research native grouped CPML axis queues" OFF)\nset_property(SOURCE src/native_extension.cc src/program.cu src/update.cu APPEND PROPERTY COMPILE_DEFINITIONS "BEAMZ_CUDA_NATIVE_GROUPED_AXES=$<BOOL:${BEAMZ_CUDA_NATIVE_GROUPED_AXES}>")\n'
    p.write_text(s)
    p = src / "mstep.cuh"
    p.write_text(p.read_text() + SHADER)
    p = src / "update.cu"
    p.write_text(p.read_text() + LAUNCHER)
    p = src / "program.cu"
    s = p.read_text()
    s = once(
        s,
        '#include "launch.h"',
        '#include "launch.h"\n#if defined(BEAMZ_CUDA_NATIVE_GROUPED_AXES) && BEAMZ_CUDA_NATIVE_GROUPED_AXES\n#include "native_grouped.h"\n#endif',
    )
    marker = "  for (const auto &b : buffers)\n    BeamzAppendBufferKey(&key, b);"
    s = once(
        s,
        marker,
        r"""#if defined(BEAMZ_CUDA_NATIVE_GROUPED_AXES) && BEAMZ_CUDA_NATIVE_GROUPED_AXES
  if(launch.first_axis_offsets&&launch.edge_axis_offsets){
    key.append("grouped-axes-v1");
    key.append(reinterpret_cast<const char*>(launch.first_axis_offsets),9*sizeof(int32_t));
    key.append(reinterpret_cast<const char*>(launch.edge_axis_offsets),9*sizeof(int32_t));
  }
#endif
"""
        + marker,
    )
    for kind, member in [("first", "first_cells"), ("edge", "edge_cells")]:
        marker = f"        return BeamzEnqueueMstepEdgeFull(\n            stream, phase, launch.{member}, extent[1], extent[2]);"
        new = (
            f"""#if defined(BEAMZ_CUDA_NATIVE_GROUPED_AXES) && BEAMZ_CUDA_NATIVE_GROUPED_AXES
        return BeamzEnqueueMstepGroupedEdge(stream,phase,launch.{member},launch.{kind}_axis_offsets,extent[1],extent[2]);
#else
"""
            + marker
            + "\n#endif"
        )
        s = once(s, marker, new)
    p.write_text(s)
    p = src / "native_extension.cc"
    s = p.read_text()
    s = once(s, "#include <algorithm>", "#include <algorithm>\n#include <array>")
    s = once(
        s,
        '#include "native_runtime.h"',
        '#include "native_runtime.h"\n#ifndef BEAMZ_CUDA_NATIVE_GROUPED_AXES\n#define BEAMZ_CUDA_NATIVE_GROUPED_AXES 0\n#endif',
    )
    s = once(
        s,
        "  nb::dict RunSchedule(const std::vector<int>& chunks) {",
        "  nb::dict RunSchedule(const std::vector<int>& chunks) {return RunScheduleImpl(chunks,false);}\n  nb::dict RunGrouped(const std::vector<int>& chunks) {return RunScheduleImpl(chunks,true);}\n  nb::dict RunScheduleImpl(const std::vector<int>& chunks,bool grouped) {",
    )
    start = s.index("  nb::dict RunScheduleImpl(")
    end = s.index("  nb::dict RunDeep4(", start)
    part = s[start:end]
    marker = "    float cuda_ms = 0, physics_ms = 0, copy_ms = 0;"
    part = once(
        part,
        marker,
        '#if BEAMZ_CUDA_NATIVE_GROUPED_AXES\n    PrepareGrouped(grouped);\n#else\n    if(grouped)throw std::invalid_argument("grouped axes support not compiled");\n#endif\n'
        + marker,
    )
    marker = "        const int error = mstep_ ? BeamzLaunchMstep(stream_.value, launch.mstep)"
    part = once(
        part,
        marker,
        "#if BEAMZ_CUDA_NATIVE_GROUPED_AXES\n        if(grouped){launch.mstep.first_axis_offsets=grouped_offsets_[0].data();launch.mstep.edge_axis_offsets=grouped_offsets_[1].data();}\n#endif\n"
        + marker,
    )
    s = s[:start] + part + s[end:]
    s = once(
        s,
        " private:\n  std::unique_lock<std::mutex> Guard() const {",
        " private:\n" + PREPARE + "\n  std::unique_lock<std::mutex> Guard() const {",
    )
    s = once(
        s,
        "  BeamzNativeAttributes attributes_;",
        MEMBERS + "\n  BeamzNativeAttributes attributes_;",
    )
    s = once(
        s,
        '    result["deep_d4_compiled"] = bool(BEAMZ_CUDA_DEEP_D4);',
        r"""    result["grouped_axes_compiled"]=bool(BEAMZ_CUDA_NATIVE_GROUPED_AXES);
#if BEAMZ_CUDA_NATIVE_GROUPED_AXES
    result["grouped_axes_ready"]=grouped_ready_;
    result["grouped_queue_mode"]=grouped_mode_;
    result["grouped_first_offsets"]=nb::cast(std::vector<int32_t>(grouped_offsets_[0].begin(),grouped_offsets_[0].end()));
    result["grouped_edge_offsets"]=nb::cast(std::vector<int32_t>(grouped_offsets_[1].begin(),grouped_offsets_[1].end()));
    result["grouped_extra_gpu_bytes"]=0;
#endif
    result["deep_d4_compiled"] = bool(BEAMZ_CUDA_DEEP_D4);""",
    )
    s = once(
        s,
        '      .def("run_deep4", &NativeSession::RunDeep4, "chunks"_a)',
        '      .def("run_grouped", &NativeSession::RunGrouped, "chunks"_a)\n      .def("run_deep4", &NativeSession::RunDeep4, "chunks"_a)',
    )
    p.write_text(s)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    a = p.parse_args()
    patch(a.source)


if __name__ == "__main__":
    main()
