#!/usr/bin/env python3
"""Patch an isolated CUDA source snapshot with an opt-in mixed D2/CPML pair.

The research schedule keeps ordinary first-step envelopes and observations,
uses frozen auxiliary copies for overlapping CPML tiles, and replaces only
covered second-step edge rows. It never edits the installed component.
"""

from __future__ import annotations

import argparse
from pathlib import Path

HEADER = r"""#pragma once
#include "launch.h"
struct BeamzHybridLaunch {
  BeamzMstepLaunch mstep{};
  BeamzBuffer shadows[12]{};
  BeamzBuffer residual{};
};
int BeamzLaunchMstepHybrid(void* stream, const BeamzHybridLaunch& hybrid);
cudaError_t BeamzEnqueueHybridCpml(cudaStream_t stream, const BeamzLaunch& h,
 const BeamzLaunch& e,const BeamzBuffer* final_fields,
 const BeamzSourceGroupLaunch* groups,int step);
"""
UPDATE = r"""
#if defined(BEAMZ_CUDA_NATIVE_HYBRID) && BEAMZ_CUDA_NATIVE_HYBRID
#include "native_hybrid.h"
cudaError_t BeamzEnqueueHybridCpml(cudaStream_t stream,const BeamzLaunch& h,
 const BeamzLaunch& e,const BeamzBuffer* final_fields,
 const BeamzSourceGroupLaunch* groups,int step) {
  PairFieldOutputs final{}; FusedHSources hs{},es{};
  hs.step=es.step=step;
  PairCpmlGrid grid{};
  for(int c=0;c<6;++c) {
    final.fields[c]=final_fields[c];
    for(int a=0;a<3;++a)grid.extent[a]=std::max(grid.extent[a],int(final_fields[c].dims[a]));
  }
  for(int c=0;c<3;++c){hs.groups[c]=groups[3+c];es.groups[c]=groups[6+c];}
  for(int a=0;a<3;++a){
    int limit=int(h.outputs[0].dims[a]);
    for(int c=1;c<3;++c)limit=std::min(limit,int(h.outputs[c].dims[a]));
    grid.field_high[a]=limit;grid.high[a]=h.uniform_cpml_thickness+2;
    grid.boundary[a]=limit-h.uniform_cpml_thickness-2;
  }
  const auto hp=MakePairCpmlPhase(h),ep=MakePairCpmlPhase(e);
  const bool half=h.inputs[31].element_type==kBeamzBF16;
#define HYBRID_MASK(M) { \
 constexpr bool face=M==1||M==2||M==4; \
 constexpr int X=face?(M==4?15:M==2?16:32):16; \
 constexpr int Y=face?(M==4?16:M==2?15:4):8; \
 constexpr int Z=face?16:4; \
 cudaError_t error; \
 if(half) error=LaunchTemporalCpmlMask<true,true,X,Y,Z,M,2>(stream,hp,ep,final,hs,es,{},grid); \
 else error=LaunchTemporalCpmlMask<true,false,X,Y,Z,M,2>(stream,hp,ep,final,hs,es,{},grid); \
 if(error!=cudaSuccess)return error; \
 }
 HYBRID_MASK(1);HYBRID_MASK(2);HYBRID_MASK(4);
 HYBRID_MASK(3);HYBRID_MASK(5);HYBRID_MASK(6);HYBRID_MASK(7);
#undef HYBRID_MASK
 return cudaSuccess;
}
#endif
"""
PREPARE = r"""
#if BEAMZ_CUDA_NATIVE_HYBRID
  void PrepareHybrid() {
    if(hybrid_ready_)return;
    if(!mstep_||!attributes_.cpml_enabled||attributes_.metric_kind!=0)
      throw std::invalid_argument("hybrid requires uniform CPML Mstep");
    BindOutputs();BeamzNativeLaunchStorage prepared;
    BeamzNativePrepare(attributes_,inputs_.data(),outputs_.data(),source_offset_,
                       monitor_offset_,workspace_count_,true,&prepared);
    const auto&h=prepared.mstep.program.h_ab;
    const auto&e=prepared.mstep.program.e_ab;
    const int thickness=h.uniform_cpml_thickness;
    if(thickness<=0||e.uniform_cpml_thickness!=thickness)
      throw std::invalid_argument("hybrid requires uniform positive CPML thickness");
    for(int i=6;i<12;++i)if(h.inputs[i].rank!=0)
      throw std::invalid_argument("hybrid requires scalar H coefficients");
    for(int c=0;c<3;++c)if(e.inputs[9+c].element_type!=kBeamzS32)
      throw std::invalid_argument("hybrid requires exact packed E coefficients");
    const int psi_type=h.inputs[31].element_type;
    if(psi_type!=kBeamzF32&&psi_type!=kBeamzBF16)
      throw std::invalid_argument("hybrid requires F32/BF16 CPML state");
    for(int t=0;t<6;++t)if(h.inputs[31+t].element_type!=psi_type||e.inputs[31+t].element_type!=psi_type)
      throw std::invalid_argument("hybrid requires homogeneous CPML precision");
    int low[3],high[3],extent[3]{};
    for(int a=0;a<3;++a){
      high[a]=int(h.outputs[0].dims[a]);
      for(int c=1;c<3;++c)high[a]=std::min(high[a],int(h.outputs[c].dims[a]));
      low[a]=thickness+2;high[a]-=thickness+2;
      for(int c=0;c<6;++c)extent[a]=std::max(extent[a],int(base_inputs_[c].dims[a]));
      if(high[a]<=low[a])throw std::invalid_argument("hybrid CPML shell leaves no interior");
    }
    for(int g=0;g<9;++g){
      const int index=source_offset_+3*g;
      const auto&coeff=base_inputs_[index];
      if(g<3&&coeff.dims[0])throw std::invalid_argument("hybrid does not admit pre-E sources");
      const auto*starts=static_cast<const int32_t*>(host_[index+2].data);
      for(int s=0;s<coeff.dims[0];++s)for(int a=0;a<3;++a)
        if(starts[3*s+a]<low[a]||int64_t(starts[3*s+a])+coeff.dims[a+1]>high[a])
          throw std::invalid_argument("hybrid source overlaps temporally owned CPML shell");
    }
    const auto*rows=static_cast<const int32_t*>(host_[133].data);
    for(int64_t i=0;i<base_inputs_[133].dims[0];++i){
      const int z=rows[i]/(extent[1]*extent[2]);
      const int y=(rows[i]/extent[2])%extent[1],x=rows[i]%extent[2];
      if(z>=low[0]&&z<high[0]&&y>=low[1]&&y<high[1]&&x>=low[2]&&x<high[2])
        hybrid_rows_.push_back(rows[i]);
    }
    size_t needed=hybrid_rows_.size()*sizeof(int32_t);
    for(int t=0;t<12;++t)needed+=host_[t<6?31+t:62+t-6].bytes;
    size_t free,total;Check(cudaMemGetInfo(&free,&total),"query hybrid admission");
    if(needed+(256ULL<<20)>free)throw std::invalid_argument("hybrid extra state exceeds free GPU memory with256MiB reserve");
    for(int t=0;t<12;++t){
      const int i=t<6?31+t:62+t-6;
      hybrid_shadow_descriptors_[t]=base_inputs_[i];
      hybrid_shadow_storage_.push_back(std::make_unique<DeviceMemory>(host_[i].bytes));
      hybrid_shadow_descriptors_[t].data=hybrid_shadow_storage_.back()->data;
      hybrid_psi_bytes_+=host_[i].bytes;
    }
    hybrid_queue_=std::make_unique<DeviceMemory>(hybrid_rows_.size()*sizeof(int32_t));
    hybrid_residual_=base_inputs_[133];hybrid_residual_.data=hybrid_queue_->data;
    hybrid_residual_.dims[0]=hybrid_rows_.size();
    if(!hybrid_rows_.empty())Check(cudaMemcpy(hybrid_queue_->data,hybrid_rows_.data(),hybrid_queue_->bytes,cudaMemcpyHostToDevice),"upload hybrid residual queue");
    resident_bytes_+=needed;hybrid_ready_=true;
  }
#endif
"""
MEMBERS = r"""
#if BEAMZ_CUDA_NATIVE_HYBRID
  bool hybrid_ready_=false;
  size_t hybrid_psi_bytes_=0;
  std::vector<int32_t>hybrid_rows_;
  std::vector<std::unique_ptr<DeviceMemory>>hybrid_shadow_storage_;
  BeamzBuffer hybrid_shadow_descriptors_[12]{},hybrid_residual_{};
  std::unique_ptr<DeviceMemory>hybrid_queue_;
#endif
"""


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(
            f"Expected one patch site: {old[:100]!r}, got{text.count(old)}"
        )
    return text.replace(old, new, 1)


def patch(source):
    src = source / "src"
    if (src / "native_hybrid.h").exists():
        raise ValueError("Already patched")
    (src / "native_hybrid.h").write_text(HEADER)
    cmake = (source / "CMakeLists.txt").read_text()
    cmake += (
        '\noption(BEAMZ_CUDA_NATIVE_HYBRID "Research native mixed CPML pair" OFF)\n'
    )
    cmake += 'set_property(SOURCE src/native_extension.cc src/program.cu src/update.cu APPEND PROPERTY COMPILE_DEFINITIONS "BEAMZ_CUDA_NATIVE_HYBRID=$<BOOL:${BEAMZ_CUDA_NATIVE_HYBRID}>")\n'
    (source / "CMakeLists.txt").write_text(cmake)
    update = (src / "update.cu").read_text() + UPDATE
    (src / "update.cu").write_text(update)
    program = (src / "program.cu").read_text()
    fn = program[program.index("int BeamzLaunchMstep(void *raw_stream,") :]
    assert fn.rstrip().endswith("}")
    fn = replace_once(
        fn,
        "int BeamzLaunchMstep(void *raw_stream, const BeamzMstepLaunch &launch) {",
        "int BeamzLaunchMstepHybrid(void *raw_stream, const BeamzHybridLaunch &hybrid) {\n  const auto& launch=hybrid.mstep;",
    )
    fn = replace_once(
        fn,
        "  for (const auto &b : buffers)\n    BeamzAppendBufferKey(&key, b);",
        '  key.append("hybrid-shell-v1");\n  for(const auto& shadow:hybrid.shadows)BeamzAppendBufferKey(&key,shadow);\n  BeamzAppendBufferKey(&key,hybrid.residual);\n  for (const auto &b : buffers)\n    BeamzAppendBufferKey(&key, b);',
    )
    fn = replace_once(
        fn,
        "auto first_edge = [&](const BeamzLaunch &phase) {",
        "auto first_edge = [&](const BeamzLaunch &input_phase) {\n      auto phase=input_phase;\n      for(int t=0;t<6;++t){phase.inputs[31+t]=hybrid.shadows[phase.phase*6+t];phase.outputs[3+t]=hybrid.shadows[phase.phase*6+t];}",
    )
    fn = fn.replace(
        "launch.edge_cells, extent[1], extent[2]",
        "hybrid.residual, extent[1], extent[2]",
    )
    marker = "      if (auto err = enqueue_core(base, dest); err != cudaSuccess)\n        return err;"
    fn = replace_once(
        fn,
        marker,
        marker
        + r"""
      auto boundary_h=base.first,boundary_e=base.second;
      for(int t=0;t<12;++t){
        const auto& original=t<6?base.first.inputs[31+t]:base.second.inputs[31+t-6];
        size_t bytes=original.element_type==kBeamzBF16?2:4;
        for(int a=0;a<original.rank;++a)bytes*=original.dims[a];
        if(auto err=cudaMemcpyAsync(hybrid.shadows[t].data,original.data,bytes,cudaMemcpyDeviceToDevice,stream);err!=cudaSuccess)return err;
        if(t<6)boundary_h.inputs[31+t]=hybrid.shadows[t];
        else boundary_e.inputs[31+t-6]=hybrid.shadows[t];
      }
      if(auto err=BeamzEnqueueHybridCpml(stream,boundary_h,boundary_e,dest,program.source_groups,step);err!=cudaSuccess)return err;
""",
    )
    program += (
        '\n#if defined(BEAMZ_CUDA_NATIVE_HYBRID) && BEAMZ_CUDA_NATIVE_HYBRID\n#include "native_hybrid.h"\n'
        + fn
        + "\n#endif\n"
    )
    (src / "program.cu").write_text(program)
    native = (src / "native_extension.cc").read_text()
    native = replace_once(
        native,
        '#include "native_runtime.h"',
        '#include "native_runtime.h"\n#ifndef BEAMZ_CUDA_NATIVE_HYBRID\n#define BEAMZ_CUDA_NATIVE_HYBRID 0\n#endif\n#if BEAMZ_CUDA_NATIVE_HYBRID\n#include "native_hybrid.h"\n#endif',
    )
    native = replace_once(
        native,
        "  nb::dict RunSchedule(const std::vector<int>& chunks) {",
        "  nb::dict RunSchedule(const std::vector<int>& chunks) { return RunScheduleImpl(chunks,false); }\n  nb::dict RunHybrid(const std::vector<int>& chunks) { return RunScheduleImpl(chunks,true); }\n  nb::dict RunScheduleImpl(const std::vector<int>& chunks,bool hybrid) {",
    )
    marker = "  nb::dict RunScheduleImpl(const std::vector<int>& chunks,bool hybrid) {\n    auto guard = Guard();\n    RequireOpen();\n    if (chunks.empty())"
    native = replace_once(
        native,
        marker,
        '  nb::dict RunScheduleImpl(const std::vector<int>& chunks,bool hybrid) {\n    auto guard = Guard();\n    RequireOpen();\n#if BEAMZ_CUDA_NATIVE_HYBRID\n    if(hybrid)PrepareHybrid();\n#else\n    if(hybrid)throw std::invalid_argument("hybrid CPML support not compiled");\n#endif\n    if (chunks.empty())',
    )
    marker = "        const int error = mstep_ ? BeamzLaunchMstep(stream_.value, launch.mstep)\n                                 : BeamzLaunchProgram(stream_.value, launch.program);"
    native = replace_once(
        native,
        marker,
        r"""        int error;
#if BEAMZ_CUDA_NATIVE_HYBRID
        if(hybrid){BeamzHybridLaunch mixed{};mixed.mstep=launch.mstep;mixed.residual=hybrid_residual_;
          std::copy(hybrid_shadow_descriptors_,hybrid_shadow_descriptors_+12,mixed.shadows);
          error=BeamzLaunchMstepHybrid(stream_.value,mixed);
        }else
#endif
        error=mstep_?BeamzLaunchMstep(stream_.value,launch.mstep):BeamzLaunchProgram(stream_.value,launch.program);""",
    )
    native = replace_once(
        native,
        " private:\n  std::unique_lock<std::mutex> Guard() const {",
        " private:\n" + PREPARE + "\n  std::unique_lock<std::mutex> Guard() const {",
    )
    native = replace_once(
        native,
        "  BeamzNativeAttributes attributes_;",
        MEMBERS + "\n  BeamzNativeAttributes attributes_;",
    )
    native = replace_once(
        native,
        '    result["deep_d4_compiled"] = bool(BEAMZ_CUDA_DEEP_D4);',
        """    result["hybrid_compiled"]=bool(BEAMZ_CUDA_NATIVE_HYBRID);
#if BEAMZ_CUDA_NATIVE_HYBRID
    result["hybrid_ready"]=hybrid_ready_;
    result["hybrid_psi_shadow_bytes"]=hybrid_psi_bytes_;
    result["hybrid_residual_cells"]=hybrid_rows_.size();
    result["hybrid_covered_second_cells"]=base_inputs_[133].dims[0]-int64_t(hybrid_rows_.size());
#endif
    result["deep_d4_compiled"] = bool(BEAMZ_CUDA_DEEP_D4);""",
    )
    native = replace_once(
        native,
        "    allocations_.clear();",
        """#if BEAMZ_CUDA_NATIVE_HYBRID
    hybrid_shadow_storage_.clear();hybrid_queue_.reset();hybrid_rows_.clear();hybrid_ready_=false;
#endif
    allocations_.clear();""",
    )
    native = replace_once(
        native,
        '      .def("run_deep4", &NativeSession::RunDeep4, "chunks"_a)',
        '      .def("run_hybrid", &NativeSession::RunHybrid, "chunks"_a)\n      .def("run_deep4", &NativeSession::RunDeep4, "chunks"_a)',
    )
    (src / "native_extension.cc").write_text(native)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    a = p.parse_args()
    patch(a.source)


if __name__ == "__main__":
    main()
