#!/usr/bin/env python3
"""Patch a private CUDA snapshot with rolling-D4 bulk/ordinary-edge execution."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HEADER = r"""#pragma once
#include <cuda_runtime_api.h>
#include "launch.h"
int BeamzLaunchRolling4(void*stream,const BeamzMstepLaunch&launch);
cudaError_t BeamzEnqueueRolling4Core(cudaStream_t stream,const BeamzLaunch&h,
 const BeamzLaunch&e,const BeamzBuffer*final_fields,const BeamzBuffer&masks,const BeamzBuffer&bounds);
"""
FACTORY = r"""
#if defined(BEAMZ_CUDA_NATIVE_ROLLING4) && BEAMZ_CUDA_NATIVE_ROLLING4
#include "native_rolling4.h"
#include "cuda_mstep_rolling_d4.cuh"
template<int Layout,int X,int Y>
cudaError_t LaunchRolling4View(cudaStream_t stream,const MstepPackedCoreView&view,
 const BeamzBuffer&masks,const BeamzBuffer&bounds,int nz,int segment){
 using T=beamz_rolling_d4::Tile<X,Y>;
 if(auto error=cudaFuncSetAttribute(beamz_rolling_d4::Kernel<X,Y,Layout>,cudaFuncAttributeMaxDynamicSharedMemorySize,T::shared_bytes);error!=cudaSuccess)return error;
 const dim3 blocks(masks.dims[1]*(32/X),masks.dims[0]*(16/Y),segment?(nz+segment-1)/segment:1);
 beamz_rolling_d4::Kernel<X,Y,Layout><<<blocks,T::threads,T::shared_bytes,stream>>>(view,static_cast<const int*>(masks.data),static_cast<const int*>(bounds.data),masks.dims[1],segment);
 return cudaGetLastError();
}
cudaError_t BeamzEnqueueRolling4Core(cudaStream_t stream,const BeamzLaunch&h,
 const BeamzLaunch&e,const BeamzBuffer*final_fields,const BeamzBuffer&masks,const BeamzBuffer&bounds){
 MstepPackedCoreView view{};view.min_y=view.min_x=INT32_MAX;view.inv=h.inv_resolution;
 for(int c=0;c<6;++c){
  if(h.inputs[6+c].rank!=0)return cudaErrorNotSupported;
  const auto&in=h.inputs[c];const auto&out=final_fields[c];
  view.input[c]={static_cast<float*>(in.data),in.row_stride,in.plane_stride};
  view.output[c]={static_cast<float*>(out.data),out.row_stride,out.plane_stride};
  view.min_y=std::min(view.min_y,int(in.dims[1]));view.min_x=std::min(view.min_x,int(in.dims[2]));
 }
 for(int c=0;c<3;++c){
  if(e.inputs[9+c].element_type!=kBeamzS32||e.inputs[6+c].element_type!=kBeamzF32)return cudaErrorNotSupported;
  view.codes[c]=static_cast<const uint32_t*>(e.inputs[9+c].data);view.tables[c]=static_cast<const float*>(e.inputs[6+c].data);
  view.table_sizes[c]=int(e.inputs[6+c].dims[0]);view.code_ranks[c]=e.inputs[9+c].rank;
  view.h_decay[c]=static_cast<const float*>(h.inputs[6+c].data);view.h_source[c]=static_cast<const float*>(h.inputs[9+c].data);
 }
 const int segment=(h.cuda_flags&beamz::cuda::abi::kBeamzMstepSegment32)?32:(h.cuda_flags&beamz::cuda::abi::kBeamzMstepSegment64)?64:(h.cuda_flags&beamz::cuda::abi::kBeamzMstepSegment128)?128:(h.cuda_flags&beamz::cuda::abi::kBeamzMstepSegment256)?256:0;
 constexpr int X=BEAMZ_CUDA_ROLLING4_X,Y=16;
 if(view.code_ranks[0]==1&&view.code_ranks[1]==1&&view.code_ranks[2]==1)return LaunchRolling4View<1,X,Y>(stream,view,masks,bounds,h.inputs[0].dims[0],segment);
 if(view.code_ranks[0]==2&&view.code_ranks[1]==2&&view.code_ranks[2]==2)return LaunchRolling4View<2,X,Y>(stream,view,masks,bounds,h.inputs[0].dims[0],segment);
 return LaunchRolling4View<0,X,Y>(stream,view,masks,bounds,h.inputs[0].dims[0],segment);
}
#endif
"""
LOOP = r"""
    for(int step=0;step+3<program.nsteps;step+=4){
      auto base=phases(current,current);
      if(auto error=inject(base,0,step);error!=cudaSuccess)return error;
      if(auto error=enqueue_core(base,dest);error!=cudaSuccess)return error;
      // Three chronological in-place envelopes. Their three-cell Q margin
      // and two-cell observation margin protect retained values; outer halo
      // results may be discarded, exactly as in the D2 first-step envelope.
      for(int substep=0;substep<3;++substep){
        auto first=phases(current,current);
        if(substep)if(auto error=inject(first,0,step+substep);error!=cudaSuccess)return error;
        if(auto error=first_edge(first.first);error!=cudaSuccess)return error;
        if(auto error=inject(first,1,step+substep);error!=cudaSuccess)return error;
        if(auto error=first_edge(first.second);error!=cudaSuccess)return error;
        if(auto error=inject(first,2,step+substep);error!=cudaSuccess)return error;
        if(auto error=observe(current,step+substep);error!=cudaSuccess)return error;
      }
      auto last=phases(current,dest);
      if(auto error=inject(last,0,step+3);error!=cudaSuccess)return error;
      if(auto error=second_edge(last.first);error!=cudaSuccess)return error;
      if(auto error=inject(last,1,step+3);error!=cudaSuccess)return error;
      if(auto error=second_edge(last.second);error!=cudaSuccess)return error;
      if(auto error=inject(last,2,step+3);error!=cudaSuccess)return error;
      if(auto error=observe(dest,step+3);error!=cudaSuccess)return error;
      current=dest;dest=current==original?launch.final_fields:original;
    }
    for(int step=(program.nsteps/4)*4;step<program.nsteps;++step){
      auto tail=phases(current,current);
      if(auto error=inject(tail,0,step);error!=cudaSuccess)return error;
      if(auto error=BeamzEnqueuePhase(stream,tail.first);error!=cudaSuccess)return error;
      if(auto error=inject(tail,1,step);error!=cudaSuccess)return error;
      if(auto error=BeamzEnqueuePhase(stream,tail.second);error!=cudaSuccess)return error;
      if(auto error=inject(tail,2,step);error!=cudaSuccess)return error;
      if(auto error=observe(current,step);error!=cudaSuccess)return error;
    }
"""


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Patch count{text.count(old)}: {old[:90]!r}")
    return text.replace(old, new, 1)


def patch(source):
    if source.resolve() == (Path(__file__).resolve().parents[1] / "cuda").resolve():
        raise ValueError("Copied source required")
    src = source / "src"
    if (src / "native_rolling4.h").exists():
        raise ValueError("Already patched")
    (src / "native_rolling4.h").write_text(HEADER)
    shutil.copy(
        Path(__file__).with_name("cuda_mstep_rolling_d4.cuh"),
        src / "cuda_mstep_rolling_d4.cuh",
    )
    p = source / "CMakeLists.txt"
    s = p.read_text()
    s += '\noption(BEAMZ_CUDA_NATIVE_ROLLING4 "Research rolling D4 interior" OFF)\nset(BEAMZ_CUDA_ROLLING4_X "16" CACHE STRING "Rolling D4 interior X tile,16 or32")\nset_property(SOURCE src/native_extension.cc src/program.cu src/update.cu APPEND PROPERTY COMPILE_DEFINITIONS "BEAMZ_CUDA_NATIVE_ROLLING4=$<BOOL:${BEAMZ_CUDA_NATIVE_ROLLING4}>;BEAMZ_CUDA_ROLLING4_X=${BEAMZ_CUDA_ROLLING4_X}")\n'
    p.write_text(s)
    p = src / "update.cu"
    p.write_text(p.read_text() + FACTORY)
    p = src / "program.cu"
    s = p.read_text()
    fn = s[s.index("int BeamzLaunchMstep(void *raw_stream,") :]
    assert fn.rstrip().endswith("}")
    fn = once(
        fn,
        "int BeamzLaunchMstep(void *raw_stream, const BeamzMstepLaunch &launch) {",
        "int BeamzLaunchRolling4(void *raw_stream,const BeamzMstepLaunch&launch){",
    )
    fn = once(
        fn,
        "      raw_stream, program);",
        '      raw_stream, program);\n  key.append("rolling-d4-interior-v1");\n  const int rolling_x=BEAMZ_CUDA_ROLLING4_X;key.append(reinterpret_cast<const char*>(&rolling_x),sizeof(rolling_x));',
    )
    start = fn.index("    auto enqueue_core =")
    end = fn.index("    auto first_edge =", start)
    fn = (
        fn[:start]
        + """    auto enqueue_core=[&](const auto&p,const BeamzBuffer*next){return BeamzEnqueueRolling4Core(stream,p.first,p.second,next,launch.core_masks,launch.bounds);};\n"""
        + fn[end:]
    )
    start = fn.index("    for (int step = 0; step + 1 < program.nsteps; step += 2) {")
    end = fn.index("    return cudaSuccess;", start)
    fn = fn[:start] + LOOP + fn[end:]
    s += (
        '\n#if defined(BEAMZ_CUDA_NATIVE_ROLLING4) && BEAMZ_CUDA_NATIVE_ROLLING4\n#include "native_rolling4.h"\n'
        + fn
        + "\n#endif\n"
    )
    p.write_text(s)
    p = src / "native_extension.cc"
    s = p.read_text()
    s = once(
        s,
        '#include "native_runtime.h"',
        '#include "native_runtime.h"\n#ifndef BEAMZ_CUDA_NATIVE_ROLLING4\n#define BEAMZ_CUDA_NATIVE_ROLLING4 0\n#endif\n#if BEAMZ_CUDA_NATIVE_ROLLING4\n#include "native_rolling4.h"\n#endif',
    )
    s = once(
        s,
        '    if (kind != "mstep" && kind != "program")',
        '    rolling_plan_depth_=AttributeInt(attributes,"rolling_plan_depth",2);\n    if (kind != "mstep" && kind != "program")',
    )
    s = once(
        s,
        "  nb::dict RunSchedule(const std::vector<int>& chunks) {",
        "  nb::dict RunSchedule(const std::vector<int>& chunks){return RunScheduleImpl(chunks,false);}\n  nb::dict RunRolling4(const std::vector<int>& chunks){return RunScheduleImpl(chunks,true);}\n  nb::dict RunScheduleImpl(const std::vector<int>& chunks,bool rolling){",
    )
    start = s.index("  nb::dict RunScheduleImpl(")
    end = s.index("  nb::list Download()", start)
    part = s[start:end]
    marker = "    RequireOpen();"
    part = once(
        part,
        marker,
        marker
        + """\n#if BEAMZ_CUDA_NATIVE_ROLLING4
    if(rolling){
      if(!mstep_||rolling_plan_depth_!=4||!attributes_.cpml_enabled||attributes_.metric_kind!=0||!BEAMZ_CUDA_MSTEP_FULL_INTERMEDIATES)
        throw std::invalid_argument("rolling4 requires its depth4 prepared CPML full-intermediate packet");
      for(int g=0;g<3;++g)if(base_inputs_[source_offset_+3*g].dims[0])throw std::invalid_argument("rolling4 pre-E source not admitted");
      for(int i=6;i<12;++i)if(base_inputs_[i].rank!=0)throw std::invalid_argument("rolling4 requires scalar H coefficients");
      for(int c=0;c<3;++c)if(base_inputs_[40+c].element_type!=kBeamzS32)throw std::invalid_argument("rolling4 requires packed E coefficients");
    }
#else
    if(rolling)throw std::invalid_argument("rolling4 support not compiled");
#endif""",
    )
    marker = "        const int error = mstep_ ? BeamzLaunchMstep(stream_.value, launch.mstep)\n                                 : BeamzLaunchProgram(stream_.value, launch.program);"
    part = once(
        part,
        marker,
        """        int error;
#if BEAMZ_CUDA_NATIVE_ROLLING4
        if(rolling)error=BeamzLaunchRolling4(stream_.value,launch.mstep);else
#endif
        error=mstep_?BeamzLaunchMstep(stream_.value,launch.mstep):BeamzLaunchProgram(stream_.value,launch.program);""",
    )
    part = once(
        part,
        "        last_bank_ = mstep_ ? ((nsteps / 2) & 1) : 0;",
        "        last_bank_=mstep_?((nsteps/(rolling?4:2))&1):0;",
    )
    marker = '    result["nsteps"] = static_cast<int>(total_steps);'
    part = once(
        part,
        marker,
        marker
        + '\n    result["algorithm"]=rolling?"rolling4_interior":"mstep_or_program";',
    )
    s = s[:start] + part + s[end:]
    s = once(
        s,
        "  BeamzNativeAttributes attributes_;",
        "  int rolling_plan_depth_=2;\n  BeamzNativeAttributes attributes_;",
    )
    s = once(
        s,
        '    result["deep_d4_compiled"] = bool(BEAMZ_CUDA_DEEP_D4);',
        '    result["rolling4_compiled"]=bool(BEAMZ_CUDA_NATIVE_ROLLING4);\n    result["rolling4_plan_depth"]=rolling_plan_depth_;\n#if BEAMZ_CUDA_NATIVE_ROLLING4\n    result["rolling4_tile_x"]=BEAMZ_CUDA_ROLLING4_X;\n#endif\n    result["deep_d4_compiled"] = bool(BEAMZ_CUDA_DEEP_D4);',
    )
    s = once(
        s,
        '      .def("run_deep4", &NativeSession::RunDeep4, "chunks"_a)',
        '      .def("run_rolling4", &NativeSession::RunRolling4, "chunks"_a)\n      .def("run_deep4", &NativeSession::RunDeep4, "chunks"_a)',
    )
    p.write_text(s)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    a = p.parse_args()
    patch(a.source)


if __name__ == "__main__":
    main()
