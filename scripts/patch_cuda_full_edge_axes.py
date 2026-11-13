"""Patch an immutable candidate snapshot: specialize full-bank edge CPML axes.

The installed component and repository CUDA sources are not modified. Existing
conservative per-axis eligibility proves which recurrence terms can be skipped;
included terms retain their original staggered CPML checks and FP32 operations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

HELPER = r'''
template<int Phase, bool Packed, int PsiType, bool HasMetallicEdges,
         int Layout, int Axes>
__device__ __forceinline__ void MstepFullEdgeAxisComponents(
    const BeamzLaunch &launch, int z, int y, int x) {
  UpdateComponent<Phase, 0, true, 0, HasMetallicEdges, true, PsiType,
                  Packed, false, false, Axes, Layout>(launch,z,y,x);
  UpdateComponent<Phase, 1, true, 0, HasMetallicEdges, true, PsiType,
                  Packed, false, false, Axes, Layout>(launch,z,y,x);
  UpdateComponent<Phase, 2, true, 0, HasMetallicEdges, true, PsiType,
                  Packed, false, false, Axes, Layout>(launch,z,y,x);
}
'''

DISPATCH = r'''
#if BEAMZ_CUDA_MSTEP_CPML_AXIS_MASK
  if constexpr(Cpml && UniformCpml) {
    const int coordinates[] = {z,y,x};
    int axes=0;
#pragma unroll
    for(int axis=0;axis<3;++axis)
      if(MstepCpmlAxisMayBeActive(coordinates[axis],launch.uniform_cpml_thickness,
          launch.outputs[0].dims[axis],launch.outputs[1].dims[axis],
          launch.outputs[2].dims[axis])) axes |= 1<<axis;
#define BEAMZ_FULL_AXIS_CASE(mask) case mask: \
    MstepFullEdgeAxisComponents<Phase,Packed,PsiType,HasMetallicEdges,CoefficientLayout,mask>(launch,z,y,x); return
    switch(axes) {
      BEAMZ_FULL_AXIS_CASE(0); BEAMZ_FULL_AXIS_CASE(1);
      BEAMZ_FULL_AXIS_CASE(2); BEAMZ_FULL_AXIS_CASE(3);
      BEAMZ_FULL_AXIS_CASE(4); BEAMZ_FULL_AXIS_CASE(5);
      BEAMZ_FULL_AXIS_CASE(6); BEAMZ_FULL_AXIS_CASE(7);
    }
#undef BEAMZ_FULL_AXIS_CASE
  }
#endif
'''


def patch(text):
    marker = "// RESEARCH_FULL_EDGE_AXIS_LOWERING"
    if marker in text:
        raise ValueError("Candidate is already patched")
    start = text.index("template <int Phase, bool Cpml, bool Packed, bool UniformCpml, bool CopyCore,")
    insert = text.index("  // The core kernel has already finished all reads", start)
    text = text[:insert] + DISPATCH + text[insert:]
    return text[:start] + marker + "\n" + HELPER + text[start:]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-source", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1] / "cuda"
    if a.candidate_source.resolve() == root.resolve():
        p.error("Use a copied candidate source, not the live repository")
    path = a.candidate_source / "src/mstep.cuh"
    path.write_text(patch(path.read_text()))


if __name__ == "__main__":
    main()
