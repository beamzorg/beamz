"""Portable proof that uniform-CPML axis pruning cannot omit an active slab."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_axis_union_covers_every_staggered_component_slab(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("A C++17 compiler is required for the portable region helper")
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "region_oracle.cc"
    source.write_text(
        r"""
#include "mstep_regions.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
int main() {
  uint32_t random=0x394be98u;
  for (int trial=0;trial<100000;++trial) {
    random=random*1664525u+1013904223u;
    int thickness=1+(random%64);
    std::array<int64_t,3> extent;
    for (auto &n:extent) {
      random=random*1664525u+1013904223u;
      n=2*thickness+random%129;
    }
    const int64_t smallest=*std::min_element(extent.begin(),extent.end());
    const int64_t largest=*std::max_element(extent.begin(),extent.end());
    for (int coordinate=0;coordinate<largest;++coordinate) {
      const bool candidate=MstepCpmlAxisMayBeActive(
          coordinate,thickness,extent[0],extent[1],extent[2]);
      bool actual=false;
      for (auto n:extent) {
        // Each component's independent packed slab excludes coordinates
        // outside that component. The union may conservatively include them.
        if (coordinate<n && (coordinate<thickness || coordinate>=n-thickness))
          actual=true;
      }
      if ((actual && !candidate) || (coordinate<smallest && actual!=candidate)) {
        std::fprintf(stderr,"Bad pruning at coord%d,t%d,extents%ld/%ld/%ld\n",
                     coordinate,thickness,extent[0],extent[1],extent[2]);
        return 1;
      }
    }
  }
}
"""
    )
    executable = tmp_path / "region_oracle"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(root / "cuda/src"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=30
    )
