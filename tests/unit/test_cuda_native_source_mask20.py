"""Actual emitted host-source facts select only the admitted immutable class."""

import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import patch_cuda_native_source_mask20 as patch

ROOT = Path(__file__).resolve().parents[2]


def test_emitted_presence_class_counts_fallback_and_snapshot(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ unavailable")
    cpp = '#include "launch.h"\n#include <stdexcept>\n#include <cassert>\n'
    cpp += patch.PRESENCE_HELPER
    cpp += r"""
int main(){
 for(unsigned mask=0;mask<64;++mask){
  BeamzSourceGroupLaunch original[9]{};
  for(int f=0;f<6;++f)original[3+f].coefficients.dims[0]=(mask&(1u<<f))?2:0;
  // Native preparation copies descriptor shapes, independently of caller data.
  BeamzSourceGroupLaunch captured[9];std::copy(original,original+9,captured);
  const uint32_t actual=BeamzNativeMovingSourcePresenceMask(captured,9);assert(actual==mask);
  for(int layout:{1,2})for(bool iso:{false,true})assert(BeamzNativeMovingSourceMask20Class(layout,iso,actual)==(layout==1&&iso&&mask==20));
  original[3].coefficients.dims[0]=original[3].coefficients.dims[0]?0:2;
  assert(BeamzNativeMovingSourcePresenceMask(captured,9)==actual);
  assert(BeamzNativeMovingSourcePresenceMask(original,9)!=(actual));
 }
 for(unsigned mask:{0u,1u,4u,16u,20u,21u,63u})assert(BeamzNativeMovingSourceMask20Class(1,true,mask)==(mask==20));
 BeamzSourceGroupLaunch invalid[9]{};int negatives=0;
 try{BeamzNativeMovingSourcePresenceMask(nullptr,9);assert(false);}catch(const std::invalid_argument&){++negatives;}
 try{BeamzNativeMovingSourcePresenceMask(invalid,8);assert(false);}catch(const std::invalid_argument&){++negatives;}
 invalid[3].coefficients.dims[0]=-1;
 try{BeamzNativeMovingSourcePresenceMask(invalid,9);assert(false);}catch(const std::invalid_argument&){++negatives;}
 assert(negatives==3);
}
"""
    # std::copy and initializer_list are host-test support, not shader changes.
    cpp = "#include <algorithm>\n#include <initializer_list>\n" + cpp
    source = tmp_path / "source.cc"
    source.write_text(cpp)
    binary = tmp_path / "source-proof"
    subprocess.run([compiler, "-std=c++17", "-O2", "-fsanitize=undefined",
                    "-I", str(ROOT / "cuda/src"), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_source_copy_hashes_and_duplicate_anchors_fail_closed():
    with pytest.raises(ValueError, match="Exact a0da"):
        patch.transformed(dict.fromkeys(patch.PARENT_HASHES, b"unreviewed"))
    with pytest.raises(ValueError, match="one exact"):
        patch.once("anchor anchor", "anchor", "changed")
