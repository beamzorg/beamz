"""Check the immutable coarse-mask encoding independently of CUDA kernels."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_observation_masks_preserve_every_raw_leaf(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    helper = Path(__file__).resolve().parents[2] / "scripts/cuda_native_observation_masks.inc"
    source = tmp_path / "oracle.cc"
    source.write_text(r'''
#include <cstdint>
#include <climits>
#include <vector>
#include <stdexcept>
#include <cassert>
struct BeamzBuffer {};
struct BeamzNativeMovingSparseObservation { std::vector<int32_t> coarse, leaf; };
BeamzNativeMovingSparseObservation fixture;
auto BeamzBuildNativeMovingSparseObservation(const BeamzBuffer*, const std::vector<int64_t>&) { return fixture; }
''' + f'#include "{helper}"\n' + r'''
int main() {
  fixture.coarse = {-1};
  for (int mask=1; mask<64; ++mask) {
    fixture.coarse.push_back(mask-1);
    for (int cell=0; cell<128; ++cell)
      for (int field=0; field<6; ++field)
        fixture.leaf.push_back((mask&(1<<field)) && cell%7==field ? int(fixture.leaf.size()) : -1);
  }
  const auto result = BeamzBuildNativeMovingMaskedObservation(nullptr, {});
  assert(result.leaf == fixture.leaf && result.coarse[0] == -1);
  for (int mask=1; mask<64; ++mask) {
    const int word=result.coarse[mask];
    assert((word&63)==mask && (word>>6)==mask-1);
    for (int cell=0; cell<128; ++cell)
      for (int field=0; field<6; ++field) {
        const int expected=fixture.leaf[(mask-1)*768+cell*6+field];
        const int actual=(word&(1<<field)) ? result.leaf[(word>>6)*768+cell*6+field] : -1;
        assert(expected==actual);
      }
  }
  for (int bad : {INT32_MAX, 63}) {
    fixture.coarse={bad};
    bool rejected=false;
    try { BeamzBuildNativeMovingMaskedObservation(nullptr, {}); }
    catch(const std::invalid_argument&) { rejected=true; }
    assert(rejected);
  }
}
''')
    executable = tmp_path / "oracle"
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
