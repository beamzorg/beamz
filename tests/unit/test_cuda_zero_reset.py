"""The compressed reset fact must distinguish every nonzero input byte."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_zero_reset_rejects_signed_zero_and_every_nonzero_byte(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    helper = Path(__file__).resolve().parents[2] / "scripts/cuda_native_zero_reset.inc"
    source = tmp_path / "oracle.cc"
    source.write_text(r'''
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cassert>
#include <limits>
#include <initializer_list>
''' + f'#include "{helper}"\n' + r'''
int main() {
  unsigned char bytes[83]{};
  assert(BeamzNativeAllPositiveZero(nullptr, 0));
  for (size_t length=0; length<=82; ++length) {
    assert(BeamzNativeAllPositiveZero(bytes+1, length));
    for (size_t index=0; index<length; ++index) {
      bytes[index+1]=128;
      assert(!BeamzNativeAllPositiveZero(bytes+1, length));
      bytes[index+1]=0;
    }
  }
  for (float value : {-0.f, 1.f, -1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
    assert(!BeamzNativeAllPositiveZero(&value, sizeof(value)));
}
''')
    executable = tmp_path / "oracle"
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
