"""Differential C++ address checks for dense canonical packed CPML slabs."""

import itertools
import shutil
import subprocess

import numpy as np
import pytest

from scripts.patch_cuda_native_cpml_strides import (
    AXES,
    INT_MAX,
    NEW,
    admitted,
    field_shape,
    slab_shape,
)


def address_cases():
    minima = [
        (1, 2, 3),
        (11, 12, 13),
        (23, 24, 25),
        (31, 49, 65),
        (64, 96, 128),
        (512, 512, 512),
        (1289, 1289, 1289),
    ]
    for axis in range(3):
        minimum = [1, 1, 1]
        minimum[axis] = INT_MAX // 48 - 1
        minima.append(tuple(minimum))
    for minimum in minima:
        assert admitted(minimum)
        for slot in range(12):
            family, term = divmod(slot, 6)
            axis = AXES[term]
            target = field_shape(minimum, family * 3 + term // 2)
            shape = slab_shape(minimum, slot)
            samples = [
                sorted({v for v in [0, 1, 11, 12, n - 13, n - 12, n - 1] if 0 <= v < n})
                for n in target
            ]
            for point in itertools.product(*samples):
                coordinate = point[axis]
                if coordinate >= 12 and coordinate < target[axis] - 12:
                    continue
                q = list(point)
                q[axis] = (
                    coordinate if coordinate < 12 else coordinate - target[axis] + 24
                )
                oracle = int(np.ravel_multi_index(tuple(q), shape, order="C"))
                yield (*minimum, slot, *point, oracle)


def test_actual_emitted_cpp_matches_numpy_slabs_with_undefined_behavior_checks(
    tmp_path,
):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler required for emitted-helper validation")
    source = tmp_path / "offset.cc"
    source.write_text(
        r"""
#include <iostream>
namespace beamz { namespace cuda { namespace yee {
int CpmlAxis(int term){constexpr int a[6]={1,0,0,2,2,1};return a[term];}
}}}
struct Parameters{int physical_n[3];};
struct Environment{
 Parameters p;
 int Packed(int slot,int z,int y,int x)const{
  const int family=slot/6,axis=beamz::cuda::yee::CpmlAxis(slot%6);
  const int q[3]={z,y,x},size=p.physical_n[axis]+family;
  return q[axis]<12?q[axis]:24+q[axis]-size;
 }
 int PsiOffset(int slot,int z,int y,int x)const{
__BODY__
 }
};
int main(){
 Environment e{};int slot,z,y,x;long long expected;size_t count=0;
 while(std::cin>>e.p.physical_n[0]>>e.p.physical_n[1]>>e.p.physical_n[2]>>slot>>z>>y>>x>>expected){
  const int actual=e.PsiOffset(slot,z,y,x);
  if(actual!=expected){std::cerr<<"case "<<count<<": "<<actual<<" != "<<expected;return 1;}
  ++count;
 }
 std::cout<<count;
 return count>10000?0:2;
}
""".replace("__BODY__", NEW)
    )
    binary = tmp_path / "offset"
    build = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert build.returncode == 0, build.stderr
    cases = list(address_cases())
    assert max(case[-1] for case in cases) > INT_MAX - 1000
    payload = "\n".join(" ".join(map(str, case)) for case in cases) + "\n"
    result = subprocess.run(
        [str(binary)], input=payload, text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) == len(cases)


@pytest.mark.parametrize(
    "minimum",
    [
        (0, 32, 32),
        (-1, 32, 32),
        (1290, 1290, 1290),
        (1, 1, INT_MAX),
        (INT_MAX // 48 + 1, 1, 1),
    ],
)
def test_invalid_or_overflowing_descriptors_cannot_support_the_fact(minimum):
    assert not admitted(minimum)
