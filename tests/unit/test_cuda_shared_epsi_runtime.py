"""Actual scalar adapter and copy-only source binding for shared Epsi runtime."""

import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.patch_cuda_native_shared_epsi_runtime import POINT, transform

ROOT = Path(__file__).resolve().parents[2]
PARENT = (
    ROOT / ".cache/cuda-five-hour-2026-11-13/source-pending-paired-on-v12/source/src"
)


def test_actual_adapter_preserves_copy_bits_global_h_and_onehot_fallback(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    adapter = POINT[: POINT.index("constexpr size_t SharedEPsiBytes")]
    code = (
        r"""
#include <cstdint>
#include <cstring>
#define __device__
#define __forceinline__ inline
struct Base {
 float *values;
 float LoadLocal(int s,int,int,int)const{return values[s];}
 void StoreLocal(int s,int,int,int,float v)const{values[s]=v;}
};
"""
        + adapter
        + r"""
static uint32_t bits(float f){uint32_t x;std::memcpy(&x,&f,4);return x;}
static float value(uint32_t x){float f;std::memcpy(&f,&x,4);return f;}
int main(){
 const uint32_t patterns[]={0,0x80000000u,0x3f800001u,0x7fc00123u,0xff800000u};
 int checks=0;
 for(bool enabled:{false,true})for(bool first:{false,true})for(bool flush:{false,true})
 for(int term=0;term<12;++term)for(uint32_t raw:patterns){
  float globals[12];for(int k=0;k<12;++k)globals[k]=value(0x3f000001u+k);
  volatile float a=value(0x3f800123u),b=value(0xbf800124u);
  Base base{globals};SharedEPsiPoint<Base> p(base,a,b,first,flush,enabled);
  const bool cache=enabled&&term>=6;
  const bool color_b=term%6==2||term%6==4||term%6==5;
  const uint32_t expected=cache&&!first?bits(color_b?b:a):bits(globals[term]);
  if(bits(p.LoadLocal(term,1,2,3))!=expected)return 1;
  const uint32_t old=bits(globals[term]);p.StoreLocal(term,1,2,3,value(raw));
  if(bits(globals[term])!=(cache&&!flush?old:raw))return 2;
  if(cache&&bits(color_b?b:a)!=raw)return 3;
  ++checks;
 }
 return checks==480?0:4;
}
"""
    )
    # initializer-list syntax deliberately exercises both boolean values.
    code = "#include <initializer_list>\n" + code
    source = tmp_path / "adapter.cc"
    source.write_text(code)
    binary = tmp_path / "adapter"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fsanitize=undefined",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True)


def test_exact_source_binding_preserves_appended_source_and_host_math():
    if not PARENT.exists():
        pytest.skip("Immutable parent artifact not retained")
    c = (PARENT / "asymmetric_native.cuh").read_bytes()
    e = (PARENT / "native_extension.cc").read_bytes()
    h = (PARENT / "native_moving.h").read_bytes()
    changed, host, header = transform(c, e, h)
    begin = b"// Private source-only storage schedule;"
    # The complete appended source-body/entry bytes end at the unchanged host
    # preparation helper; compare by exact common anchors.
    left = c.index(begin)
    right = c.index(b"inline bool Prepare(", left)
    left2 = changed.index(begin)
    right2 = changed.index(b"inline bool Prepare(", left2)
    assert c[left:right] == changed[left2:right2]
    assert b"__fmul_rn(__ldg(s.coefficients+offset)" in changed
    assert b"RuntimePhase" not in changed
    assert b"84516" in changed and b"84516" in host
    assert b"shared_epsi_state_independent" in header
    assert b"append_bytes(&shared_selector,sizeof(bool))" in host
    assert host.index(
        b"BeamzCompactNativeMovingUniformObservers(moving_plan_"
    ) < host.index(b"moving_shared_epsi_axes_={0,0,0};")
    with pytest.raises(ValueError, match="Exact frozen"):
        transform(c + b"\n", e, h)


def test_debug_hit_marks_only_actual_retained_reads_and_clears_every_invocation():
    if not PARENT.exists():
        pytest.skip("Immutable parent artifact not retained")
    c, e, _ = transform(
        (PARENT / "asymmetric_native.cuh").read_bytes(),
        (PARENT / "native_extension.cc").read_bytes(),
        (PARENT / "native_moving.h").read_bytes(),
        debug_hits=True,
    )
    point = c[
        c.index(b"template<class Base> struct SharedEPsiPoint") : c.index(
            b"constexpr size_t SharedEPsiBytes"
        )
    ]
    assert (
        point.index(b"if(slot<6||!enabled||first)return Base::LoadLocal")
        < point.index(b"atomicOr(this->p.bad,8u<<")
        < point.index(b"return State(slot)")
    )
    assert b"if(bad&~56)throw" in e  # original errors1/2/4 stay fail-closed
    assert e.index(
        b"moving_shared_epsi_debug_raw_=0;\n      Check(cudaMemsetAsync"
    ) < e.index(b"for(size_t chunk=0;chunk<chunks.size();++chunk)")
    assert b"moving_shared_epsi_debug_raw_=bad;" in e
    assert b"moving_shared_epsi_validation_only" in e
