"""Actual import-prefix equivalence for asynchronous trail staging."""

import shutil
import subprocess

import pytest

from scripts.patch_cuda_native_async_stage import resolve_body
from scripts.patch_cuda_native_import_packets import PACKETS


def test_resolved_addresses_preserve_all_packet_bits_and_error_flags(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler needed for import-resolution differential check")
    original = PACKETS[
        : PACKETS.index(
            "template<int Depth,bool Fast>\n__device__ __forceinline__ void StageLowX("
        )
    ]
    resolved = resolve_body(original)
    source = tmp_path / "resolver.cc"
    source.write_text(
        r"""
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#define __device__
#define __forceinline__ inline
constexpr int BZ=7,BY=15,BX=31,SurfacePoints=735;
struct Field{int z,y,x;};struct Phase{Field own[3];};
struct Parameters{Phase phase[2];float*trails;int*bad;};
int atomicOr(int*p,int value){const int old=*p;*p|=value;return old;}
float __int_as_float(int bits){float value;std::memcpy(&value,&bits,4);return value;}
int Surface(int z,int y,int x){
 if(z==BZ-1)return y*BX+x;
 if(y==BY-1)return BY*BX+z*BX+x;
 if(x==BX-1)return BY*BX+(BZ-1)*BX+z*(BY-1)+y;
 return -1;
}
__ORIGINAL__
struct AsyncTrailView{const float*point;unsigned mask;bool bad;};
template<int Depth,bool Fast>
AsyncTrailView Resolve(const Parameters&p,const size_t*bases,int version,
 int oz,int oy,int ox,int dz,int dy,int dx){
__RESOLVED__
}
uint32_t Bits(float value){uint32_t out;std::memcpy(&out,&value,4);return out;}
template<bool Safe> bool Check(size_t&cases){
 constexpr uint32_t samples[]={0u,0x80000000u,1u,0x00800000u,0x3f800001u,
  0xbf123456u,0x7fc12345u,0x7f800000u,0xff800000u,0x7fffffffu};
 std::vector<float>memory(8*16*6*SurfacePoints);
 for(size_t i=0;i<memory.size();++i){const auto bits=samples[i%10];std::memcpy(memory.data()+i,&bits,4);}
 int bad=0;Parameters p{};p.trails=memory.data();p.bad=&bad;
 for(const int extent:{1,24,64}){
  p.phase[0].own[0]={extent,extent+7,extent+16};
  p.phase[0].own[1]={extent,extent+8,extent+15};
  for(const int origin:{0,7,21,84})for(const int n:{1,4,16})for(int invalid=0;invalid<2;++invalid){
   size_t bases[8];for(int k=0;k<8;++k)bases[k]=(invalid&&k%2)?size_t(-1):size_t(k)*16*6*SurfacePoints;
   for(int z=0;z<=BZ;++z)for(int y=0;y<=BY;++y)for(int x=0;x<=BX;++x){
    bad=0;const auto packet=ImportPacket<16,Safe>(p,bases,n,origin,origin+2,origin+5,z,y,x);const int expected_bad=bad;
    bad=0;const auto view=Resolve<16,Safe>(p,bases,n,origin,origin+2,origin+5,z,y,x);
    if(bad!=expected_bad)return false;
    const std::array<float,6>actual={packet.hx,packet.hy,packet.hz,packet.ex,packet.ey,packet.ez};
    for(int field=0;field<6;++field){uint32_t expected=0;
     if(view.point&&(view.mask&(1u<<field))){
      if(reinterpret_cast<uintptr_t>(view.point+field*SurfacePoints)%4)return false;
      std::memcpy(&expected,view.point+field*SurfacePoints,4);
     }else if(view.bad&&(view.mask&(1u<<field)))expected=0x7fffffffu;
     if(Bits(actual[field])!=expected)return false;
    }
    ++cases;
   }
  }
 }
 return true;
}
int main(){size_t cases=0;const bool ok=Check<false>(cases)&&Check<true>(cases);
 std::cout<<cases;return ok&&cases>500000?0:1;}
""".replace("__ORIGINAL__", original).replace("__RESOLVED__", resolved)
    )
    binary = tmp_path / "resolver"
    build = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-fsanitize=address,undefined",
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
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) == 589824


def test_import_resolver_rejects_changed_zero_or_error_semantics():
    for old, new in [
        ("if(!mask)return {};", "if(!mask)return {1.f};"),
        ("atomicOr(p.bad,2)", "atomicOr(p.bad,1)"),
        ("version-1", "version"),
    ]:
        with pytest.raises(ValueError, match="Unreviewed original import"):
            resolve_body(PACKETS.replace(old, new))
