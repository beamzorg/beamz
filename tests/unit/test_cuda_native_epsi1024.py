"""Actual emitted private-state callbacks preserve warm data and publication."""

import shutil
import subprocess

import pytest

from scripts.probe_cuda_native_epsi1024 import storage_source


def test_ir_one_axis_colors_have_one_distinct_pair_and_only_e_cache():
    source, proof = storage_source()
    assert proof["axis_terms"] == {0: [1, 2], 1: [0, 5], 2: [3, 4]}
    assert proof["color_b_mask"] == 52
    assert "if(slot<6)return Base::LoadLocal" in source
    assert "if(slot<6){Base::StoreLocal" in source
    assert "return first?Base::LoadLocal" in source
    assert "State(slot)=value;if(flush)Base::StoreLocal" in source


@pytest.mark.parametrize("cache_colors", [1, 2])
def test_actual_cpp_callback_retains_epsi_and_flushes_exactly_when_requested(tmp_path, cache_colors):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++17 compiler required")
    helper, _ = storage_source(cache_colors)
    source = r'''
#include <cassert>
#include <cstdio>
#define __device__
#define __forceinline__ inline
struct Data {float psi[12][4]{},expected[12][4]{};int loads=0,stores=0;};
struct Base {Data&d;
 float LoadLocal(int slot,int z,int,int)const{++d.loads;return d.psi[slot][z];}
 void StoreLocal(int slot,int z,int,int,float value)const{++d.stores;d.psi[slot][z]=value;}
};
'''+helper+r'''
int main(){int checks=0;const int pairs[3][2]={{1,2},{0,5},{3,4}};
 for(const auto&pair:pairs){Data data;Base base{data};EPsiStorage<true,4> cache;
  for(int slot=0;slot<12;++slot)for(int j=0;j<4;++j)data.psi[slot][j]=data.expected[slot][j]=float(100*slot+j);
  for(int n=1;n<=5;++n)for(int j=0;j<4;++j){bool fresh=n==1||(n==3&&j==2),flush=n==5||(n==2&&j==1);
   if(n==3&&j==2)for(int slot=6;slot<12;++slot)data.psi[slot][j]=data.expected[slot][j]=float(2000+100*slot+j);
   PrivateEPsiPoint<Base> point(base,cache.a[j],@COLOR_B@fresh,flush);
   for(int term:pair){const int slot=term+6;const int loads=data.loads,stores=data.stores;
    const bool cached=@CACHED@;
    const float old=point.LoadLocal(slot,j,0,0);assert(old==data.expected[slot][j]);assert(data.loads-loads==int(!cached||fresh));++checks;
    const float value=old+float(n);data.expected[slot][j]=value;point.StoreLocal(slot,j,0,0,value);
    assert(data.stores-stores==int(!cached||flush));if(!cached||flush)assert(data.psi[slot][j]==value);++checks;
   }
   const int loads=data.loads,stores=data.stores;
   const float h=point.LoadLocal(2,j,0,0);point.StoreLocal(2,j,0,0,h+1.f);
   assert(data.loads==loads+1&&data.stores==stores+1&&data.psi[2][j]==h+1.f);++checks;
  }
  for(int term:pair)for(int j=0;j<4;++j)assert(data.psi[6+term][j]==data.expected[6+term][j]);
 }
 std::printf("checks=%d errors=0\n",checks);return 0;
}
'''
    source = source.replace("@COLOR_B@", "cache.b[j]," if cache_colors == 2 else "")
    source = source.replace("@CACHED@", "true" if cache_colors == 2 else "term==0||term==1||term==3")
    cpp, binary = tmp_path / "state.cc", tmp_path / "state"
    cpp.write_text(source)
    subprocess.run([compiler, "-std=c++17", "-O2", "-ffp-contract=off", str(cpp), "-o", str(binary)],
                   check=True, capture_output=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert "checks=300 errors=0" in result.stdout
