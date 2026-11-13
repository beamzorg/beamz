"""Packed CPML owner-chain and exact scalar adapter gates for private histories."""

from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.cuda_native_geometry_ir import derive_geometry
from scripts.patch_cuda_traced_lifetime_private_psi import POINT, transform
from scripts.prove_cuda_stationary_psi_cache import prove, prove_owner_boundary_classes


def parents():
    directory = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source/src")
    if not directory.is_dir():
        pytest.skip("Frozen research source required")
    return tuple((directory / name).read_text() for name in ("asymmetric_native.cuh", "native_extension.cc", "native_moving.h"))


def test_actual_single_axis_terms_have_distinct_colors_in_each_independent_phase():
    terms = derive_geometry()["cpml_terms"]
    for phase in (0, 1):
        for axis in range(3):
            active = [int(term["term"][-1]) for term in terms if term["phase"] == phase and term["axis_zyx"] == axis]
            assert len(active) == 2
            assert {int(term in (2, 4, 5)) for term in active} == {0, 1}


@pytest.mark.parametrize("minimum,depth", [((32, 35, 37), 4), ((64, 96, 128), 16), ((65, 97, 129), 16)])
def test_exact_current_owned_geometry_routes_both_H_and_E_packed_recurrences(minimum, depth):
    result = prove(minimum, depth, owned=(7, 15, 31), mode="single-axis", load_policy="first-owned")
    stats = result["statistics"]
    assert stats["owned_updates"] == depth * stats["sampled_physical_recurrences"]
    assert stats["direct_updates"] > 0
    if min(minimum) >= 64:
        assert stats["cached_updates"] > 0
        assert stats["high_publications"] > 0
        assert stats["final_publications"] > 0


def test_current_geometry_startup_and_incoming_owner_classes_are_exhaustive():
    result = prove_owner_boundary_classes((7, 15, 31))
    assert result["startup_classes"] == 4096
    assert result["incoming_classes"] == 841
    assert not result["future_overwrite_before_consumer"]


@pytest.mark.parametrize("option", ["omit_h_publication", "omit_e_publication", "omit_low_refresh", "wrong_e_extent", "future_publication"])
def test_invalid_cache_handoff_or_packed_phase_extent_is_rejected(option):
    with pytest.raises(AssertionError):
        prove((64, 96, 128), 16, owned=(7, 15, 31), mode="all-cache", load_policy="first-owned", **{option: True})


def test_actual_cpp_adapter_preserves_bits_colors_and_lazy_load_store_counts(tmp_path):
    if not shutil.which("c++"):
        pytest.skip("Host C++ compiler required")
    source = r'''#include <cstdint>
#include <cstring>
#include <cstdio>
#define __device__
#define __forceinline__ inline
struct Data{float global[12];int loads[12]{},stores[12]{};};
struct Base{Data&data;
float LoadLocal(int s,int,int,int)const{++data.loads[s];return data.global[s];}
void StoreLocal(int s,int,int,int,float value)const{++data.stores[s];data.global[s]=value;}
};
'''+POINT+r'''
float F(uint32_t bits){float value;std::memcpy(&value,&bits,4);return value;}
uint32_t B(float value){uint32_t bits;std::memcpy(&bits,&value,4);return bits;}
int main(){
 const int axes[]={1,0,0,2,2,1};
 const uint32_t patterns[]={0,0x80000000u,1,0x80000001u,0x3f800001u,0x7f800000u,0xff800000u,0x7fc12345u};
 int checks=0;
 for(int axis=0;axis<3;++axis){
  Data data{};for(int s=0;s<12;++s)data.global[s]=F(patterns[s%8]);
  Base base{data};PrivatePsiHistory<true,4>bank{};
  for(int step=1;step<=4;++step){
   for(int phase=0;phase<2;++phase){
    auto env=phase?PrivatePsiHistoryPoint<Base>(base,bank.ea[0],bank.eb[0],step==1,step==4)
                  :PrivatePsiHistoryPoint<Base>(base,bank.ha[0],bank.hb[0],step==1,step==4);
    for(int term=0;term<6;++term)if(axes[term]==axis){const int s=phase*6+term;
     const uint32_t expected=patterns[(s+step-1)%8];
     if(B(env.LoadLocal(s,17,19,23))!=expected)return 1;
     env.StoreLocal(s,17,19,23,F(patterns[(s+step)%8]));
     if(data.loads[s]!=1||data.stores[s]!=(step==4?1:0))return 2;
     if(B(data.global[s])!=patterns[(s+(step==4?4:0))%8])return 3;
     ++checks;
    }
   }
  }
  // A fresh entering physical point must ignore retained slot contents.
  for(int s=0;s<12;++s)data.global[s]=F(patterns[(s+5)%8]);
  for(int phase=0;phase<2;++phase){
   auto env=phase?PrivatePsiHistoryPoint<Base>(base,bank.ea[0],bank.eb[0],true,true)
                 :PrivatePsiHistoryPoint<Base>(base,bank.ha[0],bank.hb[0],true,true);
   for(int term=0;term<6;++term)if(axes[term]==axis){const int s=phase*6+term;
    if(B(env.LoadLocal(s,5,7,11))!=patterns[(s+5)%8])return 4;
    env.StoreLocal(s,5,7,11,F(patterns[(s+6)%8]));
    if(B(data.global[s])!=patterns[(s+6)%8]||data.loads[s]!=2||data.stores[s]!=2)return 5;
    ++checks;
   }
  }
 }
 std::printf("%d\n",checks);return checks==60?0:6;
}
'''
    cpp, binary = tmp_path / "adapter.cc", tmp_path / "adapter"
    cpp.write_text(source)
    subprocess.run(["c++", "-std=c++17", "-O2", str(cpp), "-o", str(binary)], check=True, capture_output=True, timeout=30)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=10)
    assert result.stdout.strip() == "60"


def test_generated_host_key_and_alias_guard_bind_actual_prepared_state():
    original, original_host, original_header = parents()
    consumer, host, header, proof = transform(original, original_host, original_header)
    assert "private_psi_state_independent=BeamzNativeIndependentPhaseState(cache_state,cache_profile)" in host
    assert "append_bytes(&moving_workspace_.private_psi_state_independent,sizeof(bool))" in host
    assert "append_bytes(&private_psi_selector,sizeof(bool))" in host
    assert "BeamzResearchPrivatePsiEnabled&&w.private_psi_state_independent" in consumer
    assert "bool private_psi_state_independent=false" in header
    assert proof["private_psi_floats_per_thread"] == 16
    for phase in (0, 1):
        for component in range(3):
            assert consumer.count(f"AdvanceNativeCell<{phase},{component},false>(point_env,read,n,z,y,x)") == 1


def test_selector_off_changes_only_host_constant_not_cuda_or_workspace():
    values = parents()
    a = transform(*values, enabled=True)
    b = transform(*values, enabled=False)
    assert a[0] == b[0] and a[2] == b[2]
    assert a[1].replace("const bool BeamzResearchPrivatePsiEnabled=true;", "const bool BeamzResearchPrivatePsiEnabled=false;", 1) == b[1]


def test_debug_hits_distinguish_actual_H_E_retained_axes_and_reset_each_invocation():
    consumer, host, _, proof = transform(*parents(), debug_hits=True)
    assert "(slot<6?8u:64u)<<beamz::cuda::yee::CpmlAxis(slot%6)" in consumer
    assert "moving_private_psi_debug_=0" in host
    assert "if(bad&~504)" in host
    assert "moving_private_psi_debug_H_zyx" in host and "moving_private_psi_debug_E_zyx" in host
    assert proof["debug_hits"]
