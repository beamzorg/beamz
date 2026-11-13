"""Packed physical CPML state cache and lazy publication owner-chain gates."""

import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.patch_cuda_native_private_psi import POINT_ENV, patch
from scripts.prove_cuda_stationary_psi_cache import prove, prove_owner_boundary_classes


@pytest.mark.parametrize("policy", ("startup-and-low", "first-owned"))
def test_packed_staggered_state_cache_routes_all_versions(policy):
    result = prove((32, 35, 37), 4, mode="all-cache", load_policy=policy)
    stats = result["statistics"]
    assert stats["owned_updates"] == 4 * stats["sampled_physical_recurrences"]
    assert stats["cached_updates"] == stats["owned_updates"]
    assert result["additional_low_x_staging_bytes"] == (1200 if policy == "startup-and-low" else 0)


def test_single_axis_admission_uses_fallback_on_corners_and_small_domains():
    small = prove((32, 35, 37), 4)
    assert small["statistics"]["cached_updates"] == 0
    assert small["statistics"]["direct_updates"] > 0
    larger = prove((64, 96, 128), 16)
    assert larger["statistics"]["cached_updates"] > 0
    assert larger["statistics"]["direct_updates"] > 0
    assert larger["ideal_all_active_psi_bytes_per_update"] < 12


def test_every_relative_startup_and_incoming_owner_boundary_class():
    result = prove_owner_boundary_classes()
    assert result["startup_classes"] == 3072
    assert result["incoming_classes"] == 747
    assert result["future_overwrite_before_consumer"] is False


@pytest.mark.parametrize("option", ("omit_h_publication", "omit_e_publication", "omit_low_refresh",
                                   "wrong_e_extent", "publish_halo", "future_publication"))
def test_rejects_missing_publication_wrong_packing_or_future_state(option):
    policy = "startup-and-low" if option == "publish_halo" else "first-owned"
    with pytest.raises(AssertionError):
        prove((32, 35, 37), 4, mode="all-cache", load_policy=policy, **{option: True})


def test_private_point_wrapper_uses_fixed_scalar_refs_and_preserves_fallback(tmp_path):
    if not shutil.which("c++"):
        pytest.skip("C++ compiler required")
    source = r'''
#include <initializer_list>
#define __device__
#define __forceinline__ inline
namespace beamz {namespace cuda {namespace yee {
int CpmlAxis(int term){const int axes[]={1,0,0,2,2,1};return axes[term];}
}}}
struct Data{float global[12];int reads[12]{},writes[12]{};};
struct Base{Data&data;
float LoadLocal(int s,int,int,int)const{++data.reads[s];return data.global[s];}
void StoreLocal(int s,int,int,int,float value)const{++data.writes[s];data.global[s]=value;}
};
'''
    source += POINT_ENV + r'''
template<int Axis>bool Check(){
 Data data{};for(int s=0;s<12;++s)data.global[s]=float(7*s);
 Base base{data};float h0=-999.f,h1=-999.f,e0=-999.f,e1=-999.f;
 for(int step=1;step<=4;++step){
  PrivatePsiPoint<Axis,Base>env(base,h0,h1,e0,e1,step==1,step==4);
  for(int s=0;s<12;++s){const bool selected=beamz::cuda::yee::CpmlAxis(s%6)==Axis;
   const float old=env.LoadLocal(s,0,0,0);if(old!=float(7*s+step-1))return false;
   env.StoreLocal(s,0,0,0,old+1.f);
   if(data.reads[s]!=(selected?1:step))return false;
   if(data.writes[s]!=(selected?(step==4?1:0):step))return false;
   if(data.global[s]!=float(7*s+(selected&&step<4?0:step)))return false;
  }
 }
 return true;
}
int main(){return Check<0>()&&Check<1>()&&Check<2>()?0:1;}
'''
    path, binary = tmp_path / "private.cc", tmp_path / "private"
    path.write_text(source)
    compiled = subprocess.run(["c++", "-std=c++17", "-O2", str(path), "-o", str(binary)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    assert subprocess.run([str(binary)], timeout=10).returncode == 0


def test_resource_patch_does_not_wire_private_axis_into_runtime_dispatch():
    from scripts.patch_cuda_native_import_packets import patch as packets
    from scripts.patch_cuda_native_io_policies import patch as policies

    root = Path(__file__).resolve().parents[2]
    source = policies(packets((root / "scripts/cuda_asymmetric_native.cuh").read_text()))
    changed = patch(source)
    assert changed.count("__syncthreads()") == source.count("__syncthreads()")
    assert "float psi_ha[CZ],psi_hb[CZ],psi_ea[CZ],psi_eb[CZ];" in changed
    assert changed.count("PrivatePsiPoint<CacheAxis,decltype(env)> point_env(") == 2
    # The source's complete rank-dispatch function remains byte-identical.
    start = "inline cudaError_t LaunchRanks("
    end = "template<int Depth,int Layout>\ninline cudaError_t DispatchClass"
    assert changed[changed.index(start):changed.index(end)] == source[source.index(start):source.index(end)]
    with pytest.raises(ValueError):
        patch(changed)


def test_strong_axis_dispatch_is_cta_uniform_and_keeps_multiaxis_fallback():
    from scripts.patch_cuda_native_import_packets import patch as packets
    from scripts.patch_cuda_native_io_policies import patch as policies

    root = Path(__file__).resolve().parents[2]
    source = policies(packets((root / "scripts/cuda_asymmetric_native.cuh").read_text()))
    changed = patch(source, strong_axis=True, cta_switch=True)
    start = changed.index("// blockIdx and immutable tile facts")
    end = changed.index("inline bool MakeSource(", start)
    wrapper = changed[start:end]
    assert "__grid_constant__ Parameters p" in wrapper
    assert "__ldg(p.flags+tiles[blockIdx.x])&14u" in wrapper
    assert "threadIdx" not in wrapper
    assert "else if constexpr(Policy==1)" in wrapper
    assert "AsymmetricBody<Depth,Layout,Fast,IsotropicH,Policy,-1>(p,tiles)" in wrapper
    for mask, axis in ((2, 0), (4, 1), (8, 2)):
        assert f"case {mask}:AsymmetricBody<Depth,Layout,Fast,IsotropicH,Policy,{axis}>" in wrapper
    # The single-axis proof must only erase other axes; selected-axis validity
    # and the ordinary class's active predicate remain delegated to Base.
    assert "CpmlAxis(slot%6)!=Axis)return false" in changed
    assert "return Base::LocalActive(slot,z,y,x);" in changed
