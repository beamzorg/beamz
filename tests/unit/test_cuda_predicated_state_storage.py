"""Predicate-aware interference, H/E persistence, and actual emitted slots."""

import copy
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.cuda_predicated_state_storage import (
    allocate,
    verify,
    verify_guard_lifetime,
)
from scripts.cuda_space_time_ir import IllegalSchedule, yee_operator
from scripts.cuda_stencil_trace import Expr, Read
from scripts.patch_cuda_native_private_psi import patch


def test_single_axis_guard_reduces_twelve_persistent_states_to_four_colors():
    report = allocate()
    assert report == allocate()
    assert len(report["interference_edges"]) == 18
    assert report["colors_per_point"] == report["optimal_lower_bound"] == 4
    assert list(report["colors"].values()) == [0, 0, 1, 0, 1, 1, 2, 2, 3, 2, 3, 3]
    control = allocate(exclusive_axis=False)
    assert control["colors_per_point"] == 12
    verify(control, allow_multiple_axes=True)


def test_allocation_rejects_multiple_axes_and_phase_aliasing():
    report = allocate()
    with pytest.raises(AssertionError):
        verify(report, allow_multiple_axes=True)
    unsafe = copy.deepcopy(report)
    unsafe["colors"].update({name: value - 2 for name, value in unsafe["colors"].items() if name.startswith("psi_E")})
    unsafe["colors_per_point"] = 2
    with pytest.raises(AssertionError, match="H/E"):
        verify(unsafe)
    verify_guard_lifetime([4] * 64)
    with pytest.raises(AssertionError):
        verify_guard_lifetime([2, 4])


def test_cross_recurrence_coupling_cannot_inherit_cpml_guard_exclusivity():
    operator = yee_operator(cpml_axes=(0, 1, 2))
    update = next(u for u in operator.updates if u.output.field == "psi_H0")
    coupled = replace(update, value=update.value + Expr("read", (Read("psi_H1", 0),)))
    operator = replace(operator, updates=tuple(coupled if u is update else u for u in operator.updates))
    with pytest.raises(IllegalSchedule):
        allocate(operator)


def colored_source():
    from scripts.patch_cuda_native_import_packets import patch as packets
    from scripts.patch_cuda_native_io_policies import patch as policies

    root = Path(__file__).resolve().parents[2]
    original = policies(packets((root / "scripts/cuda_asymmetric_native.cuh").read_text()))
    return patch(original, colored=True, cta_switch=True)


def test_one_colored_cta_body_has_uniform_single_axis_guard_and_generic_fallback():
    source = colored_source()
    begin = source.index("// blockIdx and immutable tile facts")
    end = source.index("inline bool MakeSource(", begin)
    wrapper = source[begin:end]
    assert "threadIdx" not in wrapper
    assert "if(axes==2||axes==4||axes==8)" in wrapper
    assert "Policy,-2>(p,tiles)" in wrapper
    assert "else AsymmetricBody<Depth,Layout,Fast,IsotropicH,Policy,-1>(p,tiles)" in wrapper
    assert "switch(axes)" not in wrapper
    assert "case 2:AsymmetricBody" not in wrapper


def test_colored_storage_pass_commutes_with_parallel_rank_launch_lowering():
    from scripts.patch_cuda_native_import_packets import patch as packets
    from scripts.patch_cuda_native_io_policies import patch as policies
    from scripts.patch_cuda_native_parallel_policies import patch as parallel

    root = Path(__file__).resolve().parents[2]
    original = policies(packets((root / "scripts/cuda_asymmetric_native.cuh").read_text()))
    assert parallel(patch(original, colored=True, cta_switch=True)) == patch(parallel(original), colored=True, cta_switch=True)


def test_actual_colored_point_routes_every_logical_axis_and_preserves_inactive_globals(tmp_path):
    if not shutil.which("c++"):
        pytest.skip("C++ compiler required")
    text = colored_source()
    begin = text.index("template<int Axis,class Base> struct PrivatePsiPoint")
    end = text.index("\n};", begin) + 4
    point = text[begin:end]
    source = r'''
#include <initializer_list>
#define __device__
#define __forceinline__ inline
namespace beamz{namespace cuda{namespace yee{
int CpmlAxis(int term){const int axes[]={1,0,0,2,2,1};return axes[term];}
}}}
struct Data{float global[12];int reads[12]{},writes[12]{};};
struct Base{Data&data;unsigned mask;
bool LocalActive(int s,int,int,int)const{return mask&(2u<<beamz::cuda::yee::CpmlAxis(s%6));}
float LoadLocal(int s,int,int,int)const{++data.reads[s];return data.global[s];}
void StoreLocal(int s,int,int,int,float value)const{++data.writes[s];data.global[s]=value;}
};
'''
    source += point + r'''
int main(){
 for(unsigned mask:{2u,4u,8u}){
  Data data{};for(int s=0;s<12;++s)data.global[s]=float(7*s);
  Base base{data,mask};float h0=-999.f,h1=-999.f,e0=-999.f,e1=-999.f;
  for(int step=1;step<=4;++step){PrivatePsiPoint<-2,Base>env(base,h0,h1,e0,e1,step==1,step==4);
   // H stores happen before E's OLD state reads: the four-color allocation
   // must preserve both families across this phase/time overlap.
   for(int s=0;s<12;++s)if(env.LocalActive(s,0,0,0)){
    float old=env.LoadLocal(s,0,0,0);if(old!=float(7*s+step-1))return 1;
    env.StoreLocal(s,0,0,0,old+1.f);
   }
  }
  for(int s=0;s<12;++s){const bool active=base.LocalActive(s,0,0,0);
   if(data.global[s]!=float(7*s+(active?4:0)))return 2;
   if(data.reads[s]!=(active?1:0)||data.writes[s]!=(active?1:0))return 3;
  }
 }
 return 0;
}
'''
    path, binary = tmp_path / "colored.cc", tmp_path / "colored"
    path.write_text(source)
    built = subprocess.run(["c++", "-std=c++17", "-O2", str(path), "-o", str(binary)],
                           capture_output=True, text=True, timeout=30)
    assert built.returncode == 0, built.stderr
    assert subprocess.run([str(binary)], timeout=10).returncode == 0
