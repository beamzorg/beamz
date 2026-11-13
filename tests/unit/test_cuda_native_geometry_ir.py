"""Canonical component-shape facts and actual emitted metadata helper parity."""

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.cuda_native_geometry_ir import (
    SUPPORT_DELTA,
    derive_geometry,
    emit_geometry,
    verify_component_shapes,
)
from scripts.cuda_space_time_ir import IllegalSchedule, yee_operator
from scripts.cuda_stencil_trace import Expr
from scripts.generate_cuda_native_cell import generate_native_cell
from scripts.patch_cuda_native_geometry import patch

ROOT = Path(__file__).resolve().parents[2]


def test_actual_traced_terms_share_only_phase_axis_extent_not_profiles():
    report = derive_geometry()
    assert len(report["derivatives"]) == 12
    assert report["removed_H_wall_guards"] == 6
    assert len(report["cpml_terms"]) == 12
    for phase in (0, 1):
        for axis in (0, 1, 2):
            terms = [t for t in report["cpml_terms"] if t["phase"] == phase and t["axis_zyx"] == axis]
            assert len(terms) == 2
            assert {t["extent"] for t in terms} == {f"N{axis}+{phase}"}
            assert all("profile pointers" in t["distinct"] for t in terms)
    assert emit_geometry() == emit_geometry()


@pytest.mark.parametrize("minimum", ((24, 25, 31), (64, 96, 128), (512, 512, 512)))
def test_actual_shape_admission_rejects_one_changed_terminal_extent(minimum):
    shapes = tuple(tuple(n + d for n, d in zip(minimum, delta, strict=True)) for delta in SUPPORT_DELTA)
    assert verify_component_shapes(shapes) == minimum
    wrong = list(shapes)
    wrong[4] = (wrong[4][0], wrong[4][1] + 1, wrong[4][2])
    with pytest.raises(IllegalSchedule):
        verify_component_shapes(wrong)


def test_future_or_noncanonical_derivative_does_not_get_native_guard_fact():
    operator = yee_operator()
    update = operator.updates[0]
    read = next(r for r in update.reads if any(r.offset))
    changed = replace(read, field="Ex")
    def mutate(value):
        if value.op == "read" and value.args[0] == read:
            return Expr("read", (changed,))
        return Expr(value.op, tuple(mutate(v) if isinstance(v, Expr) else v for v in value.args))

    updates = (replace(update, value=mutate(update.value)),) + operator.updates[1:]
    with pytest.raises(IllegalSchedule):
        derive_geometry(replace(operator, updates=updates))


def test_emitted_geometry_matches_native_guards_and_cpml_indices(tmp_path):
    if not shutil.which("c++"):
        pytest.skip("C++ compiler required")
    header = tmp_path / "generated.cuh"
    header.write_text(emit_geometry()["source"])
    source = tmp_path / "main.cc"
    source.write_text(r'''
#include <bit>
#include <cstdint>
#define __device__
#define __forceinline__ inline
float __fsub_rn(float a,float b){volatile float v=a-b;return v;}
float __fmul_rn(float a,float b){volatile float v=a*b;return v;}
#include "generated.cuh"
template<int Phase> bool Check(){
 using namespace beamz::cuda::trace_geometry;
 for(int extent:{24,25,31,128})for(int p=0;p<extent+Phase;++p){
  const int full_extent=extent+Phase;
  if(CpmlPacked<Phase>(p,extent)!=(p<12?p:24+p-full_extent))return false;
  if(CpmlActive<Phase>(p,extent)!=(p<12||p>=full_extent-12))return false;
  for(uint32_t bits:{0u,0x80000000u,0x3f234567u,0xc0123456u}){
   const float center=std::bit_cast<float>(bits),adjacent=std::bit_cast<float>(bits^0x80000000u);
   const int source_extent=extent+1-Phase;
   float expected;
   if((!Phase&&p+1>=source_extent)||(Phase&&(p==0||p==source_extent)))expected=0.f;
   else expected=__fmul_rn(Phase?__fsub_rn(center,adjacent):__fsub_rn(adjacent,center),.3125f);
   if(std::bit_cast<uint32_t>(Difference<Phase>(p,extent,center,adjacent,.3125f))!=std::bit_cast<uint32_t>(expected))return false;
  }
 }
 return true;
}
int main(){return Check<0>()&&Check<1>()?0:1;}
'''.replace('#include <bit>', '#include <bit>\n#include <initializer_list>'))
    binary = tmp_path / "probe"
    compiled = subprocess.run(["c++", "-std=c++20", "-O2", "-ffp-contract=off", str(source), "-o", str(binary)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run([str(binary)], timeout=10)
    assert executed.returncode == 0


def test_geometry_patch_preserves_schedule_and_fails_closed_on_stale_functor():
    source = (ROOT / "scripts/cuda_asymmetric_native.cuh").read_text()
    cell = generate_native_cell()["source"]
    changed = patch(source, cell_source=cell)
    assert changed.count("__syncthreads()") == source.count("__syncthreads()")
    assert changed.count("__shfl_sync(") == source.count("__shfl_sync(")
    assert "coordinate+1>=size" not in changed
    assert "coordinate==p.physical_n[axis]" in changed
    assert "out->physical_n[axis]=minimum[axis]" in changed
    assert changed.count("const int size=p.physical_n[axis]+family;") == 2
    with pytest.raises(IllegalSchedule, match="canonical Operator"):
        patch(source, cell_source=cell + "// stale\n")
    with pytest.raises(IllegalSchedule):
        patch(changed, cell_source=cell)


def test_geometry_lowering_commutes_with_independent_queue_and_import_passes():
    from scripts.patch_cuda_native_import_packets import patch as imports
    from scripts.patch_cuda_native_scalar_queue import patch as queues

    source = (ROOT / "scripts/cuda_asymmetric_native.cuh").read_text()
    cell = generate_native_cell()["source"]
    assert patch(imports(queues(source)), cell_source=cell) == imports(queues(patch(source, cell_source=cell)))


def test_actual_original_and_patched_environment_methods_match(tmp_path):
    if not shutil.which("c++"):
        pytest.skip("C++ compiler required")
    source = (ROOT / "scripts/cuda_asymmetric_native.cuh").read_text()
    changed = patch(source, cell_source=generate_native_cell()["source"])

    def environment(text, name):
        start = text.index("template<int Layout,bool Fast,bool IsotropicH> struct Environment {")
        end = text.index("\n};", start) + 4
        return text[start:end].replace("struct Environment {", f"struct {name} {{", 1)

    prelude = r'''
#include <algorithm>
#include <bit>
#include <cstdint>
#include <initializer_list>
#define __device__
#define __forceinline__ inline
using std::min;using std::max;
template<class T>T __ldg(const T*p){return *p;}
float __fsub_rn(float a,float b){volatile float v=a-b;return v;}
float __fmul_rn(float a,float b){volatile float v=a*b;return v;}
float __fadd_rn(float a,float b){volatile float v=a+b;return v;}
namespace beamz { namespace cuda { namespace moving {constexpr int kMayHaveSources=16,kMayObserve=32;}
namespace yee {
int CpmlAxis(int term){const int axis[]={1,0,0,2,2,1};return axis[term];}
float ScaleYeeDifference(float d,float inv){return __fmul_rn(d,inv);}
float PackedMaterialSource(const float*,int,const uint32_t*,int){return 0.f;}
}}}
struct Field{float*data;int z,y,x,row,plane;};
struct Psi{const float*input,*a,*b,*inv;float*output;int row,plane;};
struct Phase{Field own[3],output[3];Psi psi[6];const float*decay[3],*source[3],*tables[3];const uint32_t*codes[3];int table_sizes[3];float inv;};
struct Source{const float*coefficients,*waveforms;const int32_t*starts,*current_step;int count,nz,ny,nx,ntime;};
struct Parameters{Phase phase[2];Source sources[6];int physical_n[3],macro_step;float h_source;};
bool Contains(const Field&f,int z,int y,int x){return z>=0&&y>=0&&x>=0&&z<f.z&&y<f.y&&x<f.x;}
const Field&InitialField(const Parameters&p,int f){return p.phase[f/3].own[f%3];}
const Field&FinalField(const Parameters&p,int f){return p.phase[f/3].output[f%3];}
void Observe(const Parameters&,int,int,int,int,int,float){}
'''
    records = "{" + ",".join("{" + ",".join(str(v) for v in (
        item["phase"], "xyz".index(item["output"][-1]),
        (0 if item["source"][0] == "H" else 3) + "xyz".index(item["source"][-1]), item["axis_zyx"])) + "}"
                              for item in derive_geometry()["derivatives"]) + "}"
    main = r'''
int main(){
 const int delta[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
 const int derivatives[][4]=RECORDS;
 for(int base:{24,31,64}){
  Parameters p{};for(int a=0;a<3;++a)p.physical_n[a]=base+a;
  for(int f=0;f<6;++f){auto&v=p.phase[f/3];const int c=f%3;
   auto&shape=v.own[c];shape.z=base+delta[f][0];shape.y=base+1+delta[f][1];shape.x=base+2+delta[f][2];shape.row=shape.x+3;shape.plane=shape.row*(shape.y+2);v.output[c]=shape;v.inv=.137f;
   for(int term=2*c;term<2*c+2;++term){const int axis=beamz::cuda::yee::CpmlAxis(term);int dims[]={shape.z,shape.y,shape.x};dims[axis]=24;v.psi[term].row=dims[2];v.psi[term].plane=dims[1]*dims[2];}
  }
  for(unsigned flags:{0u,2u,4u,8u,14u}){
   OriginalEnvironment<1,false,false> old{p,static_cast<uint8_t>(flags)};
   ChangedEnvironment<1,false,false> now{p,static_cast<uint8_t>(flags)};
   for(int slot=0;slot<12;++slot){const int phase=slot/6,term=slot%6,axis=beamz::cuda::yee::CpmlAxis(term);const auto&f=p.phase[phase].own[term/2];const int dims[]={f.z,f.y,f.x};
    for(int coord=0;coord<dims[axis];++coord){int point[]={1,1,1};point[axis]=coord;const int z=point[0],y=point[1],x=point[2];
     if(old.LocalActive(slot,z,y,x)!=now.LocalActive(slot,z,y,x)||old.Packed(slot,z,y,x)!=now.Packed(slot,z,y,x)||old.PsiOffset(slot,z,y,x)!=now.PsiOffset(slot,z,y,x))return 1;
    }
   }
   for(const auto&r:derivatives){const int phase=r[0],c=r[1],field=r[2],axis=r[3];const auto&f=p.phase[phase].own[c];const int dims[]={f.z,f.y,f.x};
    for(int coord=0;coord<dims[axis];++coord){int point[]={0,0,0};point[axis]=coord;
     for(uint32_t bits:{0u,0x80000000u,0x3f234567u,0xc0123456u}){float a=std::bit_cast<float>(bits),b=std::bit_cast<float>(bits^0x80000000u);
      if(std::bit_cast<uint32_t>(old.Difference(phase,field,axis,point[0],point[1],point[2],a,b))!=std::bit_cast<uint32_t>(now.Difference(phase,field,axis,point[0],point[1],point[2],a,b)))return 2;
     }
    }
   }
  }
 }
 return 0;
}
'''.replace("RECORDS", records)
    path = tmp_path / "environment.cc"
    path.write_text(prelude + environment(source, "OriginalEnvironment") + environment(changed, "ChangedEnvironment") + main)
    binary = tmp_path / "environment"
    compiled = subprocess.run(["c++", "-std=c++20", "-O2", "-ffp-contract=off", str(path), "-o", str(binary)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run([str(binary)], timeout=10)
    assert executed.returncode == 0
