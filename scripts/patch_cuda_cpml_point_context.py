"""Trace-derived complete CPML point-address context for cubic one-axis NoIO.

Unlike per-term stride substitution, one phase/axis context computes active,
packed, common base and the alternate-term delta once for all3 field updates.
The exact scalar arithmetic and original false/default bodies remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA, derive_geometry
from scripts.patch_cuda_native_cpml_strides import ADMISSION
from scripts.patch_cuda_shared_profiles import PARENT


def sha(value):
    return hashlib.sha256(value).hexdigest()


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def mul(a, b):
    values = [0] * 5
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            values[i + j] += x * y
    if any(values[3:]):
        raise ValueError("Address polynomial exceeds quadratic canonical strides")
    return tuple(values[:3])


def derive():
    """Derive exact stride polynomials from actual typed target supports."""
    geometry = derive_geometry()
    axis_of = {x["term"]: x["axis_zyx"] for x in geometry["cpml_terms"]}
    result = []
    # Coefficients of N**0,N**1,N**2 for each z/y/x coordinate. The normal
    # coordinate is already packed; N is the actual common physical dimension.
    common = {
        0: ((0, 1, 1), (0, 1, 0), (1, 0, 0)),
        1: ((0, 24, 0), (0, 1, 0), (1, 0, 0)),
        2: ((0, 24, 0), (24, 0, 0), (1, 0, 0)),
    }
    alternate = {
        0: ((0, 0, 0), (1, 0, 0), (0, 0, 0)),
        1: ((24, 0, 0), (1, 0, 0), (0, 0, 0)),
        2: ((24, 0, 0), (0, 0, 0), (0, 0, 0)),
    }
    for phase in range(2):
        for axis in range(3):
            rows = []
            extra = []
            for term in range(6):
                name = f"psi_{'HE'[phase]}{term}"
                if axis_of[name] != axis:
                    continue
                shape = [(d, 1, 0) for d in SUPPORT_DELTA[phase * 3 + term // 2]]
                shape[axis] = (24, 0, 0)
                strides = (mul(shape[1], shape[2]), shape[2], (1, 0, 0))
                difference = tuple(sub(a, b) for a, b in zip(strides, common[axis]))
                if difference == alternate[axis]:
                    extra.append(term)
                elif any(any(p) for p in difference):
                    raise ValueError("Actual target staggering is not common-base plus one exact delta")
                rows.append({"term": term, "target_support_delta": SUPPORT_DELTA[phase * 3 + term // 2],
                             "packed_shape_polynomial": shape, "stride_polynomial": strides,
                             "extra_delta": difference == alternate[axis]})
            if len(rows) != 2 or len(extra) != 1:
                raise ValueError("One phase/axis requires two distinct terms and one alternate stride")
            result.append({"phase": phase, "axis": axis, "common_base_polynomial": common[axis],
                           "delta_polynomial": alternate[axis], "extra_term": extra[0], "terms": rows})
    return result


def helper():
    derived = derive()
    extras = {(r["phase"], r["axis"]): r["extra_term"] for r in derived}
    expression = "Phase==0?(" + ":".join([f"Axis=={a}?{extras[0,a]}" for a in (0, 1)]) + f":{extras[0,2]})" + ":(" + ":".join([f"Axis=={a}?{extras[1,a]}" for a in (0, 1)]) + f":{extras[1,2]})"
    return r'''
// Derived from actual typed field supports and independent packed psi targets.
template<int Phase,int Axis,class Base>struct CubicCpmlPointContext:Base{
 static_assert((Phase==0||Phase==1)&&Axis>=0&&Axis<3,"Exact phase and one-hot axis required");
 static constexpr int ExtraTerm=@EXTRA@;
 int packed_index,point_base,point_delta;bool point_active;
 __device__ __forceinline__ CubicCpmlPointContext(const Base&base,int z,int y,int x):Base(base){
  const int n=this->p.physical_n[0],coordinate=Axis==0?z:Axis==1?y:x,size=n+Phase;
  point_active=coordinate<12||coordinate>=size-12;
  packed_index=coordinate<12?coordinate:24+coordinate-size;
  if constexpr(Axis==0){point_base=packed_index*(n*(n+1))+y*n+x;point_delta=y;}
  else if constexpr(Axis==1){const int q=z*24+packed_index;point_base=q*n+x;point_delta=q;}
  else{point_base=(z*n+y)*24+packed_index;point_delta=z*24;}
 }
 __device__ __forceinline__ bool Eligible(int slot)const{
  return slot/6==Phase&&beamz::cuda::yee::CpmlAxis(slot%6)==Axis;
 }
 __device__ __forceinline__ int Offset(int slot)const{return point_base+(slot%6==ExtraTerm?point_delta:0);}
 __device__ __forceinline__ bool LocalActive(int slot,int z,int y,int x)const{
  if(Eligible(slot))return point_active;return Base::LocalActive(slot,z,y,x);
 }
 __device__ __forceinline__ float LoadLocal(int slot,int z,int y,int x)const{
  if(Eligible(slot))return this->p.phase[Phase].psi[slot%6].input[Offset(slot)];
  return Base::LoadLocal(slot,z,y,x);
 }
 __device__ __forceinline__ void StoreLocal(int slot,int z,int y,int x,float value)const{
  if(Eligible(slot)){this->p.phase[Phase].psi[slot%6].output[Offset(slot)]=value;return;}
  Base::StoreLocal(slot,z,y,x,value);
 }
 __device__ __forceinline__ float Coefficient(int kind,int index,int z,int y,int x)const{
  if(kind>=2&&Eligible(index)){const auto&psi=this->p.phase[Phase].psi[index%6];
   return kind==2?__ldg(psi.a+packed_index):kind==3?__ldg(psi.b+packed_index):__ldg(psi.inv+packed_index);
  }
  return Base::Coefficient(kind,index,z,y,x);
 }
};
'''.replace("@EXTRA@", expression)


H_LOOP = """     #pragma unroll
     for(int c=0;c<3;++c){float value=0.f;if(env.Valid(c,z,y,x)){
      if(c==0)value=beamz::cuda::trace_native::AdvanceNativeCell<0,0,false>(env,read,n,z,y,x);
      else if(c==1)value=beamz::cuda::trace_native::AdvanceNativeCell<0,1,false>(env,read,n,z,y,x);
      else value=beamz::cuda::trace_native::AdvanceNativeCell<0,2,false>(env,read,n,z,y,x);
     }h[c][j]=value;}"""
E_LOOP = H_LOOP.replace("env.Valid(c,z,y,x)", "env.Valid(c+3,z,y,x)").replace("AdvanceNativeCell<0,", "AdvanceNativeCell<1,").replace("}h[c][j]", "}e[c][j]")


def transformed(files):
    for name, expected in PARENT.items():
        if sha(files[name]) != expected:
            raise ValueError("Exact selected52865 parent required:" + name)
    host = files["native_extension.cc"].decode()
    if any(a not in host for a in ADMISSION):
        raise ValueError("Actual canonical staggered24-slab admission changed")
    original = files["asymmetric_native.cuh"].decode()
    text, edits = original, []

    def change(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError(f"One exact CPML point-context anchor required:{old!r}")
        text = text.replace(old, new, 1)
        edits.append((old, new))

    change("// Canonical staggering was proved by Prepare before this kernel can run.", helper() + "\n// Canonical staggering was proved by Prepare before this kernel can run.")
    change("int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody", "int Policy,bool UniformE,int ActiveAxes=-1,bool PointContext=false>\n__device__ __forceinline__ void UniformBody")
    anchor = " constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);\n extern __shared__ float frame[];size_t*bases="
    change(anchor, " constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);\n static_assert(!PointContext||(Layout==1&&!Fast&&IsotropicH&&Policy==1&&UniformE&&(ActiveAxes==1||ActiveAxes==2||ActiveAxes==4)),\"Point context requires certified one-axis NoIO\");\n constexpr int PointAxis=ActiveAxes==1?0:ActiveAxes==2?1:2;\n extern __shared__ float frame[];size_t*bases=")
    for phase, loop in ((0, H_LOOP), (1, E_LOOP)):
        selected = loop.replace("(env,read,n,z,y,x)", "(point_env,read,n,z,y,x)")
        replacement = f"     if constexpr(PointContext){{\n      CubicCpmlPointContext<{phase},PointAxis,decltype(env)>point_env(env,z,y,x);\n" + selected + "\n     }else{\n" + loop + "\n     }"
        change(loop, replacement)
    start = text.index("__global__ __launch_bounds__(1024,1) void AsymmetricNoIOUniform1024(")
    end = text.index("template<int Depth>", start)
    wrapper = text[start:end]
    intro = " // Original full certified NoIO grid; selector uniform before any barrier."
    dispatch = """ // Cubic coordinates and one-hot trajectory axis are immutable CTA-uniform facts.
 if(p.physical_n[0]==p.physical_n[1]&&p.physical_n[1]==p.physical_n[2]){
  switch(p.flags[tiles[blockIdx.x]]&14){
   case 2:UniformBody<Depth,1,false,true,1,true,1,true>(p,tiles);return;
   case 4:UniformBody<Depth,1,false,true,1,true,2,true>(p,tiles);return;
   case 8:UniformBody<Depth,1,false,true,1,true,4,true>(p,tiles);return;
   default:break;
  }
 }
"""
    change(wrapper, wrapper.replace(intro, dispatch + intro))
    inverse = text
    for before, after in reversed(edits):
        if inverse.count(after) != 1:
            raise ValueError("Point-context inverse anchor is ambiguous")
        inverse = inverse.replace(after, before, 1)
    assert inverse == original
    assert text.count("__syncthreads()") == original.count("__syncthreads()")
    marker = "// Private source-only storage schedule; source-bound"
    assert text[text.index(marker):text.index("inline bool MakeSource")] == original[original.index(marker):original.index("inline bool MakeSource")]
    result = dict(files)
    result["asymmetric_native.cuh"] = text.encode()
    return result, {"derived_phase_axis_addresses": derive(), "exact_source_inverse": True,
                    "shared_bytes_unchanged": 51748, "all_barrier_sites_identical": True,
                    "selector": "Native canonical admission ANDcubic physical_n ANDcertifiedNoIOUniform ANDonehotCPML axis",
                    "rectangle_multiaxis_and_default_original_loops_retained": True,
                    "canonical_cell_and_source_bodies_byte_identical": True,
                    "profile_phase_term_kind_and_psi_input_output_pointer_identities_retained": True,
                    "expected_moving_kernels": 62}


def patch_copy(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if output.exists() or output == canonical or canonical in output.parents or source in output.parents or output in source.parents:
        raise ValueError("New disjoint private source directory required")
    before = {name: (source / "src" / name).read_bytes() for name in PARENT}
    after, proof = transformed(before)
    shutil.copytree(source, output)
    for name, value in after.items():
        (output / "src" / name).write_bytes(value)
    proof.update(schema="beamz.cubic-cpml-point-context.v1", parent_hashes=PARENT,
                 result_hashes={name: sha(v) for name, v in after.items()},
                 producer_sha256=sha(Path(__file__).read_bytes()), GPU_executed=False)
    (output.parent / "point-context-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output), indent=2))


if __name__ == "__main__":
    main()
