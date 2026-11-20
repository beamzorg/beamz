"""Canonical Yee-cell graph and deterministic CUDA emission; stdlib build dependency."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

# Each component has two signed curl edges: (source component, Z/Y/X axis).
_CURL = (((2, 1), (1, 0)), ((0, 0), (2, 2)), ((1, 2), (0, 1)))


def _cell() -> str:
    return (Path(__file__).resolve().parents[1] / "src/physics/cell.cuh").read_text()


def _terms() -> str:
    text = "struct CurlTerm {int source_component;int derivative_axis;};\n"
    for i, name in enumerate(("First", "Second")):
        values = [f"CurlTerm{{{edges[i][0]},{edges[i][1]}}}" for edges in _CURL]
        text += f"__host__ __device__ constexpr CurlTerm {name}CurlTerm(int component){{return component==0?{values[0]}:component==1?{values[1]}:{values[2]};}}\n"
    axes = [edge[1] for edges in _CURL for edge in edges]
    expression = str(axes[-1])
    for i in reversed(range(5)): expression = f"term=={i}?{axes[i]}:({expression})"
    return text + f"__host__ __device__ constexpr int CpmlAxis(int term){{return {expression};}}\n__host__ __device__ constexpr float CpmlSign(int term){{return term%2==0?1.0f:-1.0f;}}\n"


def _adapter() -> str:
    return '// Compatibility arithmetic interface, backed by the canonical cell graph.\n#pragma once\n#include "cell.cuh"\nnamespace beamz::cuda::generated {\ntemplate<int Phase,int Component,class ReadField,class ReadCoefficient>\n__device__ __forceinline__ float AdvanceYeeCell(float inv,ReadField read,ReadCoefficient coefficient){\n struct Environment {\n  float inv;ReadCoefficient coefficient;\n  __device__ bool Valid(int,int,int,int)const{return true;}\n  __device__ bool LocalActive(int,int,int,int)const{return false;}\n  __device__ float LoadLocal(int,int,int,int)const{return 0.f;}\n  __device__ void StoreLocal(int,int,int,int,float)const{}\n  __device__ float Coefficient(int kind,int field,int,int,int)const{return coefficient(kind,field%3);}\n  __device__ float Difference(int phase,int,int,int,int,int,float center,float adjacent)const{\n   return __fmul_rn(phase?__fsub_rn(center,adjacent):__fsub_rn(adjacent,center),inv);\n  }\n  __device__ float Finish(int,int,int,int,int,float value)const{return value;}\n } env{inv,coefficient};\n return beamz::cuda::trace_native::AdvanceNativeCell<Phase,Component,false>(env,read,1,0,0,0);\n}\n}\n'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=Path(".")); parser.add_argument("--fingerprint", action="store_true"); args = parser.parse_args()
    if args.fingerprint:
        primitives = Path(__file__).resolve().parents[1] / "src/physics/yee_primitives.cuh"
        print(hashlib.sha256((_cell() + _terms() + _adapter()).encode() + primitives.read_bytes()).hexdigest())
        raise SystemExit(0)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, source in (("cell.cuh", _cell()), ("curl_terms.inc", _terms()), ("yee_cell_generated.cuh", _adapter())):
        target = args.output / name
        if not target.exists() or target.read_text() != source: target.write_text(source)
