"""Host-admitted canonical Yee geometry facts and metadata/guard lowering.

Prove facts from actual traced dependencies and six component support shapes.
This pass does not alias CPML profiles or auxiliary strides: only their logical
axis coordinate/extent is shared. All FP32 subtraction/scaling nodes remain.
The emitted helper is experimental and has no automatic production dispatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_space_time_ir import (
    FIELDS,
    IllegalSchedule,
    validate_operator,
    yee_operator,
)
from scripts.cuda_stencil_trace import serialize_updates

# Physical support increments in ZYX order, distinct from vector-component XYZ.
SUPPORT_DELTA = ((0, 0, 1), (0, 1, 0), (1, 0, 0),
                 (1, 1, 0), (1, 0, 1), (0, 1, 1))


def verify_component_shapes(shapes):
    if (len(shapes) != 6 or any(len(s) != 3 for s in shapes)
            or any(type(n) is not int or n < 1 for s in shapes for n in s)):
        raise IllegalSchedule("Six positive integer ZYX component shapes required")
    minimum = tuple(min(s[a] for s in shapes) for a in range(3))
    expected = tuple(tuple(n + d for n, d in zip(minimum, delta, strict=True)) for delta in SUPPORT_DELTA)
    if tuple(map(tuple, shapes)) != expected:
        raise IllegalSchedule("Actual component shapes do not match canonical Yee staggering")
    return minimum


def derive_geometry(operator=None):
    operator = yee_operator(cpml_axes=(0, 1, 2)) if operator is None else operator
    validate_operator(operator)
    specs = operator.specs
    shapes = {name: SUPPORT_DELTA[index] for index, name in enumerate(FIELDS)}
    fields = {u.output.field: u for u in operator.updates if specs[u.output.field].role == "field"}
    if set(fields) != set(FIELDS):
        raise IllegalSchedule("Exactly six canonical Yee fields required")
    derivatives = []
    for output, update in fields.items():
        phase = specs[output].phase
        adjacent_reads = [r for r in update.reads if r.field in fields and any(r.offset)]
        if len(adjacent_reads) != 2:
            raise IllegalSchedule("Canonical Yee field requires two axial derivatives")
        for read in adjacent_reads:
            axes = [axis for axis, delta in enumerate(read.offset) if delta]
            if len(axes) != 1 or read.offset[axes[0]] != (1 if phase == 0 else -1):
                raise IllegalSchedule("Canonical difference direction required")
            axis = axes[0]
            component = "xyz".index(output[-1])
            vector_axis = 2 - axis
            if (vector_axis == component or read.field[-1] != "xyz"[3 - component - vector_axis]
                    or read.version != phase):
                raise IllegalSchedule("Derivative is not the canonical tangential Yee component/version")
            if specs[read.field].phase == phase or shapes[output][axis] != phase or shapes[read.field][axis] != 1 - phase:
                raise IllegalSchedule("Derivative support does not prove canonical axis geometry")
            derivatives.append({"output": output, "source": read.field, "phase": phase,
                                "axis_zyx": axis, "output_extent": f"N{axis}+{phase}",
                                "source_extent": f"N{axis}+{1-phase}",
                                "guard": "unreachable-after-output-valid" if not phase else "coordinate==0 or coordinate==Naxis"})
    terms = []
    for update in operator.updates:
        name = update.output.field
        if specs[name].role != "local-state":
            continue
        consumers = [field for field, value in fields.items() if any(r.field == name for r in value.reads)]
        if len(consumers) != 1:
            raise IllegalSchedule("Canonical local recurrence must correct exactly one field")
        shifted = [r for r in update.reads if r.field in fields and any(r.offset)]
        if len(shifted) != 1 or sum(bool(d) for d in shifted[0].offset) != 1:
            raise IllegalSchedule("Canonical CPML recurrence requires one axial derivative")
        axis = next(a for a, delta in enumerate(shifted[0].offset) if delta)
        phase = specs[name].phase
        owner = consumers[0]
        if shapes[owner][axis] != phase or shifted[0] not in fields[owner].reads:
            raise IllegalSchedule("CPML output support cannot share canonical phase-axis extent")
        terms.append({"term": name, "output": owner, "phase": phase, "axis_zyx": axis,
                      "extent": f"N{axis}+{phase}",
                      "active": "coord<12 or coord>=Naxis+phase-12",
                      "packed": "coord<12 ? coord : coord+24-Naxis-phase",
                      "distinct": ["profile pointers", "psi row/plane strides", "psi storage"]})
    report = {"operator_sha256": hashlib.sha256(json.dumps(serialize_updates(operator.updates), sort_keys=True).encode()).hexdigest(),
              "shape_predicate": "verify all six actual component shapes against N+SUPPORT_DELTA",
              "requires": ["output support predicate passed", "Cartesian Yee stencil", "CPML12 when used"],
              "derivatives": sorted(derivatives, key=lambda x: (x["output"], x["axis_zyx"])),
              "cpml_terms": sorted(terms, key=lambda x: x["term"]),
              "shared_cpml_extent_expressions": 6,
              "removed_H_wall_guards": sum(d["phase"] == 0 for d in derivatives),
              "arithmetic_change": "none; retain explicit RN subtract then RN multiply"}
    report["geometry_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    return report


def emit_geometry(operator=None):
    report = derive_geometry(operator)
    source = """// Generated by scripts/cuda_native_geometry_ir.py; research opt-in.
#pragma once
namespace beamz { namespace cuda { namespace trace_geometry {
// Require host-verified canonical component shapes and a valid output point.
template<int Phase> __device__ __forceinline__ int CpmlPacked(int coordinate,int minimum_extent){
static_assert(Phase==0||Phase==1,"H/E phase required");
return coordinate<12?coordinate:coordinate+24-minimum_extent-Phase;
}
template<int Phase> __device__ __forceinline__ bool CpmlActive(int coordinate,int minimum_extent){
static_assert(Phase==0||Phase==1,"H/E phase required");
return coordinate<12||coordinate>=minimum_extent+Phase-12;
}
template<int Phase> __device__ __forceinline__ float Difference(int coordinate,int minimum_extent,float center,float adjacent,float inverse_spacing){
static_assert(Phase==0||Phase==1,"H/E phase required");
if constexpr(Phase==1){if(coordinate==0||coordinate==minimum_extent)return 0.f;}
return __fmul_rn(Phase?__fsub_rn(center,adjacent):__fsub_rn(adjacent,center),inverse_spacing);
}
} } } // namespace beamz::cuda::trace_geometry
"""
    return {"source": source, "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "geometry": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = emit_geometry()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result["source"])
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps({k: v for k, v in result.items() if k != "source"}, indent=2))
    if not args.output and not args.manifest:
        print(json.dumps(result["geometry"], indent=2))


if __name__ == "__main__":
    main()
