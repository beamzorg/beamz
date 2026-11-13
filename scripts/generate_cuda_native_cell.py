"""Emit the native point functor from the same typed cell/control-flow IR.

No field-storage or scheduler is embedded. A winning owned-cache schedule can
supply its cached versioned reads while native Env supplies packed material,
canonical wall differences, local CPML, PEC/source finish and real supports.
Field IDs0Hx..5Ez; local-term IDs0H0..5H5,6E0..11E5 are separate namespaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from scripts.cuda_codegen_audit import source_costs
    from scripts.cuda_space_time_codegen import emit_cell_group, mutable_groups
    from scripts.cuda_space_time_ir import (
        FIELDS,
        IllegalSchedule,
        validate_operator,
        validate_stream_signature,
        yee_operator,
    )
    from scripts.cuda_stencil_trace import Expr
except ModuleNotFoundError:
    from cuda_codegen_audit import source_costs
    from cuda_space_time_codegen import emit_cell_group, mutable_groups
    from cuda_space_time_ir import (
        FIELDS,
        IllegalSchedule,
        validate_operator,
        validate_stream_signature,
        yee_operator,
    )
    from cuda_stencil_trace import Expr


def generate_native_cell(operator=None, *, profile_bindings=None, read_binding=None):  # noqa: C901
    operator = yee_operator(cpml_axes=(0, 1, 2)) if operator is None else operator
    legality = validate_operator(operator)
    validate_stream_signature(operator)
    if profile_bindings is not None:
        from scripts.cuda_readonly_profile_bindings import verify_bindings

        verify_bindings(operator, profile_bindings)
    specs = operator.specs
    ids = {field: i for i, field in enumerate(FIELDS)}
    local_ids = {}
    for tensor in operator.tensors:
        if tensor.role == "local-state":
            name = tensor.name
            if not name.startswith("psi_") or len(name) != 6 or name[4] not in "HE" or name[5] not in "012345":
                raise IllegalSchedule("Native local-term ABI admits canonical named H/E CPML terms only")
            local_ids[name] = 6 * int(name[4] == "E") + int(name[5])
    ids.update(local_ids)
    lines = ["// Deterministic output of scripts/generate_cuda_native_cell.py; research opt-in.",
             "#pragma once", "#include <cuda_bf16.h>",
             "namespace beamz { namespace cuda { namespace trace_native {",
             "template<int Phase,int Component,bool CheckSupport=true,class Environment,class ReadField>",
             "__device__ __forceinline__ float AdvanceNativeCell(Environment&env,ReadField read,int version,int z,int y,int x){",
             "static_assert(Phase==0||Phase==1,\"Yee phase must be H/E\");",
             "static_assert(Component>=0&&Component<3,\"Yee component must be x/y/z\");"]
    reports = []
    point_blocks = {}
    for phase in (0, 1):
        for field, group in mutable_groups(operator, phase):
            c = "xyz".index(field[-1])
            lines += [f"if constexpr(Phase=={phase}&&Component=={c}){{",
                      f"if constexpr(CheckSupport){{if(!env.Valid({ids[field]},z,y,x))return 0.f;}}",
                      "float result=0.f;"]

            def read_name(read):
                if specs[read.field].role == "local-state":
                    return f"env.LoadLocal({ids[read.field]},z,y,x)"
                dz, dy, dx = read.offset
                v = "version" if read.version else "(version-1)"
                return f"read({ids[read.field]},{v},{dz},{dy},{dx})"

            def coefficients(read):
                if read.name == "inv_spacing":
                    return "env.inv_spacing"
                for prefix, kind in (("decay_", 0), ("source_", 1), ("a_", 2), ("b_", 3), ("inv_kappa_", 4)):
                    if read.name.startswith(prefix) and read.name[len(prefix):] in ids:
                        dz, dy, dx = read.offset
                        coordinates = f"z+({dz}),y+({dy}),x+({dx})" if read.axes else "0,0,0"
                        index = ids[read.name[len(prefix):]]
                        if kind >= 2 and profile_bindings is not None:
                            index = profile_bindings.slot(read)
                        return f"env.Coefficient({kind},{index},{coordinates})"
                raise IllegalSchedule("Unknown native coefficient signature")

            def parameters(value, phase=phase, read_fun=read_name):
                if value.op == "mul_rn":
                    difference, scale = value.args
                    scale_name = scale.args[0] if scale.op == "parameter" else scale.args[0].name if scale.op == "coefficient" else None
                    if scale_name == "inv_spacing" and difference.op == "sub" and all(a.op == "read" for a in difference.args):
                        operands = [a.args[0] for a in difference.args]
                        center = next((r for r in operands if not any(r.offset)), None)
                        adjacent = next((r for r in operands if any(r.offset)), None)
                        if center and adjacent and center.field == adjacent.field and center.version == adjacent.version:
                            axis = next(a for a, d in enumerate(adjacent.offset) if d)
                            return Expr("parameter", (f"env.Difference({phase},{ids[center.field]},{axis},z,y,x,({read_fun(center)}),({read_fun(adjacent)}))",))
                if value.op == "parameter":
                    name = value.args[0]
                    if name == "inv_spacing":
                        return Expr("parameter", ("env.inv_spacing",))
                    if name in ("corrected_0", "corrected_1"):
                        return value
                    for prefix, kind in (("decay_", 0), ("source_", 1)):
                        if name.startswith(prefix) and name[len(prefix):] in ids:
                            return Expr("parameter", (f"env.Coefficient({kind},{ids[name[len(prefix):]]},z,y,x)",))
                    raise IllegalSchedule("Unknown native parameter")
                return Expr(value.op, tuple(parameters(a,phase,read_fun) if isinstance(a, Expr) else a for a in value.args))

            cell = emit_cell_group(group, field, specs, ids, read_name, coefficients,
                                   parameters, "result", f"native_{field}", native_boundaries=True)
            lines += [cell["source"], f"return env.Finish({ids[field]},version,z,y,x,result);", "}"]
            if read_binding is not None:
                def bound_read(read):
                    if specs[read.field].role == "local-state":
                        return f"env.LoadLocal({ids[read.field]},z,y,x)"
                    return read_binding(field,read)
                bound=emit_cell_group(group,field,specs,ids,bound_read,coefficients,
                    lambda value:parameters(value,phase,bound_read),"result",f"native_{field}",native_boundaries=True)
                point_blocks[field] = ("float result=0.f;\n"+bound["source"]+
                                       f"\nresult=env.Finish({ids[field]},n,z,y,x,result);")
            reports.append({"field": field, "phase": phase, "component": c,
                            "native_term_scopes": cell["native_term_scopes"],
                            "cell_ssa_liveness": cell["liveness"]})
    lines += ["return 0.f;", "}", "} } } // namespace beamz::cuda::trace_native", ""]
    source = "\n".join(lines)
    result = {"source": source, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "field_ids": {f: i for i, f in enumerate(FIELDS)}, "local_term_ids": local_ids,
            "legality": legality, "groups": reports, "static_source_costs": source_costs(source),
            "environment_contract": {
                "read": "float read(fieldID,macroLocalVersion,dz,dy,dx); n in1..Depth, caller supplies cached n-1/n values; Env adds macro_offset for source/DFT exactly once; fixed Yee signature",
                "Valid": "actual staggered component support; may skip only after caller's explicit support proof",
                "Difference": "native forward/backward boundary branch + exact RN scaling of supplied operands; no field reload",
                "Coefficient": "kind0decay/1source uses fieldID; kind2a/3b/4invk uses canonical localTermID",
                "LocalActive": "exact packed CPML support predicate; false skips profile/psi loads and stores",
                "LoadLocal": "ordinary mutable packed psi at physical point, canonical localTermID namespace",
                "StoreLocal": "store after old-psi final use; BF16 round already emitted; unique owned point only",
                "Finish": "mask PEC after recurrence, then H/E point source before opposite-phase use; reject pre_E until separate lifecycle",
                "dispatch": "verify scalar specialization facts against actual packet if operator was specialized"}}
    if profile_bindings is not None:
        result["profile_bindings"] = profile_bindings.evidence()
        result["environment_contract"]["dispatch"] += "; verify_profile_class against actual host inputs"
    if read_binding is not None:
        result['point_blocks']=point_blocks
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpml-axes", nargs="*", type=int, default=(0, 1, 2))
    p.add_argument("--no-cpml", action="store_true")
    p.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--check", action="store_true")
    p.add_argument("--scalar-h-class", action="store_true", help="Compile a class variant; dispatcher must verify current scalar class before use")
    a = p.parse_args()
    operator = yee_operator(cpml_axes=None if a.no_cpml else tuple(a.cpml_axes), precision=a.precision)
    evidence = None
    if a.scalar_h_class:
        try:
            from scripts.cuda_trace_specialize import (
                MagneticScalarClass,
                specialize_scalar_class,
            )
        except ModuleNotFoundError:
            from cuda_trace_specialize import (
                MagneticScalarClass,
                specialize_scalar_class,
            )
        operator, evidence = specialize_scalar_class(operator, MagneticScalarClass())
    result = generate_native_cell(operator)
    if evidence is not None:
        result["scalar_class_specialization"] = evidence
    if a.check:
        if not a.output.is_file() or a.output.read_text() != result["source"]:
            p.error("Generated native cell differs")
    else:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(result["source"])
    if a.manifest:
        a.manifest.write_text(json.dumps({k: v for k, v in result.items() if k != "source"}, sort_keys=True, indent=2) + "\n")
    print(result["source_sha256"])


if __name__ == "__main__":
    main()
