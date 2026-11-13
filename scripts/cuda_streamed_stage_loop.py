"""Compact runtime-stage lowering with statically named register queues.

The affine schedule and fixed-XY ownership are unchanged. Named scalar queues
rotate after each component's old-value last use, avoiding dynamic register-array
indexing while emitting only one H/E phase body. Queue moves trade instructions
for a much smaller code footprint; CUDA resource/performance gates are mandatory.
"""

from __future__ import annotations

import hashlib

try:
    from scripts.cuda_space_time_ir import FIELDS, IllegalSchedule, compile_stream
    from scripts.cuda_stencil_trace import Expr
except ModuleNotFoundError:
    from cuda_space_time_ir import FIELDS, IllegalSchedule, compile_stream
    from cuda_stencil_trace import Expr


def generate_looped(operator, shape, *, relative_callbacks=False, native_boundaries=False):  # noqa: C901
    try:
        from scripts.cuda_space_time_codegen import (
            ENVIRONMENT_CONTRACT,
            emit_cell_group,
            mutable_groups,
        )
    except ModuleNotFoundError:
        from cuda_space_time_codegen import (
            ENVIRONMENT_CONTRACT,
            emit_cell_group,
            mutable_groups,
        )
    plan = compile_stream(operator, shape)
    specs = operator.specs
    tensor_ids = {name: i for i, name in enumerate(FIELDS)}
    tensor_ids.update({t.name: i + 6 for i, t in enumerate(t for t in operator.tensors if t.role == "local-state")})
    lengths = {}
    for field in FIELDS:
        versions = sorted(f["version"] for f in plan["frontier"] if f["tensor"] == field)
        if versions != list(range(1, len(versions) + 1)):
            raise IllegalSchedule("Named queue lowering requires a contiguous version frontier")
        lengths[field] = len(versions)
        if len(versions) not in (shape.depth, shape.depth - 1):
            raise IllegalSchedule("Queue lifetime is not a canonical Yee stage frontier")
    components = plan["shared_components"]
    offsets, words = {}, 0
    for family in "HE":
        for c in components[family]:
            offsets[family, c] = words
            words += shape.threads
    code = ["// Deterministic compact-stage output of cuda_space_time_codegen.py.",
            "#pragma once", "#include <cuda_bf16.h>",
            "namespace beamz { namespace cuda { namespace streamed_ir {",
            f'inline constexpr char SemanticSHA256[]="{plan["semantic_sha256"]}";',
            f"inline constexpr int Depth={shape.depth},Threads={shape.threads},SharedBytes={4*words};",
            "template<class Environment> __device__ __forceinline__",
            "void RunStreamedTile(Environment &env,float*exchange,int oz,int oy,int ox){",
            "const int q=threadIdx.x,qx=q&31,qy=q>>5;"]
    for field, length in lengths.items():
        code.extend(f"float queue_{field}_{i}=0.f;" for i in range(length))
    code += [f"for(int iz=0;iz<{shape.tile_zyx[0]};++iz){{"]
    for name in ("old_h", "old_e", "current_h", "current_e"):
        code.append("float " + ",".join(f"{name}_{c}=0.f" for c in "xyz") + ";")
    code += ["#pragma unroll 1", "for(int n=1;n<=Depth;++n){",
             f"const bool owned_xy=qx+n-{shape.depth}>=0&&qx+n-{shape.depth}<{shape.tile_zyx[2]}&&qy+n-{shape.depth}>=0&&qy+n-{shape.depth}<{shape.tile_zyx[1]};"]
    if shape.depth > 1:
        code += ["if(n>1){", *[f"exchange[{offsets['E', c]}+q]=old_e_{'xyz'[c]};" for c in components["E"]], "}"]
    code.append("__syncthreads(); // A: publish E-old; retire prior E readers.")
    reports = []
    for phase, family in enumerate("HE"):
        if phase:
            code += [f"exchange[{offsets['H', c]}+q]=current_h_{'xyz'[c]};" for c in components["H"]]
            code.append("__syncthreads(); // B: publish H-current; retire H readers.")
        code += ["{", f"const int z=oz+iz-n,y=oy-{shape.depth}+qy,x=ox-{shape.depth}+qx;"]
        input_holder = "old_e" if phase == 0 else "current_h"
        dx = 1 if phase == 0 else -1
        for c in "yz":
            code.append(f"const float shuffled_{c}=__shfl_sync(0xffffffffu,{input_holder}_{c},(qx+({dx}))&31);")
        for field, group in mutable_groups(operator, phase):
            component, lower = field[-1], family.lower()
            length = lengths[field]
            captured = f"queue_{field}_0" if length else "0.f"
            conditional = length != shape.depth
            if conditional:
                captured = f"(n<Depth?({captured}):0.f)"
            output = f"next_{lower}_{component}"
            code += ["{", f"const float captured_old={captured};", f"float {output}=0.f;",
                     f"if(owned_xy&&env.Valid({tensor_ids[field]},z,y,x)){{"]

            def read_name(read, phase=phase):
                if specs[read.field].role == "local-state":
                    return f"env.LoadLocal({tensor_ids[read.field]},z,y,x)"
                dz, dy, dx = read.offset
                dt = read.version - 1
                version = "n" if not dt else "(n-1)"
                rz = f"iz+({dz+dt})"
                ry, rx = f"qy+n+({dy+dt-shape.depth})", f"qx+n+({dx+dt-shape.depth})"
                external = (f"env.ReadRelative({tensor_ids[read.field]},{version},{rz},{ry},{rx})" if relative_callbacks else
                            f"env.Read({tensor_ids[read.field]},{version},z+({dz}),y+({dy}),x+({dx}))")
                if read.field[0] == ("E" if phase else "H"):
                    local = f"old_{read.field[0].lower()}_{read.field[-1]}"
                elif read.offset[0]:
                    local = f"{'old_h' if phase else 'current_e'}_{read.field[-1]}"
                elif dy:
                    c = "xyz".index(read.field[-1])
                    # E exchange is absent at T1; every previous-version E
                    # operand then takes the global branch unconditionally.
                    local = f"exchange[{offsets.get((read.field[0],c),0)}+q+({dy})*32]"
                elif dx:
                    local = f"shuffled_{read.field[-1]}"
                else:
                    local = f"{'current_h' if phase else 'old_e'}_{read.field[-1]}"
                initial_guard = "n>1&&" if dt else ""
                owned = (f"{initial_guard}{rz}>=0&&{rz}<{shape.tile_zyx[0]}&&"
                         f"{ry}>=0&&{ry}<{shape.tile_zyx[1]}&&{rx}>=0&&{rx}<{shape.tile_zyx[2]}")
                return f"(({owned})?({local}):({external}))"

            def coefficients(read):
                if read.name == "inv_spacing":
                    return "env.inv_spacing"
                for prefix, kind in (("decay_", 0), ("source_", 1), ("a_", 2), ("b_", 3), ("inv_kappa_", 4)):
                    if read.name.startswith(prefix) and read.name[len(prefix):] in tensor_ids:
                        dz, dy, dx = read.offset
                        coordinates = f"z+({dz}),y+({dy}),x+({dx})" if read.axes else "0,0,0"
                        return f"env.Coefficient({kind},{tensor_ids[read.name[len(prefix):]]},{coordinates})"
                raise IllegalSchedule("Unsupported coefficient signature")

            def parameters(value, phase=phase, read_name=read_name):
                if native_boundaries and value.op == "mul_rn":
                    difference, scale = value.args
                    name = scale.args[0] if scale.op == "parameter" else scale.args[0].name if scale.op == "coefficient" else None
                    if name == "inv_spacing" and difference.op == "sub" and all(a.op == "read" for a in difference.args):
                        operands = [a.args[0] for a in difference.args]
                        center = next((r for r in operands if not any(r.offset)), None)
                        adjacent = next((r for r in operands if any(r.offset)), None)
                        if center and adjacent and center.field == adjacent.field and center.version == adjacent.version:
                            axis = next(a for a, d in enumerate(adjacent.offset) if d)
                            return Expr("parameter", (f"env.Difference({phase},{tensor_ids[center.field]},{axis},z,y,x,({read_name(center)}),({read_name(adjacent)}))",))
                if value.op == "parameter":
                    name = value.args[0]
                    if name == "inv_spacing":
                        return Expr("parameter", ("env.inv_spacing",))
                    if name in ("corrected_0", "corrected_1"):
                        return value
                    for prefix, kind in (("decay_", 0), ("source_", 1)):
                        if name.startswith(prefix) and name[len(prefix):] in tensor_ids:
                            return Expr("parameter", (f"env.Coefficient({kind},{tensor_ids[name[len(prefix):]]},z,y,x)",))
                    raise IllegalSchedule("Unsupported parameter")
                return Expr(value.op, tuple(parameters(a) if isinstance(a, Expr) else a for a in value.args))

            cell = emit_cell_group(group, field, specs, tensor_ids, read_name,
                                   coefficients, parameters, output, f"loop_{field}",
                                   native_boundaries=native_boundaries)
            code += [cell["source"], f"{output}=env.Finish({tensor_ids[field]},n,z,y,x,{output});", "}",
                     f"old_{lower}_{component}=captured_old;", f"current_{lower}_{component}={output};"]
            if length:
                if conditional:
                    code.append("if(n<Depth){")
                code.extend(f"queue_{field}_{i}=queue_{field}_{i+1};" for i in range(length - 1))
                code.append(f"queue_{field}_{length-1}={output};")
                if conditional:
                    code.append("}")
            code.append("}")
            reports.append({"field": field, "cell_ssa_liveness": cell["liveness"],
                            "native_term_scopes": cell["native_term_scopes"]})
        if phase:
            code.append("if(owned_xy){")
            if relative_callbacks:
                values = ",".join(f"current_{f[0].lower()}_{f[-1]}" for f in FIELDS)
                code.append(f"env.PublishPacketRelative(n,iz,qy+n-{shape.depth},qx+n-{shape.depth},{values});")
            else:
                code.extend(f"env.Publish({i},n,z,y,x,current_{f[0].lower()}_{f[-1]});" for i, f in enumerate(FIELDS))
            code.append("}")
        code.append("}")
    code += ["}", "}", "}", "} } }", ""]
    source = "\n".join(code)
    try:
        from scripts.cuda_codegen_audit import source_costs
    except ModuleNotFoundError:
        from cuda_codegen_audit import source_costs
    return {"source": source, "static_source_costs": source_costs(source), "plan": plan, "tensor_ids": tensor_ids,
            "groups": reports, "environment_contract": {**ENVIRONMENT_CONTRACT,
                **({"ReadRelative": "Read with exact transformed source u-origin in[-1,B-1]",
                    "PublishPacketRelative": "Publish6 fields with owned destination u-origin"} if relative_callbacks else {}),
                **({"Difference": "Native wall branches/RN scaling using supplied stage operands",
                    "LocalActive": "Exact packed CPML support; false path skips all recurrence/profile arithmetic"} if native_boundaries else {})},
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "stage_control_lowering": {"kind": "runtime-loop/fixed-named-scalar-queues",
                "queue_lengths": lengths, "persistent_scalars_per_thread": sum(lengths.values()),
                "phase_body_copies": 2, "history_has_dynamic_array_index": False,
                "queue_assignments_per_u_z_plane": sum(length * (shape.depth if length == shape.depth else shape.depth - 1) for length in lengths.values()),
                "code_budget_review": {"static_sass_instructions": 2048, "branch_sites": 128,
                    "scope": "review thresholds, not performance predictions or automatic promotion"}},
            "relative_callback_lowering": {"enabled": relative_callbacks},
            "native_boundary_lowering": {"enabled": native_boundaries}}
