"""Generate a complete fixed-plane Yee tile from the schedule/storage IR.

Experimental device helper only. Environment callbacks provide real global
state, material/source/CPML/publication semantics; no alternate native ABI is
invented. Version0 is the invocation's initial state, versions1..T its updates.
The source-owner guard always precedes local reads. No GPU execution here.
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
    from scripts.cuda_space_time_ir import (
        FIELDS,
        IllegalSchedule,
        StreamShape,
        compile_stream,
        yee_operator,
    )
    from scripts.cuda_stencil_trace import Expr, Update, emit_cuda, emit_cuda_block
    from scripts.cuda_trace_cpml_lower import lower_group
except ModuleNotFoundError:
    from cuda_space_time_ir import (
        FIELDS,
        IllegalSchedule,
        StreamShape,
        compile_stream,
        yee_operator,
    )
    from cuda_stencil_trace import Expr, Update, emit_cuda, emit_cuda_block
    from cuda_trace_cpml_lower import lower_group


ENVIRONMENT_CONTRACT = {
    "inv_spacing": "float member; canonical direct inverse spacing",
    "Read": "float Read(int fieldID,int version,int z,int y,int x); initial0 or external staged value; safe physical bounds",
    "Valid": "bool Valid(int fieldID,int z,int y,int x); true staggered component support",
    "Coefficient": "float Coefficient(int kind,int index,int z,int y,int x); kind0decay/1source indexes fieldID,2a/3b/4invk indexes local-stateID",
    "LoadLocal": "float LoadLocal(int localStateID,int z,int y,int x); ordinary mutable previous psi; outside support0",
    "StoreLocal": "void StoreLocal(int localStateID,int z,int y,int x,float); after all group SSA reads; canonical storage round already emitted",
    "Finish": "float Finish(int fieldID,int version,int z,int y,int x,float); point-local source injection before neighbor use; no evolving-neighbor reads",
    "Publish": "void Publish(int fieldID,int version,int z,int y,int x,float); includes zero values outside physical bounds, callback admits valid stores; unique field writer; raw observations/trails/final stores only; deferred DFT reduction",
}


def mutable_groups(operator, phase):
    """Partition outputs by shared mutable-old reads before scheduling stores.

    This prevents a psi store from changing another expression's old-psi read.
    The current backend admits exactly one primary field output per group.
    """
    specs = operator.specs
    pending = [u for u in operator.updates if specs[u.output.field].phase == phase]
    groups = []
    while pending:
        group = [pending.pop(0)]
        state_reads = {r.field for r in group[0].reads if specs[r.field].role == "local-state"}
        changed = True
        while changed:
            changed = False
            for update in tuple(pending):
                other = {r.field for r in update.reads if specs[r.field].role == "local-state"}
                if state_reads.intersection(other):
                    pending.remove(update)
                    group.append(update)
                    state_reads.update(other)
                    changed = True
        fields = [u.output.field for u in group if specs[u.output.field].role == "field"]
        if len(fields) != 1:
            raise IllegalSchedule("Mutable SSA group must have exactly one primary Yee field")
        groups.append((fields[0], tuple(group)))
    return tuple(sorted(groups))


def emit_cell_group(group, field, specs, tensor_ids, read_name, coefficient_name,
                    parameters, output_name, prefix, *, native_boundaries):
    """Emit one last-use-checked mutable component group without scheduling it."""
    lines = []
    primary = next(u for u in group if u.output.field == field)
    native_terms = native_boundaries and primary.value.args[1].op == "add"
    term_report = []
    if native_terms:
        terms, field_value = lower_group(field, group, specs)
        for term_index, term in enumerate(terms):
            derivative = emit_cuda(parameters(term.derivative), read_name, coefficient_name)
            lines += [f"const float derivative_{term_index}={derivative};",
                       f"float corrected_{term_index}=__fmul_rn({term.sign!r}f,derivative_{term_index});"]
            if term.state is not None:
                sid = tensor_ids[term.state]
                a = emit_cuda(parameters(term.a), read_name, coefficient_name)
                b = emit_cuda(parameters(term.b), read_name, coefficient_name)
                invk = emit_cuda(parameters(term.inv_kappa), read_name, coefficient_name)
                lines += [f"if(env.LocalActive({sid},z,y,x)) {{",
                           f"const float next_psi=__fmaf_rn({b},env.LoadLocal({sid},z,y,x),__fmul_rn({a},derivative_{term_index}));",
                           f"corrected_{term_index}=__fmul_rn({term.sign!r}f,__fmaf_rn(derivative_{term_index},{invk},next_psi));"]
                stored = "__bfloat162float(__float2bfloat16_rn(next_psi))" if term.precision == "bfloat16" else "next_psi"
                lines += [f"env.StoreLocal({sid},z,y,x,{stored});", "}"]
            term_report.append({"state": term.state, "guarded": term.state is not None,
                                "precision": term.precision, "old_state_last_use": "recurrence/correction before store"})
        transformed = (Update(primary.output, parameters(field_value)),)
    else:
        for update in group:
            if specs[update.output.field].role == "local-state":
                lines.append(f"float result_{tensor_ids[update.output.field]}=0.f;")
        transformed = tuple(Update(u.output, parameters(u.value)) for u in group)
    block = emit_cuda_block(transformed, read_name=read_name,
                            coefficient_name=coefficient_name,
                            write_name=lambda dst, output_name=output_name, field=field: output_name if dst.field == field else f"result_{tensor_ids[dst.field]}",
                            prefix=prefix)
    lines.append(block["source"])
    if not native_terms:
        # Generic SSA must read every mutable input before stores.
        for update in group:
            if specs[update.output.field].role == "local-state":
                sid = tensor_ids[update.output.field]
                lines.append(f"env.StoreLocal({sid},z,y,x,result_{sid});")
    return {"source": "\n".join(lines), "liveness": block["liveness"],
            "native_term_scopes": term_report}


def generate(operator=None, shape=None, *, warp_x=True, metadata_remat=False, materialized=False, native_boundaries=False, relative_callbacks=False, loop_stages=False):  # noqa: C901
    operator = yee_operator() if operator is None else operator
    shape = StreamShape() if shape is None else shape
    if loop_stages:
        if materialized or metadata_remat or not warp_x:
            raise IllegalSchedule("Compact stage-loop currently admits direct warp-X lowering without provider/remat variants")
        try:
            from scripts.cuda_streamed_stage_loop import generate_looped
        except ModuleNotFoundError:
            from cuda_streamed_stage_loop import generate_looped
        return generate_looped(operator, shape, relative_callbacks=relative_callbacks,
                               native_boundaries=native_boundaries)
    plan = compile_stream(operator, shape, warp_x=warp_x, materialized=materialized)
    specs = operator.specs
    tensor_ids = {field: i for i, field in enumerate(FIELDS)}
    tensor_ids.update({t.name: i + len(FIELDS) for i, t in enumerate(t for t in operator.tensors if t.role == "local-state")})
    history = {(f["tensor"], f["version"]) for f in plan["frontier"]}
    routes = {(a["stage"], a["output"], a["source"], a["version"], tuple(a["offset"])): a for a in plan["accesses"]}
    offsets, total = {}, 0
    for family in "HE":
        for c in plan["shared_components"][family]:
            offsets[family, c] = total
            total += shape.threads
    def holder_name(value):
        return value.replace("_previous_", "_").replace("_current_", "_")

    def provider_loads(stage, prefixes):
        if not materialized:
            return []
        grouped = {}
        for item in plan["provider_materialization"]["providers"]:
            prefix, component = item["holder"].rsplit("_", 1)
            if item["stage"] == stage and prefix in prefixes:
                key = prefix, item["version"], item["physical_z_offset"], item["source"][0]
                grouped.setdefault(key, {})[component] = item
        lines = ["{"]
        if not relative_callbacks:
            lines.append(f"const int provider_z=oz+iz-{stage},provider_y=oy-{shape.depth}+qy,provider_x=ox-{shape.depth}+qx;")
        for (prefix, version, dz, family), components in sorted(grouped.items()):
            lo = shape.depth - stage
            needs = []
            for c in "xyz":
                if c not in components:
                    needs.append("0u")
                    continue
                predicates = [f"(qy>={lo+dy}&&qy<{lo+dy+shape.tile_zyx[1]}&&qx>={lo+dx}&&qx<{lo+dx+shape.tile_zyx[2]})"
                              for dy, dx in components[c]["xy_offsets"]]
                needs.append(f"(({ '||'.join(predicates) })?{1 << 'xyz'.index(c)}u:0u)")
            delta = dz + version - stage
            inside = (f"iz+({delta})>=0&&iz+({delta})<{shape.tile_zyx[0]}&&"
                      f"qy+({version-shape.depth})>=0&&qy+({version-shape.depth})<{shape.tile_zyx[1]}&&"
                      f"qx+({version-shape.depth})>=0&&qx+({version-shape.depth})<{shape.tile_zyx[2]}")
            mask_name = f"provider_mask_{prefix}_{version}_{dz+1}"
            outside = "true" if version == 0 else f"!({inside})"
            lines.append(f"const unsigned {mask_name}=({outside})?({'|'.join(needs)}):0u;")
            packet_method = "ReadPacketRelative" if relative_callbacks else "ReadPacket"
            packet_coords = f"iz+({delta}),qy+({version-shape.depth}),qx+({version-shape.depth})" if relative_callbacks else f"provider_z+({dz}),provider_y,provider_x"
            lines.append(f"if({mask_name})env.{packet_method}({mask_name},{int(family=='E')},{version},{packet_coords},{holder_name(prefix+'_x')},{holder_name(prefix+'_y')},{holder_name(prefix+'_z')});")
        lines.append("}")
        return lines

    source = ["// Deterministic output of scripts/cuda_space_time_codegen.py.",
              "// Callback contract is recorded in the companion manifest; research only.",
              "#pragma once", "#include <cuda_bf16.h>",
              "namespace beamz { namespace cuda { namespace streamed_ir {",
              f'inline constexpr char SemanticSHA256[]="{plan["semantic_sha256"]}";',
              f"inline constexpr int Depth={shape.depth}, Threads={shape.threads}, SharedBytes={4 * total + (12 if metadata_remat else 0)};",
              "template<class Environment> __device__ __forceinline__",
              "void RunStreamedTile(Environment &env,float *exchange,int oz,int oy,int ox) {",
              "const int q=threadIdx.x,qx=q&31,qy=q>>5;"]
    if metadata_remat:
        source += [f"volatile int*origin=reinterpret_cast<volatile int*>(exchange+{total});",
                   "if(q==0){origin[0]=oz;origin[1]=oy;origin[2]=ox;}"]
    for field, version in sorted(history, key=lambda p: (p[1], p[0])):
        source.append(f"float history_{field}_{version}=0.f;")
    source += [f"for(int iz=0;iz<{shape.tile_zyx[0]};++iz) {{"]
    for name in ("old_h", "old_e", "current_h", "current_e"):
        source.append("float " + ",".join(f"{name}_{c}=0.f" for c in "xyz") + ";")
    group_reports = []
    for n in range(1, shape.depth + 1):
        source += ["{", f"const bool owned_xy=qx>={shape.depth-n}&&qx<{shape.depth-n+shape.tile_zyx[2]}&&qy>={shape.depth-n}&&qy<{shape.depth-n+shape.tile_zyx[1]};"]
        source += provider_loads(n, ("old_h_previous", "old_e_previous", "current_e_previous"))
        if n > 1 or materialized:
            source += [f"exchange[{offsets['E', c]}+q]=old_e_{'xyz'[c]};" for c in plan["shared_components"]["E"]]
        source.append("__syncthreads(); // A: publish previous-E exchange; retire previous E's H readers.")
        for phase, family in enumerate("HE"):
            if phase:
                source += provider_loads(n, ("old_h_current", "current_h"))
                source += [f"exchange[{offsets['H', c]}+q]=current_h_{'xyz'[c]};" for c in plan["shared_components"]["H"]]
                source.append("__syncthreads(); // B: publish H exchange; retire H's previous-E readers.")
            # Opaque readonly origins are published by the first A barrier.
            # Their phase-local loads break cross-barrier address-expression
            # liveness without a mutable phase-counter protocol.
            source.append("{")
            origins = ("origin[0]", "origin[1]", "origin[2]") if metadata_remat else ("oz", "oy", "ox")
            source += [f"const int z={origins[0]}+iz-{n};",
                       f"const int y={origins[1]}-{shape.depth}+qy,x={origins[2]}-{shape.depth}+qx;"]
            shuffle_names = {}
            candidates = [a for a in plan["accesses"] if a["stage"] == n and a["phase"] == phase and a["route"] == "warp-shuffle"]
            for route in candidates:
                key = route["holder"], route["offset"][2]
                if key not in shuffle_names:
                    name = f"shuffle_{n}_{phase}_{len(shuffle_names)}"
                    shuffle_names[key] = name
                    # All32 lanes participate before either owned/valid predicate.
                    source.append(f"const float {name}=__shfl_sync(0xffffffffu,{holder_name(key[0])},(qx+({key[1]}))&31);")
            for field, group in mutable_groups(operator, phase):
                component = field[-1]
                family_lower = family.lower()
                output_name = f"next_{family_lower}_{component}"
                previous = f"history_{field}_{n}" if (field, n) in history else "0.f"
                source += ["{", f"const float captured_old={previous};", f"float {output_name}=0.f;",
                           f"if(owned_xy&&env.Valid({tensor_ids[field]},z,y,x)) {{"]

                def read_name(read, output=field, phase=phase, n=n, shuffle_names=shuffle_names):
                    version = read.version + n - 1
                    if specs[read.field].role == "local-state":
                        return f"env.LoadLocal({tensor_ids[read.field]},z,y,x)"
                    dz, dy, dx = read.offset
                    external = (f"env.ReadRelative({tensor_ids[read.field]},{version},iz+({dz+version-n}),qy+({dy+version-shape.depth}),qx+({dx+version-shape.depth}))"
                                if relative_callbacks else f"env.Read({tensor_ids[read.field]},{version},z+({dz}),y+({dy}),x+({dx}))")
                    if not version and not materialized:
                        return external
                    # Route keys are identical for a group's primary/psi consumers.
                    route = routes[n, output, read.field, version, read.offset]
                    if route["route"] == "shared-exchange":
                        c = "xyz".index(read.field[-1])
                        local = f"exchange[{offsets[read.field[0], c]}+q+({dy})*32+({dx})]"
                    elif route["route"] == "warp-shuffle":
                        local = shuffle_names[route["holder"], dx]
                    else:
                        local = holder_name(route["holder"])
                    if materialized:
                        lo = shape.depth - n
                        inside_union = (lo + dy >= 0 and lo + dy + shape.tile_zyx[1] <= shape.threads // 32 and
                                        lo + dx >= 0 and lo + dx + shape.tile_zyx[2] <= 32)
                        if inside_union:
                            return local
                        return f"((qy+({dy})>=0&&qy+({dy})<{shape.threads//32}&&qx+({dx})>=0&&qx+({dx})<32)?({local}):({external}))"
                    uz_delta = dz + version - n
                    uy_delta, ux_delta = dy + version - shape.depth, dx + version - shape.depth
                    predicate = (f"iz+({uz_delta})>=0&&iz+({uz_delta})<{shape.tile_zyx[0]}&&"
                                 f"qy+({uy_delta})>=0&&qy+({uy_delta})<{shape.tile_zyx[1]}&&"
                                 f"qx+({ux_delta})>=0&&qx+({ux_delta})<{shape.tile_zyx[2]}")
                    return f"(({predicate})?({local}):({external}))"

                def coefficient_name(read):
                    name = read.name
                    z, y, x = read.offset
                    if name == "inv_spacing":
                        return "env.inv_spacing"
                    for prefix, kind in (("decay_", 0), ("source_", 1), ("a_", 2), ("b_", 3), ("inv_kappa_", 4)):
                        if name.startswith(prefix) and name[len(prefix):] in tensor_ids:
                            coordinates = f"z+({z}),y+({y}),x+({x})" if read.axes else "0,0,0"
                            return f"env.Coefficient({kind},{tensor_ids[name[len(prefix):]]},{coordinates})"
                    raise IllegalSchedule(f"Unsupported coefficient callback signature: {name}")

                def parameters(value, phase=phase):
                    if native_boundaries and value.op == "mul_rn":
                        difference, scale = value.args
                        scale_name = scale.args[0] if scale.op == "parameter" else scale.args[0].name if scale.op == "coefficient" else None
                        if scale_name == "inv_spacing" and difference.op == "sub" and all(a.op == "read" for a in difference.args):
                            derivative_reads = [a.args[0] for a in difference.args]
                            center = next((r for r in derivative_reads if not any(r.offset)), None)
                            adjacent = next((r for r in derivative_reads if any(r.offset)), None)
                            if center is not None and adjacent is not None and center.field == adjacent.field and center.version == adjacent.version:
                                axis = next(a for a, d in enumerate(adjacent.offset) if d)
                                expected = 1 if phase == 0 else -1
                                if adjacent.offset[axis] != expected or sum(bool(d) for d in adjacent.offset) != 1:
                                    raise IllegalSchedule("Native derivative lowering requires canonical axial direction")
                                # Native callback owns only wall branching/ScaleYeeDifference;
                                # actual versioned field operands still come from this plan.
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
                        raise IllegalSchedule(f"Unsupported parameter: {name}")
                    return Expr(value.op, tuple(parameters(a) if isinstance(a, Expr) else a for a in value.args))

                cell = emit_cell_group(group, field, specs, tensor_ids, read_name,
                                       coefficient_name, parameters, output_name,
                                       f"s{n}_{family}_{component}", native_boundaries=native_boundaries)
                source.append(cell["source"])
                source += [f"{output_name}=env.Finish({tensor_ids[field]},{n},z,y,x,{output_name});", "}",
                           f"old_{family_lower}_{component}=captured_old;",
                           f"current_{family_lower}_{component}={output_name};"]
                if (field, n) in history:
                    source.append(f"history_{field}_{n}={output_name};")
                source.append("}")
                group_reports.append({"stage": n, "field": field,
                                      "cell_ssa_liveness": cell["liveness"],
                                      "native_term_scopes": cell["native_term_scopes"],
                                      "mutable_outputs": [u.output.field for u in group if specs[u.output.field].role == "local-state"]})
            if phase:
                source.append("if(owned_xy) {")
                if relative_callbacks:
                    values = ",".join(f"current_{f[0].lower()}_{f[-1]}" for f in FIELDS)
                    source.append(f"env.PublishPacketRelative({n},iz,qy+({n-shape.depth}),qx+({n-shape.depth}),{values});")
                else:
                    source.extend(f"env.Publish({i},{n},z,y,x,current_{f[0].lower()}_{f[-1]});" for i, f in enumerate(FIELDS))
                source.append("}")
            source.append("}")
        source.append("}")
    source += ["}", "}", "} } } // namespace beamz::cuda::streamed_ir", ""]
    code = "\n".join(source)
    try:
        from scripts.cuda_codegen_audit import source_costs
    except ModuleNotFoundError:
        from cuda_codegen_audit import source_costs
    return {"source": code, "static_source_costs": source_costs(code), "plan": plan, "tensor_ids": tensor_ids,
            "environment_contract": {**ENVIRONMENT_CONTRACT,
                **({"ReadPacket": "void ReadPacket(unsigned mask,int family,int version,int z,int y,int x,float&cx,float&cy,float&cz); update requested components only, invalid requested components0; one shared ownership/surface computation"} if materialized else {}),
                **({"ReadRelative": "float ReadRelative(int field,int version,int rz,int ry,int rx); source u-origin in[-1,B-1]",
                    "ReadPacketRelative": "ReadPacket signature with transformed owner-relative source coordinates",
                    "PublishPacketRelative": "void PublishPacketRelative(int version,int rz,int ry,int rx,float hx,float hy,float hz,float ex,float ey,float ez); owned destination u-origin, batch metadata once"} if relative_callbacks else {}),
                **({"Difference": "float Difference(int phase,int sourceFieldID,int axis,int z,int y,int x,float center,float adjacent); native wall branches + exact ScaleYeeDifference; never reload fields",
                    "LocalActive": "bool LocalActive(int localStateID,int z,int y,int x); exact native packed CPML support; inactive terms skip all recurrence/profile arithmetic"} if native_boundaries else {})},
            "relative_callback_lowering": {"enabled": relative_callbacks,
                "coordinates": "source u-origin supplied directly; no physical-to-owner reconstruction",
                "envelope": "every source component in [-1,B-1]; each publication point owned"},
            "native_boundary_lowering": {"enabled": native_boundaries,
                "requires": "Difference uses actual source support/metallic bits and canonical RN scaling; Finish masks PEC after psi stores before source"},
            "metadata_rematerialization": {"enabled": metadata_remat,
                "shared_bytes": 12 if metadata_remat else 0,
                "protocol": "immutable origins initialized once; first A publishes; phase-local volatile reads"},
            "groups": group_reports, "source_sha256": hashlib.sha256(code.encode()).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tile", nargs=3, type=int, default=(64, 13, 29))
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--threads", type=int, default=512)
    p.add_argument("--cpml-axes", nargs="*", type=int)
    p.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    p.add_argument("--no-warp-x", action="store_true")
    p.add_argument("--metadata-remat", action="store_true")
    p.add_argument("--materialize-providers", action="store_true")
    p.add_argument("--native-boundaries", action="store_true")
    p.add_argument("--relative-callbacks", action="store_true")
    p.add_argument("--loop-stages", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    result = generate(yee_operator(cpml_axes=a.cpml_axes, precision=a.precision),
                      StreamShape(tuple(a.tile), a.depth, a.threads), warp_x=not a.no_warp_x,
                      metadata_remat=a.metadata_remat, materialized=a.materialize_providers, native_boundaries=a.native_boundaries, relative_callbacks=a.relative_callbacks, loop_stages=a.loop_stages)
    if a.check:
        if not a.output.is_file() or a.output.read_text() != result["source"]:
            p.error("Generated source is missing or differs")
    else:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(result["source"])
    if a.manifest:
        a.manifest.write_text(json.dumps({k: v for k, v in result.items() if k != "source"}, sort_keys=True, indent=2) + "\n")
    print(result["source_sha256"])


if __name__ == "__main__":
    main()
