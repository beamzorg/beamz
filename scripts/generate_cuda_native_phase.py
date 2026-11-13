"""Schedule canonical NoIO Yee terms by shared immutable CPML profile lifetime.

This is a phase-wide IR lowering, not a new set of FDTD equations. Two logical
recurrences using one certified same-axis profile are evaluated together. The
three profile values retire before the next axis; every output reconstructs
the original rounded expression. Mutable psi identities remain independent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_readonly_profile_bindings import (
    EqualAxisProfileClass,
    bind_profiles,
    verify_bindings,
)
from scripts.cuda_space_time_codegen import mutable_groups
from scripts.cuda_space_time_ir import (
    FIELDS,
    IllegalSchedule,
    validate_operator,
    validate_stream_signature,
    yee_operator,
)
from scripts.cuda_stencil_trace import Expr, emit_cuda, serialize_updates
from scripts.cuda_trace_axis_guards import AxisGuardClass, prune_inactive_axes
from scripts.cuda_trace_cached_read_facts import prove_addresses, verify_fact
from scripts.cuda_trace_cpml_lower import key, lower_group
from scripts.cuda_trace_runtime_uniform import (
    UniformElectricRegionClass,
    specialize_runtime_e,
)
from scripts.cuda_trace_specialize import MagneticScalarClass, specialize_scalar_class
from scripts.generate_cuda_runtime_fact_cell import verify_rounded_rewrite


@dataclass(frozen=True)
class ScheduledTerm:
    component: int
    ordinal: int
    slot: int
    term: object


def plan_phase(operator, phase, bindings):
    """Reject cross-component old-state dependencies before store reordering."""
    validate_operator(operator)
    validate_stream_signature(operator)
    verify_bindings(operator, bindings)
    if phase not in (0, 1):
        raise IllegalSchedule("Canonical H/E phase required")
    groups = mutable_groups(operator, phase)
    if len(groups) != 3:
        raise IllegalSchedule("Exactly three independent mutable component groups required")
    scheduled, outputs, states = [], {}, set()
    for field, group in groups:
        component = FIELDS.index(field) - 3 * phase
        if component not in range(3):
            raise IllegalSchedule("Mixed phase output")
        terms, output = lower_group(field, group, operator.specs)
        if len(terms) != 2:
            raise IllegalSchedule("Canonical two-term curl required")
        outputs[component] = output
        substitutions = {f"corrected_{ordinal}": term.corrected for ordinal, term in enumerate(terms)}

        def reconstruct(value, substitutions=substitutions):
            if value.op == "parameter" and value.args[0] in substitutions:
                return substitutions[value.args[0]]
            return Expr(value.op, tuple(reconstruct(arg) if isinstance(arg, Expr) else arg for arg in value.args))

        if key(reconstruct(output)) != key(next(u.value for u in group if u.output.field == field)):
            raise IllegalSchedule("Scheduled corrections do not reconstruct exact output rounding")
        for ordinal, term in enumerate(terms):
            slot = 6 * phase + 2 * component + ordinal
            expected = f"psi_{'HE'[phase]}{slot % 6}"
            if term.state != expected or term.state in states or term.precision != "float32":
                raise IllegalSchedule("Independent canonical FP32 local recurrence required")
            states.add(term.state)
            scheduled.append(ScheduledTerm(component, ordinal, slot, term))
    pairs = {}
    for item in scheduled:
        term = item.term
        reads = tuple(expression.args[0] for expression in (term.a, term.b, term.inv_kappa))
        indices = tuple(bindings.slot(read) for read in reads)
        if len(set(indices)) != 1 or any(read.offset != (0, 0, 0) for read in reads):
            raise IllegalSchedule("One unshifted readonly profile triplet required")
        pairs.setdefault(indices[0], []).append(item)
    if len(pairs) != 3 or any(len(pair) != 2 for pair in pairs.values()):
        raise IllegalSchedule("Exactly three certified same-axis profile pairs required")
    # lower_group verifies each stored next-state is the exact recurrence used
    # by its correction. mutable_groups rejects any old-state edge linking two
    # primary fields, so reordering these stores cannot change another root.
    proof = {"phase": phase, "fields": sorted(outputs), "distinct_logical_state_ids": sorted(states),
             "profile_pairs": {str(slot): [item.slot for item in pair] for slot, pair in sorted(pairs.items())},
             "field_expression_keys": {str(c): repr(key(value)) for c, value in outputs.items()},
             "per_output_rounding_reconstruction": True,
             "old_state_partition": "Each old psi belongs to exactly one canonical recurrence/correction; no cross-component old-state edges",
             "runtime_requires": "NoIO and disjoint native psi allocations; immutable profiles cannot overlap writable psi"}
    ordered = sorted(pairs.items(), key=lambda item: item[1][0].term.a.args[0].axes)
    proof["executed_profile_order"] = [slot for slot, _ in ordered]
    return tuple((slot, tuple(pair)) for slot, pair in ordered), outputs, proof


def generate(operator=None, *, cached_read_fact=None, axis_guard=None):  # noqa: C901
    if axis_guard is not None and cached_read_fact is not None:
        raise IllegalSchedule("Axis pruning and speculative reads require separate admitted experiments")
    if cached_read_fact is not None:
        verify_fact(cached_read_fact)
    canonical = yee_operator(cpml_axes=(0, 1, 2))
    if operator is None:
        operator, _ = specialize_scalar_class(canonical, MagneticScalarClass())
        operator, _ = specialize_runtime_e(operator, UniformElectricRegionClass())
    verify_rounded_rewrite(canonical, operator)
    pruned_operator, guard_proof = (operator, None) if axis_guard is None else prune_inactive_axes(operator, axis_guard)
    bindings = bind_profiles(operator, EqualAxisProfileClass())
    specs = operator.specs
    ids = {name: i for i, name in enumerate(FIELDS)}
    ids.update({f"psi_{family}{term}": 6 * phase + term for phase, family in enumerate("HE") for term in range(6)})
    lines = ["// Generated by scripts/generate_cuda_native_phase.py; admitted NoIO phase only.",
             "#pragma once", "namespace beamz { namespace cuda { namespace trace_native_phase {",
             "struct PhaseResult {float x,y,z;};",
             "template<int Phase,class Environment,class ReadField>",
             "__device__ __forceinline__ PhaseResult AdvanceNativePhase(Environment&env,ReadField read,int version,int z,int y,int x){",
             'static_assert(Phase==0||Phase==1,"Canonical phase required");', "PhaseResult result{0.f,0.f,0.f};"]
    if cached_read_fact is not None:
        lines.insert(-1, 'static_assert(!Environment::HasSources&&!Environment::HasObserve,"Speculation is admitted for NoIO only");')
    if axis_guard is not None:
        lines[4] = "template<int Phase,int AdmittedNativeAxisFlags,class Environment,class ReadField>"
        lines.insert(-1, f'static_assert(AdmittedNativeAxisFlags=={axis_guard.native_mask},"Dispatch must prove this exact immutable native axis class");')
    reports = []
    for phase in (0, 1):
        pairs, outputs, proof = plan_phase(operator, phase, bindings)
        if axis_guard is not None:
            pruned_pairs = []
            for representative, pair in pairs:
                if pair[0].term.a.args[0].axes[0] in axis_guard.axes:
                    pruned_pairs.append((representative, pair))
                else:
                    raw_pair = []
                    for item in pair:
                        term = item.term
                        raw_correction = Expr("mul_rn", (Expr("constant", (term.sign,)), term.derivative))
                        raw_pair.append(replace(item, term=replace(term, corrected=raw_correction, state=None, a=None, b=None, inv_kappa=None)))
                    pruned_pairs.append((None, tuple(raw_pair)))
            pairs = tuple(pruned_pairs)
            proof["removed_inactive_logical_states"] = [s for s in guard_proof["removed_local_states"] if s.startswith("psi_" + "HE"[phase])]
            proof["retained_logical_states"] = sorted(item.term.state for _, pair in pairs for item in pair if item.term.state is not None)
            proof["original_logical_state_ids"] = proof["distinct_logical_state_ids"]
            proof["distinct_logical_state_ids"] = proof["retained_logical_states"]
            proof["executed_profile_order"] = [slot for slot, _ in pairs if slot is not None]
            proof["profile_pairs"] = {str(slot): [item.slot for item in pair] for slot, pair in pairs if slot is not None}
        reports.append(proof)
        lines.append(f"if constexpr(Phase=={phase}){{")
        for c in range(3):
            lines.append(f"const bool valid_{c}=env.Valid({3 * phase + c},z,y,x);")
        for slot in range(6 * phase, 6 * phase + 6):
            lines.append(f"float correction_{slot}=0.f;")

        def read_name(read):
            if specs[read.field].role == "local-state":
                raise IllegalSchedule("Mutable state reads must remain in their admitted recurrence")
            dz, dy, dx = read.offset
            v = "version" if read.version else "(version-1)"
            return f"read({ids[read.field]},{v},{dz},{dy},{dx})"

        def coefficient_name(read):
            if any(read.offset):
                raise IllegalSchedule("Runtime coefficient class admits unshifted physical cells only")
            for prefix, kind in (("source_", 1), ("decay_", 0)):
                if read.name.startswith(prefix) and read.name[len(prefix):] in ids:
                    coordinates = "z,y,x" if read.axes else "0,0,0"
                    return f"env.Coefficient({kind},{ids[read.name[len(prefix):]]},{coordinates})"
            raise IllegalSchedule("Profile leaf escaped its scoped shared load")

        def native(value, phase=phase):
            if value.op == "mul_rn":
                difference, scale = value.args
                scale_name = scale.args[0] if scale.op == "parameter" else scale.args[0].name if scale.op == "coefficient" else None
                if scale_name == "inv_spacing" and difference.op == "sub" and all(a.op == "read" for a in difference.args):
                    operands = [a.args[0] for a in difference.args]
                    center = next((r for r in operands if not any(r.offset)), None)
                    adjacent = next((r for r in operands if any(r.offset)), None)
                    if center and adjacent and center.field == adjacent.field and center.version == adjacent.version:
                        axis = next(a for a, d in enumerate(adjacent.offset) if d)
                        return Expr("parameter", (f"env.Difference({phase},{ids[center.field]},{axis},z,y,x,({read_name(center)}),({read_name(adjacent)}))",))
            return Expr(value.op, tuple(native(arg) if isinstance(arg, Expr) else arg for arg in value.args))

        completed, emitted = set(), set()

        def emit_output(c, phase=phase, outputs=outputs):
            def corrected(name):
                if name in ("corrected_0", "corrected_1"):
                    return f"correction_{6 * phase + 2 * c + int(name[-1])}"
                raise IllegalSchedule("Unlowered phase parameter")

            expression = emit_cuda(native(outputs[c]), read_name, coefficient_name, corrected)
            field = FIELDS[3 * phase + c]
            return f"if(valid_{c})result.{'xyz'[c]}=env.Finish({ids[field]},version,z,y,x,{expression});"

        for representative, pair in pairs:
            lines.append("{")
            for item in pair:
                derivative = emit_cuda(native(item.term.derivative), read_name, coefficient_name)
                if cached_read_fact is None:
                    lines += [f"float derivative_{item.slot}=0.f;",
                              f"if(valid_{item.component}){{derivative_{item.slot}={derivative};",
                              f"correction_{item.slot}=__fmul_rn({item.term.sign!r}f,derivative_{item.slot});}}"]
                else:
                    lines += [f"const float derivative_{item.slot}={derivative};",
                              f"correction_{item.slot}=__fmul_rn({item.term.sign!r}f,derivative_{item.slot});"]
            if representative is not None:
                condition = "||".join(f"valid_{item.component}" for item in pair)
                lines += [f"if(({condition})&&env.LocalActive({representative},z,y,x)){{",
                          f"const float profile_a=env.Coefficient(2,{representative},z,y,x);",
                          f"const float profile_b=env.Coefficient(3,{representative},z,y,x);",
                          f"const float profile_inv=env.Coefficient(4,{representative},z,y,x);"]
                for item in pair:
                    lines += [f"if(valid_{item.component}){{",
                              f"const float next_psi=__fmaf_rn(profile_b,env.LoadLocal({item.slot},z,y,x),__fmul_rn(profile_a,derivative_{item.slot}));",
                              f"correction_{item.slot}=__fmul_rn({item.term.sign!r}f,__fmaf_rn(derivative_{item.slot},profile_inv,next_psi));",
                              f"env.StoreLocal({item.slot},z,y,x,next_psi);", "}"]
                lines.append("}")
            lines.append("}")
            completed.update(item.slot for item in pair)
            for c in range(3):
                required = {6 * phase + 2 * c, 6 * phase + 2 * c + 1}
                if c not in emitted and required.issubset(completed):
                    lines.append(emit_output(c))
                    emitted.add(c)
        if emitted != {0, 1, 2}:
            raise IllegalSchedule("Phase output has an unscheduled correction")
        lines.append("}")
    lines += ["return result;", "}", "} } } // namespace beamz::cuda::trace_native_phase", ""]
    source = "\n".join(lines)
    evidence = {"schema": "beamz.NoIO-phase-profile-last-use.v1", "phases": reports,
                "profile_bindings": bindings.evidence(), "operator": serialize_updates(operator.updates),
                "ordering": "Every per-output FP32 operator and operand order unchanged; independent term stores scheduled by profile lifetime",
                "scope": "NoIO only; actual native support/walls callbacks retained; unique disjoint packed psi allocations required",
                "profile_load_sites": 18 if axis_guard is None else 6 * len(axis_guard.axes),
                "profile_guard_sites": 6 if axis_guard is None else 2 * len(axis_guard.axes),
                "peak_scoped_profile_values": 3 if axis_guard is None or axis_guard.axes else 0, "declared_corrections": 6,
                "output_last_use": "Emit each output immediately after both original ordered corrections are available; retire its old own read"}
    if cached_read_fact is not None:
        source = source.replace("trace_native_phase", "trace_native_phase_speculated")
        evidence["cached_read_speculation"] = {"fact": asdict(cached_read_fact), "address_proof": prove_addresses(operator),
            "effect": "Only raw cached-field derivative/signed-multiply may execute when output is physically invalid; profile/psi/Finish/output guards retained",
            "floating_environment": "CUDA arithmetic has no user-observable floating trap/exception-flag side effects; invalid unused values cannot affect valid outputs"}
    if axis_guard is not None:
        source = source.replace("trace_native_phase", f"trace_native_phase_axes_{axis_guard.native_mask}")
        evidence["operator"] = serialize_updates(pruned_operator.updates)
        evidence["profile_bindings"] = bind_profiles(pruned_operator, EqualAxisProfileClass()).evidence()
        evidence["axis_guard_pruning"] = guard_proof
        evidence["scope"] += "; exact captured native axis flags must select this entry before any point call"
    evidence["proof_sha256"] = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    return {"source": source, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "field_ids": {name: i for i, name in enumerate(FIELDS)},
            "local_term_ids": {name: index for name, index in ids.items() if name.startswith("psi_")}, "proof": evidence}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--active-axes", type=int, nargs="*", choices=(0, 1, 2))
    a = p.parse_args()
    result = generate(axis_guard=None if a.active_axes is None else AxisGuardClass(tuple(a.active_axes)))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(result["source"])
    if a.manifest:
        a.manifest.write_text(json.dumps({k: v for k, v in result.items() if k != "source"}, indent=2) + "\n")
    print(json.dumps({"source_sha256": result["source_sha256"], "proof_sha256": result["proof"]["proof_sha256"]}))


if __name__ == "__main__":
    main()
