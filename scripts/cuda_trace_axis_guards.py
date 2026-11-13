"""Prune CPML recurrences using an admitted immutable native tile-axis guard.

An absent axis means LocalActive is false throughout the owned trajectory, not
that its profile values equal an identity. Inactive correction remains exactly
RN(sign * derivative), including signed zero. Field arithmetic is not folded.
Native planner correctness and the selected tile's class remain caller duties.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from scripts.cuda_native_geometry_ir import derive_geometry
from scripts.cuda_space_time_codegen import mutable_groups
from scripts.cuda_space_time_ir import (
    FIELDS,
    IllegalSchedule,
    Operator,
    validate_operator,
)
from scripts.cuda_stencil_trace import Expr, Update, serialize_updates
from scripts.cuda_trace_cpml_lower import key, lower_group


@dataclass(frozen=True)
class AxisGuardClass:
    axes: tuple[int, ...]

    def __post_init__(self):
        if self.axes != tuple(sorted(set(self.axes))) or any(type(a) is not int or a not in (0, 1, 2) for a in self.axes):
            raise IllegalSchedule("Sorted unique ZYX axis guard required")

    @property
    def native_mask(self):
        return sum(2 << axis for axis in self.axes)

    def verify_flags(self, flags):
        if type(flags) is not int or flags not in range(256) or flags & 14 != self.native_mask:
            raise IllegalSchedule("Actual immutable tile-axis flags do not match the graph class")


def guard_from_native_flags(flags):
    if type(flags) is not int or flags not in range(256):
        raise IllegalSchedule("One immutable native uint8 tile flag required")
    return AxisGuardClass(tuple(a for a in range(3) if flags & (2 << a)))


def prune_inactive_axes(operator, fact):
    """Drop unreachable local writes/reads; preserve every surviving FP node."""
    if not isinstance(fact, AxisGuardClass):
        raise IllegalSchedule("Explicit axis guard class required")
    validate_operator(operator)
    geometry = derive_geometry(operator)
    term_axes = {term["term"]: term["axis_zyx"] for term in geometry["cpml_terms"]}
    specs, removed, replacements = operator.specs, set(), {}
    witnesses = []
    for phase in (0, 1):
        groups = mutable_groups(operator, phase)
        if {field for field, _ in groups} != set(FIELDS[3 * phase:3 * phase + 3]):
            raise IllegalSchedule("Exactly three canonical fields per phase required")
        for field, group in groups:
            terms, _ = lower_group(field, group, specs)
            if len(terms) != 2 or any(t.state is None for t in terms):
                raise IllegalSchedule("The input must retain the complete six-term guarded CPML graph")
            for ordinal, term in enumerate(terms):
                axes = term.a.args[0].axes
                if (axes != (term_axes[term.state],) or term.b.args[0].axes != axes
                        or term.inv_kappa.args[0].axes != axes):
                    raise IllegalSchedule("Recurrence profiles must share one physical axis")
                axis = axes[0]
                if axis in fact.axes:
                    continue
                raw = Expr("mul_rn", (Expr("constant", (term.sign,)), term.derivative))
                replacements[key(term.corrected)] = raw
                removed.add(term.state)
                witnesses.append({"field": field, "term": ordinal, "state": term.state, "axis": axis,
                                  "reason": "immutable native axis bit absent => LocalActive false at every valid owned point",
                                  "inactive_expression": repr(key(raw))})
    if len({u.output.field for u in operator.updates if specs[u.output.field].role == "local-state"}) != 12:
        raise IllegalSchedule("Exactly twelve canonical local recurrence definitions required")

    def rewrite(value):
        if key(value) in replacements:
            return replacements[key(value)]
        return Expr(value.op, tuple(rewrite(a) if isinstance(a, Expr) else a for a in value.args))

    updates = tuple(Update(u.output, rewrite(u.value)) for u in operator.updates if u.output.field not in removed)
    result = Operator(updates, tuple(t for t in operator.tensors if t.name not in removed))
    validate_operator(result)
    before = {u.output.field: u for u in operator.updates if u.output.field in FIELDS}
    after = {u.output.field: u for u in result.updates if u.output.field in FIELDS}
    for field in FIELDS:
        if {r for r in before[field].reads if r.field in FIELDS} != {r for r in after[field].reads if r.field in FIELDS}:
            raise IllegalSchedule("Guard pruning changed an evolving field dependency")
    evidence = {"schema": "beamz.cpml-immutable-axis-guard-pruning.v1", "axes_zyx": fact.axes,
                "geometry_sha256": geometry["geometry_sha256"],
                "native_flag_mask": fact.native_mask, "removed_local_states": sorted(removed),
                "witnesses": witnesses, "before": serialize_updates(operator.updates), "after": serialize_updates(updates),
                "admission": "Native planner flags must conservatively cover every actual H/E CPML activation over this owned trajectory; verify_flags(actual captured tile flags) before choosing this graph",
                "unchanged": ["all evolving field/version/offset reads", "ordered signed derivative addition", "surviving recurrence/FMA operand order", "inactive RN signed derivative multiply", "runtime coefficient values"]}
    evidence["proof_sha256"] = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    return result, evidence
