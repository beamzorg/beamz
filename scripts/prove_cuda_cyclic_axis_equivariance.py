"""Ordered Yee/CPML graph equivariance under proper cyclic coordinate maps.

This is a typed IR/source-remap feasibility proof, not a backend rotation or a
speed result. All RN/FMA operand order, phase/version and BF16-store nodes are
retained. Observation packets must preserve original world reduction order.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA, derive_geometry
from scripts.cuda_space_time_ir import (
    FIELDS,
    IllegalSchedule,
    Operator,
    Tensor,
    validate_operator,
    yee_operator,
)
from scripts.cuda_stencil_trace import (
    CoefficientRead,
    Expr,
    Read,
    Update,
    serialize_updates,
)

TERM_AXIS = {int(t["term"][-1]): t["axis_zyx"] for t in derive_geometry()["cpml_terms"]}


@dataclass(frozen=True)
class AxisMap:
    # Old XYZ component/positive-coordinate axis -> new XYZ axis.
    xyz: tuple[int, int, int]

    def __post_init__(self):
        if (not isinstance(self.xyz, tuple) or len(self.xyz) != 3
                or any(type(a) is not int for a in self.xyz) or sorted(self.xyz) != [0, 1, 2]):
            raise IllegalSchedule("A bijective integer XYZ axis map is required")

    @property
    def odd(self):
        return sum(self.xyz[i] > self.xyz[j] for i in range(3) for j in range(i + 1, 3)) % 2 == 1

    @property
    def zyx(self):
        return tuple(2 - self.xyz[2 - a] for a in range(3))

    def vector(self, old):
        if len(old) != 3:
            raise IllegalSchedule("Three ZYX coordinates required")
        out = [None] * 3
        for a, target in enumerate(self.zyx):
            out[target] = old[a]
        return tuple(out)

    def field(self, name):
        if name not in FIELDS:
            raise IllegalSchedule("Canonical H/E field component required")
        return name[0] + "xyz"[self.xyz["xyz".index(name[1])]]

    def term(self, term):
        if type(term) is not int or term not in range(6):
            raise IllegalSchedule("Canonical CPML term0..5 required")
        new_component = self.xyz[term // 2]
        new_axis = self.zyx[TERM_AXIS[term]]
        return next(t for t in range(6) if t // 2 == new_component and TERM_AXIS[t] == new_axis)

    def name(self, name):
        if name in FIELDS:
            return self.field(name)
        if name == "inv_spacing":
            return name
        match = re.fullmatch(r"psi_([HE])([0-5])", name)
        if match:
            return f"psi_{match[1]}{self.term(int(match[2]))}"
        match = re.fullmatch(r"(decay|source)_([HE][xyz])", name)
        if match:
            return match[1] + "_" + self.field(match[2])
        match = re.fullmatch(r"(a|b|inv_kappa)_(psi_[HE][0-5])", name)
        if match:
            return match[1] + "_" + self.name(match[2])
        raise IllegalSchedule(f"Unknown semantic axis label cannot be silently rotated: {name}")

    def source_group(self, group):
        if type(group) is not int or group not in range(9):
            raise IllegalSchedule("Canonical preE/postH/postE group required")
        return 3 * (group // 3) + self.xyz[group % 3]

    def presence(self, mask):
        if type(mask) is not int or mask not in range(64):
            raise IllegalSchedule("Six-component post-source mask required")
        return sum(1 << FIELDS.index(self.field(f)) for i, f in enumerate(FIELDS) if mask & (1 << i))

    def inverse(self):
        return AxisMap(tuple(self.xyz.index(a) for a in range(3)))


@dataclass(frozen=True)
class SourceBatch:
    """Ordered source IDs in one preE/postH/postE component group."""

    group: int
    starts: tuple[tuple[int, int, int], ...]
    slab_zyx: tuple[int, int, int]

    def __post_init__(self):
        if (type(self.group) is not int or self.group not in range(9)
                or not isinstance(self.starts, tuple) or not isinstance(self.slab_zyx, tuple) or len(self.slab_zyx) != 3
                or any(type(n) is not int or n < 1 for n in self.slab_zyx)
                or any(not isinstance(p, tuple) or len(p) != 3 or any(type(x) is not int for x in p) for p in self.starts)):
            raise IllegalSchedule("Typed component/source-ID/spatial slab batch required")

    def rotated(self, rotation):
        if not isinstance(rotation, AxisMap) or rotation.odd:
            raise IllegalSchedule("Sources require the same admitted proper cyclic map")
        return SourceBatch(rotation.source_group(self.group), tuple(rotation.vector(p) for p in self.starts), rotation.vector(self.slab_zyx))

    def accesses(self, point, current_step, macro_step, version, ntime):
        if type(ntime) is not int or ntime < 1:
            raise IllegalSchedule("Positive waveform row length required")
        time = max(0, min(current_step + macro_step + version - 1, ntime - 1))
        result = []
        for source, start in enumerate(self.starts):
            offset = tuple(x - a for x, a in zip(point, start, strict=True))
            if all(0 <= x < n for x, n in zip(offset, self.slab_zyx, strict=True)):
                result.append((source, offset, time))
        return result


def cyclic(steps):
    if type(steps) is not int or steps not in range(3):
        raise IllegalSchedule("Cyclic rotation step0/1/2 required")
    return AxisMap(tuple((c + steps) % 3 for c in range(3)))


def _relabel(operator, rotation, negative=None):
    """Unchecked parity exists only to produce the odd-map counterexample."""
    memo = {}

    def name(n):
        if negative == "unmapped-profile" and n.startswith(("a_psi", "b_psi", "inv_kappa_psi")):
            return n
        if negative == "unmapped-state" and n.startswith("psi_"):
            return n
        return rotation.name(n)

    def value(v):
        if id(v) in memo:
            return memo[id(v)]
        if v.op == "read":
            r = v.args[0]
            args = (Read(name(r.field), r.version, rotation.vector(r.offset)),)
        elif v.op == "coefficient":
            r = v.args[0]
            args = (CoefficientRead(name(r.name), tuple(sorted(rotation.zyx[a] for a in r.axes)), rotation.vector(r.offset), r.dtype),)
        elif v.op == "parameter":
            args = (name(v.args[0]),)
        elif v.op == "constant":
            args = v.args  # Preserve exact FP32 constants/sign, no evaluation.
        elif v.op in {"add", "sub", "neg", "mul_rn", "fma_rn", "round_bf16"}:
            args = tuple(value(a) for a in v.args)
            if negative == "reversed-subtraction" and v.op == "sub":
                args = args[::-1]
        else:
            raise IllegalSchedule(f"Unsupported ordered arithmetic node: {v.op}")
        out = Expr(v.op, args)
        memo[id(v)] = out
        return out

    updates = tuple(Update(Read(name(u.output.field), u.output.version, rotation.vector(u.output.offset)), value(u.value)) for u in operator.updates)
    tensors = tuple(sorted(Tensor(name(t.name), t.phase, t.role, t.precision) for t in operator.tensors))
    return Operator(updates, tensors)


def rotate_operator(operator, rotation):
    if not isinstance(rotation, AxisMap) or rotation.odd:
        raise IllegalSchedule("Only orientation-preserving cyclic permutations are admitted")
    validate_operator(operator)
    result = _relabel(operator, rotation)
    validate_operator(result)
    return result


def signature(operator):
    ordered = tuple(sorted(operator.updates, key=lambda u: u.output.field))
    return {"tensors": [(t.name, t.phase, t.role, t.precision) for t in sorted(operator.tensors)],
            "ordered_graph": serialize_updates(ordered)}


def supports(rotation, minimum=(35, 49, 65)):
    rotated_minimum = rotation.vector(minimum)
    for f in FIELDS:
        old = tuple(n + d for n, d in zip(minimum, SUPPORT_DELTA[FIELDS.index(f)], strict=True))
        new = tuple(n + d for n, d in zip(rotated_minimum, SUPPORT_DELTA[FIELDS.index(rotation.field(f))], strict=True))
        assert rotation.vector(old) == new
    checks = 0
    for phase in (0, 1):
        for term in range(6):
            axis, target_axis = TERM_AXIS[term], TERM_AXIS[rotation.term(term)]
            assert rotation.zyx[axis] == target_axis
            for coordinate in range(-1, minimum[axis] + phase + 2):
                before = coordinate < 12 or coordinate >= minimum[axis] + phase - 12
                after = coordinate < 12 or coordinate >= rotated_minimum[target_axis] + phase - 12
                packed_before = coordinate if coordinate < 12 else coordinate + 24 - minimum[axis] - phase
                packed_after = coordinate if coordinate < 12 else coordinate + 24 - rotated_minimum[target_axis] - phase
                assert before == after and packed_before == packed_after
                checks += 1
    return checks


def f32(value):
    return struct.unpack("!f", struct.pack("!f", value))[0]


def dft_reorder_witness():
    rotation = cyclic(2)  # World normalX becomes internal normalZ.
    points = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]
    internal = [rotation.vector(p) for p in points]
    order = sorted(range(4), key=lambda i: internal[i])
    values = [f32(4e8), f32(-4e8), f32(13.0), 0.0]

    def ordered_sum(indices):
        result = 0.0
        for i in indices:
            # Four normalized bilinear weights1/4 are valid interpolation.
            # Their products are exact powers-of-two scaling. This example
            # needs one rounded addition, not a general Python FMA oracle.
            result = f32(result + f32(values[i] * 0.25))
        return result

    before, after = ordered_sum(range(4)), ordered_sum(order)
    assert order == [0, 2, 1, 3] and before == 3.25 and after == 0.0
    return {"world_points": points, "internal_points": internal,
            "world_raw_order": [0, 1, 2, 3], "sorted_internal_order": order,
            "weights": [0.25] * 4, "values": values,
            "original_fma_gather_result": before, "reordered_fma_gather_result": after}


def source_remaps():
    batches = checks = active = 0
    for group in range(9):
        for count in range(4):
            before = SourceBatch(group, tuple((s - 1, 2 - s, -2 + s) for s in range(count)), (3, 4, 2))
            points = list(itertools.product(range(-3, 7, 2), repeat=3)) + list(before.starts)
            for steps in range(3):
                rotation = cyclic(steps)
                after = before.rotated(rotation)
                assert after.group // 3 == before.group // 3
                assert len(after.starts) == len(before.starts)
                for point in points:
                    for current, macro, version, ntime in ((-3, 0, 1, 7), (2, 3, 4, 11), (8, 16, 16, 17)):
                        expected = before.accesses(point, current, macro, version, ntime)
                        actual = after.accesses(rotation.vector(point), current, macro, version, ntime)
                        unrotate = [(s, rotation.inverse().vector(offset), time) for s, offset, time in actual]
                        assert unrotate == expected  # Ordered source IDs and time retained.
                        checks += 1
                        active += len(actual)
                batches += 1
    return {"typed_batches": batches, "point_clock_checks": checks, "active_source_accesses": active,
            "coefficient_rule": "Transpose each source's spatial tensor exactly; preserve source-ID and waveform/source-time row order"}


def report():
    cases = []
    for precision in ("float32", "bfloat16"):
        for count in range(4):
            for axes in itertools.combinations(range(3), count):
                before = yee_operator(cpml_axes=axes, precision=precision)
                for steps in range(3):
                    rotation = cyclic(steps)
                    actual = rotate_operator(before, rotation)
                    expected_axes = tuple(sorted(rotation.zyx[a] for a in axes))
                    expected = yee_operator(cpml_axes=expected_axes, precision=precision)
                    assert signature(actual) == signature(expected)
                    cases.append({"precision": precision, "old_active_axes": axes, "steps_xyz": steps,
                                  "new_active_axes": expected_axes, "outputs": len(actual.updates),
                                  "ordered_graph_sha256": hashlib.sha256(json.dumps(signature(actual), sort_keys=True).encode()).hexdigest()})
    for steps in range(3):
        op = yee_operator()
        assert signature(rotate_operator(op, cyclic(steps))) == signature(op)
    before = yee_operator(cpml_axes=(0, 1, 2))
    negatives = {}
    odd = AxisMap((1, 0, 2))
    assert signature(_relabel(before, odd)) != signature(before)
    negatives["signless-odd-map"] = "Actual field curl operand/sign order differs; public pass rejects this parity"
    for name in ("unmapped-profile", "unmapped-state", "reversed-subtraction"):
        assert signature(_relabel(before, cyclic(1), name)) != signature(before)
        negatives[name] = "Actual ordered graph differs from independently traced canonical equations"
    return {"scope": __doc__, "graph_cases": cases, "ordered_cpml_graph_cases": len(cases),
            "plain_yee_graph_cases": 3, "negative_controls": negatives,
            "support_and_cpml_coordinate_checks": sum(supports(cyclic(s)) for s in range(3)),
            "worldX_to_internalZ": {"steps_xyz": 2, "old_to_new_zyx": cyclic(2).zyx,
                                    "transpose_old_axes_for_new_storage": (2, 0, 1),
                                    "source_fields": {"Hz": cyclic(2).field("Hz"), "Ey": cyclic(2).field("Ey")},
                                    "source_presence_mask": [20, cyclic(2).presence(20)],
                                    "source_groups": {g: cyclic(2).source_group(g) for g in (5, 7)}},
            "dft_reduction_order_counterexample": dft_reorder_witness(),
            "source_batch_address_proof": source_remaps(),
            "limits": "Cell graph isomorphism is proved; world tensor/material/psi/source/observation data remaps, aliases, boundary flags, storage schedules and actual backend admission remain engineering obligations. No GPU or performance result."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = report()
    r["proof_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps({"CPML_cases": r["ordered_cpml_graph_cases"], "plain_cases": 3,
                      "coordinate_checks": r["support_and_cpml_coordinate_checks"], "negative_controls": len(r["negative_controls"])}))


if __name__ == "__main__":
    main()
