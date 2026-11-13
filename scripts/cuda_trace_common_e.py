"""Verified equality of three admitted runtime electric scalar leaves.

The enclosing UniformElectricRegionClass still has to prove spatial material
uniformity. This additional class equates its three runtime values by FP32 bits;
the class/kernel identity never contains those values or resolution bits.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from scripts.cuda_space_time_ir import IllegalSchedule, Operator, validate_operator
from scripts.cuda_stencil_trace import CoefficientRead, Expr, Update, serialize_updates


@dataclass(frozen=True)
class CommonElectricScalarClass:
    schema: str = "beamz.equal-admitted-runtime-electric-scalars.v1"


def common_electric_scalar_class(values):
    values = np.asarray(values)
    if values.dtype != np.dtype("float32") or values.shape != (3,) or not values.flags.c_contiguous or not np.isfinite(values).all():
        return None
    bits = values.view(np.uint32)
    return CommonElectricScalarClass() if bits[0] == bits[1] == bits[2] else None


def verify_common_e(fact, values):
    if fact != CommonElectricScalarClass() or common_electric_scalar_class(values) != fact:
        raise IllegalSchedule("Current certified runtime E scalars do not satisfy exact equality")


def specialize_common_e(operator, fact):
    if fact != CommonElectricScalarClass():
        raise IllegalSchedule("Unknown common runtime electric class")
    validate_operator(operator)
    members = {f"source_E{c}" for c in "xyz"}
    seen, memo = set(), {}

    def transform(value):
        if id(value) in memo:
            return memo[id(value)]
        if value.op == "coefficient" and value.args[0].name in members:
            read = value.args[0]
            if read.axes or any(read.offset) or read.dtype != "float32":
                raise IllegalSchedule("Spatial source must first receive its independent runtime region certificate")
            seen.add(read.name)
            result = Expr("coefficient", (CoefficientRead("source_Ex", ()),))
        else:
            if value.op == "parameter" and value.args[0] in members:
                raise IllegalSchedule("Common E class requires typed runtime coefficient leaves")
            result = Expr(value.op, tuple(transform(arg) if isinstance(arg, Expr) else arg for arg in value.args))
        memo[id(value)] = result
        return result

    result = Operator(tuple(Update(u.output, transform(u.value)) for u in operator.updates), operator.tensors)
    if seen != members:
        raise IllegalSchedule("Three distinct admitted source leaves required before commoning")
    if [u.reads for u in result.updates] != [u.reads for u in operator.updates]:
        raise IllegalSchedule("Readonly scalar alias changed evolving state reads")
    evidence = {"class": asdict(fact), "representative": "source_Ex", "members": sorted(members),
                "operator": serialize_updates(result.updates),
                "dispatch_requires": "Existing complete-trajectory UniformE region certificate AND exact finite equality of its actual private common_e[3] runtime bits",
                "arithmetic": "Only typed scalar leaf names changed; every node/operand/rounding/store/field dependency remains"}
    evidence["specialization_sha256"] = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    return result, evidence
