"""Verified readonly CPML profile aliases, separate from logical state identity.

Equal same-phase/same-axis profiles can use one runtime pointer. Logical a/b/inv
leaves retain their names so guarded recurrence recognition and mutable ψ
ownership remain unchanged. No coefficient is baked and no arithmetic is folded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from scripts.cuda_native_geometry_ir import derive_geometry, verify_component_shapes
from scripts.cuda_space_time_ir import IllegalSchedule
from scripts.cuda_stencil_trace import coefficient_reads

AXES = (1, 0, 0, 2, 2, 1)
REPRESENTATIVE = (0, 1, 1, 3, 3, 0)
KINDS = ("a", "b", "inv_kappa")


@dataclass(frozen=True)
class EqualAxisProfileClass:
    schema: str = "beamz.cpml12-equal-runtime-phase-axis-profiles.v1"


def profile_input_index(phase, term, kind):
    """Prepared full native Mstep ABI (H base13, E base44)."""
    if phase not in (0, 1) or term not in range(6) or kind not in range(3):
        raise ValueError("Canonical phase/term/profile kind required")
    return 13 + 31 * phase + 3 * term + kind


def equal_axis_profile_class(inputs):
    if len(inputs) < 62:
        return None
    fields = [np.asarray(value) for value in inputs[:6]]
    try:
        minimum = verify_component_shapes(tuple(tuple(value.shape) for value in fields))
    except IllegalSchedule:
        return None
    if min(minimum) < 24 or any(value.dtype != np.dtype("float32") for value in fields):
        return None
    for phase in (0, 1):
        for term, axis in enumerate(AXES):
            expected_shape = tuple(24 if a == axis else 1 for a in range(3))
            for kind in range(3):
                array = np.asarray(inputs[profile_input_index(phase, term, kind)])
                representative = np.asarray(inputs[profile_input_index(phase, REPRESENTATIVE[term], kind)])
                if (array.dtype != np.dtype("float32") or array.shape != expected_shape
                        or not array.flags.c_contiguous or not np.isfinite(array).all()
                        or representative.dtype != array.dtype or representative.shape != array.shape
                        or not representative.flags.c_contiguous):
                    return None
                if not np.array_equal(array.view(np.uint32), representative.view(np.uint32)):
                    return None
    return EqualAxisProfileClass()


def verify_profile_class(fact, inputs):
    if fact != EqualAxisProfileClass() or equal_axis_profile_class(inputs) != fact:
        raise IllegalSchedule("Current packet does not satisfy the equal runtime profile class")


@dataclass(frozen=True)
class ProfileBindings:
    # Tuple avoids a mutable mapping silently changing after evidence was hashed.
    bindings: tuple[tuple[str, int], ...]
    operator_sha256: str
    geometry_sha256: str
    schema: str = EqualAxisProfileClass().schema

    def slot(self, logical_read):
        mapping = dict(self.bindings)
        if logical_read.name not in mapping:
            raise IllegalSchedule("Readonly profile binding is absent")
        slot = mapping[logical_read.name]
        if (slot not in range(12) or logical_read.axes != (AXES[slot % 6],)
                or any(logical_read.offset) or logical_read.dtype != "float32"):
            raise IllegalSchedule("Readonly profile binding changes physical indexing")
        return slot

    def evidence(self):
        result = {"class": {"schema": self.schema}, "bindings": dict(self.bindings),
                  "operator_sha256": self.operator_sha256, "geometry_sha256": self.geometry_sha256,
                  "dispatch_requires": "verify_profile_class(fact, actual_host_inputs), including canonical six-field supports, before dispatch",
                  "unchanged": ["logical coefficient names", "active guards", "mutable psi IDs/strides",
                                "all arithmetic nodes and rounding", "runtime coefficient values"]}
        result["binding_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        return result


def bind_profiles(operator, fact):
    if fact != EqualAxisProfileClass():
        raise IllegalSchedule("Unsupported profile alias class")
    geometry = derive_geometry(operator)
    term_info = {t["term"]: t for t in geometry["cpml_terms"]}
    bindings = {}
    for update in operator.updates:
        for read in coefficient_reads(update.value):
            for kind in KINDS:
                prefix = kind + "_"
                if not read.name.startswith(prefix):
                    continue
                state = read.name[len(prefix):]
                if state not in term_info:
                    raise IllegalSchedule("Profile leaf does not identify a proved canonical recurrence")
                info = term_info[state]
                if read.axes != (info["axis_zyx"],) or any(read.offset) or read.dtype != "float32":
                    raise IllegalSchedule("Profile alias requires an unshifted FP32 axis profile")
                if len(state) != 6 or state[4] not in "HE" or state[5] not in "012345":
                    raise IllegalSchedule("Canonical native local-term namespace required")
                phase, term = info["phase"], int(state[5])
                if AXES[term] != info["axis_zyx"]:
                    raise IllegalSchedule("Profile term axis disagrees with prepared ABI")
                representative = REPRESENTATIVE[term]
                target = f"psi_{state[4]}{representative}"
                if target not in term_info or term_info[target]["axis_zyx"] != info["axis_zyx"]:
                    raise IllegalSchedule("Representative recurrence absent or differently indexed")
                bindings[read.name] = 6 * phase + representative
    return ProfileBindings(tuple(sorted(bindings.items())), geometry["operator_sha256"], geometry["geometry_sha256"])


def verify_bindings(operator, bindings):
    if bindings != bind_profiles(operator, EqualAxisProfileClass()):
        raise IllegalSchedule("Readonly profile bindings do not match the proved current Operator")
