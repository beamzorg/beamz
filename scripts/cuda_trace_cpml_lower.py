"""Recognize canonical traced CPML terms for guarded, term-local lowering.

The field/psi DAG is never algebraically reassociated. Matching proves the old
psi's final uses, permitting its store after recurrence+correction and before
the next independent term. Inactive terms must execute only sign*derivative;
identity-profile FMAs are not an exact substitute for every signed-zero input.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

try:
    from scripts.cuda_space_time_ir import IllegalSchedule
    from scripts.cuda_stencil_trace import Expr
except ModuleNotFoundError:
    from cuda_space_time_ir import IllegalSchedule
    from cuda_stencil_trace import Expr


def key(value):
    if isinstance(value, Expr):
        args = tuple(key(a) for a in value.args)
        if value.op == "constant":
            args = (struct.pack("!f", value.args[0]),)
        return value.op, args
    return value


@dataclass(frozen=True)
class Term:
    derivative: Expr
    sign: float
    corrected: Expr
    state: str | None = None
    a: Expr | None = None
    b: Expr | None = None
    inv_kappa: Expr | None = None
    precision: str = "float32"


def lower_group(field, updates, specs):
    definitions = {u.output.field: u.value for u in updates}
    root = definitions[field]
    if root.op != "fma_rn" or root.args[1].op != "add":
        raise IllegalSchedule("Native CPML field must retain canonical FMA/signed-term addition")
    terms = []
    used_states = set()
    for corrected in root.args[1].args:
        if corrected.op != "mul_rn" or corrected.args[0].op != "constant" or corrected.args[0].args[0] not in (1., -1.):
            raise IllegalSchedule("CPML correction requires an explicit exact +/-1 multiplier")
        sign, body = corrected.args[0].args[0], corrected.args[1]
        if body.op != "fma_rn":
            # Statically inactive axis; raw derivative keeps the signed multiply.
            terms.append(Term(body, sign, corrected))
            continue
        derivative, inv_kappa, next_psi = body.args
        if any(specs[r.field].role != "field" for r in _reads(derivative)):
            raise IllegalSchedule("CPML derivatives may not consume mutable local state")
        if next_psi.op != "fma_rn" or next_psi.args[1].op != "read" or next_psi.args[2].op != "mul_rn":
            raise IllegalSchedule("CPML recurrence must remain fma(b,oldpsi,mul(a,derivative))")
        b, old, product = next_psi.args
        a, same_derivative = product.args
        state = old.args[0].field
        if state not in definitions or specs[state].role != "local-state" or state in used_states:
            raise IllegalSchedule("CPML term requires a unique local recurrence output")
        if old.args[0].version != 0 or any(old.args[0].offset):
            raise IllegalSchedule("CPML recurrence is not old same-point state")
        for coefficient, prefix in ((a, "a_"), (b, "b_"), (inv_kappa, "inv_kappa_")):
            if coefficient.op != "coefficient" or coefficient.args[0].name != prefix + state:
                raise IllegalSchedule("CPML profile does not belong to its recurrence")
            if any(coefficient.args[0].offset):
                raise IllegalSchedule("Native CPML profiles must use the unshifted physical recurrence coordinate")
        if key(derivative) != key(same_derivative):
            raise IllegalSchedule("CPML recurrence/correction derivatives differ")
        stored = definitions[state]
        precision = specs[state].precision
        if precision == "bfloat16":
            if stored.op != "round_bf16":
                raise IllegalSchedule("BF16 recurrence storage must retain its explicit round")
            stored = stored.args[0]
        if key(stored) != key(next_psi):
            raise IllegalSchedule("Stored recurrence and unrounded corrected recurrence differ")
        # No other expression in the group may consume this old state through
        # a noncanonical path. Replacing the whole signed term in the field
        # must eliminate it completely, leaving only the stored recurrence.
        used_states.add(state)
        terms.append(Term(derivative, sign, corrected, state, a, b, inv_kappa, precision))
    expected_states = {u.output.field for u in updates if specs[u.output.field].role == "local-state"}
    if used_states != expected_states:
        raise IllegalSchedule("Unconsumed/local-state output prevents term-local stores")
    substitutions = {key(t.corrected): Expr("parameter", (f"corrected_{i}",)) for i, t in enumerate(terms)}

    def replace(node):
        replacement = substitutions.get(key(node))
        if replacement is not None:
            return replacement
        return Expr(node.op, tuple(replace(a) if isinstance(a, Expr) else a for a in node.args))

    field_value = replace(root)
    if any(r.field in used_states for r in _reads(field_value)):
        raise IllegalSchedule("Field still consumes old psi after its proposed store")
    return tuple(terms), field_value


def _reads(value):
    if value.op == "read":
        return value.args
    return tuple(r for a in value.args if isinstance(a, Expr) for r in _reads(a))
