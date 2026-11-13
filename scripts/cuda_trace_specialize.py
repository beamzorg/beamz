"""Proof-carrying specialization of traced immutable scalar coefficients.

This substitutes exact FP32 constants, without reassociating or folding any
arithmetic. A generated-kernel caller must verify the witnesses against its
actual host packet before dispatch; the ordinary operator remains the fallback.
No evolving field, CPML state, spatial material or source waveform is a fact.
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass

import numpy as np

from scripts.cuda_space_time_ir import Operator
from scripts.cuda_stencil_trace import CoefficientRead, Expr, Update, serialize_updates


@dataclass(frozen=True)
class ScalarFact:
    name: str
    input_index: int
    fp32_bits: int

    def __post_init__(self):
        indices = {f"{kind}_H{c}": base + i
                   for kind, base in (("decay", 6), ("source", 9))
                   for i, c in enumerate("xyz")}
        if indices.get(self.name) != self.input_index:
            raise ValueError("Scalar fact name does not match the prepared ABI input")
        if type(self.fp32_bits) is not int or not 0 <= self.fp32_bits <= 0xffffffff:
            raise ValueError("Scalar fact requires exact uint32 coefficient bits")
        if not np.isfinite(self.value):
            raise ValueError("Nonfinite coefficients cannot be scalar facts")

    @property
    def value(self):
        return struct.unpack("<f", struct.pack("<I", self.fp32_bits))[0]


def scalar_bits(value):
    value = np.asarray(value)
    if value.dtype != np.dtype("float32") or value.shape != ():
        raise ValueError("A scalar fact requires a scalar FP32 host input")
    if not np.isfinite(value):
        raise ValueError("Nonfinite coefficients cannot be scalar facts")
    return int(value.view(np.uint32))


def magnetic_facts(inputs):
    """Prove H decay/source values from the Cartesian prepared native ABI.

    Each of the six coefficients is independently proven. Dense or non-FP32
    coefficients remain dynamic, including mixed scalar/dense packets.
    """
    if len(inputs) < 12:
        raise ValueError("Prepared packet does not contain the H coefficients")
    facts = []
    for kind, base in (("decay", 6), ("source", 9)):
        for c, component in enumerate("xyz"):
            try:
                bits = scalar_bits(inputs[base + c])
            except ValueError:
                continue
            facts.append(ScalarFact(f"{kind}_H{component}", base + c, bits))
    return tuple(facts)


def verify(facts, inputs):
    """Fail closed on stale evidence before selecting a compiled variant."""
    for fact in facts:
        if fact.input_index < 0 or fact.input_index >= len(inputs):
            raise ValueError(f"Scalar fact input is absent: {fact.name}")
        if scalar_bits(inputs[fact.input_index]) != fact.fp32_bits:
            raise ValueError(f"Scalar coefficient changed: {fact.name}")


def specialize(operator, facts):
    """Return a new operator and dispatch evidence, preserving rounding nodes."""
    facts = tuple(facts)
    if len({f.name for f in facts}) != len(facts):
        raise ValueError("Scalar coefficient facts must have unique names")
    constants = {f.name: f for f in facts}
    used = set()

    def transform(node):
        if node.op in {"parameter", "coefficient"}:
            name = node.args[0] if node.op == "parameter" else node.args[0].name
            if name in constants:
                # A scalar witnesses every logical offset; no spatial sampling.
                used.add(name)
                return Expr("constant", (constants[name].value,))
        return Expr(node.op, tuple(transform(a) if isinstance(a, Expr) else a
                                  for a in node.args))

    result = Operator(tuple(Update(u.output, transform(u.value))
                            for u in operator.updates), operator.tensors)
    evidence = {"facts": [asdict(f) for f in facts],
                "substituted": sorted(used),
                "unused": sorted(set(constants) - used),
                "dispatch_requires": "verify(facts, actual_host_inputs) before dispatch",
                "arithmetic": "unchanged nodes and explicit rounding; no algebraic folding",
                "operator": serialize_updates(result.updates)}
    evidence["specialization_sha256"] = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result, evidence


@dataclass(frozen=True)
class MagneticScalarClass:
    """Value-independent kernel class; the common H source stays a parameter."""
    schema: str = "beamz.scalar-lossless-isotropic-H.v1"


def magnetic_scalar_class(inputs):
    """Admit exact decay1 and equal finite scalar sources, preserving sign bits."""
    facts = magnetic_facts(inputs)
    if len(facts) != 6:
        return None
    values = {fact.name: fact.fp32_bits for fact in facts}
    if any(values[f"decay_H{c}"] != 0x3f800000 for c in "xyz"):
        return None
    if any(values[f"source_H{c}"] != values["source_Hx"] for c in "yz"):
        return None
    return MagneticScalarClass()


def verify_scalar_class(fact, inputs):
    if fact != MagneticScalarClass() or magnetic_scalar_class(inputs) != fact:
        raise ValueError("Packet does not satisfy the scalar lossless isotropic H class")


def specialize_scalar_class(operator, fact):
    """Alias runtime scalar sources and bake decay1; no per-resolution JIT.

    Coefficient callbacks must honor axes=(): the source has no spatial
    dependence. The compiler may pass zero coordinates to shorten liveness.
    """
    if fact != MagneticScalarClass():
        raise ValueError("Unsupported scalar coefficient class")
    used = set()

    def transform(node):
        if node.op in {"parameter", "coefficient"}:
            name = node.args[0] if node.op == "parameter" else node.args[0].name
            if name in {f"decay_H{c}" for c in "xyz"}:
                used.add(name)
                return Expr("constant", (1.,))
            if name in {f"source_H{c}" for c in "xyz"}:
                used.add(name)
                return Expr("coefficient", (CoefficientRead("source_Hx", ()),))
        return Expr(node.op, tuple(transform(a) if isinstance(a, Expr) else a
                                  for a in node.args))

    result = Operator(tuple(Update(u.output, transform(u.value))
                            for u in operator.updates), operator.tensors)
    evidence = {"class": asdict(fact), "substituted": sorted(used),
                "runtime_source": {"name": "source_Hx", "input_index": 9, "axes": []},
                "dispatch_requires": "verify_scalar_class(fact, actual_host_inputs)",
                "arithmetic": "unchanged nodes/rounding; runtime common source value",
                "operator": serialize_updates(result.updates)}
    evidence["specialization_sha256"] = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result, evidence
