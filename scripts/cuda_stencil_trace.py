"""Experimental cell-level tracing for CUDA schedule generation.

This is an opt-in research frontend, not a Simulation backend. Trace one cell
per phase; never expand Python objects for the material grid. Spatial offsets
use BeamZ's (z, y, x) convention and versions count complete H/E timesteps.
The explicit multiply/FMA nodes preserve yee_primitives.cuh's rounding points.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True, order=True)
class Read:
    field: str
    version: int
    offset: tuple[int, int, int] = (0, 0, 0)


@dataclass(frozen=True, eq=False)
class Expr:
    op: str
    args: tuple

    def __add__(self, other):
        return Expr("add", (self, expression(other)))

    def __sub__(self, other):
        return Expr("sub", (self, expression(other)))

    def __mul__(self, other):
        return mul_rn(self, other)

    def __radd__(self, other):
        return expression(other) + self

    def __rsub__(self, other):
        return expression(other) - self

    def __rmul__(self, other):
        return mul_rn(other, self)

    def __neg__(self):
        return Expr("neg", (self,))

    def __bool__(self):
        raise TypeError("Data-dependent branching is unsupported in a stencil trace")

    def __eq__(self, other):
        raise TypeError("Data-dependent comparisons are unsupported in a stencil trace")

    def __ne__(self, other):
        raise TypeError("Data-dependent comparisons are unsupported in a stencil trace")


def expression(value):
    if isinstance(value, Expr):
        return value
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except OverflowError as error:
            raise ValueError("Stencil constants must fit finite float32") from error
        if not math.isfinite(number):
            raise ValueError("Stencil constants must be finite")
        if abs(number) > 3.4028234663852886e38:
            raise ValueError("Stencil constants must fit finite float32")
        rounded = struct.unpack("!f", struct.pack("!f", number))[0]
        return Expr("constant", (rounded,))
    raise TypeError(f"Unsupported stencil operand: {type(value).__name__}")


def mul_rn(a, b):
    return Expr("mul_rn", (expression(a), expression(b)))


def fma_rn(a, b, c):
    return Expr("fma_rn", (expression(a), expression(b), expression(c)))


@dataclass(frozen=True)
class Field:
    name: str
    version: int

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("Field name must be an identifier")
        if type(self.version) is not int or self.version < 0:
            raise ValueError("Field version must be a nonnegative integer")

    def __getitem__(self, offset):
        if (
            not isinstance(offset, tuple)
            or len(offset) != 3
            or any(type(i) is not int for i in offset)
        ):
            raise TypeError("Stencil reads require three constant integer offsets")
        return Expr("read", (Read(self.name, self.version, offset),))


@dataclass(frozen=True, order=True)
class CoefficientRead:
    """Read-only FP32 coefficient with explicit varying logical axes.

    Empty axes means scalar, one axis an independent profile, and all three
    axes a dense material. Packed addressing remains a backend lowering choice.
    These reads are not evolving state and have no timestep version.
    """

    name: str
    axes: tuple[int, ...]
    offset: tuple[int, int, int] = (0, 0, 0)
    dtype: str = "float32"


@dataclass(frozen=True)
class Coefficient:
    name: str
    axes: tuple[int, ...] = ()
    dtype: str = "float32"

    def __post_init__(self):
        if (
            not isinstance(self.name, str)
            or not self.name.isidentifier()
            or not isinstance(self.axes, tuple)
            or any(type(axis) is not int for axis in self.axes)
            or self.axes not in ((), (0,), (1,), (2,), (0, 1, 2))
            or self.dtype != "float32"
        ):
            raise ValueError(
                "Coefficients require an identifier, scalar/profile/cell axes, and float32"
            )

    def __getitem__(self, offset):
        if (
            not isinstance(offset, tuple)
            or len(offset) != 3
            or any(type(i) is not int for i in offset)
            or any(offset[a] and a not in self.axes for a in range(3))
        ):
            raise TypeError(
                "Coefficient offsets require three integers along varying axes"
            )
        return Expr(
            "coefficient", (CoefficientRead(self.name, self.axes, offset, self.dtype),)
        )


def parameter(name):
    return Expr("parameter", (name,))


def round_bf16(value):
    """Round storage to BF16 then represent the stored value again as FP32."""
    return Expr("round_bf16", (expression(value),))


@dataclass(frozen=True)
class Update:
    output: Read
    value: Expr

    @property
    def reads(self):
        return reads(self.value)


def reads(value):
    if value.op == "read":
        return frozenset(value.args)
    return frozenset(
        read for arg in value.args if isinstance(arg, Expr) for read in reads(arg)
    )


def coefficient_reads(value):
    if value.op == "coefficient":
        return frozenset(value.args)
    return frozenset(
        read
        for arg in value.args
        if isinstance(arg, Expr)
        for read in coefficient_reads(arg)
    )


def cpml_term(derivative, old_psi, a, b, inv_kappa, sign=1, *, precision="float32"):
    """Canonical CUDA CPML recurrence and signed corrected derivative.

    BF16 rounds the stored recurrence only. The current field correction uses
    unrounded FP32 next_psi, exactly as CorrectCpml in update.cu does.
    """
    if precision not in {"float32", "bfloat16"}:
        raise ValueError("CPML precision must be float32 or bfloat16")
    next_psi = fma_rn(b, old_psi, mul_rn(a, derivative))
    stored_psi = round_bf16(next_psi) if precision == "bfloat16" else next_psi
    corrected = mul_rn(sign, fma_rn(derivative, inv_kappa, next_psi))
    return stored_psi, corrected


def trace_cpml_step(version=0, *, active_axes=(0, 1, 2), precision="float32"):
    """Trace representative homogeneous CPML region with active-axis specialization.

    Boundary/packed-coordinate admission is deliberately not modeled here.
    ``active_axes`` describes a face/edge/corner region already proven by the
    caller. Coefficient profiles are read-only one-axis values. Storage psi
    names include phase and term, so local recurrences cannot alias silently.
    """
    if type(version) is not int or version < 0:
        raise ValueError("version must be a nonnegative integer")
    if precision not in {"float32", "bfloat16"}:
        raise ValueError("CPML precision must be float32 or bfloat16")
    if (
        not isinstance(active_axes, tuple)
        or any(type(a) is not int for a in active_axes)
        or tuple(sorted(set(active_axes))) != active_axes
        or any(a not in (0, 1, 2) for a in active_axes)
    ):
        raise ValueError("active_axes must be a sorted unique tuple of z,y,x axes")
    terms = (((2, 1), (1, 0)), ((0, 0), (2, 2)), ((1, 2), (0, 1)))
    origin, updates = (0, 0, 0), []
    inv = Coefficient("inv_spacing")[origin]
    for phase, family in enumerate("HE"):
        opposite_kind = "E" if phase == 0 else "H"
        opposite = tuple(Field(opposite_kind + c, version + phase) for c in "xyz")
        for c, component in enumerate("xyz"):
            signed_terms = []
            for term, (source_component, axis) in enumerate(terms[c]):
                offset = tuple(
                    (1 if phase == 0 else -1) if a == axis else 0 for a in range(3)
                )
                center, adjacent = (
                    opposite[source_component][origin],
                    opposite[source_component][offset],
                )
                derivative = mul_rn(
                    adjacent - center if phase == 0 else center - adjacent, inv
                )
                sign = 1 if term == 0 else -1
                if axis in active_axes:
                    name = f"psi_{family}{2 * c + term}"
                    psi, corrected = cpml_term(
                        derivative,
                        Field(name, version)[origin],
                        Coefficient("a_" + name, (axis,))[origin],
                        Coefficient("b_" + name, (axis,))[origin],
                        Coefficient("inv_kappa_" + name, (axis,))[origin],
                        sign,
                        precision=precision,
                    )
                    updates.append(Update(Read(name, version + 1), psi))
                else:
                    corrected = mul_rn(sign, derivative)
                signed_terms.append(corrected)
            field = family + component
            old = Field(field, version)[origin]
            source = Coefficient("source_" + field, (0, 1, 2))[origin]
            decay = Coefficient("decay_" + field, (0, 1, 2))[origin]
            updates.append(
                Update(
                    Read(field, version + 1),
                    fma_rn(
                        -source if phase == 0 else source,
                        signed_terms[0] + signed_terms[1],
                        mul_rn(decay, old),
                    ),
                )
            )
    return tuple(updates)


def yee_phase(old, opposite, decay, source, inv_spacing, *, magnetic):
    """Python cell equations corresponding to CUDA's canonical Yee primitives.

    This is deliberately explicit about numerical operations. It does not
    trace arbitrary Python/tensor programs, boundaries, sources or observers.
    """
    terms = (((2, 1), (1, 0)), ((0, 0), (2, 2)), ((1, 2), (0, 1)))
    origin = (0, 0, 0)

    def derivative(component, axis):
        offset = tuple((1 if magnetic else -1) if a == axis else 0 for a in range(3))
        center = opposite[component][origin]
        adjacent = opposite[component][offset]
        difference = adjacent - center if magnetic else center - adjacent
        return mul_rn(difference, inv_spacing)

    result = []
    for c, (a, b) in enumerate(terms):
        curl = derivative(*a) - derivative(*b)
        result.append(
            fma_rn(
                -source[c] if magnetic else source[c],
                curl,
                mul_rn(decay[c], old[c][origin]),
            )
        )
    return tuple(result)


def trace_yee_step(version=0):
    if type(version) is not int or version < 0:
        raise ValueError("version must be a nonnegative integer")
    h = tuple(Field("H" + c, version) for c in "xyz")
    e = tuple(Field("E" + c, version) for c in "xyz")
    inv = parameter("inv_spacing")
    updates = []
    for magnetic in (True, False):
        names = tuple(("H" if magnetic else "E") + c for c in "xyz")
        old = h if magnetic else e
        opposite = e if magnetic else tuple(Field("H" + c, version + 1) for c in "xyz")
        values = yee_phase(
            old,
            opposite,
            tuple(parameter("decay_" + name) for name in names),
            tuple(parameter("source_" + name) for name in names),
            inv,
            magnetic=magnetic,
        )
        updates.extend(
            Update(Read(name, version + 1), value)
            for name, value in zip(names, values, strict=True)
        )
    return tuple(updates)


def dependency_cone(field, depth):
    """Exact initial field samples needed for one output after ``depth`` steps.

    Coefficients are time-invariant parameters in this first frontend. This
    computes dependencies, not a legal parallel schedule or storage allocation.
    """
    if field not in {kind + c for kind in "HE" for c in "xyz"}:
        raise ValueError("Unknown Yee field")
    if type(depth) is not int or depth < 1:
        raise ValueError("depth must be a positive integer")
    definitions = {
        update.output: update.value
        for version in range(depth)
        for update in trace_yee_step(version)
    }

    @lru_cache(None)
    def expand(name, version):
        if version == 0:
            return frozenset((Read(name, 0),))
        return frozenset(
            Read(
                parent.field,
                0,
                tuple(a + b for a, b in zip(parent.offset, read.offset, strict=True)),
            )
            for read in reads(definitions[Read(name, version)])
            for parent in expand(read.field, read.version)
        )

    return expand(field, depth)


def emit_cuda(value, read_name=None, coefficient_name=None, parameter_name=None):
    """Emit a scalar expression; read_name selects shared/register/global loads."""
    if value.op == "read":
        read = value.args[0]
        if read_name:
            return read_name(read)
        z, y, x = read.offset
        return f"load_{read.field}_v{read.version}({z}, {y}, {x})"
    if value.op == "parameter":
        return parameter_name(value.args[0]) if parameter_name else value.args[0]
    if value.op == "coefficient":
        read = value.args[0]
        if coefficient_name:
            return coefficient_name(read)
        if not read.axes:
            return read.name
        z, y, x = read.offset
        return f"coefficient_{read.name}({z}, {y}, {x})"
    if value.op == "constant":
        return f"{value.args[0]!r}f"
    args = [
        emit_cuda(arg, read_name, coefficient_name, parameter_name)
        for arg in value.args
    ]
    if value.op in {"add", "sub"}:
        return f"__f{'add' if value.op == 'add' else 'sub'}_rn({args[0]}, {args[1]})"
    if value.op == "neg":
        return f"(-{args[0]})"
    if value.op == "mul_rn":
        return f"__fmul_rn({args[0]}, {args[1]})"
    if value.op == "fma_rn":
        return f"__fmaf_rn({args[0]}, {args[1]}, {args[2]})"
    if value.op == "round_bf16":
        return f"__bfloat162float(__float2bfloat16_rn({args[0]}))"
    raise ValueError(f"Unsupported operation: {value.op}")


def serialize_updates(updates, *, cse=True):
    """Deterministic typed DAG with optional exact structural commoning.

    CSE never reorders operands or reassociates arithmetic. Read versions,
    affine offsets, coefficient axes and BF16 storage-round nodes are part of
    the key. FP32 constant bits preserve negative zero. This is cell semantics,
    not a spatial schedule or inter-thread synchronization proof.
    """
    # Retain roots when the caller supplies a generator. Otherwise previously
    # visited Expr objects can die between yields and Python can recycle their
    # ids, making identity memoization return an unrelated earlier graph node.
    updates = tuple(updates)
    nodes, seen, common = [], {}, {}
    arities = {"neg": 1, "add": 2, "sub": 2, "mul_rn": 2, "fma_rn": 3, "round_bf16": 1}

    def visit(value):
        identity = id(value)
        if identity in seen:
            return seen[identity]
        node = {"op": value.op}
        if value.op == "read":
            read = value.args[0]
            node.update(field=read.field, version=read.version, offset=read.offset)
            key = value.op, read
        elif value.op == "coefficient":
            read = value.args[0]
            node.update(
                name=read.name, axes=read.axes, offset=read.offset, dtype=read.dtype
            )
            key = value.op, read
        elif value.op in {"constant", "parameter"}:
            node["value"] = value.args[0]
            key = (
                (value.op, struct.pack("!f", value.args[0]))
                if value.op == "constant"
                else (value.op, value.args[0])
            )
        else:
            if value.op not in arities or len(value.args) != arities[value.op]:
                raise ValueError(f"Unsupported operation or arity: {value.op}")
            node["inputs"] = [visit(arg) for arg in value.args]
            key = value.op, *node["inputs"]
        if cse and key in common:
            seen[identity] = common[key]
            return common[key]
        index = len(nodes)
        nodes.append(node)
        seen[identity] = index
        common[key] = index
        return index

    outputs = []
    for update in updates:
        outputs.append(
            {
                "field": update.output.field,
                "version": update.output.version,
                "offset": update.output.offset,
                "value": visit(update.value),
            }
        )
    return {
        "schema": "beamz.cell-dag.v2",
        "arithmetic": "fp32-rn-explicit-fma",
        "nodes": nodes,
        "outputs": outputs,
    }


def cell_liveness(graph):
    """Conservative straight-line scalar lifetimes, not a CUDA register count.

    Include leaves and output stores. Spatial stage rings and thread mapping
    are deliberately separate resource analyses. The estimate materializes
    every graph node; a CUDA compiler may fold constants or reuse registers.
    """
    nodes, outputs = graph["nodes"], graph["outputs"]
    last = list(range(len(nodes)))
    uses = [0] * len(nodes)
    for consumer, node in enumerate(nodes):
        for parent in node.get("inputs", ()):
            if type(parent) is not int or not 0 <= parent < consumer:
                raise ValueError("Cell DAG must be topologically ordered")
            last[parent] = max(last[parent], consumer)
            uses[parent] += 1
    for index, output in enumerate(outputs, start=len(nodes)):
        parent = output["value"]
        if type(parent) is not int or not 0 <= parent < len(nodes):
            raise ValueError("Cell DAG output references an invalid node")
        last[parent] = max(last[parent], index)
        uses[parent] += 1
    live, peak = set(), 0
    for index in range(len(nodes) + len(outputs)):
        live = {value for value in live if last[value] >= index}
        if index < len(nodes):
            live.add(index)
        peak = max(peak, len(live))
    return {
        "last_use": last,
        "use_counts": uses,
        "peak_live_scalars": peak,
        "peak_live_bytes": 4 * peak,
        "scope": "straight-line cell DAG with materialized FP32 values; excludes spatial staging",
    }


def emit_cuda_block(
    updates, *, read_name=None, coefficient_name=None, write_name=None, prefix="cell"
):
    """Emit topological scalar SSA with exact CSE and explicit output stores.

    Output callbacks receive the complete versioned affine destination. A
    caller must keep this block in a region where coefficients and field
    versions are stable. BF16 nodes require cuda_bf16.h in the containing CU.
    """
    if not isinstance(prefix, str) or not prefix.isidentifier():
        raise ValueError("CUDA SSA prefix must be an identifier")
    graph = serialize_updates(updates)
    lines = []
    for index, node in enumerate(graph["nodes"]):
        op = node["op"]
        if op == "read":
            value = Expr(
                op, (Read(node["field"], node["version"], tuple(node["offset"])),)
            )
        elif op == "coefficient":
            value = Expr(
                op,
                (
                    CoefficientRead(
                        node["name"],
                        tuple(node["axes"]),
                        tuple(node["offset"]),
                        node["dtype"],
                    ),
                ),
            )
        elif op in {"constant", "parameter"}:
            value = Expr(op, (node["value"],))
        else:
            value = Expr(
                op, tuple(parameter(f"{prefix}_{parent}") for parent in node["inputs"])
            )
        lines.append(
            f"const float {prefix}_{index} = {emit_cuda(value, read_name, coefficient_name)};"
        )
    for output in graph["outputs"]:
        destination = Read(output["field"], output["version"], tuple(output["offset"]))
        name = (
            write_name(destination)
            if write_name
            else f"out_{destination.field}_v{destination.version}"
        )
        lines.append(f"{name} = {prefix}_{output['value']};")
    return {
        "source": "\n".join(lines),
        "graph": graph,
        "liveness": cell_liveness(graph),
    }


def manifest(depth):
    fields = {}
    for name in (kind + c for kind in "HE" for c in "xyz"):
        cone = dependency_cone(name, depth)
        fields[name] = {
            "initial_sample_count": len(cone),
            "lower": [min(read.offset[a] for read in cone) for a in range(3)],
            "upper": [max(read.offset[a] for read in cone) for a in range(3)],
        }
    return {
        "schema": "beamz.experimental.cell-trace.v1",
        "scope": "uniform isotropic Yee interior; no boundaries, sources or monitors",
        "depth": depth,
        "axis_order": ["z", "y", "x"],
        "fields": fields,
        "cell_graph": serialize_updates(trace_yee_step()),
        "cell_liveness": cell_liveness(serialize_updates(trace_yee_step())),
        "cuda_updates": {u.output.field: emit_cuda(u.value) for u in trace_yee_step()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest(args.depth), indent=2) + "\n")


if __name__ == "__main__":
    main()
