"""Affine schedule/storage compiler for traced Yee cells (experimental).

Cell arithmetic stays in cuda_stencil_trace. This module proves an owned moving
box order, chooses register/warp/shared access routes, and derives the fixed-Z
frontier retained by a u_z-plane streaming schedule. It never expands a domain.
All register/resource numbers are source-level requirements, not ptxas results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from scripts.cuda_stencil_trace import (
        Read,
        Update,
        coefficient_reads,
        serialize_updates,
        trace_cpml_step,
        trace_yee_step,
    )
except ModuleNotFoundError:
    from cuda_stencil_trace import (
        Read,
        Update,
        coefficient_reads,
        serialize_updates,
        trace_cpml_step,
        trace_yee_step,
    )


FIELDS = tuple(f + c for f in "HE" for c in "xyz")


class IllegalSchedule(ValueError):
    """An explicitly checked dependence/storage admission failed."""


@dataclass(frozen=True, order=True)
class Tensor:
    name: str
    phase: int
    role: str = "field"
    precision: str = "float32"


@dataclass(frozen=True)
class Operator:
    updates: tuple[Update, ...]
    tensors: tuple[Tensor, ...]

    @property
    def specs(self):
        return {tensor.name: tensor for tensor in self.tensors}


@dataclass(frozen=True)
class StreamShape:
    tile_zyx: tuple[int, int, int] = (64, 13, 29)
    depth: int = 4
    threads: int = 512
    physical_row: int = 32

    def validate(self):
        if len(self.tile_zyx) != 3 or any(type(n) is not int or n < 1 for n in self.tile_zyx):
            raise IllegalSchedule("Tile requires three positive integer dimensions")
        if type(self.depth) is not int or self.depth < 1:
            raise IllegalSchedule("Depth must be a positive integer")
        if type(self.physical_row) is not int or type(self.threads) is not int or self.physical_row != 32 or self.threads < 32 or self.threads % 32:
            raise IllegalSchedule("The warp-X lowering requires complete width32 warps")
        _, y, x = self.tile_zyx
        if x + self.depth - 1 > self.physical_row:
            raise IllegalSchedule("Owned stage union exceeds the fixed physical X row")
        if y + self.depth - 1 > self.threads // self.physical_row:
            raise IllegalSchedule("Owned stage union exceeds the physical Y thread rows")
        if self.threads > 1024:
            raise IllegalSchedule("CUDA block thread limit exceeded")


@dataclass(frozen=True, order=True)
class Access:
    output: str
    stage: int
    phase: int
    source: str
    version: int
    offset: tuple[int, int, int]
    transformed_delta: tuple[int, int, int]
    plane: str
    route: str
    holder: str
    external_fallback: bool
    full_warp: bool = False


@dataclass(frozen=True, order=True)
class Frontier:
    tensor: str
    version: int
    producer_phase: int
    consumers: tuple[tuple[int, int], ...]
    storage: str = "same-thread-register"


@dataclass(frozen=True)
class Barrier:
    stage: int
    label: str
    retires: tuple[str, ...]
    publishes: tuple[str, ...]


def yee_operator(*, cpml_axes=None, precision="float32"):
    if cpml_axes is None:
        updates = trace_yee_step()
    else:
        updates = trace_cpml_step(active_axes=tuple(cpml_axes), precision=precision)
    tensors = []
    for update in updates:
        name = update.output.field
        if name in FIELDS:
            tensors.append(Tensor(name, int(name.startswith("E"))))
        else:
            # Frontend supplies this semantic role; the compiler verifies it.
            family = name.removeprefix("psi_")[0]
            tensors.append(Tensor(name, int(family == "E"), "local-state", precision))
    return Operator(tuple(updates), tuple(sorted(tensors)))


def validate_operator(operator, skew=(1, 1, 1)):
    """Validate affine ownership/time and local-state recurrence semantics."""
    if len(skew) != 3 or any(type(s) is not int for s in skew):
        raise IllegalSchedule("Integer 3D ownership skew required")
    specs = operator.specs
    if len(specs) != len(operator.tensors):
        raise IllegalSchedule("Tensor declarations must be unique")
    definitions = {u.output.field: u for u in operator.updates}
    if len(definitions) != len(operator.updates) or set(definitions) != set(specs):
        raise IllegalSchedule("Exactly one definition per declared tensor required")
    coefficients = {}
    dependencies = []
    for update in operator.updates:
        dst = update.output
        target = specs[dst.field]
        if target.phase not in (0, 1) or target.role not in ("field", "local-state"):
            raise IllegalSchedule("Only two-phase field/local-state tensors admitted")
        if target.precision not in ("float32", "bfloat16") or (target.role == "field" and target.precision != "float32"):
            raise IllegalSchedule("Field storage is FP32; local state may be FP32/BF16")
        if dst.version != 1 or any(dst.offset):
            raise IllegalSchedule("Canonical operator outputs must be version1 at origin")
        for read in sorted(update.reads):
            if read.field not in specs:
                raise IllegalSchedule(f"Undeclared evolving state: {read.field}")
            source = specs[read.field]
            dt = read.version - dst.version
            if dt > 0 or dt < -1:
                raise IllegalSchedule("Streaming frontend admits only current/previous versions")
            if not dt and source.phase >= target.phase:
                raise IllegalSchedule("Current-version producer does not precede consumer phase")
            delta = tuple(d + dt * s for d, s in zip(read.offset, skew, strict=True))
            if any(d > 0 for d in delta):
                raise IllegalSchedule(f"Dependence points to a future owner: {dst.field}<-{read}")
            if delta[0] not in (0, -1):
                raise IllegalSchedule("Fixed-plane schedule needs more than one previous Z frontier")
            if source.role == "local-state" and (any(read.offset) or dt != -1 or source.phase != target.phase):
                raise IllegalSchedule("Local-state requires same-point previous-version same-phase reads")
            dependencies.append({"output": dst.field, "source": read.field,
                                 "version_delta": dt, "offset": read.offset,
                                 "transformed_delta": delta})
        if target.role == "local-state" and not any(r.field == dst.field for r in update.reads):
            raise IllegalSchedule("Local-state update must carry its previous recurrence")
        for read in coefficient_reads(update.value):
            key = (read.axes, read.dtype)
            if read.name in coefficients and coefficients[read.name] != key:
                raise IllegalSchedule("Conflicting readonly coefficient declaration")
            if read.dtype != "float32" or read.axes not in ((), (0,), (1,), (2,), (0, 1, 2)) or any(read.offset[a] and a not in read.axes for a in range(3)):
                raise IllegalSchedule("Unsupported readonly coefficient type/affinity")
            coefficients[read.name] = key
    return {"skew": skew, "dependences": dependencies,
            "coefficients": [{"name": n, "axes": a, "dtype": d,
                              "scope": "scalar" if not a else "profile" if len(a) == 1 else "material",
                              "placement": "phase-local-rematerialization",
                              "stationary": True}
                             for n, (a, d) in sorted(coefficients.items())],
            "local_state": [{"name": t.name, "placement": "mutable-global-in-place",
                              "requires": "unique physical recurrence; ordinary loads; all SSA before stores"}
                             for t in operator.tensors if t.role == "local-state"]}


def validate_stream_signature(operator):
    """The current lowering admits axial Yee reads, not arbitrary custom physics."""
    specs = operator.specs
    if {t.name for t in operator.tensors if t.role == "field"} != set(FIELDS):
        raise IllegalSchedule("Streamed Yee lowering requires the six declared field components")
    for update in operator.updates:
        target = specs[update.output.field]
        for read in update.reads:
            source = specs[read.field]
            if source.role != "field":
                continue
            if source.phase == target.phase:
                # Psi equations have no own-family field read in the canonical trace.
                if read.field != update.output.field or read.version != 0 or any(read.offset):
                    raise IllegalSchedule("Same-family lowering admits only the old own component")
            else:
                direction = 1 if target.phase == 0 else -1
                if read.version != target.phase or sum(bool(d) for d in read.offset) > 1 or any(d not in (0, direction) for d in read.offset):
                    raise IllegalSchedule("Opposite reads must use the canonical axial Yee phase/version")


def _holder(read, stage, phase, *, materialized_initial=False):
    """Named old/current register value available at the consumer phase."""
    c = read.field[-1]
    if read.version == 0 and not materialized_initial:
        return "initial-global"
    if read.field.startswith("H"):
        if phase == 0:
            return f"old_h_previous_{c}"
        return f"old_h_current_{c}" if read.offset[0] else f"current_h_{c}"
    if read.offset[0]:
        return f"current_e_previous_{c}"
    return f"old_e_previous_{c}"


def materialize_providers(accesses, shape):
    """Union inverse-stencil provider footprints in the physical thread plane.

    This is O(depth*plane), not a full-domain expansion. Consumers outside the
    owned box are never computed: their needed values are boundary loads only.
    Separate first/interior u_z counts expose the imported previous-Z frontier.
    """
    grouped = {}
    by, bx = shape.tile_zyx[1:]
    rows = shape.threads // shape.physical_row
    for read in accesses:
        if read.route == "local-recurrence-global":
            continue
        key = read.stage, read.source, read.version, read.offset[0]
        item = grouped.setdefault(key, {"holder": read.holder, "phases": set(),
                                      "offsets": set(), "outputs": set()})
        item["phases"].add(read.phase)
        item["offsets"].add(read.offset[1:])
        item["outputs"].add(read.output)
    providers = []
    for (stage, field, version, dz), item in sorted(grouped.items()):
        lo = shape.depth - stage
        footprint = {(lo + y + dy, lo + x + dx)
                     for dy, dx in item["offsets"]
                     for y in range(by) for x in range(bx)}
        in_union = {(y, x) for y, x in footprint if 0 <= y < rows and 0 <= x < shape.physical_row}
        transformed_y = [y - shape.depth + version for y, _ in footprint]
        transformed_x = [x - shape.depth + version for _, x in footprint]
        delta_z = dz + version - stage
        if (min(transformed_y) < -1 or max(transformed_y) >= by or
                min(transformed_x) < -1 or max(transformed_x) >= bx or delta_z not in (-1, 0)):
            raise IllegalSchedule("Provider materialization exceeds the one-cell lower-neighbor ownership envelope")
        external = {}
        for plane_name, iz in (("first", 0), ("interior", 1)):
            external[plane_name] = sum(version == 0 or
                not (0 <= iz + dz + version - stage < shape.tile_zyx[0] and
                     0 <= y - shape.depth + version < by and
                     0 <= x - shape.depth + version < bx) for y, x in footprint)
        providers.append({"stage": stage, "source": field, "version": version,
                          "physical_z_offset": dz, "holder": item["holder"],
                          "transformed_xy_bounds": [[min(transformed_y), max(transformed_y)],
                                                    [min(transformed_x), max(transformed_x)]],
                          "transformed_z_delta": delta_z, "one_cell_lower_neighbor_admitted": True,
                          "consumer_phases": sorted(item["phases"]),
                          "xy_offsets": sorted(item["offsets"]),
                          "consumer_outputs": sorted(item["outputs"]),
                          "provider_points": len(footprint),
                          "thread_union_points": len(in_union),
                          "extra_apron_points": len(footprint - in_union),
                          "footprint_rows": [[y, min(x for yy, x in footprint if yy == y),
                                                max(x for yy, x in footprint if yy == y)]
                                               for y in sorted({yy for yy, _ in footprint})],
                          "external_unique_scalar_fetches": external,
                          "placement": "materialize once at provider thread; apron callback outside thread union"})
    # Raw read-edge traffic is intentionally labeled: cell CSE may reduce it.
    operands = {name: 0 for name in ("first", "interior")}
    for read in accesses:
        if read.route == "local-recurrence-global":
            continue
        lo = shape.depth - read.stage
        dz, dy, dx = read.offset
        for name, iz in (("first", 0), ("interior", 1)):
            operands[name] += sum(read.version == 0 or
                not (0 <= iz + dz + read.version - read.stage < shape.tile_zyx[0] and
                     0 <= lo + y + dy - shape.depth + read.version < by and
                     0 <= lo + x + dx - shape.depth + read.version < bx)
                for y in range(by) for x in range(bx))
    return {"providers": providers,
            "counts": {"raw_external_read_edges": operands,
                       "unique_external_scalar_fetches": {name: sum(p["external_unique_scalar_fetches"][name] for p in providers)
                                                          for name in operands},
                       "extra_apron_scalar_points": sum(p["extra_apron_points"] for p in providers)},
            "scope": "unclipped physical-domain source points; requested scalar payload, not DRAM; raw edges exclude cell CSE"}


def compile_stream(operator, shape=None, *, warp_x=True, materialized=False):
    """Lower dependence classes to a two-exchange-frame physical-XY schedule.

    The local route is legal ONLY when the source-owner predicate holds.
    Every external source uses the immutable-initial or staged-global callback;
    no halo updates are duplicated and no negative E apron is silently assumed.
    """
    shape = StreamShape() if shape is None else shape
    shape.validate()
    legality = validate_operator(operator)
    validate_stream_signature(operator)
    specs = operator.specs
    access, frontier = [], {}
    shared_components = {"H": set(), "E": set()}
    barriers = []
    for n in range(1, shape.depth + 1):
        for update in operator.updates:
            target = specs[update.output.field]
            for base in sorted(update.reads):
                read = Read(base.field, base.version + n - 1, base.offset)
                source = specs[read.field]
                delta = tuple(d + read.version - n for d in read.offset)
                if source.role == "local-state":
                    route, plane, holder = "local-recurrence-global", "same-physical-point", "mutable-psi"
                elif read.version == 0 and not materialized:
                    route, plane, holder = "initial-global", "initial", "immutable-initial"
                else:
                    plane = "previous" if delta[0] == -1 else "current"
                    holder = _holder(read, n, target.phase, materialized_initial=materialized)
                    if read.offset[1] or (read.offset[2] and not warp_x):
                        route = "shared-exchange"
                        shared_components[read.field[0]].add("xyz".index(read.field[-1]))
                    elif read.offset[2]:
                        route = "warp-shuffle"
                    else:
                        route = "same-thread-register"
                    if delta[0] == -1 and read.version > 0:
                        key = read.field, read.version
                        frontier.setdefault(key, set()).add((n, target.phase))
                access.append(Access(update.output.field, n, target.phase,
                                     read.field, read.version, read.offset, delta,
                                     plane, route, holder, source.role == "field",
                                     route == "warp-shuffle"))
        barriers.extend((
            Barrier(n, "A", ("previous E phase H-frame readers",), ("previous-plane E exchange",)),
            Barrier(n, "B", ("current H phase E-frame readers",), ("current-plane H exchange",)),
        ))
    required = tuple(Frontier(f, v, specs[f].phase, tuple(sorted(consumers)))
                     for (f, v), consumers in sorted(frontier.items(), key=lambda p: (p[0][1], p[0][0])))
    # Local psi is not included in the field frontier: its local recurrence
    # stays in the single ordinary global bank by monotone physical ownership.
    frame_components = {family: sorted(c) for family, c in shared_components.items()}
    plane_words = shape.threads
    frame_words = {f: len(c) * plane_words for f, c in frame_components.items()}
    graph = serialize_updates(operator.updates)
    semantic_hash = hashlib.sha256(json.dumps(graph, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    plan = {
        "schema": "beamz.space-time-stream.v1", "semantic_sha256": semantic_hash,
        "arithmetic": graph["arithmetic"], "shape": asdict(shape),
        "ownership": "u=p+version", "materialized_providers_enabled": materialized, "plane_order": "u_z ascending; stage then H/E",
        "legality": legality, "accesses": [asdict(a) for a in access],
        "frontier": [asdict(f) for f in required],
        "shared_components": frame_components, "shared_frame_words": frame_words,
        "provider_materialization": materialize_providers(access, shape),
        "barriers": [asdict(b) for b in barriers],
        "resources": {"persistent_field_scalars_per_thread": len(required),
                      "rolling_old_h_e_scalars_per_thread": 6,
                      "current_h_e_scalars_per_thread": 6,
                      "exchange_shared_bytes": 4 * sum(frame_words.values()),
                      "barriers_per_u_z_plane": len(barriers),
                      "owned_xy_cells_per_stage": math.prod(shape.tile_zyx[1:]),
                      "owned_to_launched_xy_ratio": math.prod(shape.tile_zyx[1:]) / shape.threads,
                      "scope": "source liveness only; excludes SSA/metadata/compiler allocation"},
        "metadata_policy": {"coordinates": "rematerialize within phase; never retain descriptor subexpressions across barriers",
                            "global_fallback": "test transformed source coordinates against the whole owned u-box",
                            "coefficients": "stationary scalar/profile/material callback local to cell SSA",
                            "observations": "point-local finish callback; deferred chronological reduction",
                            "sources": "point-local finish before dependent field phase"},
    }
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tile", type=int, nargs=3, default=(64, 13, 29))
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--threads", type=int, default=512)
    p.add_argument("--cpml-axes", type=int, nargs="*")
    p.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    plan = compile_stream(yee_operator(cpml_axes=a.cpml_axes, precision=a.precision),
                          StreamShape(tuple(a.tile), a.depth, a.threads))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
    print(json.dumps(plan["resources"], sort_keys=True))


if __name__ == "__main__":
    main()
