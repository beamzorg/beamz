"""Execute the compiled two-frame stream plan with independent version tokens.

This models actual reused H/E frame contents, private stage histories, global
owner fallbacks and local CPML in-place state. It checks schedule/storage, not
FP32 arithmetic, CUDA convergence, launch resources or speed.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from scripts.cuda_space_time_ir import (
        StreamShape,
        compile_stream,
        yee_operator,
    )
    from scripts.prove_cuda_systolic_schedule import component_shapes
except ModuleNotFoundError:
    from cuda_space_time_ir import StreamShape, compile_stream, yee_operator
    from prove_cuda_systolic_schedule import component_shapes


ZERO = ("zero",)


def prove(shape=(5, 6, 7), tile=(3, 3, 5), depth=2, *, cpml=False,  # noqa: C901
          current_e_frame=False, stale_h_frame=False, omit_barrier_a=False,
          early_psi_store=False, wrong_old_h=False, materialized=False,
          omit_provider_load=False):  # noqa: C901
    threads = 32 * (tile[1] + depth - 1)
    config = StreamShape(tile, depth, threads)
    operator = yee_operator(cpml_axes=(0, 1, 2) if cpml else None)
    plan = compile_stream(operator, config, materialized=materialized)
    specs, shapes = operator.specs, component_shapes(shape)
    equations = {phase: [u for u in operator.updates if specs[u.output.field].phase == phase] for phase in (0, 1)}
    routes = {(a["stage"], a["output"], a["source"], a["version"], tuple(a["offset"])): a for a in plan["accesses"]}
    retained = {(f["tensor"], f["version"]) for f in plan["frontier"]}
    produced, psi = {}, {}
    stats = Counter()
    grid = tuple((n + depth + b) // b for n, b in zip(shape, tile, strict=True))
    owners = itertools.product(*(range(n) for n in grid))

    def valid(field, p):
        return all(0 <= a < b for a, b in zip(p, shapes[field], strict=True))

    def owner(p, version):
        return tuple((a + version) // b for a, b in zip(p, tile, strict=True))

    def expected(field, version, p):
        return (field, version, p) if valid(field, p) else ZERO

    def global_read(field, version, p):
        if not valid(field, p):
            return ZERO
        if version == 0:
            return expected(field, version, p)
        key = field, version, p
        assert key in produced, ("Global producer has not executed", key)
        return produced[key]

    for box in sorted(owners, key=lambda b: (sum(b), b)):
        origin = tuple(a * b for a, b in zip(box, tile, strict=True))
        yy = [origin[1] - depth + q // 32 for q in range(threads)]
        xx = [origin[2] - depth + q % 32 for q in range(threads)]
        # All previous-Z sources at the first plane are external owners, so
        # initialized zero GPRs are safe; the owner guard must precede the route.
        history = {(field, version): [ZERO] * threads for field, version in retained}
        frame = {"H": {}, "E": {}}
        pending_h_reads = set()
        for uz in range(origin[0], origin[0] + tile[0]):
            old_h_previous = {c: [ZERO] * threads for c in "xyz"}
            old_e_previous = {c: [ZERO] * threads for c in "xyz"}
            current_e_previous = {c: [ZERO] * threads for c in "xyz"}
            for n in range(1, depth + 1):
                pz = uz - n
                active = [owner((pz, yy[q], xx[q]), n) == box for q in range(threads)]
                old_h_current = {c: list(history.get(("H" + c, n), [ZERO] * threads)) for c in "xyz"}
                def load_providers(holders, prefixes, n=n, pz=pz, yy=yy, xx=xx, box=box):
                    if not materialized or omit_provider_load:
                        return
                    for item in plan["provider_materialization"]["providers"]:
                        prefix, c = item["holder"].rsplit("_", 1)
                        if item["stage"] != n or prefix not in prefixes:
                            continue
                        lo = depth - n
                        for q in range(threads):
                            qy, qx = divmod(q, 32)
                            needed = any(lo + dy <= qy < lo + dy + tile[1] and lo + dx <= qx < lo + dx + tile[2]
                                         for dy, dx in item["xy_offsets"])
                            if not needed:
                                continue
                            p = pz + item["physical_z_offset"], yy[q], xx[q]
                            if item["version"] == 0 or owner(p, item["version"]) != box:
                                holders[prefix][c][q] = global_read(item["source"], item["version"], p)
                                stats["provider_scalar_loads"] += valid(item["source"], p)
                            else:
                                assert holders[prefix][c][q] == expected(item["source"], item["version"], p), ("Bad local provider holder", item, p)

                load_providers({"old_h_previous": old_h_previous,
                                "old_e_previous": old_e_previous,
                                "current_e_previous": current_e_previous},
                               ("old_h_previous", "old_e_previous", "current_e_previous"))
                # Republish the PREVIOUS plane for this stage from private GPRs.
                e_values = current_e_previous if current_e_frame else old_e_previous
                for c in plan["shared_components"]["E"]:
                    for q in range(threads):
                        frame["E"][c, q] = e_values["xyz"[c]][q]
                # A publishes E and retires previous E-phase H-frame readers.
                if not omit_barrier_a:
                    pending_h_reads.clear()
                cur_h = {c: [ZERO] * threads for c in "xyz"}
                cur_e = {c: [ZERO] * threads for c in "xyz"}
                for phase, _family in enumerate("HE"):
                    holders = {"old_h_previous": old_h_previous,
                               "old_h_current": old_h_current,
                               "old_e_previous": old_e_previous,
                               "current_e_previous": current_e_previous,
                               "current_h": cur_h}
                    shuffle = {}
                    for route in plan["accesses"]:
                        if route["stage"] != n or route["phase"] != phase or route["route"] != "warp-shuffle":
                            continue
                        holder_name, c = route["holder"].rsplit("_", 1)
                        values = holders[holder_name][c]
                        dx = route["offset"][2]
                        # Every full warp executes; inactive inputs are defined zero.
                        for q in range(threads):
                            source_q = (q // 32) * 32 + ((q % 32 + dx) % 32)
                            shuffle[route["source"], tuple(route["offset"]), q] = values[source_q]
                        stats["full_warp_shuffle_participations"] += threads
                    writes = []
                    for q in range(threads):
                        if not active[q]:
                            continue
                        p = pz, yy[q], xx[q]
                        # All SSA reads happen before ANY local recurrence store.
                        for update in equations[phase]:
                            field = update.output.field
                            if not valid(field, p):
                                continue
                            for base in sorted(update.reads):
                                version = base.version + n - 1
                                source = tuple(a + b for a, b in zip(p, base.offset, strict=True))
                                route = routes[n, field, base.field, version, base.offset]
                                if specs[base.field].role == "local-state":
                                    actual = psi.get((base.field, source), expected(base.field, 0, source)) if valid(base.field, source) else ZERO
                                elif not valid(base.field, source):
                                    actual = ZERO
                                elif ((not materialized and (version == 0 or owner(source, version) != box)) or
                                      (materialized and not (0 <= q // 32 + base.offset[1] < threads // 32 and
                                                             0 <= q % 32 + base.offset[2] < 32))):
                                    actual = global_read(base.field, version, source)
                                    stats["global_operands"] += 1
                                elif route["route"] == "shared-exchange":
                                    source_q = q + base.offset[1] * 32 + base.offset[2]
                                    assert 0 <= source_q < threads and xx[source_q] == source[2] and yy[source_q] == source[1]
                                    key = "xyz".index(base.field[-1]), source_q
                                    assert key in frame[base.field[0]], ("Uninitialized exchange frame", n, route)
                                    actual = frame[base.field[0]][key]
                                    if phase:
                                        pending_h_reads.add(key)
                                    stats["shared_operands"] += 1
                                elif route["route"] == "warp-shuffle":
                                    actual = shuffle[base.field, base.offset, q]
                                    stats["shuffle_operands"] += 1
                                else:
                                    name, c = route["holder"].rsplit("_", 1)
                                    actual = holders[name][c][q]
                                    if wrong_old_h and name == "old_h_current":
                                        actual = cur_h[c][q]
                                    stats["register_operands"] += 1
                                assert actual == expected(base.field, version, source), ("Incorrect route value", box, uz, n, phase, route, p, actual, expected(base.field, version, source))
                            value = expected(field, n, p)
                            writes.append((field, n, p, q, value))
                            if early_psi_store and specs[field].role == "local-state":
                                psi[field, p] = value
                    for field, version, p, q, value in writes:
                        key = field, version, p
                        assert key not in produced, "Duplicate owned update"
                        produced[key] = value
                        if specs[field].role == "local-state":
                            psi[field, p] = value
                        else:
                            (cur_h if phase == 0 else cur_e)[field[-1]][q] = value
                    if not phase:
                        # Persist only owned outputs, before ghost providers fill
                        # temporary current-H registers for cross-thread consumers.
                        for c in "xyz":
                            if ("H" + c, n) in retained:
                                history["H" + c, n] = list(cur_h[c])
                        load_providers({"old_h_current": old_h_current, "current_h": cur_h},
                                       ("old_h_current", "current_h"))
                        # H frame writes must wait until A retires previous E readers.
                        write_set = {(c, q) for c in plan["shared_components"]["H"] for q in range(threads)}
                        assert not pending_h_reads.intersection(write_set), "H-frame overwrite races a previous E reader"
                        if not stale_h_frame:
                            for c, q in write_set:
                                frame["H"][c, q] = cur_h["xyz"[c]][q]
                        # B publishes H; all H E-frame readers have retired.
                    else:
                        old_e_next = {c: list(history.get(("E" + c, n), [ZERO] * threads)) for c in "xyz"}
                        for c in "xyz":
                            if ("E" + c, n) in retained:
                                history["E" + c, n] = list(cur_e[c])
                        old_e_previous = old_e_next
                        current_e_previous = cur_e
                old_h_previous = old_h_current
    expected_updates = depth * sum(math.prod(shapes[t.name]) for t in operator.tensors)
    assert len(produced) == expected_updates
    return {"shape": shape, "tile": tile, "depth": depth, "cpml": cpml,
            "updates": len(produced), "executed_over_useful": 1.0,
            "persistent_scalars_per_thread": len(retained), "materialized": materialized,
            "shared_bytes": plan["resources"]["exchange_shared_bytes"],
            **stats, "result": "pass"}


def main():
    cases = [prove(depth=n, cpml=c) for n in (1, 2, 4) for c in (False, True)]
    negatives = []
    for name, options in (("publish current E instead of previous", {"current_e_frame": True}),
                          ("stale H frame", {"stale_h_frame": True}),
                          ("overwrite H before retirement barrier", {"omit_barrier_a": True}),
                          ("current H in place of previous Z H", {"wrong_old_h": True}),
                          ("local psi store before all SSA reads", {"cpml": True, "early_psi_store": True})):
        try:
            prove(**options)
        except AssertionError:
            negatives.append(name)
        else:
            raise AssertionError(f"Negative control escaped: {name}")
    print(json.dumps({"scope": __doc__, "cases": cases, "negative_controls_rejected": negatives}, indent=2))


if __name__ == "__main__":
    main()
