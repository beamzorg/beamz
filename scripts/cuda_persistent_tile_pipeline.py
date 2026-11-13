"""Resident FIFO tile-step pipeline oracle, separate from completion-ready DAGs.

Jobs are assigned only by executing workers. Claim waits use the explicit old
holder and ALL its raw consumers; RAW communication waits only on claim tags
and ready STEP counters. Counter reset precedes release claim publication.
This is a CPU method proof, never a GPU throughput prediction.
"""

from __future__ import annotations

import heapq
import itertools
import random

try:
    from scripts.cuda_persistent_wavefront_dag import build as build_dag
except ModuleNotFoundError:
    from cuda_persistent_wavefront_dag import build as build_dag


class PipelineHazard(AssertionError):
    pass


def prepare(counts, slots, frontier, *, jobs=None, locality=False):
    dag = build_dag(counts, slots, frontier, epochs=4)
    count = len(dag["ranks"])
    raw_pred = [set() for _ in range(count)]
    for src, targets in enumerate(dag["raw_successors"]):
        for dst in targets:
            raw_pred[dst].add(src)
    war_pred = [set() for _ in range(count)]
    for new, old in dag["previous_slot_holder"].items():
        # Do not derive this as allPreds-RAW: an edge can impose BOTH STEP
        # readiness and full-completion reader retirement.
        war_pred[new] = {old} | dag["raw_successors"][old]
    if jobs is None:
        if not locality:
            jobs = sorted(range(count), key=lambda t: (dag["ranks"][t], t))
        else:
            ny, nx = counts[1:]

            def morton(t):
                z, y, x = t//(ny*nx), (t//nx) % ny, t % nx
                return sum(((v >> bit) & 1) << (3*bit+axis) for bit in range(21) for axis, v in enumerate((x, y, z)))

            indegree = [len(x) for x in dag["predecessors"]]
            ready = [(morton(t), t) for t, n in enumerate(indegree) if n == 0]
            heapq.heapify(ready)
            jobs = []
            while ready:
                _, t = heapq.heappop(ready)
                jobs.append(t)
                for child in dag["successors"][t]:
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        heapq.heappush(ready, (morton(child), child))
    jobs = tuple(jobs)
    if len(jobs) != count or set(jobs) != set(range(count)):
        raise ValueError("One exact tile permutation required")
    inverse = {t: i for i, t in enumerate(jobs)}
    for t in jobs:
        if any(inverse[p] >= inverse[t] for p in raw_pred[t] | war_pred[t]):
            raise ValueError("Every RAW and explicit slot-retirement predecessor must have a lower job ID")
    dag.update({"jobs": jobs, "job_index": inverse, "raw_predecessors": raw_pred,
                "claim_completion_predecessors": war_pred})
    return dag


def windows(counts, extent, depth):
    """Exact selected P5 first/last inequalities; extent is physical N(Z,Y,X)."""
    ny, nx = counts[1:]
    owned = (7, 15, 31)
    result = []
    for tile in range(counts[0]*ny*nx):
        q = (tile//(ny*nx), (tile//nx) % ny, tile % nx)
        origins = tuple(a*b for a, b in zip(q, owned, strict=True))
        first = max(1, *(o-n for o, n in zip(origins, extent, strict=True)))
        last = min(depth, *(o+b-1 for o, b in zip(origins, owned, strict=True)))
        # Every skipped owned box is wholly outside the union of all six
        # staggered supports. Imported native-support masks are thereforezero.
        for step in range(1, depth+1):
            if first <= step <= last:
                continue
            low = tuple(o-step for o in origins)
            high = tuple(o+b-1-step for o, b in zip(origins, owned, strict=True))
            if not any(lo > n or hi < 0 for lo, hi, n in zip(low, high, extent, strict=True)):
                raise PipelineHazard(("Skipped tile version has a potentially supported field", tile, step))
        result.append((first, last))
    return tuple(result)


def simulate(dag, *, depth=16, workers=4, seed=19, step_times=None,
             claim_delays=None, worker_start_delays=None, tile_windows=None,
             negative=None, timeout=100000, publication_steps=1, cache_upstream_floor=False, overlap_h=False,startup_previous=False):
    n = len(dag["jobs"])
    if workers < 1 or depth < 1 or publication_steps not in (1, 2, 4, 16):
        raise ValueError("Positive workers and depth required")
    rng = random.Random(seed)
    times = step_times or [[rng.randrange(1, 9) for _ in range(depth)] for _ in range(n)]
    delays = claim_delays or [rng.randrange(4) for _ in range(n)]
    starts = worker_start_delays or [rng.randrange(7) for _ in range(workers)]
    spans = tile_windows or [(1, depth)]*n
    if len(times) != n or any(len(x) != depth or min(x) < 1 for x in times) or len(delays) != n or len(starts) != workers or len(spans) != n:
        raise ValueError("Durations, worker/claim delays and tile windows differ")
    epoch = 17
    owners = [None]*(4*dag["frontier"])
    ready = [0]*len(owners)
    actual_published = [0]*n
    claimed, completed = [epoch-1]*n, [epoch-1]*n
    done = set()
    assigned = set()
    states = [{"state": "idle", "start": starts[i], "worker": i} for i in range(workers)]
    next_job = 0
    clock = 0
    step_starts = [[None]*depth for _ in range(n)]
    step_ready = [[None]*depth for _ in range(n)]
    import_starts = [[None]*depth for _ in range(n)]
    overlap_starts = claim_checks = ready_checks = 0
    slot_resets_before_tag = 0
    reset_events = []
    publication_count = 0
    upstream_floor = [0]*n
    floor_hits = 0

    def owner_slot(t):
        return (dag["ranks"][t] & 3)*dag["frontier"]+dag["slots"][t]

    def raw_ready(t, step):
        nonlocal ready_checks, floor_hits
        if cache_upstream_floor and step>0 and step<=upstream_floor[t]:
            # These checks are proof-only instrumentation, not CUDA polls:
            # full WAR must keep every parent slot owned by that parent.
            for parent in dag["raw_predecessors"][t]:
                if owners[owner_slot(parent)]!=parent or actual_published[parent]<step:
                    raise PipelineHazard(("Cached acquired floor invalidated by slot reuse",t,parent,step))
            floor_hits+=1
            return True
        floor=depth
        for parent in dag["raw_predecessors"][t]:
            ready_checks += 1
            slot = owner_slot(parent)
            if owners[slot] != parent or claimed[parent] != epoch or ready[slot] < step:
                return False
            if actual_published[parent] < step:
                raise PipelineHazard(("Ready claim observed stale data from previous holder", t, parent, step, ready[slot]))
            floor=min(floor,ready[slot])
        if cache_upstream_floor:upstream_floor[t]=floor
        return True

    while len(done) != n:
        progress = False
        while reset_events and reset_events[0][0] <= clock:
            _, slot = heapq.heappop(reset_events)
            ready[slot] = 0
            progress = True
        order = list(range(workers))
        rng.shuffle(order)
        for wi in order:
            s = states[wi]
            state = s["state"]
            if state == "running-H" and s["until"] <= clock:
                s["state"]="wait-import"
                s["stalled_since"]=clock
                progress=True
            if s["state"]=="wait-import":
                t,step=s["tile"],s["step"]
                if negative=='skip-import-gate' or raw_ready(t,step):
                    for parent in dag['raw_predecessors'][t]:
                        if actual_published[parent]<step:
                            raise PipelineHazard(('Import started before current parent publication',t,parent,step))
                    import_starts[t][step-1]=clock
                    # H/E split sums to the original per-step work duration.
                    e_time=times[t][step-1]-max(1,times[t][step-1]//2)
                    s.update({"state":"running","until":clock+e_time,"stalled_since":clock})
                    progress=True
            if state == "running" and s["until"] <= clock:
                t, step = s["tile"], s["step"]
                # One all-writer fence can publish the entire earlier group.
                # End-window publication is mandatory even when unaligned.
                publish = step % publication_steps == 0 or step == s["last"]
                if negative == "omit-window-end-publication":
                    publish = step % publication_steps == 0
                if publish:
                    actual_published[t] = step
                    ready[owner_slot(t)] = step
                    publication_count += 1
                    for old_step in range(step):
                        if step_ready[t][old_step] is None:
                            step_ready[t][old_step] = clock
                s["state"] = "wait-next" if step < s["last"] else "completing"
                s["until"] = clock+1
                s["stalled_since"] = clock
                progress = True
            if s["state"] == "completing" and s["until"] <= clock:
                t = s["tile"]
                actual_published[t] = depth
                ready[owner_slot(t)] = depth
                completed[t] = epoch
                done.add(t)
                s["state"] = "idle"
                progress = True
            if s["state"] == "idle" and s["start"] <= clock and next_job < n:
                t = dag["jobs"][next_job]
                if t in assigned:
                    raise PipelineHazard("FIFO job assigned twice")
                assigned.add(t)
                next_job += 1
                s.update({"state": "claim", "tile": t, "until": clock+delays[t], "stalled_since": clock})
                progress = True
            if s["state"] == "claim" and s["until"] <= clock:
                t = s["tile"]
                deps = dag["claim_completion_predecessors"][t]
                claim_checks += len(deps)
                if negative == "drop-WAR":
                    deps = set()
                elif negative == "subtract-RAW-from-WAR":
                    deps = deps-dag["raw_predecessors"][t]
                if all(completed[p] == epoch for p in deps):
                    old = dag["previous_slot_holder"].get(t)
                    slot = owner_slot(t)
                    if old is not None and (old not in done or not dag["raw_successors"][old] <= done):
                        raise PipelineHazard(("Old consumer poisoned by premature trail-slot reuse", old, t))
                    if owners[slot] != old:
                        raise PipelineHazard(("Slot holder differs from actual previous owner", t, old, owners[slot]))
                    if negative == "reset-after-tag" and old is not None:
                        owners[slot] = t
                        heapq.heappush(reset_events, (clock+1000, slot))
                    else:
                        ready[slot] = 0
                        slot_resets_before_tag += 1
                        owners[slot] = t
                    claimed[t] = epoch
                    upstream_floor[t] = 0
                    first, last = spans[t]
                    s.update({"state": "wait-first", "step": first, "last": last, "stalled_since": clock})
                    if first > last:
                        s.update({"state": "completing", "until": clock+1})
                    progress = True
            if s["state"] in ("wait-first", "wait-next"):
                t = s["tile"]
                step = s["step"] if s["state"] == "wait-first" else s["step"]+1
                target = step+1 if negative == "future-step" else step
                if startup_previous and s['state']=='wait-first':target=step-1
                if target > depth and negative == "future-step":
                    okay = False
                elif overlap_h and s["state"]=="wait-next":
                    # This is a proof-only check, not another CUDA poll: old
                    # ghost E and psi ownership already arrived in step-1.
                    okay=True
                    for parent in dag["raw_predecessors"][t]:
                        if actual_published[parent]<step-1:
                            raise PipelineHazard(("H started before previous E/psi owner publication",t,parent,step))
                else:
                    okay = raw_ready(t, target)
                if okay:
                    if any(completed[p] != epoch for p in dag["raw_predecessors"][t]):
                        overlap_starts += 1
                    step_starts[t][step-1] = clock
                    state="running-H" if overlap_h else "running"
                    duration=max(1,times[t][step-1]//2) if overlap_h else times[t][step-1]
                    if not overlap_h:import_starts[t][step-1]=clock
                    s.update({"state": state, "step": step, "until": clock+duration, "stalled_since": clock})
                    progress = True
            if s["state"] in ("claim", "wait-first", "wait-next", "wait-import") and clock-s["stalled_since"] > timeout:
                raise PipelineHazard(("Bounded-progress watchdog aborted stalled wait", s["state"], s["tile"]))
        if progress:
            continue
        future = [s["until"] for s in states if s["state"] in ("running", "running-H", "completing", "claim") and s["until"] > clock]
        future += [s["start"] for s in states if s["state"] == "idle" and s["start"] > clock and next_job < n]
        future += [x[0] for x in reset_events if x[0] > clock]
        if not future:
            raise PipelineHazard(("Deadlock / watchdog rejection", [(s["state"], s.get("tile")) for s in states]))
        clock = min(future)
    if len(assigned) != n or next_job != n or any(x != epoch for x in completed):
        raise PipelineHazard("Incomplete or duplicated persistent coverage")
    return {"tiles": n, "workers": workers, "depth": depth, "simulated_ticks": clock,
            "claim_completion_checks": claim_checks, "RAW_step_checks": ready_checks,
            "steps_started_before_RAW_parent_completion": overlap_starts,
            "ready_resets_before_claim": slot_resets_before_tag,
            "publication_steps": publication_steps, "in_body_publications": publication_count,
            "upstream_floor_hits":floor_hits,
            "step_starts": step_starts, "step_ready": step_ready,
            "import_starts":import_starts,"H_overlap":overlap_h,
            "scope": "CPU discrete-event protocol proof; ticks are not CUDA timings"}
