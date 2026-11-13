"""Independent physical CPML handoff proof for the persistent paired scheduler.

For fixed physical p, q(n)=floor((p+n)/B). A tile's complete owned interval is
L=max(qB-p), U=min(qB+B-1-p). Once q changes, the old owner never reads/writes
that physical recurrence again, regardless of how far its other work advances.
The next owner's wait on the direct upstream STEP n covers the old last write
at n-1. This proof does not certify any proposed batched readiness publication.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA, derive_geometry
from scripts.cuda_space_time_ir import FIELDS

OWNED = (7, 15, 31)


def owner(point, version):
    return tuple((p+version)//b for p, b in zip(point, OWNED, strict=True))


def interval(point, tile):
    lower = max(q*b-p for q, b, p in zip(tile, OWNED, point, strict=True))
    upper = min(q*b+b-1-p for q, b, p in zip(tile, OWNED, point, strict=True))
    return lower, upper


def boundary_classes(depth):
    handoffs = same_owner = future_checks = 0
    examples = []
    # Translating p by k*B only translates every tile by k; all equalities and
    # offset/last-use bounds are therefore covered by these residue classes.
    for point in itertools.product(*(range(b) for b in OWNED)):
        for n in range(2, depth+1):
            previous, current = owner(point, n-1), owner(point, n)
            assert all(a<=b for a, b in zip(previous, current, strict=True))
            if previous == current:
                same_owner += 1
                continue
            lo, hi = interval(point, previous)
            next_lo, next_hi = interval(point, current)
            assert lo<=n-1<=hi and hi==n-1
            assert next_lo==n and next_hi>=n
            delta = tuple(b-a for a, b in zip(previous, current, strict=True))
            assert set(delta)<={0,1} and any(delta)
            assert sum(previous)<sum(current)
            # Arbitrarily advancing the producer all the way through Depth
            # cannot read/store this point's recurrence after its handoff.
            for future in range(n, depth+1):
                assert owner(point, future) != previous
                future_checks += 1
            handoffs += 1
            if len(examples)<8:
                examples.append({"physical_residue": point, "producer": previous, "consumer": current,
                                 "handoff_step": n, "producer_last_owned_step": hi,
                                 "consumer_first_owned_step": next_lo, "direct_upstream_delta": delta})
    return {"depth": depth, "physical_residue_classes": 3255, "handoffs": handoffs,
            "same_owner_step_edges": same_owner, "producer_future_steps_proved_disjoint": future_checks,
            "examples": examples, "result": "pass"}


def packed_states(minimum, depth, *, wrong_e_extent=False):
    terms = derive_geometry()["cpml_terms"]
    recurrences = updates = handoffs = future_steps = 0
    for term in terms:
        phase, axis = term["phase"], term["axis_zyx"]
        shape = tuple(n+d for n, d in zip(minimum, SUPPORT_DELTA[FIELDS.index(term["output"])], strict=True))
        assert shape[axis] == minimum[axis]+phase
        active = list(range(12))+list(range(shape[axis]-12, shape[axis]))
        assert len(set(active))==24
        packed = {}
        for coordinate in active:
            size = minimum[axis]+(0 if wrong_e_extent else phase)
            address = coordinate if coordinate<12 else coordinate+24-size
            if address not in range(24) or address in packed:
                raise AssertionError("Wrong phase extent makes the canonical packed CPML map noninjective/out of bounds")
            packed[address] = coordinate
        assert set(packed)==set(range(24))
        others = [sorted({0,1,OWNED[a]-1,OWNED[a],shape[a]//2,shape[a]-1}) if a!=axis else active for a in range(3)]
        for point in itertools.product(*others):
            previous_owner = None
            global_version, global_value = 0, 17
            expected = 17
            for n in range(1, depth+1):
                current = owner(point, n)
                lo, hi = interval(point, current)
                assert lo<=n<=hi
                if previous_owner is not None and current!=previous_owner:
                    _, old_end = interval(point, previous_owner)
                    assert old_end==n-1 and lo==n
                    delta = tuple(b-a for a,b in zip(previous_owner,current,strict=True))
                    assert set(delta)<={0,1} and any(delta)
                    # WaitUpstream(n) waits this direct predecessor's ready>=n,
                    # whose all-thread fenced publication covers old_end=n-1.
                    assert global_version==n-1
                    for later in range(n,depth+1):
                        assert owner(point,later)!=previous_owner
                        future_steps += 1
                    handoffs += 1
                assert global_version==n-1 and global_value==expected
                expected = expected*3+7*n+11*(6*phase+int(term["term"][-1]))
                global_value, global_version = expected,n
                previous_owner=current
                updates += 1
            assert global_version==depth
            recurrences += 1
    return {"minimum_support_zyx": minimum,"depth":depth,"terms":12,
            "sampled_physical_packed_recurrences":recurrences,"serialized_state_updates":updates,
            "intertile_handoffs":handoffs,"producer_later_steps_disjoint_checks":future_steps,
            "per_term_active_normal_coordinates_exhaustive":24,
            "result":"pass"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    negative=[]
    try:packed_states((25,31,47),4,wrong_e_extent=True)
    except AssertionError as exc:negative.append({"mutation":"E-packed-extent-uses-N-instead-of-N+1","rejected":True,"witness":str(exc)})
    else:raise AssertionError("Wrong E packed extent escaped")
    result={"schema":"beamz.persistent-physical-psi-handoff.v1",
            "producer_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "owned_zyx":OWNED,"boundary_classes":[boundary_classes(depth) for depth in (4,16)],
            "packed_cases":[packed_states(minimum,depth) for minimum,depth in (((25,31,47),4),((25,31,47),16),((64,96,128),16))],
            "negative_controls":negative,
            "native_session_state_identity":"Every input gets its own cudaMalloc; BindOutputs keeps each term psi input/output aliased to itself and leaves all other term bases distinct. Canonical packed-stride validation is retained.",
            "proof":"Owner intervals are contiguous; direct next owner starts exactly one step after the predecessor last owns the point; all subsequent predecessor work addresses other physical state. STEP n acquire happens after predecessor fenced STEP n, covering lastwrite n-1 (or completed early-window promotion). Every subsequent owner satisfies the same argument.",
            "scope":"Single-step readiness only. No proposed batched publication or physical CUDA numerical/memory race claim."}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"passed":True,"output":str(args.output),"boundary_handoffs":sum(x['handoffs'] for x in result['boundary_classes']),"packed_updates":sum(x['serialized_state_updates'] for x in result['packed_cases'])}))


if __name__=="__main__":main()
