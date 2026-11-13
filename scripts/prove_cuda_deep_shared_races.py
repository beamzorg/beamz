"""Conservative CTA epoch proof for one/two-frame field-plane communication.

Each (half-stage, frame) denotes the ENTIRE X/Z shared plane. Two events on the
same plane are treated as potentially conflicting even when coordinates differ.
Thus disjoint epochs/planes suffice for arbitrary warp arrival order. Private
register/psi histories are not inter-thread stores and are outside this proof.
This does not replace GPU racecheck/synccheck or arithmetic/version parity.
"""

from __future__ import annotations

import json


def prove(depth, frames, *, pre_barrier, post_barrier=True):
    events = []
    epoch = 0
    # Use steady-state waves: ramp predicates only REMOVE accesses from these.
    for wave in range(2 * depth, 2 * depth + 4):
        old, new = (wave - 1) % frames, wave % frames
        for k in range(0, 2 * depth, 2):
            read_planes = [(k, old)] + ([(k - 1, old)] if k else [])
            for plane in read_planes:
                events.append((epoch, "read", plane, (wave, k)))
            if pre_barrier:
                epoch += 1
            write_planes = [(k, new)] + ([(k - 1, new)] if k else [])
            for plane in write_planes:
                events.append((epoch, "write", plane, (wave, k)))
            if post_barrier:
                epoch += 1
            # E's Y-neighbor input consumes H's newly published current plane.
            events.append((epoch, "read", (k, new), (wave, k + 1)))
    for i, a in enumerate(events):
        for b in events[i + 1:]:
            if a[0] != b[0] or a[2] != b[2] or (a[1] == b[1] == "read"):
                continue
            raise AssertionError(("Possible cross-warp shared-plane race", a, b))
    return {"depth": depth, "frames": frames, "pre_overwrite_barrier": pre_barrier,
            "post_publication_barrier": post_barrier, "plane_events": len(events)}


def main():
    checks = [prove(d, f, pre_barrier=f == 1) for d in (1, 2, 4, 8) for f in (1, 2)]
    rejected = []
    for label, kwargs in (("one frame without pre-overwrite barrier", {"frames": 1, "pre_barrier": False}),
                          ("two frames without post-publication barrier", {"frames": 2, "pre_barrier": False, "post_barrier": False})):
        try:
            prove(4, **kwargs)
        except AssertionError:
            rejected.append(label)
        else:
            raise AssertionError(f"Negative control escaped: {label}")
    print(json.dumps({"scope": __doc__, "checks": checks,
                      "negative_controls_rejected": rejected}, indent=2))


if __name__ == "__main__":
    main()
