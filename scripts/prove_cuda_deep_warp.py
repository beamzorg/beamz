"""Independent integer-version proof for the mixed warp/shared/register schedule.

No CUDA/JAX/device import. Each token carries its field, timestep and physical
coordinate; shared planes, selective Y histories, and private CPML histories
have separate stores. Deliberate wrong-shuffle and ramp-retirement controls
must be rejected. This proves dependencies/liveness, not floating arithmetic.
"""

from __future__ import annotations

import argparse
import json

try:
    from scripts.cuda_stencil_trace import trace_cpml_step
except ModuleNotFoundError:
    from cuda_stencil_trace import trace_cpml_step


def prove(tile=(5, 4, 25), depth=4, mask=7, *, bad_shuffle=False, bad_psi_ramp=False,  # noqa: C901
          field_frames=1, bad_frame=False):  # noqa: C901
    tz, ty, tx = tile
    stages = 2 * depth
    widths = [tx + stages - 1 - k for k in range(stages)]
    heights = [ty + stages - 1 - k for k in range(stages)]
    lows = [(k + 1) // 2 for k in range(stages)]
    assert widths[0] <= 32 and heights[0] * 32 <= 1024
    selected = []
    for k in range(stages):
        family = "E" if k % 2 else "H"
        selected.append(
            tuple(
                u
                for u in trace_cpml_step(
                    k // 2, active_axes=tuple(a for a in range(3) if mask & (1 << a))
                )
                if u.output.field.startswith(family)
                or u.output.field.startswith("psi_" + family)
            )
        )
    shared = [[{} for _ in range(field_frames)] for _ in range(stages - 1)]
    previous_y = [{} for _ in range(stages - 1)]
    previous_psi = [{} for _ in range(stages - 2)]
    writes = set()
    reads = 0

    def active(k, qx, qy):
        return (
            lows[k] <= qx < lows[k] + widths[k] and lows[k] <= qy < lows[k] + heights[k]
        )

    def fieldkey(field, qy, qx):
        return field, qy, qx

    def plane(k, previous=False):
        return shared[k][(wave - (previous and not bad_frame)) % field_frames]

    for wave in range(tz + stages - 1):
        current = {"H": {}, "E": {}}
        current_psi = {"H": {}, "E": {}}
        old = {"H": {}, "E": {}}
        for k in range(stages):
            phase = k % 2
            family = "E" if phase else "H"
            z = wave - depth - k // 2
            if not phase and k:
                for qy in range(heights[0]):
                    for qx in range(32):
                        if wave >= k and active(k - 1, qx, qy):
                            for c in "xyz":
                                key = fieldkey("E" + c, qy, qx)
                                old["E"][key] = (
                                    previous_y[k - 1] if c == "y" else plane(k - 1, True)
                                )[key]
            if phase:
                for key, value in previous_y[k - 1].items():
                    old["H"][key] = value
            pending = {}
            pending_psi = {}
            for qy in range(heights[0]):
                for qx in range(32):
                    coordinate = (z, qy - depth, qx - depth)
                    if wave >= k and active(k, qx, qy):
                        for update in selected[k]:
                            for read in update.reads:
                                reads += 1
                                dz, dy, dx = read.offset
                                key = fieldkey(read.field, qy + dy, qx + dx)
                                expected = (
                                    read.field,
                                    read.version,
                                    tuple(
                                        a + b
                                        for a, b in zip(
                                            coordinate, read.offset, strict=True
                                        )
                                    ),
                                )
                                if read.field.startswith("psi_"):
                                    if k < 2:
                                        assert read.version == 0
                                        continue
                                    actual = previous_psi[k - 2].get(key)
                                elif read.field.startswith(family):
                                    if k < 2:
                                        assert read.version == 0
                                        continue
                                    actual = old[family].get(key)
                                elif k == 0:
                                    assert read.version == 0
                                    continue
                                elif not phase:
                                    if dz:
                                        actual = current["E"].get(key)
                                    elif dy:
                                        actual = plane(k - 1, True).get(key)
                                    elif dx:
                                        actual = old["E"].get(
                                            (
                                                read.field,
                                                qy + (1 if bad_shuffle else 0),
                                                qx + dx,
                                            )
                                        )
                                    else:
                                        actual = old["E"].get(key)
                                elif dz:
                                    actual = old["H"].get(key)
                                elif dy:
                                    actual = plane(k - 1).get(key)
                                elif dx:
                                    actual = current["H"].get(
                                        (
                                            read.field,
                                            qy + (1 if bad_shuffle else 0),
                                            qx + dx,
                                        )
                                    )
                                else:
                                    actual = current["H"].get(key)
                                assert actual == expected, (wave, k, actual, expected)
                            token = (
                                update.output.field,
                                update.output.version,
                                coordinate,
                            )
                            target = (
                                pending_psi
                                if update.output.field.startswith("psi_")
                                else pending
                            )
                            target[fieldkey(update.output.field, qy, qx)] = token
                            owned = (
                                depth <= qx < depth + tx
                                and depth <= qy < depth + ty
                                and 0 <= z < tz
                            )
                            if k >= stages - 2 and owned:
                                assert token not in writes
                                writes.add(token)
            if k >= 2 and wave >= k - 2:
                for key, token in current[family].items():
                    if key[0].endswith("y") and active(k - 2, key[2], key[1]):
                        previous_y[k - 2][key] = token
                if not bad_psi_ramp or wave >= k:
                    for key, token in current_psi[family].items():
                        if active(k - 2, key[2], key[1]):
                            previous_psi[k - 2][key] = token
            if not phase:
                for key, token in plane(k, True).items():
                    if wave > k and active(k, key[2], key[1]):
                        old["H"][key] = token
            current[family].update(pending)
            current_psi[family].update(pending_psi)
            if not phase:
                if wave >= k:
                    plane(k).update(
                        {
                            key: t
                            for key, t in current["H"].items()
                            if not key[0].endswith("y") and active(k, key[2], key[1])
                        }
                    )
                if k and wave >= k - 1:
                    plane(k - 1).update(
                        {
                            key: t
                            for key, t in current["E"].items()
                            if not key[0].endswith("y")
                            and active(k - 1, key[2], key[1])
                        }
                    )
        if wave >= stages - 2:
            previous_y[stages - 2].update(
                {
                    key: t
                    for key, t in current["H"].items()
                    if key[0].endswith("y") and active(stages - 2, key[2], key[1])
                }
            )
    expected = {
        (u.output.field, depth, (z, y, x))
        for u in trace_cpml_step(
            depth - 1, active_axes=tuple(a for a in range(3) if mask & (1 << a))
        )
        for z in range(tz)
        for y in range(ty)
        for x in range(tx)
    }
    assert writes == expected, (len(writes), len(expected), writes ^ expected)
    return {
        "tile_zyx": tile,
        "depth": depth,
        "mask": mask,
        "versioned_reads_checked": reads,
        "unique_final_outputs": len(writes),
        "shared_field_frames": field_frames,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output")
    p.add_argument("--field-frames", type=int, choices=(1, 2), default=1)
    args = p.parse_args()
    checks = [
        prove((3, 3, min(7, 33 - 2 * d)), d, m, field_frames=args.field_frames)
        for d in (1, 2, 4, 8)
        for m in (0, 1, 2, 4, 7)
    ]
    for control in ("bad_shuffle", "bad_psi_ramp"):
        try:
            prove((3, 3, 7), 4, 7, **{control: True})
        except AssertionError:
            pass
        else:
            raise AssertionError(f"Negative control escaped proof: {control}")
    result = {
        "checks": checks,
        "negative_controls_rejected": ["wrong warp row", "missing psi ramp retirement"],
    }
    if args.field_frames == 2:
        try:
            prove((3, 3, 7), 4, 7, field_frames=2, bad_frame=True)
        except AssertionError:
            result["negative_controls_rejected"].append("wrong shared frame age")
        else:
            raise AssertionError("Wrong-frame negative control escaped proof")
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
