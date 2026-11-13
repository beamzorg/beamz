"""Integer/version proof for register-history, single-shared-plane scheduling."""

import pytest

from scripts.cuda_stencil_register import register_resource_estimate
from scripts.cuda_stencil_rolling import rolling_resource_estimate
from scripts.cuda_stencil_trace import trace_yee_step


def verify_register_schedule(
    tile,
    depth,
    *,
    early_e_publication=False,
    current_z=False,
    e_history="register",
    h_history="register",
):
    tz, ty, tx = tile
    stages = 2 * depth
    widths = [tx + stages - 1 - k for k in range(stages)]
    heights = [ty + stages - 1 - k for k in range(stages)]
    lows = [(k + 1) // 2 for k in range(stages)]
    previous = [{} for _ in range(stages - 1)]
    shared = [{} for _ in range(stages - 1)]
    current = {"H": {}, "E": {}}
    old_e = {}
    old_h = {}
    written = set()

    def active(k, qx, qy):
        return (
            lows[k] <= qx < lows[k] + widths[k] and lows[k] <= qy < lows[k] + heights[k]
        )

    for wave in range(tz + stages - 1):
        for k in range(stages):
            phase, family = k % 2, "E" if k % 2 else "H"
            z = wave - depth - k // 2
            if early_e_publication and not phase and k:
                shared[k - 1] = current["E"].copy()
            for qy in range(heights[0]):
                for qx in range(widths[0]):
                    coordinate = (z, qy - depth, qx - depth)
                    pending = {}
                    if wave >= k and active(k, qx, qy):
                        if k and phase == 0 and e_history == "shared":
                            for component in "xyz":
                                key = "E" + component, qy, qx
                                old_e[key] = shared[k - 1][key]
                        updates = trace_yee_step(k // 2)[3 * phase : 3 * phase + 3]
                        for update in updates:
                            for read in update.reads:
                                dz, dy, dx = read.offset
                                key = (read.field, qy + dy, qx + dx)
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
                                if read.field.startswith(family):
                                    if k < 2:
                                        assert read.version == 0
                                        continue
                                    actual = (
                                        old_e
                                        if phase and e_history == "shared"
                                        else old_h
                                        if not phase and h_history == "shared"
                                        else previous[k - 2]
                                    ).get(key)
                                elif k == 0:
                                    assert read.version == 0
                                    continue
                                elif phase == 0:
                                    if dz:
                                        actual = current["E"].get(key)
                                    elif dy or dx:
                                        actual = shared[k - 1].get(key)
                                    else:
                                        actual = (
                                            old_e
                                            if e_history == "shared"
                                            else previous[k - 1]
                                        ).get(key)
                                elif dz:
                                    actual = (
                                        current["H"]
                                        if current_z
                                        else old_h
                                        if h_history == "shared"
                                        else previous[k - 1]
                                    ).get(key)
                                elif dy or dx:
                                    actual = shared[k - 1].get(key)
                                else:
                                    actual = current["H"].get(key)
                                assert actual == expected, (
                                    tile,
                                    depth,
                                    wave,
                                    k,
                                    actual,
                                    expected,
                                )
                            token = (
                                update.output.field,
                                update.output.version,
                                coordinate,
                            )
                            pending[update.output.field, qy, qx] = token
                            if k == stages - 1:
                                hfield = "H" + update.output.field[1]
                                assert current["H"].get((hfield, qy, qx)) == (
                                    hfield,
                                    depth,
                                    coordinate,
                                )
                                assert token not in written
                                written.add(token)
                    if (
                        k >= 2
                        and not (
                            (phase and e_history == "shared")
                            or (not phase and h_history == "shared")
                        )
                        and wave >= k - 2
                        and active(k - 2, qx, qy)
                    ):
                        for component in "xyz":
                            key = family + component, qy, qx
                            previous[k - 2][key] = current[family][key]
                    if (
                        phase == 0
                        and h_history == "shared"
                        and wave > k
                        and active(k, qx, qy)
                    ):
                        for component in "xyz":
                            key = "H" + component, qy, qx
                            old_h[key] = shared[k][key]
                    current[family].update(pending)
            if phase == 0:
                if wave >= k:
                    for key, token in current["H"].items():
                        if active(k, key[2], key[1]):
                            shared[k][key] = token
                if k and wave >= k - 1:
                    for key, token in current["E"].items():
                        if active(k - 1, key[2], key[1]):
                            shared[k - 1][key] = token
        if h_history == "register" and wave >= stages - 2:
            for key, token in current["H"].items():
                if active(stages - 2, key[2], key[1]):
                    previous[stages - 2][key] = token
    assert written == {
        ("E" + component, depth, (z, y, x))
        for component in "xyz"
        for z in range(tz)
        for y in range(ty)
        for x in range(tx)
    }


@pytest.mark.parametrize("tile", [(1, 1, 1), (2, 3, 5), (8, 8, 16)])
@pytest.mark.parametrize("depth", [1, 2, 4, 8])
@pytest.mark.parametrize("e_history", ["register", "shared"])
@pytest.mark.parametrize("h_history", ["register", "shared"])
def test_single_shared_plane_register_history_preserves_versions(
    tile, depth, e_history, h_history
):
    verify_register_schedule(tile, depth, e_history=e_history, h_history=h_history)


@pytest.mark.parametrize(
    "control", [{"early_e_publication": True}, {"current_z": True}]
)
def test_missing_register_schedule_dependency_is_rejected(control):
    with pytest.raises(AssertionError):
        verify_register_schedule((2, 3, 5), 4, **control)


@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_single_shared_plane_halves_storage_but_requires_register_history(depth):
    tile = (32, 8, 16)
    info = register_resource_estimate(tile, depth)
    assert (
        info["shared_bytes"] * 2
        == rolling_resource_estimate(tile, depth)["shared_bytes"]
    )
    assert info["persistent_history_scalars_per_thread"] > 0
