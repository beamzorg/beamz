"""Integer dependency oracle for the experimental rolling-plane schedule.

Every staged value carries its exact component, time version, and unwrapped
coordinate. This checks schedule/storage correctness independently of floating
point arithmetic and CUDA execution; it does not prove device synchronization.
"""

import pytest

from scripts.cuda_stencil_trace import trace_yee_step


def _verify_ring_schedule(tile, depth, *, planes=2, shift_e=True):
    tz, ty, tx = tile
    stages = 2 * depth
    shapes = [
        (tx + stages - 1 - stage, ty + stages - 1 - stage) for stage in range(stages)
    ]
    rings = [{} for _ in range(stages - 1)]
    updates = [
        trace_yee_step(stage // 2)[(stage % 2) * 3 : (stage % 2 + 1) * 3]
        for stage in range(stages)
    ]
    written = set()
    for wave in range(tz + stages - 1):
        for stage in range(stages):
            if wave < stage:
                continue
            sx, sy = shapes[stage]
            phase = stage % 2
            z = wave - depth - stage // 2
            for iy in range(sy):
                for ix in range(sx):
                    coordinate = (
                        z,
                        -depth + (stage + 1) // 2 + iy,
                        -depth + (stage + 1) // 2 + ix,
                    )
                    for update in updates[stage]:
                        for read in update.reads:
                            dz, dy, dx = read.offset
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
                            if read.field[0] == update.output.field[0]:
                                if stage < 2:
                                    # Immutable initial-bank reads are version 0.
                                    assert read.version == 0
                                    continue
                                producer = stage - 2
                                ry, rx = iy + dy + 1, ix + dx + 1
                            else:
                                if stage == 0:
                                    assert read.version == 0
                                    continue
                                producer = stage - 1
                                shift = phase if shift_e else 0
                                ry, rx = iy + dy + shift, ix + dx + shift
                            assert 0 <= rx < shapes[producer][0]
                            assert 0 <= ry < shapes[producer][1]
                            actual = rings[producer].get(
                                (read.field, (z + dz) % planes, ry, rx)
                            )
                            assert actual == expected, (
                                tile,
                                depth,
                                wave,
                                stage,
                                actual,
                                expected,
                            )
                        value = (
                            update.output.field,
                            update.output.version,
                            coordinate,
                        )
                        if stage < stages - 1:
                            rings[stage][(update.output.field, z % planes, iy, ix)] = (
                                value
                            )
                        else:
                            hfield = "H" + update.output.field[1]
                            actual = rings[stage - 1].get(
                                (hfield, z % planes, iy + 1, ix + 1)
                            )
                            assert actual == (hfield, depth, coordinate)
                            assert all(
                                0 <= p < extent
                                for p, extent in zip(coordinate, tile, strict=True)
                            )
                            assert value not in written
                            written.add(value)
    expected_writes = {
        ("E" + component, depth, (z, y, x))
        for component in "xyz"
        for z in range(tz)
        for y in range(ty)
        for x in range(tx)
    }
    assert written == expected_writes


@pytest.mark.parametrize("tile", [(1, 1, 1), (2, 3, 5), (8, 8, 16)])
@pytest.mark.parametrize("depth", [1, 2, 3, 4, 8])
def test_rolling_rings_preserve_every_dependency_and_owned_output(tile, depth):
    _verify_ring_schedule(tile, depth)


@pytest.mark.parametrize("corruption", [{"planes": 1}, {"shift_e": False}])
def test_ring_oracle_rejects_overwritten_planes_and_wrong_stage_origins(corruption):
    with pytest.raises(AssertionError):
        _verify_ring_schedule((2, 3, 5), 2, **corruption)
