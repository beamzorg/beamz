"""Production-coordinate proof for the experimental packed D2 register core.

Models actual staggered component extents, clamped discarded halo loads,
quadrant ownership, segmented z windows and two-plane asynchronous input-ring
reuse. Tokens propagate bad coordinates instead of accepting a relabeled halo.
This is a host dependency proof, not a CUDA memory-order or racecheck substitute.
"""

from types import SimpleNamespace

import pytest

from beamz.lattice import component_shapes
from beamz.simulation.cuda.mstep import build_mstep_geometry
from scripts.cuda_stencil_trace import trace_yee_step

FIELDS = ("Hx", "Hy", "Hz", "Ex", "Ey", "Ez")
IDS = {name: c for c, name in enumerate(FIELDS)}
H_READS = tuple(update.reads for update in trace_yee_step(0)[:3])
E_READS = tuple(update.reads for update in trace_yee_step(0)[3:])


def _geometry(shape, pml):
    supports = component_shapes(shape)
    shapes = tuple(supports[name] for name in FIELDS)
    terms = (
        tuple(
            SimpleNamespace(
                axis=axis, component=name, slab=SimpleNamespace(low=pml, high=pml)
            )
            for name in FIELDS
            for axis in {"x": (1, 0), "y": (0, 2), "z": (2, 1)}[name[1]]
        )
        if pml
        else ()
    )
    return build_mstep_geometry(
        shapes, SimpleNamespace(h_terms=terms, e_terms=()), (None,) * 9
    ), shapes


def _verify_tile(shapes, tile, origin, bounds, mask, *, early_prefetch=False):  # noqa: C901
    # Keep the four ordered phases and input reuse together for schedule review.
    tx, ty = tile
    ox, oy = origin
    zb, zt = bounds
    min_y, min_x = (min(shape[a] for shape in shapes) for a in (1, 2))
    input_ring, h1, e1, h2 = {}, {}, {}, {}
    outputs = set()

    def global_read(c, z, qy, qx):
        # Exact native admission: only discarded transverse halos may clamp.
        y, x = min(max(oy - 2 + qy, 0), min_y - 1), min(max(ox - 2 + qx, 0), min_x - 1)
        nz, ny, nx = shapes[c]
        linear = (z * ny + y) * nx + x
        assert 0 <= z < nz and 0 <= linear < nz * ny * nx
        dz, rest = divmod(linear, ny * nx)
        dy, dx = divmod(rest, nx)
        assert (dz, dy, dx) == (z, y, x)
        return c, 0, (z, y, x), True

    def fill_plane(z):
        for y in range(ty + 4):
            for x in range(tx + 4):
                for c in range(3):
                    input_ring[c, z & 1, y, x] = global_read(c + 3, z, y, x)

    def check_reads(c, version, coordinate, getter):
        valid = True
        contract = H_READS[c] if c < 3 else E_READS[c - 3]
        for read in contract:
            parent = getter(IDS[read.field], read.offset)
            # E2/H2 reuse the one-step contract at relative version +1.
            expected_version = read.version + version - 1
            expected = (
                IDS[read.field],
                expected_version,
                tuple(a + b for a, b in zip(coordinate, read.offset, strict=True)),
            )
            valid &= parent is not None and parent[:3] == expected and parent[3]
        return c, version, coordinate, valid

    fill_plane(zb - 2)
    fill_plane(zb - 1)
    for wave in range(zt - zb + 3):
        # Every register-like value is recreated within this wave. Only shared
        # planes and the asynchronous input ring carry state across waves.
        current_h, current_e, old_h, old_e = {}, {}, {}, {}
        p = zb + wave - 2
        for y in range(ty + 3):
            for x in range(tx + 3):
                for c in range(3):
                    if wave:
                        old_h[c, y, x] = h1[c, y, x]

                    def first_h(field, offset, c=c, y=y, x=x, p=p):
                        dz, dy, dx = offset
                        if field < 3:
                            return global_read(field, p, y, x)
                        return input_ring.get((field - 3, (p + dz) & 1, y + dy, x + dx))

                    current_h[c, y, x] = check_reads(
                        c, 1, (p, oy - 2 + y, ox - 2 + x), first_h
                    )
                    h1[c, y, x] = current_h[c, y, x]
        if early_prefetch and wave + 1 < zt - zb + 3:
            fill_plane(p + 2)
        if wave >= 1:
            for y in range(1, ty + 3):
                for x in range(1, tx + 3):
                    for c in range(3):

                        def first_e(field, offset, c=c, y=y, x=x, p=p, old_h=old_h):
                            dz, dy, dx = offset
                            if field >= 3:
                                return input_ring.get((field - 3, p & 1, y, x))
                            if dz:
                                return old_h.get((field, y, x))
                            return h1.get((field, y + dy, x + dx))

                        current_e[c, y, x] = check_reads(
                            c + 3, 1, (p, oy - 2 + y, ox - 2 + x), first_e
                        )
        if not early_prefetch and wave + 1 < zt - zb + 3:
            fill_plane(p + 2)
        if wave >= 2:
            for y in range(1, ty + 2):
                for x in range(1, tx + 2):
                    for c in range(3):
                        old_e[c, y, x] = e1[c, y, x]
                    for c in range(3):

                        def second_h(
                            field,
                            offset,
                            c=c,
                            y=y,
                            x=x,
                            old_h=old_h,
                            current_e=current_e,
                        ):
                            dz, dy, dx = offset
                            if field < 3:
                                return old_h.get((field, y, x))
                            component = field - 3
                            if dz:
                                return current_e.get((component, y, x))
                            return e1.get((component, y + dy, x + dx))

                        next_value = check_reads(
                            c, 2, (p - 1, oy - 2 + y, ox - 2 + x), second_h
                        )
                        if wave >= 3:
                            old_h[c, y, x] = h2[c, y, x]
                        current_h[c, y, x] = next_value
            # Block publication occurs only after all H2 old-plane reads.
            for y in range(1, ty + 2):
                for x in range(1, tx + 2):
                    for c in range(3):
                        h2[c, y, x] = current_h[c, y, x]
        if wave >= 1:
            for y in range(1, ty + 3):
                for x in range(1, tx + 3):
                    for c in range(3):
                        e1[c, y, x] = current_e[c, y, x]
        if wave >= 3:
            for y in range(2, ty + 2):
                for x in range(2, tx + 2):
                    quadrant = (2 if y - 2 >= 8 else 0) + (1 if x - 2 >= 16 else 0)
                    if not mask & (1 << quadrant):
                        continue
                    for c in range(3):

                        def second_e(
                            field, offset, c=c, y=y, x=x, old_e=old_e, old_h=old_h
                        ):
                            dz, dy, dx = offset
                            if field >= 3:
                                return old_e.get((field - 3, y, x))
                            if dz:
                                return old_h.get((field, y, x))
                            return h2.get((field, y + dy, x + dx))

                        coordinate = p - 1, oy - 2 + y, ox - 2 + x
                        token = check_reads(c + 3, 2, coordinate, second_e)
                        assert token[3], (
                            tile,
                            origin,
                            bounds,
                            mask,
                            "invalid E dependency",
                            token,
                        )
                        assert current_h[c, y, x] == (c, 2, coordinate, True)
                        for field in (c, c + 3):
                            # Final global output/material offsets use p-1.
                            assert global_read(field, p - 1, y, x)[2] == coordinate
                            output = field, coordinate
                            assert output not in outputs
                            outputs.add(output)
    expected = {
        (field, (z, oy + y, ox + x))
        for field in range(6)
        for z in range(zb, zt)
        for y in range(ty)
        for x in range(tx)
        if mask & (1 << ((2 if y >= 8 else 0) + (1 if x >= 16 else 0)))
    }
    assert outputs == expected
    return len(outputs)


@pytest.mark.parametrize("tile", [(32, 16), (32, 8), (16, 16), (16, 8)])
@pytest.mark.parametrize("shape,pml", [((29, 41, 73), 0), ((37, 65, 97), 12)])
def test_native_register_mapping_with_staggered_extents_and_partial_masks(
    tile, shape, pml
):
    geometry, shapes = _geometry(shape, pml)
    positions = {}
    for row, line in enumerate(geometry.core_masks):
        for column, mask in enumerate(line):
            if mask:
                positions.setdefault(int(mask), (column, row))
    assert positions
    visited = 0
    tx, ty = tile
    for mask, (column, row) in positions.items():
        for oy in range(row * 16, row * 16 + 16, ty):
            for ox in range(column * 32, column * 32 + 32, tx):
                shift = ((oy % 16) // 8) * 2 + (ox % 32) // 16
                shifted = mask >> shift
                if not shifted:
                    continue
                zb, zt = map(int, geometry.bounds)
                for bounds in ((zb, min(zb + 3, zt)), (zt - 1, zt)):
                    visited += _verify_tile(shapes, tile, (ox, oy), bounds, shifted)
    assert visited > 0


def test_async_prefetch_before_own_e1_read_is_rejected():
    geometry, shapes = _geometry((29, 41, 73), 0)
    with pytest.raises(AssertionError, match="invalid E dependency"):
        _verify_tile(
            shapes,
            (16, 8),
            (32, 16),
            (int(geometry.bounds[0]), int(geometry.bounds[0]) + 3),
            1,
            early_prefetch=True,
        )
