"""Independent integer proof of inactive-version pruning and late startup."""

import importlib.util
import itertools
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OWNED = (7, 15, 31)


def supports(n):
    z, y, x = n
    return (
        (z, y, x + 1),
        (z, y + 1, x),
        (z + 1, y, x),
        (z + 1, y + 1, x),
        (z + 1, y, x + 1),
        (z, y + 1, x + 1),
    )


def valid(shape, p):
    return all(0 <= a < b for a, b in zip(p, shape, strict=False))


def window(origin, maximum, depth):
    first = max(1, *(a - m + 1 for a, m in zip(origin, maximum, strict=False)))
    last = min(depth, *(a + b - 1 for a, b in zip(origin, OWNED, strict=False)))
    return first, last


@pytest.mark.parametrize(
    "extent,depth",
    [((3, 5, 9), 4), ((17, 19, 37), 4), ((33, 49, 65), 16), ((64, 96, 128), 16)],
)
def test_valid_owner_versions_and_late_ghosts(extent, depth):
    shapes = supports(extent)
    maximum = tuple(max(s[a] for s in shapes) for a in range(3))
    counts = tuple(
        1 + (n + depth + b - 1) // b for n, b in zip(extent, OWNED, strict=False)
    )
    late = empty = live = 0
    # Rectangle intersection gives an independent valid-owned test per field;
    # no use of the proposed first/last formula in this oracle.
    for tile in itertools.product(*(range(c) for c in counts)):
        origin = tuple(a * b for a, b in zip(tile, OWNED, strict=False))
        first, last = window(origin, maximum, depth)
        empty += first > last
        for n in range(1, depth + 1):
            for shape in shapes:
                intersects = all(
                    max(0, o - n) < min(size, o + b - n)
                    for o, b, size in zip(origin, OWNED, shape, strict=False)
                )
                if intersects:
                    assert first <= n <= last
                    live += 1
        if not (1 < first <= last):
            continue
        late += 1
        prev = first - 1
        # First omitted previous owned volume is disjoint from every support.
        assert all(
            any(
                max(0, o - prev) >= min(size, o + b - prev)
                for o, b, size in zip(origin, OWNED, shape, strict=False)
            )
            for shape in shapes
        )
        # At late startup only the low ghost union is imported. Every valid
        # cache position is such a ghost, and its producer is a lower rank
        # with that exact version inside its own active window.
        for relative in itertools.product(*(range(b + 1) for b in OWNED)):
            p = tuple(o - prev - 1 + r for o, r in zip(origin, relative, strict=False))
            if not any(valid(shape, p) for shape in shapes):
                continue
            lower = tuple(int(r == 0) for r in relative)
            assert any(lower)
            producer = tuple(t - d for t, d in zip(tile, lower, strict=False))
            assert min(producer) >= 0
            assert sum(producer) < sum(tile)
            producer_origin = tuple(
                t * b for t, b in zip(producer, OWNED, strict=False)
            )
            pfirst, plast = window(producer_origin, maximum, depth)
            assert pfirst <= prev <= plast
            # Import maps to a high-union owned point of that producer.
            local = tuple(
                c - (o - prev) for c, o in zip(p, producer_origin, strict=False)
            )
            assert all(0 <= a < b for a, b in zip(local, OWNED, strict=False))
            assert any(a == b - 1 for a, b in zip(local, OWNED, strict=False))
    assert live and empty
    if depth == 16:
        assert late


def test_patch_keeps_actual_versions_and_fast_startup():
    spec = importlib.util.spec_from_file_location(
        "time_window", ROOT / "scripts/patch_cuda_native_time_window.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The reusable generator emits the same register-frame source structure.
    source = (ROOT / "scripts/cuda_native_register_frame_body.inc").read_text()
    prefix = "constexpr size_t SharedBytes=3ULL*PAD*sizeof(float)+8*sizeof(size_t)+sizeof(int)+6ULL*LowXPoints*sizeof(float);\n"
    patched = module.patch(prefix + source)
    assert "ImportPacket<Depth,false>(p,bases,prev" in patched
    assert "for(int n=(Fast?1:window[0]);n<=(Fast?Depth:window[1]);++n)" in patched
    assert "if(n==Depth)" in patched
    assert "if constexpr(Fast){" in patched
    assert "if(!threadIdx.x)*phase_step=first" in patched
    assert patched.index("if(window[0]>window[1])return") > patched.index(
        "p.owner_tags[owner]=epoch"
    )
    assert patched.index("if constexpr(!Fast)__syncthreads()") < patched.index(
        "ImportPacket<Depth,false>(p,bases,prev"
    )


def test_negative_skipped_endpoint_and_zero_late_ghost_have_witnesses():
    shapes = supports((33, 49, 65))
    maximum = (34, 50, 66)
    # This high-Z tile first enters at n=2. Removing that first version loses
    # an actual support point, and pretending its previous cache is all zero
    # loses a valid prior-version low ghost.
    origin = (35, 15, 31)
    first, last = window(origin, maximum, 16)
    assert first == 2
    assert any(valid(shape, (33, 13, 29)) for shape in shapes)
    prev = first - 1
    ghost = (origin[0] - prev - 1, origin[1] - prev, origin[2] - prev)
    assert any(valid(shape, ghost) for shape in shapes)
    assert ghost == (33, 14, 30)
    assert first <= last
    # A low tile's final useful owned point is p=0 at n=Bz-1. Dropping last
    # incorrectly prunes it even though later owned Z intervals are all below0.
    first, last = window((0, 15, 31), maximum, 16)
    assert last == 6
    assert any(valid(shape, (0, 9, 25)) for shape in shapes)
