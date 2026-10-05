#!/usr/bin/env python3
"""Checkerboard (split) temporal tiling of the bare Yee stencil, level-tracked.

Per pass of K steps on xy tiles (z untiled, full columns):

1. Black tiles advance in place using only their own tile: a cell component
   is updated to level L only if all its inputs are at exactly the level the
   update needs (H^L reads E^(L-1); E^L reads H^L). Inputs outside the domain
   are zero and always ready; inputs in other tiles never are.
2. White tiles see black cells from black's result B with black's levels and
   white cells (own and diagonal) from A at level 0. They compute only what
   their outputs need (backward demand from their own tile plus the black
   cells they own), and report how far that reaches beyond the tile.

Because every read checks the exact level, an overwritten (too new) or stale
input shows up as a cell that never reaches level K, never as a wrong value.
The merged result is compared bitwise with plain stepping, and component
updates are counted against plain's 6 per cell per step.

    checker.py [--shape NZ NY NX] [--tile TY TX] [--k K] [--passes P]
"""
import argparse
import numpy as np

H, E = ("hx", "hy", "hz"), ("ex", "ey", "ez")
# Inputs of each component as (source, axis, offset), axis 0/1/2 = z/y/x,
# in the order of bare.fut's hupd/eupd: (a, a', b, b').
DEPS = {
    "hx": [("ez", 1, 0), ("ez", 1, 1), ("ey", 0, 0), ("ey", 0, 1)],
    "hy": [("ex", 0, 0), ("ex", 0, 1), ("ez", 2, 0), ("ez", 2, 1)],
    "hz": [("ey", 2, 0), ("ey", 2, 1), ("ex", 1, 0), ("ex", 1, 1)],
    "ex": [("hz", 1, 0), ("hz", 1, -1), ("hy", 0, 0), ("hy", 0, -1)],
    "ey": [("hx", 0, 0), ("hx", 0, -1), ("hz", 2, 0), ("hz", 2, -1)],
    "ez": [("hy", 2, 0), ("hy", 2, -1), ("hx", 1, 0), ("hx", 1, -1)],
}
CH = CB = np.float32(0.5)


def shift(a, axis, off, fill):
    """b[p] = a[p + off] along axis, `fill` outside."""
    if off == 0:
        return a
    b = np.full_like(a, fill)
    n = a.shape[axis]
    src = [slice(None)] * a.ndim
    dst = [slice(None)] * a.ndim
    if off > 0:
        src[axis], dst[axis] = slice(off, n), slice(0, n - off)
    else:
        src[axis], dst[axis] = slice(0, n + off), slice(-off, n)
    b[tuple(dst)] = a[tuple(src)]
    return b


def update(c, v):
    """New value of component c everywhere (bare.fut arithmetic, float32)."""
    a, a1, b, b1 = (shift(v[s], ax, o, np.float32(0)) for s, ax, o in DEPS[c])
    if c in H:
        return v[c] - CH * ((a1 - a) - (b1 - b))
    return v[c] + CB * ((a - a1) - (b - b1))


def plain(v, steps):
    v = dict(v)
    for _ in range(steps):
        for group in (H, E):
            new = {c: update(c, v) for c in group}
            v.update(new)
    return v


def demand(lev, target, K):
    """need[c, L]: cells where component c must be advanced to level L so that
    every `target` cell ends with all components at level K. Backward from
    the targets through DEPS; cells already at a level need nothing more."""
    want = {(c, L): np.zeros_like(target) for c in lev for L in range(K + 1)}
    for c in lev:
        want[c, K] = target.copy()
    need = {}
    for L in range(K, 0, -1):
        for group, inlev in ((E, L), (H, L - 1)):
            for c in group:
                m = want[c, L] & (lev[c] < L)
                need[c, L] = m
                want[c, L - 1] |= m
                for s, ax, o in DEPS[c]:
                    want[s, inlev] |= m if ax == 0 else shift(m, ax - 1, -o, False)
    return need


def advance(v, lev, win, K, count, masks=None):
    """Level-tracked advance inside window `win` (2D bool, xy); greedy, or
    restricted to the `need` masks.

    lev[c] is a 2D int array (levels are uniform along z). Cells outside the
    window may be read only if outside the domain (shift fill = wildcard)."""
    BIG = -10**6
    lw = {c: np.where(win, lev[c], BIG) for c in lev}
    for L in range(1, K + 1):
        for group, need in ((H, L - 1), (E, L)):
            ready = {}
            for c in group:
                r = win & (lw[c] == L - 1)
                if masks is not None:
                    r &= masks[c, L]
                for s, ax, o in DEPS[c]:
                    if ax == 0:      # z neighbour: same column, same level
                        r &= lw[s] == need
                    else:            # xy neighbour; outside the domain is ready
                        r &= shift(lw[s], ax - 1, o, need) == need
                ready[c] = r
            new = {c: update(c, v) for c in group}
            for c in group:
                v[c] = np.where(ready[c][None], new[c], v[c])
                lw[c] = lw[c] + ready[c]
                count[0] += int(ready[c].sum()) * v[c].shape[0]
    for c in lev:
        lev[c] = np.where(win, lw[c], lev[c])


def run(shape, TY, TX, K, passes, verbose=True):
    nz, ny, nx = shape
    rng = np.random.default_rng(1)
    A = {c: rng.standard_normal(shape).astype(np.float32) for c in H + E}
    ref = plain(A, K * passes)
    nty, ntx = -(-ny // TY), -(-nx // TX)
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    tile_of = (jj // TY) * ntx + ii // TX
    black_tile = ((jj // TY) + (ii // TX)) % 2 == 0
    count = [0]
    dup = owned_black = 0
    margin = [0, 0]
    red = [0, 0]
    cur = A
    for _ in range(passes):
        # Phase 1: black tiles, own tile only.
        B = {c: cur[c].copy() for c in cur}
        LB = {c: np.zeros((ny, nx), int) for c in cur}
        for ty in range(nty):
            for tx in range(ntx):
                if (ty + tx) % 2:
                    continue
                win = tile_of == ty * ntx + tx
                v = {c: cur[c].copy() for c in cur}
                lev = {c: np.zeros((ny, nx), int) for c in cur}
                advance(v, lev, win, K, count)
                for c in cur:
                    B[c][:, win] = v[c][:, win]
                    LB[c][win] = lev[c][win]
        done_b = np.logical_and.reduce([LB[c] == K for c in cur]) & black_tile
        # Phase 2: white tiles. Each unfinished black cell is owned by the
        # orthogonal white neighbour across its nearest tile edge (ties: x
        # first), considering only edges with a tile on the other side.
        y0, x0 = (jj // TY) * TY, (ii // TX) * TX
        dist = np.stack([ii - x0, np.minimum(x0 + TX, nx) - 1 - ii,
                         jj - y0, np.minimum(y0 + TY, ny) - 1 - jj])
        exists = np.stack([ii // TX > 0, ii // TX < ntx - 1, jj // TY > 0, jj // TY < nty - 1])
        side = np.argmin(np.where(exists, dist, 10**9), axis=0)
        dty, dtx = np.array([0, 0, -1, 1])[side], np.array([-1, 1, 0, 0])[side]
        owner = np.where(black_tile, tile_of + dty * ntx + dtx, tile_of)
        owner = np.where(done_b, -2, owner)
        out = {c: B[c].copy() for c in cur}
        for ty in range(nty):
            for tx in range(ntx):
                if not (ty + tx) % 2:
                    continue
                t = ty * ntx + tx
                target = owner == t
                v = {c: np.where(black_tile[None], B[c], cur[c]) for c in cur}
                lev = {c: np.where(black_tile, LB[c], 0) for c in cur}
                need = demand(lev, target, K)
                reach = np.logical_or.reduce(list(need.values())) | target
                rj, ri = np.nonzero(reach)
                margin[0] = max(margin[0], ty * TY - rj.min(), rj.max() - ((ty + 1) * TY - 1))
                margin[1] = max(margin[1], tx * TX - ri.min(), ri.max() - ((tx + 1) * TX - 1))
                for (c, L), m in need.items():
                    other = tile_of != t
                    red[0] += int((m & other & ~black_tile).sum()) * nz      # diagonal whites
                    red[1] += int((m & black_tile & (owner != t)).sum()) * nz  # others' black cells
                advance(v, lev, np.ones_like(target), K, count, need)
                fin = np.logical_and.reduce([lev[c] == K for c in cur])
                if not fin[target].all():
                    raise SystemExit(f"white tile {t}: {np.sum(~fin[target])} owned cells unfinished")
                owned_black += int((target & black_tile).sum())
                for c in cur:
                    out[c][:, target] = v[c][:, target]
        cur = out
    diff = sum(int(np.sum(cur[c].view(np.uint32) != ref[c].view(np.uint32))) for c in cur)
    plain_updates = 6 * nz * ny * nx * K * passes
    if verbose:
        print(f"shape {shape} tile {TY}x{TX} K={K} passes={passes}: "
              f"bit-diffs {diff}, updates {count[0] / plain_updates:.4f}x plain, "
              f"black cells finished by white {owned_black / passes:.0f}/pass, "
              f"white reach beyond tile y {margin[0]} x {margin[1]}, "
              f"redundant: in other white tiles {red[0] / plain_updates:.4f}, "
              f"in black cells owned elsewhere {red[1] / plain_updates:.4f}")
    return diff, count[0] / plain_updates


def advance_inplace(v, lev, mask, target, count):
    """Greedy in-place advance of the cells in `mask` (2D bool) towards level
    `target`, reading every other cell at whatever level it holds now: a
    component moves from level l to l + 1 only when each input is at exactly
    the level that update needs (H^(l+1) reads E^l, E^(l+1) reads H^(l+1)).
    Cells outside `mask` stay as they are, as in a single-buffered state where
    the other colour is idle. Returns the number of component-cell updates."""
    OUT = -10**6     # level seen outside the domain: always acceptable
    done = 0
    while True:
        moved = 0
        for group in (H, E):
            ready = {}
            for c in group:
                l = lev[c]
                need = l if c in H else l + 1
                r = mask & (l < target)
                for s, ax, o in DEPS[c]:
                    if ax == 0:      # z neighbour: same column, same level
                        r &= lev[s] == need
                    else:
                        nb = shift(lev[s], ax - 1, o, OUT)
                        r &= (nb == need) | (nb == OUT)
                ready[c] = r
            new = {c: update(c, v) for c in group}
            for c in group:
                v[c] = np.where(ready[c][None], new[c], v[c])
                lev[c] = lev[c] + ready[c]
                n = int(ready[c].sum())
                moved += n
                count[0] += n * v[c].shape[0]
        done += moved
        if not moved:
            return done


def dilate(mask, r):
    """Cells within r of `mask` along y and x (box)."""
    out = mask.copy()
    for _ in range(r):
        grown = out.copy()
        for ax in (0, 1):
            for o in (-1, 1):
                grown |= shift(out, ax, o, False)
        out = grown
    return out


def advance_turn(hist, lev, mask, target, depth, halo, hcomps, stats):
    """One turn: the tiles in `mask` advance to `target`, greedily. Every
    computed level is kept in hist[c][L], and lev[c] is each cell's level.

    Reads: own cells at any level (the tile keeps its levels in rings); other
    cells at most `depth` levels older than they held when the turn began (a
    history of that depth kept at tile borders; 0 is in-place storage).
    The turn may also compute components `hcomps` of other cells within
    `halo` cells of its tiles, privately: those values serve this turn only
    and are discarded. Afterwards a backward pass finds which of them own
    updates actually needed; only those count as redundant work (stats
    "redundant"), with the furthest one's distance (stats "reach")."""
    OUT = -10**6
    win = dilate(mask, halo)
    glev = {c: lev[c].copy() for c in lev}
    llev = {c: lev[c].copy() for c in lev}
    lhist = {c: hist[c].copy() for c in hist}
    own_done = {}
    halo_done = {}
    while True:
        moved = 0
        for L in range(1, target + 1):
            for group in (H, E):
                ready = {}
                for c in group:
                    may = mask | (win & ~mask & (c in hcomps))
                    r = may & (llev[c] == L - 1)
                    need = L - 1 if c in H else L
                    for d, ax, o in DEPS[c]:
                        if ax == 0:
                            nb, ng, own = llev[d], glev[d], mask
                        else:
                            nb = shift(llev[d], ax - 1, o, OUT)
                            ng = shift(glev[d], ax - 1, o, OUT)
                            own = shift(mask, ax - 1, o, True)
                        ok = (nb >= need) & (own | (need >= ng - depth))
                        r &= ok | (nb == OUT)
                    ready[c] = r
                if not any(r.any() for r in ready.values()):
                    continue
                vin = {c: lhist[c][L - 1] for c in group}
                for c in (E if group is H else H):
                    vin[c] = lhist[c][L - 1] if group is H else lhist[c][L]
                for c in group:
                    new = update(c, vin)
                    lhist[c][L] = np.where(ready[c][None], new, lhist[c][L])
                    llev[c] = llev[c] + ready[c]
                    own_done[c, L] = own_done.get((c, L), False) | (ready[c] & mask)
                    halo_done[c, L] = halo_done.get((c, L), False) | (ready[c] & ~mask)
                    moved += int(ready[c].sum())
        if not moved:
            break
    nz = hist[H[0]].shape[1]
    # Which private (halo) values did own updates need, directly or not?
    used = {}
    for L in range(target, 0, -1):
        for group in (E, H):
            for c in group:
                P = own_done.get((c, L), False) | used.get((c, L), False)
                if not np.any(P):
                    continue
                reads = [(c, L - 1, P)]
                need = L - 1 if c in H else L
                for d, ax, o in DEPS[c]:
                    reads.append((d, need, P if ax == 0 else shift(P, ax - 1, -o, False)))
                for d, l, m in reads:
                    if (d, l) in halo_done:
                        used[d, l] = used.get((d, l), False) | (m & halo_done[d, l])
    for (c, L), m in own_done.items():
        stats["updates"] += int(m.sum()) * nz
        hist[c][L] = np.where(m[None], lhist[c][L], hist[c][L])
    for c in lev:
        lev[c] = np.where(mask, llev[c], lev[c])
    for (c, L), m in used.items():
        n = int(m.sum())
        if n:
            stats["redundant"] += n * nz
            stats["comps"].add(c)
            for r in range(1, halo + 1):
                if (m & ~dilate(mask, r - 1) & dilate(mask, r)).any():
                    stats["reach"] = max(stats["reach"], r)


def stagger(shape, TY, TX, steps, lead=1, K=2, verbose=True, depth=None, halo=0, hcomps=H):
    """Time-skewed checkerboard: white tiles advance `lead` steps, then the
    colours take turns advancing K steps each (black to 2, white to 3, black
    to 4, ...) until both reach `steps`. Within a turn every tile of the
    colour advances greedily, reading the other colour's cells at their
    current levels.

    depth=None: one in-place state, own cells only (no history, no halo).
    Otherwise see advance_turn: `depth` levels of border history, and a
    private halo of `halo` cells for components `hcomps` of the other colour.

    Reports, per turn, how many of the colour's component-cells fall short of
    the turn's target, then whether every cell reached `steps`, how many
    extra turns that took, bit equality with plain stepping, and the work."""
    nz, ny, nx = shape
    rng = np.random.default_rng(1)
    A = {c: rng.standard_normal(shape).astype(np.float32) for c in H + E}
    ref = plain(A, steps)
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    black = ((jj // TY) + (ii // TX)) % 2 == 0
    # Distance of each cell from its tile's boundary, to locate the lag.
    edge_dist = np.minimum.reduce([jj % TY, TY - 1 - jj % TY, ii % TX, TX - 1 - ii % TX])
    v = {c: A[c].copy() for c in A}
    hist = {c: np.full((steps + 1,) + shape, np.nan, np.float32) for c in A}
    for c in A:
        hist[c][0] = A[c]
    lev = {c: np.zeros((ny, nx), int) for c in A}
    count = [0]
    stats = {"updates": 0, "redundant": 0, "reach": 0, "comps": set()}
    turns = [("white", min(lead, steps))]
    goal = {"white": min(lead, steps), "black": 0}
    while goal["white"] < steps or goal["black"] < steps:
        for t in ("black", "white"):
            goal[t] = min(goal[t] + K, steps)
            turns.append((t, goal[t]))

    def turn(mask, target):
        if depth is None:
            advance_inplace(v, lev, mask, target, count)
        else:
            advance_turn(hist, lev, mask, target, depth, halo, hcomps, stats)

    log = []
    for colour, target in turns:
        mask = ~black if colour == "white" else black
        turn(mask, target)
        short = {c: mask & (lev[c] < target) for c in lev}
        nshort = sum(int(m.sum()) for m in short.values())
        anyshort = np.logical_or.reduce(list(short.values()))
        reach = int(edge_dist[anyshort].max()) + 1 if anyshort.any() else 0
        lagmax = max(int((target - lev[c])[mask].max()) for c in lev)
        log.append((colour, target, nshort, reach, lagmax))
    # Drain: alternate the colours until nothing moves.
    drain = 0
    while True:
        before = sum(int(lev[c].sum()) for c in lev)
        for m in (black, ~black):
            turn(m, steps)
        if sum(int(lev[c].sum()) for c in lev) == before:
            break
        drain += 2
    if depth is not None:
        v = {c: hist[c][steps] for c in hist}
        count[0] = stats["updates"]
    stuck = sum(int((lev[c] < steps).sum()) for c in lev)
    diff = sum(int(np.sum((v[c].view(np.uint32) != ref[c].view(np.uint32)) & (lev[c] == steps)[None]))
               for c in v)
    plain_updates = 6 * nz * ny * nx * steps
    if verbose:
        print(f"stagger shape {shape} tile {TY}x{TX} lead {lead} K {K} steps {steps} "
              + ("in place, own cells only" if depth is None else
                 f"border history {depth}, halo {halo} ({''.join(sorted(hcomps))})"))
        for t, g, n, d, lm in log:
            print(f"  {t:5s} -> {g:3d}: {n:7d} component-cells short, "
                  f"up to {d} cells from the tile edge, lag up to {lm}")
        print(f"  drain turns after the schedule: {drain}; stuck component-cells: {stuck}; "
              f"bit-diffs among finished cells: {diff}; own updates {count[0] / plain_updates:.4f}x plain"
              + ("" if depth is None else
                 f"; redundant halo updates {stats['redundant'] / plain_updates:.4f}x plain, "
                 f"used up to {stats['reach']} cells out, components {sorted(stats['comps'])}"))
    return stuck, diff, count[0] / plain_updates


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shape", type=int, nargs=3, default=[5, 37, 70])
    p.add_argument("--tile", type=int, nargs=2, default=[4, 8])
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--stagger", type=int, metavar="STEPS",
                   help="model the time-skewed checkerboard for STEPS steps instead")
    p.add_argument("--lead", type=int, default=1, help="white's head start (stagger)")
    p.add_argument("--depth", type=int, help="levels of history kept at tile borders (stagger)")
    p.add_argument("--halo", type=int, default=0, help="private halo width into the other colour (stagger)")
    p.add_argument("--halo-comps", default="H", choices=["H", "E", "EH"],
                   help="which fields the halo may compute (stagger)")
    a = p.parse_args()
    if a.stagger:
        hc = {"H": H, "E": E, "EH": H + E}[a.halo_comps]
        stagger(tuple(a.shape), *a.tile, a.stagger, a.lead, a.k, depth=a.depth, halo=a.halo, hcomps=hc)
    else:
        run(tuple(a.shape), *a.tile, a.k, a.passes)
