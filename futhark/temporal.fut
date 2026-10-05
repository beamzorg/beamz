-- Two-step temporal tiling for the GPU builds (see TEMPORAL_TILING.md).
--
-- The run state is stored in TY x TX tiles in (y, x), whole z columns, as
-- [tile][z][lane][3] arrays, one for E and one for H. A *core* cell lies in a
-- tile quadrant with no CPML slab, source or boundary anywhere in its window
-- (the quadrant plus a two-cell halo); one lean intra-block kernel advances
-- the interior planes of every tile with a core quadrant by two steps,
-- streaming z planes through rings in shared memory. Everything else, the
-- *edge* region (other quadrants and the z caps), takes two plain steps cell
-- by cell with yee.fut's update. A pass from state A runs:
--
--   1. the core kernel, from A into a fresh state C (in the non-core
--      quadrants of its tiles it writes values that step 3 overwrites);
--   2. plain step 1 on the edge cells and the core cells within one cell of
--      them (and those monitors sample), from A into a scratch state T that
--      holds only the quadrants of (tile, plane) items holding such cells;
--   3. plain step 2 on the edge cells, from T into C.
--
-- Step 2's E reads H at t + 2 from C, which the core kernel already holds
-- next to the edge region, and every read of T is of a cell step 1 covered,
-- so all updates are exact and nothing but the margin is computed twice.
-- CPML memory lives in yee.fut's packed slabs; only edge cells touch it.
--
-- Tiles are large (16 x 32) because the core kernel is limited by memory
-- bandwidth, and a larger window re-reads less of its neighbours' halo;
-- quadrants keep the edge region as fine as 8 x 16 tiles did.
--
-- The core kernel writes its result columns straight to global memory
-- (#[intrablock_result_global], from the Futhark checkout BeamZ builds with);
-- a whole column would not fit in shared memory.

import "yee"

def TY : i64 = 16
def TX : i64 = 32
def LY : i32 = 4
def LX : i32 = 5
def WY : i64 = TY + 4
def WX : i64 = TX + 4

-- Run storage extent and tiles per tile row.
type geo = {P0: i64, P1: i64, P2: i64, ntx: i64}

-- E or H in tiled storage.
type tiled [nt][P0] = [nt][P0][TY * TX][3]f32

-- A store of E or H rows ([3] per lane): the whole state as tiled storage,
-- or a compact subset (the scratch state T) stored as [n][1][TY * TX / 4],
-- with a slot per quadrant of an item (tile, plane), at (tile * P0 + plane)
-- * 4 + quadrant. A store comes with a `place`: whether it is compact, and
-- its slots. Plain data rather than a function, so that callers are not
-- specialised per kind of store (which doubled compile time); the flag is a
-- literal that folds after inlining.
type store [a][b][c] = [a][b][c][3]f32
type~ place = (bool, []i32)

def whole : place = (false, [])

-- Lanes of a compact store's slot.
def QL : i64 = TY * TX / 4

-- Quadrant of lane l, and its lane within the quadrant.
def lane_quadrant (l: i64) : (i64, i64) =
  let ly = i32.i64 l >> LX
  let lx = i32.i64 l & (i32.i64 TX - 1)
  let (hy, hx) = (i32.i64 TY / 2, i32.i64 TX / 2)
  in (i64.i32 ((if ly >= hy then 2 else 0) + (if lx >= hx then 1 else 0)),
      i64.i32 (((ly & (hy - 1)) << (LX - 1)) | (lx & (hx - 1))))

-- Where lane l of item (t, k) lives in a store: (item, plane, lane).
def item_at (g: geo) ((compact, slot): place) (t: i64) (k: i64) (l: i64) : (i64, i64, i64) =
  if compact then let (q, m) = lane_quadrant l
                  in (i64.i32 (#[unsafe] slot[(t * g.P0 + k) * 4 + q]), 0, m)
  else (t, k, l)

def instore (g: geo) (k: i64) (j: i64) (i: i64) : bool =
  k >= 0 && k < g.P0 && j >= 0 && j < g.P1 && i >= 0 && i < g.P2

def tile_of (g: geo) (j: i64) (i: i64) : i64 =
  i64.i32 ((i32.i64 j >> LY) * i32.i64 g.ntx + (i32.i64 i >> LX))

def lane (j: i64) (i: i64) : i64 =
  i64.i32 (((i32.i64 j & (i32.i64 TY - 1)) << LX) | (i32.i64 i & (i32.i64 TX - 1)))

-- Core cells are chosen per quadrant of a tile, so that the edge region
-- keeps a finer granularity than the core kernel's tiles. Quadrant 2 * qy +
-- qx of a tile holds rows [qy * TY / 2, (qy + 1) * TY / 2) and columns
-- likewise.
def quadrant (j: i64) (i: i64) : i64 =
  (if (j & (TY - 1)) >= TY / 2 then 2 else 0) + (if (i & (TX - 1)) >= TX / 2 then 1 else 0)

-- Whether storage cell (j, i) of tile t is a core cell, given each tile's
-- mask of core quadrants.
def core_cell [nt] (cmask: [nt]i64) (t: i64) (j: i64) (i: i64) : bool =
  (#[unsafe] cmask[t] >> quadrant j i) & 1 != 0

-- Component c at a storage cell, 0 outside the store.
def fat [a][b][d] (g: geo) (at: place) (f: store [a][b][d]) (c: i64) (k: i64) (j: i64) (i: i64) : f32 =
  if instore g k j i then let (u, v, w) = item_at g at (tile_of g j i) k (lane j i) in #[unsafe] f[u, v, w, c] else 0

-- Component c as a field accessor for yee.fut's update, which reads only
-- inside the store.
def tfield [a][b][d] (g: geo) (at: place) (f: store [a][b][d]) (c: i64) : field =
  \k j i -> let (u, v, w) = item_at g at (tile_of g j i) k (lane j i) in #[unsafe] f[u, v, w, c]

-- Storage coordinates of lane l of item it = tile * P0 + plane.
def item_cell (g: geo) (it: i64) (l: i64) : (i64, i64, i64) =
  let it = i32.i64 it
  let P0 = i32.i64 g.P0
  let t = it / P0
  let ntx = i32.i64 g.ntx
  let l = i32.i64 l
  let ty = t / ntx
  in (i64.i32 (it - t * P0), i64.i32 ((ty << LY) + (l >> LX)), i64.i32 (((t - ty * ntx) << LX) + (l & (i32.i64 TX - 1))))

-- Cells are numbered by their tiled row, (tile * P0 + plane) * TY * TX + lane;
-- every tiled array has fewer than 2^31 of them. (tile, plane, lane, j, i).
def cell_coords (g: geo) (r: i32) : (i64, i64, i64, i64, i64) =
  let it = r >> (LY + LX)
  let t = it / i32.i64 g.P0
  let l = r & (i32.i64 (TY * TX) - 1)
  let (k, j, i) = item_cell g (i64.i32 it) (i64.i32 l)
  in (i64.i32 t, k, i64.i32 l, j, i)

-- Row-major (P0, P1, P2) offsets, as the runtime computes them.
def decode (g: geo) (o: i64) : (i64, i64, i64) =
  let k = o / (g.P1 * g.P2)
  let r = o - k * g.P1 * g.P2
  let j = r / g.P2
  in (k, j, r - j * g.P2)

-- Tiled row of a run-storage offset (-1 for none).
def tiled_row (g: geo) (o: i64) : i64 =
  if o < 0 || o >= g.P0 * g.P1 * g.P2 then -1
  else let (k, j, i) = decode g o in (tile_of g j i * g.P0 + k) * (TY * TX) + lane j i

-- Components (each at most the store) to tiled storage and back. A flat
-- scalar tabulate: a per-cell row would be built transposed and copied, and a
-- multi-dimensional one can fuse into a per-tile kernel that stores it
-- interleaved.
def to_tiled [a0][a1][a2][b0][b1][b2][c0][c1][c2] (g: geo) (nt: i64) (P0: i64)
             (a: [a0][a1][a2]f32) (b: [b0][b1][b2]f32) (c: [c0][c1][c2]f32) : *tiled [nt][P0] =
  let t = tabulate (nt * P0 * (TY * TX) * 3) (\q ->
            let r = q / 3
            let (_, j, i) = item_cell g (r / (TY * TX)) (r % (TY * TX))
            let k = (r / (TY * TX)) % P0
            let at [n0][n1][n2] (v: [n0][n1][n2]f32) =
              if inside (n0, n1, n2) k j i then #[unsafe] v[k, j, i] else 0
            let s = q % 3
            in if s == 0 then at a else if s == 1 then at b else at c)
  in unflatten (unflatten (unflatten t))

-- Component c of tiled storage, cropped to (a, b, d).
def untile [nt][P0] (g: geo) (f: tiled [nt][P0]) (c: i64) (a: i64) (b: i64) (d: i64) : *[a][b][d]f32 =
  tabulate3 a b d (\k j i -> #[unsafe] f[tile_of g j i, k, lane j i, c])

-- Coefficients of the core kernel, which hard-codes the common configuration:
-- isotropic grid, scalar H coefficients and E decay, codebook E sources.
type corecf = {inv: f32, hd0: f32, hd1: f32, hd2: f32, hs0: f32, hs1: f32, hs2: f32,
               ed0: f32, ed1: f32, ed2: f32, ne: (dims, dims, dims)}

-- E source coefficient at an inside cell of a component of dims (_, b, d),
-- from its 8-bit codes and codebook.
def esource (codes: []i32) (table: []f32) ((_, b, d): dims) (k: i64) (j: i64) (i: i64) : f32 =
  let l = i32.i64 ((k * b + j) * d + i)
  let v = (#[unsafe] codes[i64.i32 (l >> 2)] >> (8 * (l & 3))) & 0xff
  in #[unsafe] table[i64.i32 v]

-- Interior updates, as field_value computes them away from slabs and
-- boundaries: v0/w0 sample the first curl term's source at the cell and its
-- neighbour, v1/w1 the second's.
def hcore (inv: f32) (decay: f32) (s: f32) (old: f32) (v0: f32) (w0: f32) (v1: f32) (w1: f32) : f32 =
  f32.fma (-s) ((w0 - v0) * inv - (w1 - v1) * inv) (decay * old)

def ecore (inv: f32) (decay: f32) (s: f32) (old: f32) (v0: f32) (w0: f32) (v1: f32) (w1: f32) : f32 =
  f32.fma s ((v0 - w0) * inv - (v1 - w1) * inv) (decay * old)

-- A ring: n planes of h x w per component. Separate per-component planes, not
-- [n][h][w][3]: reading the interleaved layout costs the core kernel 5%.
type~ ring = ([][][]f32, [][][]f32, [][][]f32)

def zring (n: i64) (h: i64) (w: i64) : *ring =
  let z () = replicate n (replicate h (replicate w 0f32)) in (z (), z (), z ())

-- Component c (a literal, so this folds) of slot s at (y, xx).
def rget (r: ring) (s: i64) (c: i64) (y: i64) (xx: i64) : f32 =
  #[unsafe] (if c == 0 then r.0[s, y, xx] else if c == 1 then r.1[s, y, xx] else r.2[s, y, xx])

-- Slot s of r := planes.
def rset [n][h][w] ((r0, r1, r2): (*[n][h][w]f32, *[n][h][w]f32, *[n][h][w]f32)) (s: i64)
    (planes: [h][w](f32, f32, f32)) : *ring =
  let (a, b, c) = unzip3 (map unzip3 planes)
  let r0[s] = a
  let r1[s] = b
  let r2[s] = c
  in (r0, r1, r2)

-- Two steps of the interior planes [zb, zt) of every tile with a core
-- quadrant (cmask), from state (ae, ah). Other tiles and planes are left
-- unwritten.
--
-- Plane p streams through four stages per iteration: H1 and E1 on p, H2 on
-- q = p - 1, then E2 on q, written out with H2. E0 (plane p of the input)
-- is loaded per iteration; H1, E1 and H2 live in rings of two planes, by
-- parity, each holding only the window part later stages read (window
-- coordinates, tile at [2, TY + 2) x [2, TX + 2)): H1 [0, WY-1) x [0, WX-1),
-- E1 [1, WY-1) x [1, WX-1) and H2 [1, WY-2) x [1, WX-2), so ring reads need
-- no clamping. Rings are read through opaque slot numbers: reading a slot
-- just written would otherwise be forwarded to the new plane, which then
-- can't be built in place (it is built apart and copied, a barrier per
-- copy).
def core_pass [nt][P0] (g: geo) (zb: i64) (zt: i64) (cf: corecf)
    (codes0: []i32) (codes1: []i32) (codes2: []i32) (table0: []f32) (table1: []f32) (table2: []f32)
    (cmask: [nt]i64) (ae: tiled [nt][P0]) (ah: tiled [nt][P0]) : (*tiled [nt][P0], *tiled [nt][P0]) =
  let (inv, hd0, hd1, hd2, hs0, hs1, hs2, ed0, ed1, ed2) =
    (cf.inv, cf.hd0, cf.hd1, cf.hd2, cf.hs0, cf.hs1, cf.hs2, cf.ed0, cf.ed1, cf.ed2)
  let (ne0, ne1, ne2) = cf.ne
  let src c k j i = if c == 0 then esource codes0 table0 ne0 k j i
                    else if c == 1 then esource codes1 table1 ne1 k j i
                    else esource codes2 table2 ne2 k j i
  let run t =
    let y0 = (t / g.ntx) * TY - 2
    let x0 = (t % g.ntx) * TX - 2
    -- Window cells outside the store are clamped into it: they only feed
    -- results in the tile's non-core quadrants, which plain stepping
    -- overwrites.
    let at y xx = (i64.max 0 (i64.min (g.P1 - 1) (y0 + y)), i64.max 0 (i64.min (g.P2 - 1) (x0 + xx)))
    let A c k y xx =
      let (j, i) = at y xx
      in if c < 3 then #[unsafe] ae[tile_of g j i, k, lane j i, c] else #[unsafe] ah[tile_of g j i, k, lane j i, c - 3]
    let o () = #[scratch] replicate P0 (replicate (TY * TX) (replicate 3 0f32))
    let (_, _, _, oe, oh) =
      loop (h1, e1, h2, oe, oh) =
        (zring 2 (WY - 1) (WX - 1), zring 2 (WY - 2) (WX - 2),
         zring 2 (WY - 3) (WX - 3), o (), o ())
      for n < (if #[unsafe] cmask[t] != 0 then zt - zb + 3 else 0) do
        let p = zb - 2 + n
        let q = p - 1
        let (sp, sq) = (p & 1, q & 1)
        let (rp, rq) = opaque (sp, sq)
        let e0 = tabulate_2d WY WX (\y xx -> (A 0 p y xx, A 1 p y xx, A 2 p y xx))
        let (e0x, e0y, e0z) = unzip3 (map unzip3 e0)
        -- Readers in window coordinates.
        let E0 c y xx = #[unsafe] (if c == 0 then e0x[y, xx] else if c == 1 then e0y[y, xx] else e0z[y, xx])
        let h1 = rset h1 sp (tabulate_2d (WY - 1) (WX - 1) (\y xx ->
          let E c = E0 c y xx
          in (hcore inv hd0 hs0 (A 3 p y xx) (E 2) (E0 2 (y + 1) xx) (E 1) (A 1 (p + 1) y xx),
              hcore inv hd1 hs1 (A 4 p y xx) (E 0) (A 0 (p + 1) y xx) (E 2) (E0 2 y (xx + 1)),
              hcore inv hd2 hs2 (A 5 p y xx) (E 1) (E0 1 y (xx + 1)) (E 0) (E0 0 (y + 1) xx))))
        let H1 s c y xx = rget h1 s c y xx
        let e1 = rset e1 sp (tabulate_2d (WY - 2) (WX - 2) (\y xx ->
          let (y, xx) = (y + 1, xx + 1)
          let (j, i) = at y xx
          let H c = H1 rp c y xx
          let H0 c = H1 (1 - rp) c y xx
          let E c = E0 c y xx
          in (ecore inv ed0 (src 0 p j i) (E 0) (H 2) (H1 rp 2 (y - 1) xx) (H 1) (H0 1),
              ecore inv ed1 (src 1 p j i) (E 1) (H 0) (H0 0) (H 2) (H1 rp 2 y (xx - 1)),
              ecore inv ed2 (src 2 p j i) (E 2) (H 1) (H1 rp 1 y (xx - 1)) (H 0) (H1 rp 0 (y - 1) xx))))
        let E1 s c y xx = rget e1 s c (y - 1) (xx - 1)
        let h2 = rset h2 sq (tabulate_2d (WY - 3) (WX - 3) (\y xx ->
          let (y, xx) = (y + 1, xx + 1)
          let E c = E1 rq c y xx
          let En c = E1 rp c y xx
          let H c = H1 rq c y xx
          in (hcore inv hd0 hs0 (H 0) (E 2) (E1 rq 2 (y + 1) xx) (E 1) (En 1),
              hcore inv hd1 hs1 (H 1) (E 0) (En 0) (E 2) (E1 rq 2 y (xx + 1)),
              hcore inv hd2 hs2 (H 2) (E 1) (E1 rq 1 y (xx + 1)) (E 0) (E1 rq 0 (y + 1) xx))))
        let H2 s c y xx = rget h2 s c (y - 1) (xx - 1)
        -- Stage 2's E on the tile, written out with H2 by one map (rows as
        -- literals: a sliced row would become a parallel dimension).
        let out = tabulate (TY * TX) (\l ->
          let (y, xx) = (l / TX + 2, l % TX + 2)
          let (j, i) = at y xx
          let H c = H2 rq c y xx
          let H0 c = H2 (1 - rq) c y xx
          let E c = E1 rq c y xx
          in ([ecore inv ed0 (src 0 q j i) (E 0) (H 2) (H2 rq 2 (y - 1) xx) (H 1) (H0 1),
               ecore inv ed1 (src 1 q j i) (E 1) (H 0) (H0 0) (H 2) (H2 rq 2 y (xx - 1)),
               ecore inv ed2 (src 2 q j i) (E 2) (H 1) (H2 rq 1 y (xx - 1)) (H 0) (H2 rq 0 (y - 1) xx)],
              [H 0, H 1, H 2]))
        let (fe, fh) = unzip out
        let r = i64.max zb q
        let oe[r] = fe
        let oh[r] = fh
        in (h1, e1, h2, oe, oh)
    in (oe, oh)
  let r = #[unsafe] #[flattening(only_intra)] #[intrablock_result_global] tabulate nt run
  in (map (.0) r, map (.1) r)

-- Interior planes, rows or columns along `axis`: no CPML slab of any term and
-- no boundary case of any component (the core kernel's assumptions).
def interior (nh: (dims, dims, dims)) (ne: (dims, dims, dims)) (slabs: [12][2]i64)
             (axis: i64) (extent: i64) : (i64, i64) =
  let ncomp (n: (dims, dims, dims)) c = if c == 0 then n.0 else if c == 1 then n.1 else n.2
  -- Output component and derivative axis of CPML term t.
  let term t = let c = (t % 6) / 2
               let a = if c == 0 then (if t % 2 == 0 then 1 else 0)
                       else if c == 1 then (if t % 2 == 0 then 0 else 2)
                       else (if t % 2 == 0 then 2 else 1)
               in (ncomp (if t < 6 then nh else ne) c, a)
  let terms = filter (\t -> (term t).1 == axis) (iota 12)
  let lo = i64.maximum (map (\t -> slabs[t, 0]) terms)
  let hs = i64.minimum (map (\t -> if slabs[t, 1] > 0 then axis_size axis (term t).0 - slabs[t, 1]
                                   else extent) terms)
  in (i64.max 1 lo, i64.min (extent - 3) (hs - 1))

-- The core planes [zb, zt) and per tile its mask of core quadrants (a
-- quadrant is core when its window, the quadrant plus a two-cell halo,
-- avoids every boundary and slab). `busy` marks quadrants (tile * 4 +
-- quadrant) whose window holds a source cell. Without `core`, every cell is
-- an edge cell.
def make_plan (g: geo) (nh: (dims, dims, dims)) (ne: (dims, dims, dims)) (slabs: [12][2]i64)
              (nt: i64) (busy: []bool) (core: bool) : (i64, i64, [nt]i64) =
  let (z0, z1) = interior nh ne slabs 0 g.P0
  let (y0, y1) = interior nh ne slabs 1 g.P1
  let (x0, x1) = interior nh ne slabs 2 g.P2
  -- Stage 1 runs two planes ahead of the first core plane and on the last.
  let (zb, zt) = (z0 + 2, z1)
  let core_quadrant t q =
    let wy = (t / g.ntx) * TY + (q / 2) * (TY / 2) - 2
    let wx = (t % g.ntx) * TX + (q % 2) * (TX / 2) - 2
    in core && zt - zb >= 4 && !(#[unsafe] busy[t * 4 + q])
       && wy >= y0 && wy + TY / 2 + 3 <= y1 && wx >= x0 && wx + TX / 2 + 3 <= x1
  in (zb, zt, tabulate nt (\t -> loop m = 0 for q < 4 do if core_quadrant t q then m | (1 << q) else m))

-- Quadrants (tile * 4 + quadrant) whose window holds one of the target
-- cells.
def busy_quadrants [K] (g: geo) (nt: i64) (target: [K]i32) : [nt * 4]bool =
  let (QY, QX) = (TY / 2, TX / 2)
  let hits =
    flatten
      (map (\o ->
              let (_, j, i) = if o < 0 then (-1, -1, -1) else decode g (i64.i32 o)
              let at qy qx =
                let ok = o >= 0 && qy >= 0 && qy * QY < g.P1 && qx >= 0 && qx * QX < g.ntx * TX
                         && j >= qy * QY - 2 && j < qy * QY + QY + 2
                         && i >= qx * QX - 2 && i < qx * QX + QX + 2
                in if ok then ((qy / 2) * g.ntx + qx / 2) * 4 + (qy % 2) * 2 + qx % 2 else -1
              let (qy0, qy1, qx0, qx1) = ((j - 2) / QY, (j + 2) / QY, (i - 2) / QX, (i + 2) / QX)
              in [at qy0 qx0, at qy0 qx1, at qy1 qx0, at qy1 qx1])
           target)
  in scatter (replicate (nt * 4) false) hits (map (const true) hits)

-- The rows r < n with p r, in an array of exactly that size (filter's result
-- keeps a block of n elements).
def compact (n: i64) (p: i32 -> bool) : []i32 =
  let flags = tabulate n (p <-< i32.i64)
  let offs = scan (+) 0 (map i64.bool flags)
  let m = if n == 0 then 0 else offs[n - 1]
  in scatter (replicate m 0) (map2 (\f o -> if f then o - 1 else -1) flags offs) (map i32.i64 (iota n))

-- The cells of the two plain steps of a pass, in the store (cells outside
-- it are never read). Step 2 covers the edge region. Its H reads step 1's E
-- at offsets 0 and +1 (along any axis) from edge cells, and that E reads H at
-- 0 and -1, so step 1 also covers every core cell within one cell of the edge
-- region in each axis, and every cell a monitor samples together with its
-- lower neighbours.
def edge_cells [nt] (g: geo) (zb: i64) (zt: i64) (cmask: [nt]i64) (samples: []i32) : ([]i32, []i32) =
  let edge t k j i = !(core_cell cmask t j i) || k < zb || k >= zt
  let tile_at j i = if instore g 0 j i then tile_of g j i else -1
  let n = nt * g.P0 * (TY * TX)
  let watched =
    let cells = flatten (map (\o -> if o < 0 then [-1, -1, -1, -1]
                                    else let (k, j, i) = decode g (i64.i32 o)
                                         let at k j i = if instore g k j i
                                                        then (tile_of g j i * g.P0 + k) * (TY * TX) + lane j i
                                                        else -1
                                         in [at k j i, at (k - 1) j i, at k (j - 1) i, at k j (i - 1)])
                             samples)
    in scatter (replicate n false) cells (map (const true) cells)
  let first r =
    let (t, k, _, j, i) = cell_coords g r
    let near jj ii = let u = tile_at jj ii in u < 0 || !(core_cell cmask u jj ii)
    in instore g k j i
       && (edge t k j i || k - 1 < zb || k + 1 >= zt
           || near (j - 1) (i - 1) || near (j - 1) (i + 1) || near (j + 1) (i - 1) || near (j + 1) (i + 1)
           || #[unsafe] watched[i64.i32 r])
  let second r = let (t, k, _, j, i) = cell_coords g r in instore g k j i && edge t k j i
  in (compact n first, compact n second)

-- Slots of a compact store holding the item quadrants ((tile * P0 + plane) *
-- 4 + quadrant) of `cells`: their count u and per item quadrant its slot, or
-- u (a spare row, read by updates whose results go unused) for none.
def item_slots (nitems: i64) (cells: []i32) : (i64, [nitems * 4]i32) =
  let at r = i64.i32 (r >> (LY + LX)) * 4 + (lane_quadrant (i64.i32 (r & (i32.i64 (TY * TX) - 1)))).0
  let hit = scatter (replicate (nitems * 4) false) (map at cells) (map (const true) cells)
  let offs = scan (+) 0 (map i32.bool hit)
  let u = if nitems == 0 then 0 else i64.i32 offs[nitems * 4 - 1]
  in (u, map2 (\h o -> if h then o - 1 else i32.i64 u) hit offs)

-- One phase of plain stepping on `cells`, as phase_next and psi_next compute
-- it: the outputs go to `dst` and the advanced CPML memories of the six curl
-- terms to q (the old ones are read from psi). `old` holds the outputs'
-- previous values and `f` the curl inputs; each store comes with its place.
-- Every slab cell must be among the cells, as q is written only there.
def edge_phase [a0][b0][c0][a1][b1][c1][a2][b2][c2][L][n]
    (g: geo) (cells: [n]i32) (phase: i64) (edges: i64) (scale: i64 -> i64 -> f32)
    (nn: (dims, dims, dims)) (lold: place) (old: store [a0][b0][c0])
    (decay: ([][][]f32, [][][]f32, [][][]f32))
    (source: (([][][]f32, []f32, []i32), ([][][]f32, []f32, []i32), ([][][]f32, []f32, []i32)))
    (m: (dims, dims, dims)) (lf: place) (f: store [a1][b1][c1])
    (slabs: [12][2]i64) (ca: [12][L]f32) (cb: [12][L]f32) (ck: [12][L]f32)
    (psi: ([][][]f32, [][][]f32, [][][]f32, [][][]f32, [][][]f32, [][][]f32))
    (ldst: place) (dst: *store [a2][b2][c2])
    (q0: *[][][]f32) (q1: *[][][]f32) (q2: *[][][]f32) (q3: *[][][]f32) (q4: *[][][]f32) (q5: *[][][]f32)
    : (*store [a2][b2][c2], *[][][]f32, *[][][]f32, *[][][]f32, *[][][]f32, *[][][]f32, *[][][]f32) =
  let value c = field_cpml phase c edges scale
  let F = tfield g lf f
  let O = tfield g lold old
  -- Where term t's memory of a cell of component dims nc lives (or -1s).
  let slot (t: i64) (axis: i64) (nc: dims) k j i =
    let p = packed (axis_coord axis k j i) (axis_size axis nc) (#[unsafe] slabs[t, 0]) (#[unsafe] slabs[t, 1])
    in if p < 0 || !(inside nc k j i) then (-1, -1, -1) else with_coord axis p k j i
  let t0 = 6 * phase
  let (rows, vals, mems, at) =
    unzip4 (map (\r ->
                   let (t, k, l, j, i) = cell_coords g r
                   let (a, a0, a1) = #[unsafe] value 0 nn.0 (O 0) decay.0 source.0 (F 2) m.2 1 (F 1) m.1 0
                                                    slabs ca cb ck psi.0 psi.1 k j i
                   let (b, b0, b1) = #[unsafe] value 1 nn.1 (O 1) decay.1 source.1 (F 0) m.0 0 (F 2) m.2 2
                                                    slabs ca cb ck psi.2 psi.3 k j i
                   let (c, c0, c1) = #[unsafe] value 2 nn.2 (O 2) decay.2 source.2 (F 1) m.1 2 (F 0) m.0 1
                                                    slabs ca cb ck psi.4 psi.5 k j i
                   in ((k, j, i), [a, b, c], (a0, a1, b0, b1, c0, c1), item_at g ldst t k l))
                cells)
  let mem t axis nc = map (\(k, j, i) -> slot t axis nc k j i) rows
  in (scatter_3d dst at vals,
      scatter_3d q0 (mem t0 1 nn.0) (map (.0) mems), scatter_3d q1 (mem (t0 + 1) 0 nn.0) (map (.1) mems),
      scatter_3d q2 (mem (t0 + 2) 0 nn.1) (map (.2) mems), scatter_3d q3 (mem (t0 + 3) 2 nn.1) (map (.3) mems),
      scatter_3d q4 (mem (t0 + 4) 2 nn.2) (map (.4) mems), scatter_3d q5 (mem (t0 + 5) 1 nn.2) (map (.5) mems))

-- Add the source groups `base`, `base + 1` and `base + 2` (the three
-- components of E or H, at one timing) to tiled storage: one histogram over
-- its rows, with -0 for the components an entry does not touch, so that they
-- stay exactly as they were. `rows` holds each entry's tiled row (-1 for
-- none).
def tinject [a][b][d][K] (g: geo) (at: place) (f: *store [a][b][d]) (step: i64) (base: i32)
    (groups: [K]i32) (rows: [K]i64) (amplitude: [K]f32) (offset: [K]i32)
    (length: [K]i32) (waves: []f32) : *store [a][b][d] =
  let (targets, values) =
    unzip (map5 (\grp r amp off len ->
                   if grp < base || grp > base + 2 || r < 0 then ((-1, -1, -1), [-0, -0, -0])
                   else let t = i64.max 0 (i64.min step (i64.i32 len - 1))
                        let v = amp * #[unsafe] waves[i64.i32 off + t]
                        let row = r / (TY * TX)
                        let c = grp - base
                        in (item_at g at (row / g.P0) (row % g.P0) (r % (TY * TX)),
                            [if c == 0 then v else -0, if c == 1 then v else -0, if c == 2 then v else -0]))
                groups rows amplitude offset length)
  in reduce_by_index_3d f (map2 (+)) [-0, -0, -0] targets values
