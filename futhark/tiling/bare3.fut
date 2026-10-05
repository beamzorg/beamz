-- Lean full-column 2.5D streaming (see bare2.fut): explicit K = 1 and K = 2
-- stages whose planes rotate through loop-carried tuples (pointer swaps, no
-- shared-memory copies), and one tabulate per output plane.
-- Layout: [tile][z][component][TY * TX]. Needs patch_cuda.py on CUDA.

import "bare"

type stiled [nt][nz] = [nt][nz][TY * TX][6]f32

def to_stiled [nz][ny][nx] (nty: i64) (ntx: i64) (s: fields ([nz][ny][nx]f32)) : *stiled [nty * ntx][nz] =
  let cs = [s.0, s.1, s.2, s.3, s.4, s.5]
  in tabulate_3d (nty * ntx) nz (TY * TX) (\t k q ->
       tabulate 6 (\c -> get cs[c] k ((t / ntx) * TY + q / TX) ((t % ntx) * TX + q % TX)))

def from_stiled [nt][nz] (ntx: i64) (ny: i64) (nx: i64) (a: stiled [nt][nz]) : fields ([nz][ny][nx]f32) =
  let u c = tabulate_3d nz ny nx (\k j i -> #[unsafe] a[(j / TY) * ntx + i / TX, k, (j % TY) * TX + i % TX, c])
  in (u 0, u 1, u 2, u 3, u 4, u 5)

type p3 = (plane, plane, plane)

-- Window geometry of one tile, with i32 arithmetic.
type geo = {nz: i32, ny: i32, nx: i32, ntx: i32, y0: i32, x0: i32}

def inside (g: geo) (z: i32) (y: i64) (x: i64) : bool =
  let gy = g.y0 + i32.i64 y
  let gx = g.x0 + i32.i64 x
  in z >= 0 && z < g.nz && gy >= 0 && gy < g.ny && gx >= 0 && gx < g.nx

def at [nt][nz] (g: geo) (a: stiled [nt][nz]) (c: i64) (z: i32) (y: i64) (x: i64) : f32 =
  let gy = g.y0 + i32.i64 y
  let gx = g.x0 + i32.i64 x
  in if inside g z y x
     then #[unsafe] a[i64.i32 ((gy / i32.i64 TY) * g.ntx + gx / i32.i64 TX), i64.i32 z,
                      i64.i32 ((gy % i32.i64 TY) * i32.i64 TX + gx % i32.i64 TX), c]
     else 0

def load3 [nt][nz] (g: geo) (a: stiled [nt][nz]) (c: i64) (z: i32) : (*plane, *plane, *plane) =
  let v = tabulate_2d WY WX (\y x -> (at g a c z y x, at g a (c + 1) z y x, at g a (c + 2) z y x))
  in (map (map (.0)) v, map (map (.1)) v, map (map (.2)) v)

-- H at plane p from E planes p (with in-plane neighbours) and p + 1; the old
-- H comes from `hself`.
def hstage (g: geo) (p: i32) (hself: i64 -> i64 -> (f32, f32, f32))
           ((ex, ey, ez): p3) ((ex1, ey1, _): p3) : (*plane, *plane, *plane) =
  let y1 y = i64.min (y + 1) (WY - 1)
  let x1 x = i64.min (x + 1) (WX - 1)
  let v = tabulate_2d WY WX (\y x ->
            if !(inside g p y x) then (0, 0, 0)
            else let (hx, hy, hz) = hself y x
                 in #[unsafe]
                    (hupd ch hx ez[y, x] ez[y1 y, x] ey[y, x] ey1[y, x],
                     hupd ch hy ex[y, x] ex1[y, x] ez[y, x] ez[y, x1 x],
                     hupd ch hz ey[y, x] ey[y, x1 x] ex[y, x] ex[y1 y, x]))
  in (map (map (.0)) v, map (map (.1)) v, map (map (.2)) v)

-- E at plane p from H planes p (with neighbours) and p - 1; old E from `eself`.
def estage (g: geo) (p: i32) (eself: i64 -> i64 -> (f32, f32, f32))
           ((hx, hy, hz): p3) ((hx0, hy0, _): p3) : (*plane, *plane, *plane) =
  let ym y = i64.max (y - 1) 0
  let xm x = i64.max (x - 1) 0
  let v = tabulate_2d WY WX (\y x ->
            if !(inside g p y x) then (0, 0, 0)
            else let (ex, ey, ez) = eself y x
                 in #[unsafe]
                    (eupd cb ex hz[y, x] hz[ym y, x] hy[y, x] hy0[y, x],
                     eupd cb ey hx[y, x] hx0[y, x] hz[y, x] hz[y, xm x],
                     eupd cb ez hy[y, x] hy[y, xm x] hx[y, x] hx[ym y, x]))
  in (map (map (.0)) v, map (map (.1)) v, map (map (.2)) v)

def pick ((a, b, c): p3) (y: i64) (x: i64) : (f32, f32, f32) =
  #[unsafe] (a[y, x], b[y, x], c[y, x])

def core6 ((ex, ey, ez): p3) ((hx, hy, hz): p3) : *[TY * TX][6]f32 =
  #[unsafe]
  tabulate (TY * TX) (\q ->
    let y = q / TX + K
    let x = q % TX + K
    in [ex[y, x], ey[y, x], ez[y, x], hx[y, x], hy[y, x], hz[y, x]])

def zero3 (_: i64) : (*plane, *plane, *plane) =
  (replicate WY (replicate WX 0), replicate WY (replicate WX 0), replicate WY (replicate WX 0))

def geometry (ntx: i64) (nz: i64) (ny: i64) (nx: i64) (t: i64) : geo =
  {nz = i32.i64 nz, ny = i32.i64 ny, nx = i32.i64 nx, ntx = i32.i64 ntx,
   y0 = i32.i64 ((t / ntx) * TY - HY), x0 = i32.i64 ((t % ntx) * TX - HX)}

-- One step per pass. Carried: E0 at z (window, for H neighbours) and H1 at
-- z - 1. The old E and H are read from global memory where only their own
-- cell is needed.
def pass1 [nt][nz] (ntx: i64) (ny: i64) (nx: i64) (src: stiled [nt][nz]) : *stiled [nt][nz] =
  let run t =
    let g = geometry ntx nz ny nx t
    let gl c z y x = at g src c z y x
    let out = #[scratch] replicate nz (replicate (TY * TX) (replicate 6 0f32))
    let (_, _, out) =
      loop (e0, h1p, out) = (load3 g src 0 0, zero3 0, out) for z < nz do
        let z = i32.i64 z
        let e0n = load3 g src 0 (z + 1)
        let h1 = hstage g z (\y x -> (gl 3 z y x, gl 4 z y x, gl 5 z y x)) e0 e0n
        let e1 = estage g z (pick e0) h1 h1p
        let out[i64.i32 z] = core6 e1 h1
        in (e0n, h1, out)
    in out
  in #[unsafe] #[flattening(only_intra)] tabulate nt run

-- Two steps per pass; stage 2 lags one plane behind stage 1.
def pass2 [nt][nz] (ntx: i64) (ny: i64) (nx: i64) (src: stiled [nt][nz]) : *stiled [nt][nz] =
  let run t =
    let g = geometry ntx nz ny nx t
    let gl c z y x = at g src c z y x
    let out = #[scratch] replicate nz (replicate (TY * TX) (replicate 6 0f32))
    let (_, _, _, _, out) =
      loop (e0, h1p, e1p, h2p, out) = (load3 g src 0 0, zero3 0, zero3 1, zero3 2, out) for z < nz + 1 do
        let z = i32.i64 z
        let e0n = load3 g src 0 (z + 1)
        let h1 = hstage g z (\y x -> (gl 3 z y x, gl 4 z y x, gl 5 z y x)) e0 e0n
        let e1 = estage g z (pick e0) h1 h1p
        let h2 = hstage g (z - 1) (pick h1p) e1p e1
        let e2 = estage g (z - 1) (pick e1p) h2 h2p
        let out[i64.i32 (i32.max 0 (z - 1))] = core6 e2 h2
        in (e0n, h1, e1, h2, out)
    in out
  in #[unsafe] #[flattening(only_intra)] tabulate nt run

-- Three and four steps per pass, each stage lagging the previous by a plane.
def pass3 [nt][nz] (ntx: i64) (ny: i64) (nx: i64) (src: stiled [nt][nz]) : *stiled [nt][nz] =
  let run t =
    let g = geometry ntx nz ny nx t
    let gl c z y x = at g src c z y x
    let out = #[scratch] replicate nz (replicate (TY * TX) (replicate 6 0f32))
    let (_, _, _, _, _, _, out) =
      loop (e0, h1p, e1p, h2p, e2p, h3p, out) =
           (load3 g src 0 0, zero3 0, zero3 1, zero3 2, zero3 3, zero3 4, out) for z < nz + 2 do
        let z = i32.i64 z
        let e0n = load3 g src 0 (z + 1)
        let h1 = hstage g z (\y x -> (gl 3 z y x, gl 4 z y x, gl 5 z y x)) e0 e0n
        let e1 = estage g z (pick e0) h1 h1p
        let h2 = hstage g (z - 1) (pick h1p) e1p e1
        let e2 = estage g (z - 1) (pick e1p) h2 h2p
        let h3 = hstage g (z - 2) (pick h2p) e2p e2
        let e3 = estage g (z - 2) (pick e2p) h3 h3p
        let out[i64.i32 (i32.max 0 (z - 2))] = core6 e3 h3
        in (e0n, h1, e1, h2, e2, h3, out)
    in out
  in #[unsafe] #[flattening(only_intra)] tabulate nt run

def pass4 [nt][nz] (ntx: i64) (ny: i64) (nx: i64) (src: stiled [nt][nz]) : *stiled [nt][nz] =
  let run t =
    let g = geometry ntx nz ny nx t
    let gl c z y x = at g src c z y x
    let out = #[scratch] replicate nz (replicate (TY * TX) (replicate 6 0f32))
    let (_, _, _, _, _, _, _, _, out) =
      loop (e0, h1p, e1p, h2p, e2p, h3p, e3p, h4p, out) =
           (load3 g src 0 0, zero3 0, zero3 1, zero3 2, zero3 3, zero3 4, zero3 5, zero3 6, out)
      for z < nz + 3 do
        let z = i32.i64 z
        let e0n = load3 g src 0 (z + 1)
        let h1 = hstage g z (\y x -> (gl 3 z y x, gl 4 z y x, gl 5 z y x)) e0 e0n
        let e1 = estage g z (pick e0) h1 h1p
        let h2 = hstage g (z - 1) (pick h1p) e1p e1
        let e2 = estage g (z - 1) (pick e1p) h2 h2p
        let h3 = hstage g (z - 2) (pick h2p) e2p e2
        let e3 = estage g (z - 2) (pick e2p) h3 h3p
        let h4 = hstage g (z - 3) (pick h3p) e3p e3
        let e4 = estage g (z - 3) (pick e3p) h4 h4p
        let out[i64.i32 (i32.max 0 (z - 3))] = core6 e4 h4
        in (e0n, h1, e1, h2, e2, h3, e3, h4, out)
    in out
  in #[unsafe] #[flattening(only_intra)] tabulate nt run

def stiled_run [nz][ny][nx] (nsteps: i64) (s: fields ([nz][ny][nx]f32)) =
  let nty = (ny + TY - 1) / TY
  let ntx = (nx + TX - 1) / TX
  let a = to_stiled nty ntx s
  -- nsteps must be a multiple of K.
  let a = if K == 1 then loop a for _i < nsteps do pass1 ntx ny nx a
          else if K == 2 then loop a for _i < nsteps / 2 do pass2 ntx ny nx a
          else if K == 3 then loop a for _i < nsteps / 3 do pass3 ntx ny nx a
          else loop a for _i < nsteps / 4 do pass4 ntx ny nx a
  in from_stiled ntx ny nx a

entry check3 (nz: i64) (ny: i64) (nx: i64) (nsteps: i64) : (i64, f32) =
  let s = init nz ny nx
  let (a0, a1, a2, a3, a4, a5) = plain_run nsteps s
  let (b0, b1, b2, b3, b4, b5) = stiled_run nsteps s
  let cmp x y = let d = map2 (\u v -> (i64.bool (f32.to_bits u != f32.to_bits v), f32.abs (u - v)))
                             (flatten_3d x) (flatten_3d y)
                in (i64.sum (map (.0) d), f32.maximum (map (.1) d))
  let r = [cmp a0 b0, cmp a1 b1, cmp a2 b2, cmp a3 b3, cmp a4 b4, cmp a5 b5]
  in (i64.sum (map (.0) r), f32.maximum (map (.1) r))

entry bench_stiled (nz: i64) (ny: i64) (nx: i64) (nsteps: i64) : f32 =
  sum6 (stiled_run nsteps (init nz ny nx))

-- Upper bound for intra-block streaming: the same z loop and loads as pass1,
-- but the output is a plain copy of the input (no stencil).
def passcopy [nt][nz] (ntx: i64) (ny: i64) (nx: i64) (src: stiled [nt][nz]) : *stiled [nt][nz] =
  let run t =
    let g = geometry ntx nz ny nx t
    let out = #[scratch] replicate nz (replicate (TY * TX) (replicate 6 0f32))
    let out =
      loop out for z < nz do
        let z = i32.i64 z
        let e = load3 g src 0 z
        let h = load3 g src 3 z
        let out[i64.i32 z] = core6 e h
        in out
    in out
  in #[unsafe] #[flattening(only_intra)] tabulate nt run

entry bench_copy (nz: i64) (ny: i64) (nx: i64) (nsteps: i64) : f32 =
  let s = init nz ny nx
  let nty = (ny + TY - 1) / TY
  let ntx = (nx + TX - 1) / TX
  let a = to_stiled nty ntx s
  let a = loop a for _i < nsteps do passcopy ntx ny nx a
  in sum6 (from_stiled ntx ny nx a)

