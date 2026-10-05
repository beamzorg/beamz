-- GENERATED from bare4.fut: same kernels with f16 field storage (bare3h.fut).
-- Full-column 2.5D streaming with ring buffers updated in place.
--
-- bare3.fut rotates freshly computed planes through loop-carried tuples.
-- Futhark double-buffers such loop parameters, and in an intra-block kernel
-- it hoists the second buffer to global memory, so every plane makes a round
-- trip through device memory per z iteration. Here every plane lives in a
-- fixed ring slot that is overwritten in place, so nothing is double-buffered.
--
-- Planes store three components per cell: [WY][WX][3]. Only values whose
-- in-plane neighbours are needed live in rings; cell-local values (old H,
-- E at z + 1) are read from global memory. The last stage computes E on the
-- core only, fused with the output.
--
-- Layout as bare3.fut: [tile][z][TY * TX][6]. Needs patch_cuda.py on CUDA.

import "bare"
import "bare3h"

type ring [n] = [n][WY][WX][3]f32

def GENERIC : bool = true
def CHUNKED : bool = false

-- Window coordinates of core cell q.
def cy (q: i64) : i64 = q / TX + HY
def cx (q: i64) : i64 = q % TX + HX

def load_plane [nt][nz] (g: geo) (a: stiled [nt][nz]) (c: i64) (z: i32) : *[WY][WX][3]f32 =
  tabulate_2d WY WX (\y x -> [at g a c z y x, at g a (c + 1) z y x, at g a (c + 2) z y x])

-- H on plane p, on the whole window: E neighbours from ring slot `e`, E at
-- p + 1 from `e1 y x`, old H from `h y x`.
def hplane [n] (g: geo) (p: i32) (es: ring [n]) (e: i64)
               (e1: i64 -> i64 -> (f32, f32)) (h: i64 -> i64 -> (f32, f32, f32))
               : *[WY][WX][3]f32 =
  let y1 y = i64.min (y + 1) (WY - 1)
  let x1 x = i64.min (x + 1) (WX - 1)
  in tabulate_2d WY WX (\y x ->
       if !(inside g p y x) then [0, 0, 0]
       else let (hx, hy, hz) = h y x
            let (ex1, ey1) = e1 y x
            let E c yy xx = #[unsafe] es[e, yy, xx, c]
            in [hupd ch hx (E 2 y x) (E 2 (y1 y) x) (E 1 y x) ey1,
                hupd ch hy (E 0 y x) ex1 (E 2 y x) (E 2 y (x1 x)),
                hupd ch hz (E 1 y x) (E 1 y (x1 x)) (E 0 y x) (E 0 (y1 y) x)])

-- E at one window cell of plane p: H neighbours from slot `h`, H at p - 1
-- from slot `h0`, old E from `e`.
def ecell [n] (g: geo) (p: i32) (hs: ring [n]) (h: i64) (h0: i64)
              (e: (f32, f32, f32)) (y: i64) (x: i64) : (f32, f32, f32) =
  if !(inside g p y x) then (0, 0, 0)
  else let ym = i64.max (y - 1) 0
       let xm = i64.max (x - 1) 0
       let H c yy xx = #[unsafe] hs[h, yy, xx, c]
       let H0 c = #[unsafe] hs[h0, y, x, c]
       in (eupd cb e.0 (H 2 y x) (H 2 ym x) (H 1 y x) (H0 1),
           eupd cb e.1 (H 0 y x) (H0 0) (H 2 y x) (H 2 y xm),
           eupd cb e.2 (H 1 y x) (H 1 y xm) (H 0 y x) (H 0 ym x))

def eplane [n] (g: geo) (p: i32) (hs: ring [n]) (h: i64) (h0: i64)
               (e: i64 -> i64 -> (f32, f32, f32)) : *[WY][WX][3]f32 =
  tabulate_2d WY WX (\y x -> let (a, b, c) = ecell g p hs h h0 (e y x) y x in [a, b, c])

-- Last stage: E on the core cells of plane p, written with H to the output.
def emit [n] (g: geo) (p: i32) (hs: ring [n]) (h: i64) (h0: i64)
             (e: i64 -> i64 -> (f32, f32, f32)) : *[TY * TX][6]f16 =
  tabulate (TY * TX) (\q ->
    let (y, x) = (cy q, cx q)
    let (ex, ey, ez) = ecell g p hs h h0 (e y x) y x
    in #[unsafe] map f16.f32 [ex, ey, ez, hs[h, y, x, 0], hs[h, y, x, 1], hs[h, y, x, 2]])

def zring (n: i64) : *ring [n] = replicate n (replicate WY (replicate WX (replicate 3 0)))

def slot3 [n] (r: ring [n]) (s: i64) (y: i64) (x: i64) : (f32, f32, f32) =
  #[unsafe] (r[s, y, x, 0], r[s, y, x, 1], r[s, y, x, 2])

-- One step per pass. Rings: E0 at z (1 slot), H1 at z and z - 1 (2 slots).
def rpass1 [nt][nz] (ntx: i64) (ny: i64) (nx: i64) (src: stiled [nt][nz]) : *stiled [nt][nz] =
  let run t =
    let g = geometry ntx nz ny nx t
    let gl c z y x = at g src c z y x
    let out = #[scratch] replicate nz (replicate (TY * TX) (replicate 6 0f16))
    let (_, _, out) =
      loop (e0, h1, out) = (zring 1, zring 2, out) for z < nz do
        let zi = i32.i64 z
        let cur = z % 2
        let e0[0] = load_plane g src 0 zi
        let h1[cur] = hplane g zi e0 0 (\y x -> (gl 0 (zi + 1) y x, gl 1 (zi + 1) y x))
                                       (\y x -> (gl 3 zi y x, gl 4 zi y x, gl 5 zi y x))
        let out[z] = emit g zi h1 cur (1 - cur) (slot3 e0 0)
        in (e0, h1, out)
    in out
  in #[unsafe] #[flattening(only_intra)] tabulate nt run

-- Two steps per pass. Rings: E0 at z (1), H1, E1 and H2 at two planes each.
-- Stage 2 works on plane z - 1.
def rpass2 [nt][nz] (ntx: i64) (ny: i64) (nx: i64) (src: stiled [nt][nz]) : *stiled [nt][nz] =
  let run t =
    let g = geometry ntx nz ny nx t
    let gl c z y x = at g src c z y x
    let out = #[scratch] replicate nz (replicate (TY * TX) (replicate 6 0f16))
    let (_, _, _, _, out) =
      loop (e0, h1, e1, h2, out) = (zring 1, zring 2, zring 2, zring 2, out) for z < nz + 1 do
        let zi = i32.i64 z
        let cur = z % 2
        let prev = 1 - cur
        let e0[0] = load_plane g src 0 zi
        let h1[cur] = hplane g zi e0 0 (\y x -> (gl 0 (zi + 1) y x, gl 1 (zi + 1) y x))
                                       (\y x -> (gl 3 zi y x, gl 4 zi y x, gl 5 zi y x))
        let e1[cur] = eplane g zi h1 cur prev (slot3 e0 0)
        -- Stage 2 on plane z - 1: E1 neighbours at z - 1 (prev), E1 at z (cur),
        -- old H1 at z - 1 (prev).
        let h2[prev] = hplane g (zi - 1) e1 prev (\y x -> #[unsafe] (e1[cur, y, x, 0], e1[cur, y, x, 1]))
                                                 (slot3 h1 prev)
        let o = i64.max 0 (z - 1)
        let out[o] = emit g (zi - 1) h2 prev cur (slot3 e1 prev)
        in (e0, h1, e1, h2, out)
    in out
  in #[unsafe] #[flattening(only_intra)] tabulate nt run

-- Any K. Stage s turns (E_s, H_s) into (E_s+1, H_s+1) on plane p = z - s.
-- hr[s] holds H_s+1 and er[s] holds E_s+1 (s < K - 1), each in two slots
-- chosen by the parity of p; E_0 at z is the single slot of e0.
def slot (p: i32) : i64 = i64.i32 (p & 1)

def rpassk [nt][nz] (ntx: i64) (ny: i64) (nx: i64) (src: stiled [nt][nz]) : *stiled [nt][nz] =
  let run t =
    let g = geometry ntx nz ny nx t
    let gl c z y x = at g src c z y x
    let out = #[scratch] replicate nz (replicate (TY * TX) (replicate 6 0f16))
    let er0 = replicate (K - 1) (zring 2)
    let (_, _, _, out) =
      loop (e0, hr, er, out) = (zring 1, replicate K (zring 2), er0, out) for z < nz + K - 1 do
        let zi = i32.i64 z
        let e0[0] = load_plane g src 0 zi
        -- Stages 0 .. K-2 write H and E rings.
        let (hr, er) =
          loop (hr, er) for s < K - 1 do
            let p = zi - i32.i64 s
            let (sp, sn, sm) = (slot p, slot (p + 1), slot (p - 1))
            let h = if s == 0
                    then hplane g p e0 0 (\y x -> (gl 0 (p + 1) y x, gl 1 (p + 1) y x))
                                         (\y x -> (gl 3 p y x, gl 4 p y x, gl 5 p y x))
                    else hplane g p er[s - 1] sp (\y x -> #[unsafe] (er[s - 1, sn, y, x, 0], er[s - 1, sn, y, x, 1]))
                                                 (slot3 hr[s - 1] sp)
            let hr[s, sp] = h
            let e = eplane g p hr[s] sp sm
                           (\y x -> if s == 0 then slot3 e0 0 y x else slot3 er[s - 1] sp y x)
            let er[s, sp] = e
            in (hr, er)
        -- Last stage: H into its ring, E on the core straight to the output.
        let s = K - 1
        let p = zi - i32.i64 s
        let (sp, sn, sm) = (slot p, slot (p + 1), slot (p - 1))
        let h = if s == 0
                then hplane g p e0 0 (\y x -> (gl 0 (p + 1) y x, gl 1 (p + 1) y x))
                                     (\y x -> (gl 3 p y x, gl 4 p y x, gl 5 p y x))
                else hplane g p er[s - 1] sp (\y x -> #[unsafe] (er[s - 1, sn, y, x, 0], er[s - 1, sn, y, x, 1]))
                                             (slot3 hr[s - 1] sp)
        let hr[s, sp] = h
        let out[i64.i32 (i32.max 0 p)] =
          emit g p hr[s] sp sm (\y x -> if s == 0 then slot3 e0 0 y x else slot3 er[s - 1] sp y x)
        in (e0, hr, er, out)
    in out
  in #[unsafe] #[flattening(only_intra)] tabulate nt run

-- Without patch_cuda.py: blocks own (tile, z chunk) and keep their ZC output
-- planes in shared memory. A chunk starts K planes early so that every stage
-- is exact on the planes the next one reads; early outputs land in plane 0
-- and are overwritten once exact. nz must be a multiple of ZC.
def rpassc [nt][nz] (ntx: i64) (ny: i64) (nx: i64) (src: stiled [nt][nz]) : *stiled [nt][nz] =
  let nzc = nz / ZC
  let run b =
    let t = b / nzc
    let z0 = (b % nzc) * ZC
    let g = geometry ntx nz ny nx t
    let gl c z y x = at g src c z y x
    let out = #[scratch] replicate ZC (replicate (TY * TX) (replicate 6 0f16))
    let er0 = replicate (K - 1) (zring 2)
    let (_, _, _, out) =
      loop (e0, hr, er, out) = (zring 1, replicate K (zring 2), er0, out) for i < ZC + 2 * K - 1 do
        let zi = i32.i64 (z0 - K + i)
        let e0[0] = load_plane g src 0 zi
        let (hr, er) =
          loop (hr, er) for s < K - 1 do
            let p = zi - i32.i64 s
            let (sp, sn, sm) = (slot p, slot (p + 1), slot (p - 1))
            let h = if s == 0
                    then hplane g p e0 0 (\y x -> (gl 0 (p + 1) y x, gl 1 (p + 1) y x))
                                         (\y x -> (gl 3 p y x, gl 4 p y x, gl 5 p y x))
                    else hplane g p er[s - 1] sp (\y x -> #[unsafe] (er[s - 1, sn, y, x, 0], er[s - 1, sn, y, x, 1]))
                                                 (slot3 hr[s - 1] sp)
            let hr[s, sp] = h
            let e = eplane g p hr[s] sp sm
                           (\y x -> if s == 0 then slot3 e0 0 y x else slot3 er[s - 1] sp y x)
            let er[s, sp] = e
            in (hr, er)
        let s = K - 1
        let p = zi - i32.i64 s
        let (sp, sn, sm) = (slot p, slot (p + 1), slot (p - 1))
        let h = if s == 0
                then hplane g p e0 0 (\y x -> (gl 0 (p + 1) y x, gl 1 (p + 1) y x))
                                     (\y x -> (gl 3 p y x, gl 4 p y x, gl 5 p y x))
                else hplane g p er[s - 1] sp (\y x -> #[unsafe] (er[s - 1, sn, y, x, 0], er[s - 1, sn, y, x, 1]))
                                             (slot3 hr[s - 1] sp)
        let hr[s, sp] = h
        let out[i64.max 0 (i64.i32 p - z0)] =
          emit g p hr[s] sp sm (\y x -> if s == 0 then slot3 e0 0 y x else slot3 er[s - 1] sp y x)
        in (e0, hr, er, out)
    in out
  let r = #[unsafe] #[flattening(only_intra)] tabulate (nt * nzc) run
  in map flatten (unflatten r) :> *stiled [nt][nz]

def rtiled_run [nz][ny][nx] (nsteps: i64) (s: fields ([nz][ny][nx]f32)) =
  let nty = (ny + TY - 1) / TY
  let ntx = (nx + TX - 1) / TX
  let a = to_stiled nty ntx s
  -- nsteps must be a multiple of K.
  let a = if CHUNKED then loop a for _i < nsteps / K do rpassc ntx ny nx a
          else if GENERIC then loop a for _i < nsteps / K do rpassk ntx ny nx a
          else if K == 1 then loop a for _i < nsteps do rpass1 ntx ny nx a
          else loop a for _i < nsteps / 2 do rpass2 ntx ny nx a
  in from_stiled ntx ny nx a

entry check4 (nz: i64) (ny: i64) (nx: i64) (nsteps: i64) : (i64, f32) =
  let s = init nz ny nx
  let (a0, a1, a2, a3, a4, a5) = plain_run nsteps s
  let (b0, b1, b2, b3, b4, b5) = rtiled_run nsteps s
  let cmp x y = let d = map2 (\u v -> (i64.bool (f32.to_bits u != f32.to_bits v), f32.abs (u - v)))
                             (flatten_3d x) (flatten_3d y)
                in (i64.sum (map (.0) d), f32.maximum (map (.1) d))
  let r = [cmp a0 b0, cmp a1 b1, cmp a2 b2, cmp a3 b3, cmp a4 b4, cmp a5 b5]
  in (i64.sum (map (.0) r), f32.maximum (map (.1) r))

entry bench_rtiled (nz: i64) (ny: i64) (nx: i64) (nsteps: i64) : f32 =
  sum6 (rtiled_run nsteps (init nz ny nx))
