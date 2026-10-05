-- GENERATED subset of bare3.fut with fields stored as f16 (storage experiment).
-- Lean full-column 2.5D streaming (see bare2.fut): explicit K = 1 and K = 2
-- stages whose planes rotate through loop-carried tuples (pointer swaps, no
-- shared-memory copies), and one tabulate per output plane.
-- Layout: [tile][z][component][TY * TX]. Needs patch_cuda.py on CUDA.

import "bare"

type stiled [nt][nz] = [nt][nz][TY * TX][6]f16

def to_stiled [nz][ny][nx] (nty: i64) (ntx: i64) (s: fields ([nz][ny][nx]f32)) : *stiled [nty * ntx][nz] =
  let cs = [s.0, s.1, s.2, s.3, s.4, s.5]
  in tabulate_3d (nty * ntx) nz (TY * TX) (\t k q ->
       tabulate 6 (\c -> f16.f32 (get cs[c] k ((t / ntx) * TY + q / TX) ((t % ntx) * TX + q % TX))))

def from_stiled [nt][nz] (ntx: i64) (ny: i64) (nx: i64) (a: stiled [nt][nz]) : fields ([nz][ny][nx]f32) =
  let u c = tabulate_3d nz ny nx (\k j i -> f32.f16 (#[unsafe] a[(j / TY) * ntx + i / TX, k, (j % TY) * TX + i % TX, c]))
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
     then f32.f16 (#[unsafe] a[i64.i32 ((gy / i32.i64 TY) * g.ntx + gx / i32.i64 TX), i64.i32 z,
                      i64.i32 ((gy % i32.i64 TY) * i32.i64 TX + gx % i32.i64 TX), c])
     else 0

def geometry (ntx: i64) (nz: i64) (ny: i64) (nx: i64) (t: i64) : geo =
  {nz = i32.i64 nz, ny = i32.i64 ny, nx = i32.i64 nx, ntx = i32.i64 ntx,
   y0 = i32.i64 ((t / ntx) * TY - HY), x0 = i32.i64 ((t % ntx) * TX - HX)}

-- One step per pass. Carried: E0 at z (window, for H neighbours) and H1 at
-- z - 1. The old E and H are read from global memory where only their own
-- cell is needed.
