-- Compile-cost probes: each entry exercises one part of temporal.fut on its
-- own, with constant materials and metrics. Compile one entry at a time with
--   futhark cuda --library --entry-point NAME cost.fut   (or futhark dev).

import "../yee"
import "../temporal"

def ctx0 [L] (g: geo) (n: dims) (slabs: [12][2]i64) (ca: [12][L]f32) : ctx [L] =
  {g, edges = 0, h_scale = \_ _ -> 1f32, e_scale = \_ _ -> 1f32,
   nh = (n, n, n), ne = (n, n, n), slabs, ca, cb = ca, ck = ca,
   hmat = \_ _ _ _ -> (1f32, 0.5f32), emat = \_ _ _ _ -> (1f32, 0.5f32)}

entry pass_only [nt][P0][Lz][L][K]
    (P1: i64) (P2: i64) (slabs: [12][2]i64) (ca: [12][L]f32)
    (target: [K]i32) (group: [K]i32) (amp: [K]f32) (off: [K]i32) (len: [K]i32) (waves: []f32)
    (splanes: [nt][P0]bool) (edge_tile: [nt]bool) (step: i64)
    (f: [nt][P0][TY * TX][6]f32) (pz: [nt][Lz][TY * TX][4]f32) (pxy: [nt][P0][TY * TX][8]f32) =
  let g = {P0, P1, P2, ntx = (P2 + TX - 1) / TX}
  let src = source_lookup (P0 * P1 * P2) target group amp off len
  let x = ctx0 g (P0, P1, P2) slabs ca
  let vals = source_values src waves group target step
  in pass x {lo = 2, hs = P0 - 2, all = false} src vals splanes edge_tile false f pz pxy

entry recompute_only [nt][P0][Lz][L][K][M]
    (P1: i64) (P2: i64) (slabs: [12][2]i64) (ca: [12][L]f32)
    (target: [K]i32) (group: [K]i32) (amp: [K]f32) (off: [K]i32) (len: [K]i32) (waves: []f32)
    (step: i64) (f: [nt][P0][TY * TX][6]f32) (pz: [nt][Lz][TY * TX][4]f32) (pxy: [nt][P0][TY * TX][8]f32)
    (cs: [M]i64) (os: [M]i64) =
  let g = {P0, P1, P2, ntx = (P2 + TX - 1) / TX}
  let src = source_lookup (P0 * P1 * P2) target group amp off len
  let x = ctx0 g (P0, P1, P2) slabs ca
  in map2 (\c o -> let (k, j, i) = decode g o
                   in recompute x {lo = 2, hs = P0 - 2, all = false} src (source_values src waves group target step) f pz pxy c k j i) cs os

entry pass2_only [nt][P0][Lz][L][K]
    (P1: i64) (P2: i64) (slabs: [12][2]i64) (ca: [12][L]f32)
    (target: [K]i32) (group: [K]i32) (amp: [K]f32) (off: [K]i32) (len: [K]i32) (waves: []f32)
    (splanes: [nt][P0]bool) (edge_tile: [nt]bool) (step: i64)
    (f: [nt][P0][TY * TX][6]f32) (pz: [nt][Lz][TY * TX][4]f32) (pxy: [nt][P0][TY * TX][8]f32) =
  let g = {P0, P1, P2, ntx = (P2 + TX - 1) / TX}
  let src = source_lookup (P0 * P1 * P2) target group amp off len
  let x = ctx0 g (P0, P1, P2) slabs ca
  let vals = source_values src waves group target step
  in pass2 x {lo = 2, hs = P0 - 2, all = false} src vals splanes edge_tile f pz pxy
