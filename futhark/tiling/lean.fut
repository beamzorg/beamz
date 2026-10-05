-- Build-up experiment for the core-case temporal kernel: the fast bare ring
-- tile (bare4.fut) with production arithmetic and features added one at a
-- time behind compile-time switches, to see what each costs.
--
--   MAT   E source coefficient from 8-bit material codes and a codebook
--   ZPML  z-slab CPML (terms 1, 2, 7, 8) with tiled memory, as temporal.fut
--
-- Fields use the production tiled layout [tile][z][TY * TX][6], all six
-- components on one (P0, P1, P2) box. Two steps per pass, ring slots updated
-- in place, components and stages written out statically.

import "../yee"

def TY : i64 = 4
def TX : i64 = 32
def MAT : bool = false
def ZPML : bool = false
-- Keep stage-1 z memory in shared rings instead of recomputing it.
def ZRING : bool = false
def SRC : bool = false
-- Use the production per-cell update (yee.cell) instead of `upd`.
def PRODCELL : bool = false
-- Production helpers: split field accessor, material/coefficient, metric.
def SPLITG : bool = false
def PRODMAT : bool = false
def PRODMETRIC : bool = false

-- Window halo (2 for K = 2). Timing experiments override HY/HX via gen.py.
def HY : i64 = 2
def HX : i64 = 2
def WY : i64 = TY + 2 * HY
def WX : i64 = TX + 2 * HX
def ZL : i64 = 12

type tf [nt][P0] = [nt][P0][TY * TX][6]f32
type geo = {P0: i64, P1: i64, P2: i64, ntx: i64}
type ring [n] = [n][WY][WX][3]f32

def zring (n: i64) : *ring [n] = replicate n (replicate WY (replicate WX (replicate 3 0)))

def instore (g: geo) (k: i64) (j: i64) (i: i64) : bool =
  k >= 0 && k < g.P0 && j >= 0 && j < g.P1 && i >= 0 && i < g.P2

def fget [nt][P0] (g: geo) (f: tf [nt][P0]) (c: i64) (k: i64) (j: i64) (i: i64) : f32 =
  if instore g k j i
  then #[unsafe] f[(j / TY) * g.ntx + i / TX, k, (j % TY) * TX + i % TX, c]
  else 0

-- z-slab row of plane k, or -1: ZL planes at each end.
def zrow (P0: i64) (k: i64) : i64 =
  if k < ZL then k else if k >= P0 - ZL then ZL + k - (P0 - ZL) else -1

type mats [C][T][S3][U9][nt][P0] = {codes: [C]i32, table: [T]f32, s_h: f32, ca: [2 * ZL]f32, cb: [2 * ZL]f32, ck: [2 * ZL]f32,
                                     slot: [S3]i32, has: [U9]bool, vals: [3][U9]f32, splanes: [nt][P0]bool,
                                     slabs: [12][2]i64, ca12: [12][2 * ZL]f32, cb12: [12][2 * ZL]f32, ck12: [12][2 * ZL]f32,
                                     kind: i64, marr: [P0]f32, one3: [1][1][1]f32, cidx: [nt]i64}

-- E source coefficient of component c at a cell.
def esrc [C][T][S3][U9][nt][P0] (m: mats [C][T][S3][U9][nt][P0]) (g: geo) (c: i64) (k: i64) (j: i64) (i: i64) : f32 =
  -- Window halo cells may lie outside the box; never read coefficients there.
  if !(instore g k j i) then 0
  else if PRODMAT then material (g.P0 - 1, g.P1 - 1, g.P2 - 1) m.one3 m.table m.codes k j i
  else if !MAT then 0.5
  else let l = ((k * g.P1 + j) * g.P2 + i) * 3 + c
       let code = (#[unsafe] m.codes[l >> 2] >> i32.i64 (8 * (l & 3))) & 0xff
       in #[unsafe] m.table[i64.i32 code]

-- v plus group g's summed sources at row o, if the cell has any.
def addsrc [C][T][S3][U9][nt][P0] (m: mats [C][T][S3][U9][nt][P0]) (g: geo) (group: i64) (o: i64)
           (k: i64) (j: i64) (i: i64) (v: f32) : f32 =
  if !SRC || !(instore g k j i) then v
  else let s = #[unsafe] m.slot[(k * g.P1 + j) * g.P2 + i]
       let e = i64.i32 s * 9 + group
       in if s < 0 || !(#[unsafe] m.has[e]) then v else v + #[unsafe] m.vals[o, e]

-- One update of component c: value and advanced memory of both terms.
-- v and w are the samples (cell, neighbour) of both curl terms; zt is the
-- local index of the curl term along z (-1 if none) and zpsi its memory.
def upd (kind: i64) (marr: []f32) (decay: [1][1][1]f32) (slabs: [12][2]i64) (ca12: [12][2 * ZL]f32) (cb12: [12][2 * ZL]f32) (ck12: [12][2 * ZL]f32)
        (phase: i64) (n: dims) (c: i64) (s: f32) (old: f32)
        (v0: f32) (w0: f32) (v1: f32) (w1: f32) (zrow: i64) (zpsi: f32)
        (ca: f32) (cb: f32) (ck: f32) (k: i64) (j: i64) (i: i64) : (f32, f32) =
  if PRODCELL
  then let (_, a0, _, a1) = route c
       let (s0, _, s1, _) = route c
       let v s = if s == s0 then v0 else v1
       let w s _ = if s == s0 then w0 else w1
       let (value, n0, n1) = cell phase c 0 (\_ _ -> 1f32) n (n, n, n) (\t e -> #[unsafe] slabs[t, e])
                                (\t p -> #[unsafe] (ca12[t, p], cb12[t, p], ck12[t, p]))
                                (\_ _ _ -> (1f32, s)) old v w (\_ _ -> zpsi) k j i
       in (value, if a0 == 0 then n0 else if a1 == 0 then n1 else 0)
  else
  if !(inside n k j i) then (0, 0)
  else
    let (_, a0, _, a1) = route c
    let sc a = if PRODMETRIC then metric kind 1 marr (axis_coord a k j i) else 1
    let d0 = dsample phase 0 n a0 (sc a0) v0 w0 k j i
    let d1 = dsample phase 0 n a1 (sc a1) v1 w1 k j i
    let old = if PRODMAT then coefficient decay k j i * old else old
    let zl = if a0 == 0 then 0 else if a1 == 0 then 1 else -1
    let slab = ZPML && zl >= 0 && zrow >= 0
    let (t0, n0) = cpml_term (slab && zl == 0) ca cb ck zpsi d0
    let (t1, n1) = cpml_term (slab && zl == 1) ca cb ck zpsi d1
    in (f32.fma (if phase == 0 then -s else s) (t0 - t1) old, n0 + n1)

-- Window planes: E0 (1 slot), H1 (2), E1 (2), H2 (2).
def pass [nt][P0][C][T][S3][U9][Lz] (g: geo) (m: mats [C][T][S3][U9][nt][P0]) (f: tf [nt][P0]) (f2: tf [nt][P0]) (pz: [nt][Lz][TY * TX][4]f32)
    : (*tf [nt][P0], *[nt][Lz][TY * TX][4]f32) =
  let n = (P0 - 1, g.P1 - 1, g.P2 - 1)
  let run t =
    let y0 = (t / g.ntx) * TY - HY
    let x0 = (t % g.ntx) * TX - HX
    let G c k j i =
      if !SPLITG then fget g f c k j i
      else if !(instore g k j i) then 0
      else let tt = (j / TY) * g.ntx + i / TX
           let core = #[unsafe] m.cidx[tt] >= 0 && k >= 3 && k < P0 - 3
           in if core then #[unsafe] f[tt, k, (j % TY) * TX + i % TX, c]
              else #[unsafe] f2[tt, P0 - 1 - k, (j % TY) * TX + i % TX, c]
    let up y = i64.min (y + 1) (WY - 1)
    let rt x = i64.min (x + 1) (WX - 1)
    let dn y = i64.max (y - 1) 0
    let lf x = i64.max (x - 1) 0
    -- z-slab memory of local z term zs (0..3: Hx, Hy, Ex, Ey) at a window cell.
    let zmem zs k y x =
      let j = y0 + y
      let i = x0 + x
      let r = zrow P0 k
      in if !ZPML || r < 0 || !(instore g k j i) then 0
         else #[unsafe] pz[(j / TY) * g.ntx + i / TX, r, (j % TY) * TX + i % TX, zs]
    let coef k = let r = zrow P0 k
                 let p = if r < 0 then 0 else r
                 in #[unsafe] (m.ca[p], m.cb[p], m.ck[p])
    let out = #[scratch] replicate P0 (replicate (TY * TX) (replicate 6 0f32))
    let oz = #[scratch] replicate Lz (replicate (TY * TX) (replicate 4 0f32))
    let zr2 () = replicate 2 (replicate WY (replicate WX (replicate 2 0f32)))
    let (_, _, _, _, _, _, out, oz) =
      loop (e0, h1, e1, h2, zmh, zme, out, oz) = (zring 1, zring 2, zring 2, zring 2, zr2 (), zr2 (), out, oz) for z < P0 + 1 do
        let p = z
        let q = z - 1
        let (ap, bp, kp) = coef p
        let (aq, bq, kq) = coef q
        let e0[0] = tabulate_2d WY WX (\y x -> [G 0 p (y0 + y) (x0 + x), G 1 p (y0 + y) (x0 + x), G 2 p (y0 + y) (x0 + x)])
        -- Stage 1 H at p: E0 from ring, E0 at p + 1 and old H from memory.
        let (h1p, zmhp) = unzip (map unzip (
          tabulate_2d WY WX (\y x ->
            let (k, j, i) = (p, y0 + y, x0 + x)
            let E c = #[unsafe] e0[0, y, x, c]
            let zr = zrow P0 k
            let (hx, mx) = upd m.kind m.marr m.one3 m.slabs m.ca12 m.cb12 m.ck12 0 n 0 m.s_h (G 3 k j i) (E 2) (#[unsafe] e0[0, up y, x, 2]) (E 1) (G 1 (k + 1) j i) zr (zmem 0 k y x) ap bp kp k j i
            let (hy, my) = upd m.kind m.marr m.one3 m.slabs m.ca12 m.cb12 m.ck12 0 n 1 m.s_h (G 4 k j i) (E 0) (G 0 (k + 1) j i) (E 2) (#[unsafe] e0[0, y, rt x, 2]) zr (zmem 1 k y x) ap bp kp k j i
            let (hz, _) = upd m.kind m.marr m.one3 m.slabs m.ca12 m.cb12 m.ck12 0 n 2 m.s_h (G 5 k j i) (E 1) (#[unsafe] e0[0, y, rt x, 1]) (E 0) (#[unsafe] e0[0, up y, x, 0]) zr 0 ap bp kp k j i
            let sp = SRC && #[unsafe] m.splanes[t, i64.min p (P0 - 1)]
            let src c v = if sp then addsrc m g (3 + c) 0 k j i v else v
            in ([src 0 hx, src 1 hy, src 2 hz], [mx, my]))))
        let h1[p & 1] = h1p
        let zmh = if ZRING then zmh with [p & 1] = zmhp else zmh
        -- Stage 1 E at p.
        let (e1p, zmep) = unzip (map unzip (
          tabulate_2d WY WX (\y x ->
            let (k, j, i) = (p, y0 + y, x0 + x)
            let H c = #[unsafe] h1[p & 1, y, x, c]
            let Hm c = #[unsafe] h1[(p - 1) & 1, y, x, c]
            let zr = zrow P0 k
            let (ex, mx) = upd m.kind m.marr m.one3 m.slabs m.ca12 m.cb12 m.ck12 1 n 0 (esrc m g 0 k j i) (#[unsafe] e0[0, y, x, 0]) (H 2) (#[unsafe] h1[p & 1, dn y, x, 2]) (H 1) (Hm 1) zr (zmem 2 k y x) ap bp kp k j i
            let (ey, my) = upd m.kind m.marr m.one3 m.slabs m.ca12 m.cb12 m.ck12 1 n 1 (esrc m g 1 k j i) (#[unsafe] e0[0, y, x, 1]) (H 0) (Hm 0) (H 2) (#[unsafe] h1[p & 1, y, lf x, 2]) zr (zmem 3 k y x) ap bp kp k j i
            let (ez, _) = upd m.kind m.marr m.one3 m.slabs m.ca12 m.cb12 m.ck12 1 n 2 (esrc m g 2 k j i) (#[unsafe] e0[0, y, x, 2]) (H 1) (#[unsafe] h1[p & 1, y, lf x, 1]) (H 0) (#[unsafe] h1[p & 1, dn y, x, 0]) zr 0 ap bp kp k j i
            let sp = SRC && #[unsafe] m.splanes[t, i64.min p (P0 - 1)]
            let src c v = if sp then addsrc m g c 1 k j i (addsrc m g (6 + c) 0 k j i v) else v
            in ([src 0 ex, src 1 ey, src 2 ez], [mx, my]))))
        let e1[p & 1] = e1p
        let zme = if ZRING then zme with [p & 1] = zmep else zme
        -- Stage 2 H at q; stage 1 z memory is recomputed from E0 (memory).
        let h2_at y x =
          let (k, j, i) = (q, y0 + y, x0 + x)
          let E c = #[unsafe] e1[q & 1, y, x, c]
          let zr = zrow P0 k
          let m1 zs s = if !ZPML || zr < 0 then 0
                        else if ZRING then #[unsafe] zmh[q & 1, y, x, zs]
                        else (cpml_term true aq bq kq (zmem zs k y x) ((G s (k + 1) j i - G s k j i) * 1)).1
          let (hx, nx) = upd m.kind m.marr m.one3 m.slabs m.ca12 m.cb12 m.ck12 0 n 0 m.s_h (#[unsafe] h1[q & 1, y, x, 0]) (E 2) (#[unsafe] e1[q & 1, up y, x, 2]) (E 1) (#[unsafe] e1[p & 1, y, x, 1]) zr (m1 0 1) aq bq kq k j i
          let (hy, ny) = upd m.kind m.marr m.one3 m.slabs m.ca12 m.cb12 m.ck12 0 n 1 m.s_h (#[unsafe] h1[q & 1, y, x, 1]) (E 0) (#[unsafe] e1[p & 1, y, x, 0]) (E 2) (#[unsafe] e1[q & 1, y, rt x, 2]) zr (m1 1 0) aq bq kq k j i
          let (hz, _) = upd m.kind m.marr m.one3 m.slabs m.ca12 m.cb12 m.ck12 0 n 2 m.s_h (#[unsafe] h1[q & 1, y, x, 2]) (E 1) (#[unsafe] e1[q & 1, y, rt x, 1]) (E 0) (#[unsafe] e1[q & 1, up y, x, 0]) zr 0 aq bq kq k j i
          let sp = SRC && #[unsafe] m.splanes[t, i64.max q 0]
          let src c v = if sp then addsrc m g (3 + c) 1 k j i v else v
          in ([src 0 hx, src 1 hy, src 2 hz], nx, ny)
        let h2[q & 1] = tabulate_2d WY WX (\y x -> (h2_at y x).0)
        -- Stage 2 E on the core, written out with H2 and z memory.
        let core =
          tabulate (TY * TX) (\l ->
            let y = l / TX + HY
            let x = l % TX + HX
            let (k, j, i) = (q, y0 + y, x0 + x)
            let H c = #[unsafe] h2[q & 1, y, x, c]
            let Hm c = #[unsafe] h2[(q - 1) & 1, y, x, c]
            let zr = zrow P0 k
            let m1 zs s = if !ZPML || zr < 0 then 0
                          else if ZRING then #[unsafe] zme[q & 1, y, x, zs - 2]
                          else (cpml_term true aq bq kq (zmem zs k y x)
                                          ((#[unsafe] h1[q & 1, y, x, s] - #[unsafe] h1[(q - 1) & 1, y, x, s]) * 1)).1
            let (ex, nx) = upd m.kind m.marr m.one3 m.slabs m.ca12 m.cb12 m.ck12 1 n 0 (esrc m g 0 k j i) (#[unsafe] e1[q & 1, y, x, 0]) (H 2) (#[unsafe] h2[q & 1, dn y, x, 2]) (H 1) (Hm 1) zr (m1 2 1) aq bq kq k j i
            let (ey, ny) = upd m.kind m.marr m.one3 m.slabs m.ca12 m.cb12 m.ck12 1 n 1 (esrc m g 1 k j i) (#[unsafe] e1[q & 1, y, x, 1]) (H 0) (Hm 0) (H 2) (#[unsafe] h2[q & 1, y, lf x, 2]) zr (m1 3 0) aq bq kq k j i
            let (ez, _) = upd m.kind m.marr m.one3 m.slabs m.ca12 m.cb12 m.ck12 1 n 2 (esrc m g 2 k j i) (#[unsafe] e1[q & 1, y, x, 2]) (H 1) (#[unsafe] h2[q & 1, y, lf x, 1]) (H 0) (#[unsafe] h2[q & 1, dn y, x, 0]) zr 0 aq bq kq k j i
            let (h, hnx, hny) = h2_at y x
            let sp = SRC && #[unsafe] m.splanes[t, i64.max q 0]
            let src c v = if sp then addsrc m g (6 + c) 1 k j i v else v
            in ([src 0 ex, src 1 ey, src 2 ez, h[0], h[1], h[2]], [hnx, hny, nx, ny]))
        let (fo, zo) = unzip core
        let out[i64.max q 0] = fo
        let r = if q >= 0 then zrow P0 q else -1
        let oz = if ZPML && r >= 0 then oz with [r] = zo else oz
        in (e0, h1, e1, h2, zmh, zme, out, oz)
    in (out, oz)
  let r = #[unsafe] #[flattening(only_intra)] #[intrablock_result_global] tabulate nt run
  in (map (.0) r, map (.1) r)

def setup (P0: i64) (P1: i64) (P2: i64) =
  let nty = (P1 + TY - 1) / TY
  let ntx = (P2 + TX - 1) / TX
  let g = {P0, P1, P2, ntx}
  let h q = let h = u32.i64 (q * 2654435761 + 12345)
            let h = (h ^ (h >> 16)) * 0x45d9f3b
            in f32.u32 ((h ^ (h >> 16)) & 0xffff) / 65536 - 0.5
  let f = tabulate_3d (nty * ntx) P0 (TY * TX) (\t k l -> map (\c -> h (((t * P0 + k) * 128 + l) * 6 + c)) (iota 6))
  let Lz = #[opaque] (2 * ZL)
  let pz = tabulate_3d (nty * ntx) Lz (TY * TX) (\t r l -> map (\c -> h (((t * 24 + r) * 128 + l) * 4 + c + 7)) (iota 4))
  let C = (P0 * P1 * P2 * 3 + 3) / 4
  let i0 = P2 / 3
  let S3 = P0 * P1 * P2
  let slot = tabulate S3 (\q -> if q % P2 == i0 then i32.i64 (q / P2) else -1)
  let U9 = P0 * P1 * 9
  let has = tabulate U9 (\e -> let g = e % 9 in g == 2 || g == 5 || g == 8)
  let vals = tabulate_2d 3 U9 (\o e -> h (e * 3 + o + 99))
  let splanes = tabulate_2d (nty * ntx) P0 (\t _ -> let tx = t % ntx in i0 >= tx * TX - 2 && i0 < tx * TX + TX + 2)
  let m = {codes = tabulate C (\q -> i32.u32 (u32.i64 q * 2654435761 & 0x01010101)),
           table = [0.5f32, 0.12f32], s_h = 0.5f32,
           ca = tabulate (2 * ZL) (\p -> 0.01 * f32.i64 p), cb = tabulate (2 * ZL) (\p -> 0.9 - 0.01 * f32.i64 p),
           ck = replicate (2 * ZL) 0.8f32, slot, has, vals, splanes,
           slabs = tabulate 12 (\t -> if t == 1 || t == 2 || t == 7 || t == 8 then [ZL, ZL] else [0, 0]),
           ca12 = replicate 12 (tabulate (2 * ZL) (\p -> 0.01 * f32.i64 p)),
           cb12 = replicate 12 (tabulate (2 * ZL) (\p -> 0.9 - 0.01 * f32.i64 p)),
           ck12 = replicate 12 (replicate (2 * ZL) 0.8f32),
           kind = #[opaque] 0i64, marr = replicate P0 1f32, one3 = [[[1f32]]],
           cidx = tabulate (nty * ntx) (\t -> if t % 3 == 0 then -1 else t)}
  in (g, m, f, pz)

-- ==
-- entry: bench
-- input { 129i64 257i64 513i64 0i64 }
-- input { 129i64 257i64 513i64 24i64 }
entry bench (P0: i64) (P1: i64) (P2: i64) (nsteps: i64) : f32 =
  let (g, m, f, pz) = setup P0 P1 P2
  let f2 = map (map (map (map (* 1.5)))) f
  let (f, _) = loop (f, pz) for _i < nsteps / 2 do pass g m f f2 pz
  in f32.sum (map (\x -> x[0, 0, 0]) f)

-- Plain one-step baseline with the same `upd` arithmetic: one H and one E
-- kernel per step over the tiled layout, no window and no redundancy. z
-- memory is dense ([P0] planes, used only on slab planes): H terms in ph, E
-- terms in pe, each written by its own kernel.
type col3 [nt][P0] = [nt][P0][TY * TX][3]f32
type col2 [nt][P0] = [nt][P0][TY * TX][2]f32

def plain_step [nt][P0][C][T][S3][U9] (g: geo) (m: mats [C][T][S3][U9][nt][P0])
               (e: col3 [nt][P0]) (h: col3 [nt][P0]) (ph: col2 [nt][P0]) (pe: col2 [nt][P0])
    : (col3 [nt][P0], col3 [nt][P0], col2 [nt][P0], col2 [nt][P0]) =
  let n = (P0 - 1, g.P1 - 1, g.P2 - 1)
  let at (a: col3 [nt][P0]) c k j i =
    if instore g k j i then #[unsafe] a[(j / TY) * g.ntx + i / TX, k, (j % TY) * TX + i % TX, c] else 0
  let coef k = let r = zrow P0 k
               let p = if r < 0 then 0 else r
               in #[unsafe] (m.ca[p], m.cb[p], m.ck[p])
  let U ph c s old v0 w0 v1 w1 zr zp (a, b, kk) k j i =
    upd m.kind m.marr m.one3 m.slabs m.ca12 m.cb12 m.ck12 ph n c s old v0 w0 v1 w1 zr zp a b kk k j i
  let mem (p: col2 [nt][P0]) zr t k l s = if ZPML && zr >= 0 then #[unsafe] p[t, k, l, s] else 0
  let ij t l = ((t / g.ntx) * TY + l / TX, (t % g.ntx) * TX + l % TX)
  let hn = tabulate_3d nt P0 (TY * TX) (\t k l ->
    let (j, i) = ij t l
    let E c kk jj ii = at e c kk jj ii
    let zr = zrow P0 k
    let cf = coef k
    let (hx, mx) = U 0 0 m.s_h (#[unsafe] h[t, k, l, 0]) (E 2 k j i) (E 2 k (j + 1) i) (E 1 k j i) (E 1 (k + 1) j i) zr (mem ph zr t k l 0) cf k j i
    let (hy, my) = U 0 1 m.s_h (#[unsafe] h[t, k, l, 1]) (E 0 k j i) (E 0 (k + 1) j i) (E 2 k j i) (E 2 k j (i + 1)) zr (mem ph zr t k l 1) cf k j i
    let (hz, _) = U 0 2 m.s_h (#[unsafe] h[t, k, l, 2]) (E 1 k j i) (E 1 k j (i + 1)) (E 0 k j i) (E 0 k (j + 1) i) zr 0 cf k j i
    let keep s v = if ZPML && zr >= 0 then v else #[unsafe] ph[t, k, l, s]
    in ([hx, hy, hz], [keep 0 mx, keep 1 my]))
  let (hn, ph') = unzip (map unzip (map (map unzip) hn))
  let en = tabulate_3d nt P0 (TY * TX) (\t k l ->
    let (j, i) = ij t l
    let H c kk jj ii = at hn c kk jj ii
    let zr = zrow P0 k
    let cf = coef k
    let (ex, mx) = U 1 0 (esrc m g 0 k j i) (#[unsafe] e[t, k, l, 0]) (H 2 k j i) (H 2 k (j - 1) i) (H 1 k j i) (H 1 (k - 1) j i) zr (mem pe zr t k l 0) cf k j i
    let (ey, my) = U 1 1 (esrc m g 1 k j i) (#[unsafe] e[t, k, l, 1]) (H 0 k j i) (H 0 (k - 1) j i) (H 2 k j i) (H 2 k j (i - 1)) zr (mem pe zr t k l 1) cf k j i
    let (ez, _) = U 1 2 (esrc m g 2 k j i) (#[unsafe] e[t, k, l, 2]) (H 1 k j i) (H 1 k j (i - 1)) (H 0 k j i) (H 0 k (j - 1) i) zr 0 cf k j i
    let keep s v = if ZPML && zr >= 0 then v else #[unsafe] pe[t, k, l, s]
    in ([ex, ey, ez], [keep 0 mx, keep 1 my]))
  let (en, pe') = unzip (map unzip (map (map unzip) en))
  in (en, hn, ph', pe')

def plain_init [nt][P0] (f: [nt][P0][TY * TX][6]f32) =
  let e = map (map (map (\x -> x[0:3] :> [3]f32))) f
  let h = map (map (map (\x -> x[3:6] :> [3]f32))) f
  let ph = map (map (map (\x -> [x[0] * 0.5, x[1] * 0.5]))) f
  let pe = map (map (map (\x -> [x[2] * 0.5, x[3] * 0.5]))) f
  in (e, h, ph, pe)

entry bench_plain (P0: i64) (P1: i64) (P2: i64) (nsteps: i64) : f32 =
  let (g, m, f, _) = setup P0 P1 P2
  let (e, h, ph, pe) = plain_init f
  let (e, _, _, _) = loop (e, h, ph, pe) for _i < nsteps do plain_step g m e h ph pe
  in f32.sum (map (\x -> x[0, 0, 0]) e)

-- Bitwise check: one tiled pass (2 steps) against two plain steps from the
-- same fields and z memory. Returns the number of differing field values and
-- differing z memories.
entry check_plain (P0: i64) (P1: i64) (P2: i64) : (i64, i64) =
  let (g, m, f, pz) = setup P0 P1 P2
  let (ft, pzt) = pass g m f f pz
  let (e, h, _, _) = plain_init f
  let dense s = map (\pt -> tabulate P0 (\k -> let r = zrow P0 k
                                              in if r >= 0 then map (\x -> [x[s], x[s + 1]]) pt[r]
                                                 else replicate (TY * TX) [0f32, 0f32])) pz
  let (e, h, ph, pe) = plain_step g m e h (dense 0) (dense 2)
  let (e, h, ph, pe) = plain_step g m e h ph pe
  let ne a b = i64.bool (f32.to_bits a != f32.to_bits b)
  let df = map3 (map3 (map3 (\x a b -> i64.sum (map2 ne x (concat a b :> [6]f32))))) ft e h
           |> flatten_3d |> i64.sum
  let dz = if !ZPML then 0
           else map3 (\pt a b -> i64.sum (tabulate P0 (\k ->
                                    let r = zrow P0 k
                                    in if r < 0 then 0
                                       else i64.sum (map3 (\x u v -> ne x[0] u[0] + ne x[1] u[1] + ne x[2] v[0] + ne x[3] v[1])
                                                          pt[r] a[k] b[k]))))
                     pzt ph pe |> i64.sum
  in (df, dz)
