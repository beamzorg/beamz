-- Bare temporal-tiling experiment: homogeneous Yee update on a box with zero
-- fields outside, no CPML, sources or monitors. Compares a plain one-step
-- schedule against 2.5D tiles that stream z planes through shared memory and
-- advance K timesteps per pass.
--
-- All six components live on the same [nz][ny][nx] cells; samples outside the
-- box read as zero. H takes forward differences, E backward differences.
--
-- Tiled state uses the layout [nty][ntx][nz][TY][TX]: each xy tile owns a
-- contiguous z column. TY, TX and K are compile-time constants (see gen.py).

def TY : i64 = 8
def TX : i64 = 32
def ZC : i64 = 8
def K : i64 = 2

-- Window halo; K unless a timing experiment decouples it (gen.py HY=.. HX=..).
def HY : i64 = K
def HX : i64 = K
def WY : i64 = TY + 2 * HY
def WX : i64 = TX + 2 * HX

type fields 'a = (a, a, a, a, a, a)

def hupd (ch: f32) (h: f32) (a: f32) (a1: f32) (b: f32) (b1: f32) : f32 =
  h - ch * ((a1 - a) - (b1 - b))

def eupd (cb: f32) (e: f32) (a: f32) (a0: f32) (b: f32) (b0: f32) : f32 =
  e + cb * ((a - a0) - (b - b0))

-- Plain reference ---------------------------------------------------------

def get [nz][ny][nx] (a: [nz][ny][nx]f32) (k: i64) (j: i64) (i: i64) : f32 =
  if k >= 0 && k < nz && j >= 0 && j < ny && i >= 0 && i < nx then #[unsafe] a[k, j, i] else 0

def plain_step [nz][ny][nx] (ch: f32) (cb: f32)
    ((ex, ey, ez, hx, hy, hz): fields ([nz][ny][nx]f32)) : fields ([nz][ny][nx]f32) =
  let hx' = tabulate_3d nz ny nx (\k j i ->
              hupd ch (#[unsafe] hx[k, j, i]) (get ez k j i) (get ez k (j + 1) i) (get ey k j i) (get ey (k + 1) j i))
  let hy' = tabulate_3d nz ny nx (\k j i ->
              hupd ch (#[unsafe] hy[k, j, i]) (get ex k j i) (get ex (k + 1) j i) (get ez k j i) (get ez k j (i + 1)))
  let hz' = tabulate_3d nz ny nx (\k j i ->
              hupd ch (#[unsafe] hz[k, j, i]) (get ey k j i) (get ey k j (i + 1)) (get ex k j i) (get ex k (j + 1) i))
  let ex' = tabulate_3d nz ny nx (\k j i ->
              eupd cb (#[unsafe] ex[k, j, i]) (get hz' k j i) (get hz' k (j - 1) i) (get hy' k j i) (get hy' (k - 1) j i))
  let ey' = tabulate_3d nz ny nx (\k j i ->
              eupd cb (#[unsafe] ey[k, j, i]) (get hx' k j i) (get hx' (k - 1) j i) (get hz' k j i) (get hz' k j (i - 1)))
  let ez' = tabulate_3d nz ny nx (\k j i ->
              eupd cb (#[unsafe] ez[k, j, i]) (get hy' k j i) (get hy' k j (i - 1)) (get hx' k j i) (get hx' k (j - 1) i))
  in (ex', ey', ez', hx', hy', hz')

-- Tiled layout --------------------------------------------------------------

-- [xy tile][z chunk][component][z in chunk][row-major TY x TX]; components
-- are Ex Ey Ez Hx Hy Hz. A block computes one (xy tile, z chunk).
type tiled [nt][nzc] = [nt][nzc][6][ZC][TY * TX]f32

def to_tiled [nz][ny][nx] (nty: i64) (ntx: i64) (nzc: i64) (s: fields ([nz][ny][nx]f32))
    : *tiled [nty * ntx][nzc] =
  let cs = [s.0, s.1, s.2, s.3, s.4, s.5]
  in tabulate_3d (nty * ntx) nzc 6 (\t zc c ->
       tabulate_2d ZC (TY * TX) (\z q ->
         get cs[c] (zc * ZC + z) ((t / ntx) * TY + q / TX) ((t % ntx) * TX + q % TX)))

def from_tiled [nt][nzc] (ntx: i64) (nz: i64) (ny: i64) (nx: i64) (a: tiled [nt][nzc])
    : fields ([nz][ny][nx]f32) =
  let u c = tabulate_3d nz ny nx (\k j i ->
              #[unsafe] a[(j / TY) * ntx + i / TX, k / ZC, c, k % ZC, (j % TY) * TX + i % TX])
  in (u 0, u 1, u 2, u 3, u 4, u 5)

type plane = [WY][WX]f32

def load [nt][nzc] (ntx: i64) (nz: i64) (ny: i64) (nx: i64) (a: tiled [nt][nzc])
         (ty: i64) (tx: i64) (c: i64) (z: i64) : *plane =
  tabulate_2d WY WX (\y x ->
    let gy = ty * TY - K + y
    let gx = tx * TX - K + x
    in if z >= 0 && z < nz && gy >= 0 && gy < ny && gx >= 0 && gx < nx
       then #[unsafe] a[(gy / TY) * ntx + gx / TX, z / ZC, c, z % ZC, (gy % TY) * TX + gx % TX]
       else 0)

-- One magnetic stage on plane p: h is H at p, e0 and e1 are E at p and p + 1.
-- Window-edge neighbours are clamped; they only pollute the halo.
def hplane (nz: i64) (ny: i64) (nx: i64) (ch: f32) (ty: i64) (tx: i64) (p: i64)
           ((hx, hy, hz): (plane, plane, plane))
           ((ex, ey, ez): (plane, plane, plane))
           ((ex1, ey1, _ez1): (plane, plane, plane)) : (*plane, *plane, *plane) =
  let valid y x = let gy = ty * TY - K + y
                  let gx = tx * TX - K + x
                  in p >= 0 && p < nz && gy >= 0 && gy < ny && gx >= 0 && gx < nx
  let y1 y = i64.min (y + 1) (WY - 1)
  let x1 x = i64.min (x + 1) (WX - 1)
  in #[unsafe]
     (tabulate_2d WY WX (\y x -> if !(valid y x) then 0
                                 else hupd ch hx[y, x] ez[y, x] ez[y1 y, x] ey[y, x] ey1[y, x]),
      tabulate_2d WY WX (\y x -> if !(valid y x) then 0
                                 else hupd ch hy[y, x] ex[y, x] ex1[y, x] ez[y, x] ez[y, x1 x]),
      tabulate_2d WY WX (\y x -> if !(valid y x) then 0
                                 else hupd ch hz[y, x] ey[y, x] ey[y, x1 x] ex[y, x] ex[y1 y, x]))

-- One electric stage on plane p: h and h0 are H at p and p - 1.
def eplane (nz: i64) (ny: i64) (nx: i64) (cb: f32) (ty: i64) (tx: i64) (p: i64)
           ((ex, ey, ez): (plane, plane, plane))
           ((hx, hy, hz): (plane, plane, plane))
           ((hx0, hy0, _hz0): (plane, plane, plane)) : (*plane, *plane, *plane) =
  let valid y x = let gy = ty * TY - K + y
                  let gx = tx * TX - K + x
                  in p >= 0 && p < nz && gy >= 0 && gy < ny && gx >= 0 && gx < nx
  let ym y = i64.max (y - 1) 0
  let xm x = i64.max (x - 1) 0
  in #[unsafe]
     (tabulate_2d WY WX (\y x -> if !(valid y x) then 0
                                 else eupd cb ex[y, x] hz[y, x] hz[ym y, x] hy[y, x] hy0[y, x]),
      tabulate_2d WY WX (\y x -> if !(valid y x) then 0
                                 else eupd cb ey[y, x] hx[y, x] hx0[y, x] hz[y, x] hz[y, xm x]),
      tabulate_2d WY WX (\y x -> if !(valid y x) then 0
                                 else eupd cb ez[y, x] hy[y, x] hy[y, xm x] hx[y, x] hx[ym y, x]))

def core (a: plane) : *[TY * TX]f32 =
  #[unsafe] tabulate (TY * TX) (\q -> a[q / TX + K, q % TX + K])

def zero_plane : plane = replicate WY (replicate WX 0)

-- Advance K steps. Stage s (0-based) turns (E_s, H_s) into (E_s+1, H_s+1) on
-- plane z - s, so each stage lags the previous one by one plane. A chunk
-- starts K planes early so that every stage is exact on the planes the next
-- one reads; the earliest outputs are overwritten once they become exact.
def tiled_pass [nt][nzc] (ntx: i64) (nz: i64) (ny: i64) (nx: i64) (ch: f32) (cb: f32)
    (src: tiled [nt][nzc]) : *tiled [nt][nzc] =
  let run b =
    let t = b / nzc
    let z0 = (b % nzc) * ZC
    let ty = t / ntx
    let tx = t % ntx
    let ld c z = load ntx nz ny nx src ty tx c z
    let lde z = (ld 0 z, ld 1 z, ld 2 z)
    let ldh z = (ld 3 z, ld 4 z, ld 5 z)
    let zero3 = (zero_plane, zero_plane, zero_plane)
    -- ep[s]: E_s on the previous plane of stage s; hp[s]: H_s+1 likewise.
    let ep = replicate K zero3
    let ep[0] = lde (z0 - K)
    let hp = replicate K zero3
    let out = replicate 6 (replicate ZC (replicate (TY * TX) 0f32))
    let (_, _, out) =
      loop (ep, hp, out) for i < ZC + 2 * K - 1 do
        let z = z0 - K + i
        let (ep, hp, e, _) =
          loop (ep, hp, enext, hin) = (ep, hp, lde (z + 1), ldh z) for s < K do
            let p = z - s
            let h = hplane nz ny nx ch ty tx p hin ep[s] enext
            let e = eplane nz ny nx cb ty tx p ep[s] h hp[s]
            let hin' = copy hp[s]
            let ep[s] = enext
            let hp[s] = h
            in (ep, hp, e, hin')
        let h = hp[K - 1]
        let o = i64.max 0 (z - (K - 1) - z0)
        let out[0, o] = core e.0
        let out[1, o] = core e.1
        let out[2, o] = core e.2
        let out[3, o] = core h.0
        let out[4, o] = core h.1
        let out[5, o] = core h.2
        in (ep, hp, out)
    in out
  in unflatten (#[flattening(only_intra)] tabulate (nt * nzc) run)

-- Entry points --------------------------------------------------------------

def init (nz: i64) (ny: i64) (nx: i64) : fields ([nz][ny][nx]f32) =
  let f seed = tabulate_3d nz ny nx (\k j i ->
                 let h = u32.i64 (((k * 7919 + j) * 104729 + i) * 31 + seed)
                 let h = (h ^ (h >> 16)) * 0x45d9f3b
                 let h = (h ^ (h >> 16)) * 0x45d9f3b
                 let h = h ^ (h >> 16)
                 in f32.u32 (h & 0xffff) / 65536 - 0.5)
  in (f 1, f 2, f 3, f 4, f 5, f 6)

def ch : f32 = 0.5
def cb : f32 = 0.5

def plain_run [nz][ny][nx] (nsteps: i64) (s: fields ([nz][ny][nx]f32)) =
  loop s for _i < nsteps do plain_step ch cb s

def tiled_run [nz][ny][nx] (nsteps: i64) (s: fields ([nz][ny][nx]f32)) =
  let nty = (ny + TY - 1) / TY
  let ntx = (nx + TX - 1) / TX
  let nzc = (nz + ZC - 1) / ZC
  let a = to_tiled nty ntx nzc s
  -- nsteps must be a multiple of K here.
  let a = loop a for _i < nsteps / K do tiled_pass ntx nz ny nx ch cb a
  in from_tiled ntx nz ny nx a

-- Plain one-step update on a z-column tiled layout [nt*nzc*ZC][TY*TX] per
-- component: same arithmetic, no shared memory; tests L2 locality alone.
type col [m] = [m][TY * TX]f32

def tget [m] (ntx: i64) (nzc: i64) (nz: i64) (ny: i64) (nx: i64) (a: col [m]) (k: i64) (j: i64) (i: i64) : f32 =
  if k >= 0 && k < nz && j >= 0 && j < ny && i >= 0 && i < nx
  then #[unsafe] a[((j / TY) * ntx + i / TX) * (nzc * ZC) + k, (j % TY) * TX + i % TX]
  else 0

def coltab (m: i64) (ntx: i64) (nzc: i64) (f: i64 -> i64 -> i64 -> f32) : *col [m] =
  tabulate_2d m (TY * TX) (\r q ->
    let t = r / (nzc * ZC)
    in f (r % (nzc * ZC)) ((t / ntx) * TY + q / TX) ((t % ntx) * TX + q % TX))

def coltiled_step [m] (ntx: i64) (nzc: i64) (nz: i64) (ny: i64) (nx: i64)
    ((ex, ey, ez, hx, hy, hz): fields (col [m])) : fields (col [m]) =
  let g = tget ntx nzc nz ny nx
  let tab f = coltab m ntx nzc (\k j i -> if k < nz && j < ny && i < nx then f k j i else 0)
  let hx' = tab (\k j i -> hupd ch (g hx k j i) (g ez k j i) (g ez k (j + 1) i) (g ey k j i) (g ey (k + 1) j i))
  let hy' = tab (\k j i -> hupd ch (g hy k j i) (g ex k j i) (g ex (k + 1) j i) (g ez k j i) (g ez k j (i + 1)))
  let hz' = tab (\k j i -> hupd ch (g hz k j i) (g ey k j i) (g ey k j (i + 1)) (g ex k j i) (g ex k (j + 1) i))
  let ex' = tab (\k j i -> eupd cb (g ex k j i) (g hz' k j i) (g hz' k (j - 1) i) (g hy' k j i) (g hy' (k - 1) j i))
  let ey' = tab (\k j i -> eupd cb (g ey k j i) (g hx' k j i) (g hx' (k - 1) j i) (g hz' k j i) (g hz' k j (i - 1)))
  let ez' = tab (\k j i -> eupd cb (g ez k j i) (g hy' k j i) (g hy' k j (i - 1)) (g hx' k j i) (g hx' k (j - 1) i))
  in (ex', ey', ez', hx', hy', hz')

def coltiled_run [nz][ny][nx] (nsteps: i64) (s: fields ([nz][ny][nx]f32)) =
  let nty = (ny + TY - 1) / TY
  let ntx = (nx + TX - 1) / TX
  let nzc = (nz + ZC - 1) / ZC
  let m = nty * ntx * nzc * ZC
  let to a = coltab m ntx nzc (\k j i -> get a k j i)
  let from a = tabulate_3d nz ny nx (\k j i -> tget ntx nzc nz ny nx a k j i)
  let s = (to s.0, to s.1, to s.2, to s.3, to s.4, to s.5)
  let (a, b, c, d, e, f) = loop s for _i < nsteps do coltiled_step ntx nzc nz ny nx s
  in (from a, from b, from c, from d, from e, from f)

def sum6 [nz][ny][nx] ((a, b, c, d, e, f): fields ([nz][ny][nx]f32)) : f32 =
  f32.sum (map (\x -> f32.sum (flatten_3d x)) [a, b, c, d, e, f])

-- Number of differing samples and max abs difference between both schedules.
entry check (nz: i64) (ny: i64) (nx: i64) (nsteps: i64) (variant: i64) : (i64, f32, f32) =
  let s = init nz ny nx
  let (a0, a1, a2, a3, a4, a5) = plain_run nsteps s
  let (b0, b1, b2, b3, b4, b5) = if variant == 0 then tiled_run nsteps s else coltiled_run nsteps s
  let cmp x y = let d = map2 (\u v -> (i64.bool (f32.to_bits u != f32.to_bits v), f32.abs (u - v)))
                             (flatten_3d x) (flatten_3d y)
                in (i64.sum (map (.0) d), f32.maximum (map (.1) d))
  let r = [cmp a0 b0, cmp a1 b1, cmp a2 b2, cmp a3 b3, cmp a4 b4, cmp a5 b5]
  in (i64.sum (map (.0) r), f32.maximum (map (.1) r), sum6 (a0, a1, a2, a3, a4, a5))

-- ==
-- entry: bench_plain bench_tiled
-- input { 128i64 256i64 512i64 64i64 }
entry bench_plain (nz: i64) (ny: i64) (nx: i64) (nsteps: i64) : f32 =
  sum6 (plain_run nsteps (init nz ny nx))

entry bench_tiled (nz: i64) (ny: i64) (nx: i64) (nsteps: i64) : f32 =
  sum6 (tiled_run nsteps (init nz ny nx))

entry bench_coltiled (nz: i64) (ny: i64) (nx: i64) (nsteps: i64) : f32 =
  sum6 (coltiled_run nsteps (init nz ny nx))
