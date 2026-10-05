-- Full-column 2.5D streaming variant of bare.fut: each block owns one xy tile
-- and streams every z plane through K pipelined stages. Futhark keeps the
-- block's output column in shared memory, so this only runs once the kernel
-- is patched to write that column to global memory (see patch_cuda.py).

import "bare"

def LTY : i32 = 2
def LTX : i32 = 5

type ctiled [nt][nz] = [nt][6][nz][TY * TX]f32

def to_ctiled [nz][ny][nx] (nty: i64) (ntx: i64) (s: fields ([nz][ny][nx]f32)) : *ctiled [nty * ntx][nz] =
  let cs = [s.0, s.1, s.2, s.3, s.4, s.5]
  in tabulate_3d (nty * ntx) 6 nz (\t c k ->
       tabulate (TY * TX) (\q -> get cs[c] k ((t / ntx) * TY + q / TX) ((t % ntx) * TX + q % TX)))

def from_ctiled [nt][nz] (ntx: i64) (ny: i64) (nx: i64) (a: ctiled [nt][nz]) : fields ([nz][ny][nx]f32) =
  let u c = tabulate_3d nz ny nx (\k j i -> #[unsafe] a[(j / TY) * ntx + i / TX, c, k, (j % TY) * TX + i % TX])
  in (u 0, u 1, u 2, u 3, u 4, u 5)

-- Window coordinates are i32 and nonnegative where they index memory, so
-- tiles and offsets are shifts and masks.
def cload [nt][nz] (ntx: i32) (ny: i32) (nx: i32) (a: ctiled [nt][nz])
          (ty: i32) (tx: i32) (c: i64) (z: i32) : *plane =
  tabulate_2d WY WX (\y x ->
    let gy = ty * i32.i64 TY - i32.i64 K + i32.i64 y
    let gx = tx * i32.i64 TX - i32.i64 K + i32.i64 x
    in if z >= 0 && z < i32.i64 nz && gy >= 0 && gy < ny && gx >= 0 && gx < nx
       then #[unsafe] a[i64.i32 ((gy >> LTY) * ntx + (gx >> LTX)), c, i64.i32 z,
                        i64.i32 (((gy & (i32.i64 TY - 1)) << LTX) | (gx & (i32.i64 TX - 1)))]
       else 0)

def cpass [nt][nz] (ntx: i64) (ny: i64) (nx: i64) (ch: f32) (cb: f32)
    (src: ctiled [nt][nz]) : *ctiled [nt][nz] =
  let run t =
    let ty = t / ntx
    let tx = t % ntx
    let ld c z = cload (i32.i64 ntx) (i32.i64 ny) (i32.i64 nx) src (i32.i64 ty) (i32.i64 tx) c (i32.i64 z)
    let lde z = (ld 0 z, ld 1 z, ld 2 z)
    let ldh z = (ld 3 z, ld 4 z, ld 5 z)
    let zero3 = (zero_plane, zero_plane, zero_plane)
    let ep = replicate K zero3
    let ep[0] = lde 0
    let hp = replicate K zero3
    let out = #[scratch] replicate 6 (replicate nz (replicate (TY * TX) 0f32))
    let (_, _, out) =
      loop (ep, hp, out) for z < nz + K - 1 do
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
        let p = i64.max 0 (z - (K - 1))
        let out[0, p] = core e.0
        let out[1, p] = core e.1
        let out[2, p] = core e.2
        let out[3, p] = core h.0
        let out[4, p] = core h.1
        let out[5, p] = core h.2
        in (ep, hp, out)
    in out
  in #[unsafe] #[flattening(only_intra)] tabulate nt run

def ctiled_run [nz][ny][nx] (nsteps: i64) (s: fields ([nz][ny][nx]f32)) =
  let nty = (ny + TY - 1) / TY
  let ntx = (nx + TX - 1) / TX
  let a = to_ctiled nty ntx s
  let a = loop a for _i < nsteps / K do cpass ntx ny nx ch cb a
  in from_ctiled ntx ny nx a

entry check2 (nz: i64) (ny: i64) (nx: i64) (nsteps: i64) : (i64, f32) =
  let s = init nz ny nx
  let (a0, a1, a2, a3, a4, a5) = plain_run nsteps s
  let (b0, b1, b2, b3, b4, b5) = ctiled_run nsteps s
  let cmp x y = let d = map2 (\u v -> (i64.bool (f32.to_bits u != f32.to_bits v), f32.abs (u - v)))
                             (flatten_3d x) (flatten_3d y)
                in (i64.sum (map (.0) d), f32.maximum (map (.1) d))
  let r = [cmp a0 b0, cmp a1 b1, cmp a2 b2, cmp a3 b3, cmp a4 b4, cmp a5 b5]
  in (i64.sum (map (.0) r), f32.maximum (map (.1) r))

entry bench_ctiled (nz: i64) (ny: i64) (nx: i64) (nsteps: i64) : f32 =
  sum6 (ctiled_run nsteps (init nz ny nx))
