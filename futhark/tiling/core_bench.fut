-- Time temporal.fut's core kernel alone on a synthetic run: storage P0 x P1
-- x P2, every tile whose window keeps PML cells away is a core tile (all four
-- quadrants), the first `cap` of them at most. Build with `futhark cuda` (the
-- checkout build.py uses, e.g. via fc.sh) and time two step counts; `digest`
-- hashes one pass's core cells, to check that a variant is bit-identical.

import "../temporal"

def setup (P0: i64) (P1: i64) (P2: i64) (pml: i64) (cap: i64) =
  let ntx = (P2 + TX - 1) / TX
  let nt = ((P1 + TY - 1) / TY) * ntx
  let g = {P0, P1, P2, ntx}
  let h q = let h = u32.i64 (q * 2654435761 + 12345)
            let h = (h ^ (h >> 16)) * 0x45d9f3b
            in f32.u32 ((h ^ (h >> 16)) & 0xffff) / 65536 - 0.5
  let a o = tabulate_3d nt P0 (TY * TX) (\t k l -> tabulate 3 (\c -> h (((t * P0 + k) * (TY * TX) + l) * 6 + o + c)))
  let a = (a 0, a 3)
  let core t = let wy = (t / ntx) * TY - 2
               let wx = (t % ntx) * TX - 2
               in wy > pml && wy + WY < P1 - pml - 2 && wx > pml && wx + WX < P2 - pml - 2
  let cnum = scan (+) 0 (map (i64.bool <-< core) (iota nt))
  let cmask = map2 (\t n -> if core t && n <= cap then 15i64 else 0) (iota nt) cnum
  let n = (P0 - 1, P1 - 1, P2 - 1)
  let C = (n.0 * n.1 * n.2 + 3) / 4
  -- Material codes, a different array per component (as in a run).
  let kb s = tabulate C (\q -> i32.u32 ((u32.i64 q * 2654435761 + s) & 0x01010101))
  let kb = (kb 0, kb 77, kb 991)
  let fb = [0.5f32, 0.12f32, 0.5, 0.12, 0.5, 0.12]
  let cf : corecf = {inv = 1, hd0 = 1, hd1 = 1, hd2 = 1, hs0 = 0.5, hs1 = 0.5, hs2 = 0.5,
            ed0 = 1, ed1 = 1, ed2 = 1, ne = (n, n, n)}
  in (g, cf, kb, fb, cmask, a, pml + 3, P0 - pml - 3, i64.sum (map (\c -> i64.bool (c != 0)) cmask))

-- ==
-- entry: bench
-- input { 129i64 257i64 513i64 12i64 100000i64 0i64 }
-- input { 129i64 257i64 513i64 12i64 100000i64 32i64 }
entry bench (P0: i64) (P1: i64) (P2: i64) (pml: i64) (cap: i64) (nsteps: i64) : f32 =
  let (g, cf, kb, fb, cmask, a, zb, zt, _) = setup P0 P1 P2 pml cap
  -- The input stays fixed (zb varies so that the pass is not hoisted): a
  -- loop-carried state would add copies to the timing.
  in loop acc = 0f32 for i < nsteps / 2 do
       let (e, h) = core_pass g (zb + (i & 1)) zt cf kb.0 kb.1 kb.2 fb[0:2] fb[2:4] fb[4:6] cmask a.0 a.1
       in acc + e[i % 7, 64, 3, 0] + h[i % 7, 64, 3, 1]

-- Core cells updated per pass (for GCUPS).
entry cells (P0: i64) (P1: i64) (P2: i64) (pml: i64) (cap: i64) : i64 =
  let (_, _, _, _, _, _, zb, zt, nc) = setup P0 P1 P2 pml cap
  in nc * (zt - zb) * TY * TX

-- Hash of the core cells after one pass.
entry digest (P0: i64) (P1: i64) (P2: i64) (pml: i64) : u64 =
  let (g, cf, kb, fb, cmask, a, zb, zt, _) = setup P0 P1 P2 pml 100000
  let (e, h) = core_pass g zb zt cf kb.0 kb.1 kb.2 fb[0:2] fb[2:4] fb[4:6] cmask a.0 a.1
  let nt = length cmask
  let v t k l c = if cmask[t] != 0 && k >= zb && k < zt
                  then let x = u64.u32 (f32.to_bits e[t, k, l, c]) * 0x9E3779B97F4A7C15
                                + u64.u32 (f32.to_bits h[t, k, l, c]) * 0xC2B2AE3D27D4EB4F
                       in x ^ (u64.i64 (((t * P0 + k) * (TY * TX) + l) * 3 + c) * 0x165667B19E3779F9)
                  else 0
  in u64.sum (flatten (flatten (flatten (tabulate_3d nt P0 (TY * TX) (\t k l -> tabulate 3 (v t k l))))))
