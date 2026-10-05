import "../../temporal"
def mk (P0: i64) (P1: i64) (P2: i64) (L: i64) (common: bool) : ctx =
  let g = {P0, P1, P2, ntx = P2 / 32}
  in {loc = {g, zb = 4, zt = P0 - 4, Zs = 8, icidx = 96, isbase = 96},
      mat = {cf = {common, ne = ((P0,P1,P2),(P0,P1,P2),(P0,P1,P2)), fcoef = 3*L, icdim = 24, ikdim = 84,
                   itdim = 90, kcode = 0, ftab = 0},
             edges = 0, nh = ((P0,P1,P2),(P0,P1,P2),(P0,P1,P2)), ne = ((P0,P1,P2),(P0,P1,P2),(P0,P1,P2)),
             kind = 0, inv = 1, L, fca = 0, fcb = L, fck = 2*L, fmet = 0, islab = 0, imdim = 72},
      src = {g, kslot = 0, kent = 0, R = 1}, ictile = 96, istile = 96, isz0 = 96, islen = 96}
entry main [nc][Zc][ns][Zs] (P0: i64) (P1: i64) (P2: i64) (L: i64) (c: bool) (fb: []f32) (ib: []i64) (kb: []i32)
    (core: [nc][Zc][TY * TX][6]f32) (shell: [ns][Zs][TY * TX][6]f32) =
  core_step (mk P0 P1 P2 L c) fb ib kb core shell
