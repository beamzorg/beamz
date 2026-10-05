-- Entry point of BeamZ's Futhark backend; the numerics live in yee.fut.

import "yee"
import "temporal"

-- Advance a whole run. Scalars that do not change between invocations are
-- static FFI attributes; clocks arrive as one-element device arrays so that
-- chunked runs never synchronize the host. Source targets and monitor plans
-- index the padded run storage.
entry program
    [z0][y0][x0][z1][y1][x1][z2][y2][x2][z3][y3][x3][z4][y4][x4][z5][y5][x5]
    [L][K][M][P][N][F][R][Q]
    (nsteps: i64) (dt: f32) (inv_resolution: f32) (edges: i64) (metric_kind: i64) (temporal: i64)
    (ex: [z0][y0][x0]f32) (ey: [z1][y1][x1]f32) (ez: [z2][y2][x2]f32)
    (hx: [z3][y3][x3]f32) (hy: [z4][y4][x4]f32) (hz: [z5][y5][x5]f32)
    (h_decay_x: [][][]f32) (h_decay_y: [][][]f32) (h_decay_z: [][][]f32)
    (h_source_x: [][][]f32) (h_source_y: [][][]f32) (h_source_z: [][][]f32)
    (e_decay_x: [][][]f32) (e_decay_y: [][][]f32) (e_decay_z: [][][]f32)
    (e_source_x: [][][]f32) (e_source_y: [][][]f32) (e_source_z: [][][]f32)
    (e_table_x: []f32) (e_table_y: []f32) (e_table_z: []f32)
    (e_codes_x: []i32) (e_codes_y: []i32) (e_codes_z: []i32)
    (h_metric_z: []f32) (h_metric_y: []f32) (h_metric_x: []f32)
    (e_metric_z: []f32) (e_metric_y: []f32) (e_metric_x: []f32)
    (cpml_slabs: [12][2]i32) (cpml_a: [12][L]f32) (cpml_b: [12][L]f32) (cpml_inv_kappa: [12][L]f32)
    (psi_h0: [][][]f32) (psi_h1: [][][]f32) (psi_h2: [][][]f32)
    (psi_h3: [][][]f32) (psi_h4: [][][]f32) (psi_h5: [][][]f32)
    (psi_e0: [][][]f32) (psi_e1: [][][]f32) (psi_e2: [][][]f32)
    (psi_e3: [][][]f32) (psi_e4: [][][]f32) (psi_e5: [][][]f32)
    (source_target: [K]i32) (source_amplitude: [K]f32) (source_offset: [K]i32)
    (source_length: [K]i32) (source_group: [K]i32) (waveforms: []f32)
    (monitor_indices: [M][6][N][P]i32) (monitor_weights: [M][6][N][P]f32)
    (monitor_freqs: [M][F]f32) (monitor_masks: [M][6]f32) (monitor_counts: [M][5]i32)
    (monitor_codes: [M][2]i32) (monitor_windows: [M][3]f32)
    (dft_re: [R]f32) (dft_im: [R]f32) (dft_weight: [Q]f32)
    (current_step: [1]i32) (time_origin: [1]f32) (elapsed_steps: [1]i32) =
  let slabs = map (map i64.i32) cpml_slabs
  let h_scale axis co =
    if axis == 0 then metric metric_kind inv_resolution h_metric_z co
    else if axis == 1 then metric metric_kind inv_resolution h_metric_y co
    else metric metric_kind inv_resolution h_metric_x co
  let e_scale axis co =
    if axis == 0 then metric metric_kind inv_resolution e_metric_z co
    else if axis == 1 then metric metric_kind inv_resolution e_metric_y co
    else metric metric_kind inv_resolution e_metric_x co
  let P0 = i64.maximum [z0, z1, z2, z3, z4, z5]
  let P1 = i64.maximum [y0, y1, y2, y3, y4, y5]
  let P2 = i64.maximum [x0, x1, x2, x3, x4, x5]
  let ne = (dims3 ex, dims3 ey, dims3 ez)
  let nh = (dims3 hx, dims3 hy, dims3 hz)
  let h_decay = (h_decay_x, h_decay_y, h_decay_z)
  let dense v = (v, [] : []f32, [] : []i32)
  let h_source = (dense h_source_x, dense h_source_y, dense h_source_z)
  let e_decay = (e_decay_x, e_decay_y, e_decay_z)
  let e_source = ((e_source_x, e_table_x, e_codes_x),
                  (e_source_y, e_table_y, e_codes_y),
                  (e_source_z, e_table_z, e_codes_z))
  let inject' (field: *[P0][P1][P2]f32) step group : *[P0][P1][P2]f32 =
    inject field step group source_group source_target source_amplitude
           source_offset source_length waveforms
  let step0 = i64.i32 current_step[0]
  let time_of s = f32.fma (f32.i64 (i64.i32 elapsed_steps[0] + s + 1)) dt time_origin[0]
  -- One plain step s of this run (all phases as separate kernels).
  let plain_step s (ex: *[P0][P1][P2]f32, ey: *[P0][P1][P2]f32, ez: *[P0][P1][P2]f32,
                    hx: *[P0][P1][P2]f32, hy: *[P0][P1][P2]f32, hz: *[P0][P1][P2]f32,
                    psi_h0: *[][][]f32, psi_h1: *[][][]f32, psi_h2: *[][][]f32,
                    psi_h3: *[][][]f32, psi_h4: *[][][]f32, psi_h5: *[][][]f32,
                    psi_e0: *[][][]f32, psi_e1: *[][][]f32, psi_e2: *[][][]f32,
                    psi_e3: *[][][]f32, psi_e4: *[][][]f32, psi_e5: *[][][]f32,
                    dft_re: *[R]f32, dft_im: *[R]f32, dft_weight: *[Q]f32) =
        let step = step0 + s
        -- 1. Pre-E sources were applied at the end of the previous iteration
        --    (or before the loop), after that step's observation.
        -- 2. Magnetic phase.
        let (hx', hy', hz') =
          phase_next 0 edges h_scale nh (hx, hy, hz) h_decay h_source ne (ex, ey, ez)
                     slabs cpml_a cpml_b cpml_inv_kappa
                     (psi_h0, psi_h1, psi_h2, psi_h3, psi_h4, psi_h5)
        let psi_h0 = psi_next 0 edges h_scale nh.0 0 (at3 ez) ne.2 1 slabs cpml_a cpml_b psi_h0
        let psi_h1 = psi_next 0 edges h_scale nh.0 1 (at3 ey) ne.1 0 slabs cpml_a cpml_b psi_h1
        let psi_h2 = psi_next 0 edges h_scale nh.1 2 (at3 ex) ne.0 0 slabs cpml_a cpml_b psi_h2
        let psi_h3 = psi_next 0 edges h_scale nh.1 3 (at3 ez) ne.2 2 slabs cpml_a cpml_b psi_h3
        let psi_h4 = psi_next 0 edges h_scale nh.2 4 (at3 ey) ne.1 2 slabs cpml_a cpml_b psi_h4
        let psi_h5 = psi_next 0 edges h_scale nh.2 5 (at3 ex) ne.0 1 slabs cpml_a cpml_b psi_h5
        -- 3. Magnetic sources.
        let hx = inject' hx' step 3
        let hy = inject' hy' step 4
        let hz = inject' hz' step 5
        -- 4. Electric phase.
        let (ex', ey', ez') =
          phase_next 1 edges e_scale ne (ex, ey, ez) e_decay e_source nh (hx, hy, hz)
                     slabs cpml_a cpml_b cpml_inv_kappa
                     (psi_e0, psi_e1, psi_e2, psi_e3, psi_e4, psi_e5)
        let psi_e0 = psi_next 1 edges e_scale ne.0 6 (at3 hz) nh.2 1 slabs cpml_a cpml_b psi_e0
        let psi_e1 = psi_next 1 edges e_scale ne.0 7 (at3 hy) nh.1 0 slabs cpml_a cpml_b psi_e1
        let psi_e2 = psi_next 1 edges e_scale ne.1 8 (at3 hx) nh.0 0 slabs cpml_a cpml_b psi_e2
        let psi_e3 = psi_next 1 edges e_scale ne.1 9 (at3 hz) nh.2 2 slabs cpml_a cpml_b psi_e3
        let psi_e4 = psi_next 1 edges e_scale ne.2 10 (at3 hy) nh.1 2 slabs cpml_a cpml_b psi_e4
        let psi_e5 = psi_next 1 edges e_scale ne.2 11 (at3 hx) nh.0 1 slabs cpml_a cpml_b psi_e5
        -- 5. Electric sources.
        let ex = inject' ex' step 6
        let ey = inject' ey' step 7
        let ez = inject' ez' step 8
        -- 6. Observe the fully constrained end-of-step fields.
        let time = time_of s
        let (dft_re, dft_im, dft_weight) =
          if M == 0 then (dft_re, dft_im, dft_weight)
          else let at f o = #[unsafe] (flatten_3d f)[o]
               let sample c o = if c == 0 then at ex o else if c == 1 then at ey o
                                else if c == 2 then at ez o else if c == 3 then at hx o
                                else if c == 4 then at hy o else at hz o
               in accumulate_dft step time dt (P0 * P1 * P2) sample
                 monitor_indices monitor_weights monitor_freqs monitor_masks
                 monitor_counts monitor_codes monitor_windows dft_re dft_im dft_weight
        -- Pre-E sources of the next step, applied after this observation. The
        -- final step selects no group rather than branching, which keeps the
        -- loop-carried E fields in their direct layout.
        let pre = if s + 1 == nsteps then -1 else 0
        let ex = inject' ex (step + 1) pre
        let ey = inject' ey (step + 1) (pre + i32.bool (pre >= 0))
        let ez = inject' ez (step + 1) (pre + 2 * i32.bool (pre >= 0))
        in (ex, ey, ez, hx, hy, hz,
            psi_h0, psi_h1, psi_h2, psi_h3, psi_h4, psi_h5,
            psi_e0, psi_e1, psi_e2, psi_e3, psi_e4, psi_e5,
            dft_re, dft_im, dft_weight)
  -- Inputs alias XLA buffers that are also outputs, so the run works on
  -- private copies and the handler copies results back at the end.
  let start () =
    (inject' (pad P0 P1 P2 ex) step0 0,
     inject' (pad P0 P1 P2 ey) step0 1,
     inject' (pad P0 P1 P2 ez) step0 2,
     pad P0 P1 P2 hx, pad P0 P1 P2 hy, pad P0 P1 P2 hz,
     copy psi_h0, copy psi_h1, copy psi_h2, copy psi_h3, copy psi_h4, copy psi_h5,
     copy psi_e0, copy psi_e1, copy psi_e2, copy psi_e3, copy psi_e4, copy psi_e5,
     copy dft_re, copy dft_im, copy dft_weight)
  -- Two-step temporal tiling (temporal.fut) on request.
  let (ex, ey, ez, hx, hy, hz,
       psi_h0, psi_h1, psi_h2, psi_h3, psi_h4, psi_h5,
       psi_e0, psi_e1, psi_e2, psi_e3, psi_e4, psi_e5,
       dft_re, dft_im, dft_weight) =
    if temporal != 2
    then let (ex, ey, ez, hx, hy, hz,
              psi_h0, psi_h1, psi_h2, psi_h3, psi_h4, psi_h5,
              psi_e0, psi_e1, psi_e2, psi_e3, psi_e4, psi_e5,
              dft_re, dft_im, dft_weight) = loop state = start () for s < nsteps do plain_step s state
         in (crop z0 y0 x0 ex, crop z1 y1 x1 ey, crop z2 y2 x2 ez,
             crop z3 y3 x3 hx, crop z4 y4 x4 hy, crop z5 y5 x5 hz,
             psi_h0, psi_h1, psi_h2, psi_h3, psi_h4, psi_h5,
             psi_e0, psi_e1, psi_e2, psi_e3, psi_e4, psi_e5,
             dft_re, dft_im, dft_weight)
    else
      -- Tiled storage is built from the inputs directly, without padded
      -- copies.
      let ntx = (P2 + TX - 1) / TX
      let nt = ((P1 + TY - 1) / TY) * ntx
      let g = {P0, P1, P2, ntx}
      -- The core kernel hard-codes the common configuration: isotropic grid,
      -- scalar H coefficients and E decay, codebook E materials. Otherwise
      -- every tile is an edge tile.
      let unit (v: [][][]f32) = length v == 1 && length v[0] == 1 && length v[0, 0] == 1
      let common = metric_kind == 0
                   && unit h_decay_x && unit h_decay_y && unit h_decay_z
                   && unit h_source_x && unit h_source_y && unit h_source_z
                   && unit e_decay_x && unit e_decay_y && unit e_decay_z
                   && length e_codes_x > 0 && length e_codes_y > 0 && length e_codes_z > 0
      let scalar (v: [][][]f32) = if unit v then #[unsafe] v[0, 0, 0] else 0
      let (zb, zt, cidx) = make_plan g nh ne slabs nt (busy_tiles g nt source_target) common
      let (first, edge) = edge_cells g zb zt cidx (flatten_4d monitor_indices)
      let cf = {inv = inv_resolution,
                hd0 = scalar h_decay_x, hd1 = scalar h_decay_y, hd2 = scalar h_decay_z,
                hs0 = scalar h_source_x, hs1 = scalar h_source_y, hs2 = scalar h_source_z,
                ed0 = scalar e_decay_x, ed1 = scalar e_decay_y, ed2 = scalar e_decay_z, ne}
      let (urows, uidx) = source_rows (nt * P0 * (TY * TX)) (map (tiled_row g <-< i64.i32) source_target)
      -- Groups g, g + 1 and g + 2: the three components at one timing.
      let inject3 [a][b] (at: place) (f: *store [a][b]) step group : *store [a][b] =
        tinject g at f step group source_group urows uidx source_amplitude source_offset source_length waveforms
      let observe [a][b] s step (at: place) (fe: store [a][b]) (fh: store [a][b])
                  (re: *[R]f32, im: *[R]f32, w: *[Q]f32) =
        if M == 0 then (re, im, w)
        else let sample c o = let (k, j, i) = decode g o
                              in if c < 3 then fat g at fe c k j i else fat g at fh (c - 3) k j i
             in accumulate_dft step (time_of s) dt (P0 * P1 * P2) sample
                  monitor_indices monitor_weights monitor_freqs monitor_masks
                  monitor_counts monitor_codes monitor_windows re im w
      -- One plain step on `cells`, from (ae, ah) (placed by la) into
      -- (de, dh) (placed by ld): H, its sources, E (curl from dh), its
      -- sources. CPML memory advances from psi_h and psi_e into qh and qe.
      let step [n][a0][b0][a1][b1] (cells: [n]i32) stp
               (la: place) (ae: store [a0][b0]) (ah: store [a0][b0])
               (ld: place) (de: *store [a1][b1]) (dh: *store [a1][b1])
               (psi_h: ([][][]f32, [][][]f32, [][][]f32, [][][]f32, [][][]f32, [][][]f32))
               (psi_e: ([][][]f32, [][][]f32, [][][]f32, [][][]f32, [][][]f32, [][][]f32))
               (qh0: *[][][]f32, qh1: *[][][]f32, qh2: *[][][]f32, qh3: *[][][]f32, qh4: *[][][]f32, qh5: *[][][]f32)
               (qe0: *[][][]f32, qe1: *[][][]f32, qe2: *[][][]f32, qe3: *[][][]f32, qe4: *[][][]f32, qe5: *[][][]f32)
               : (*store [a1][b1], *store [a1][b1],
                  (*[][][]f32, *[][][]f32, *[][][]f32, *[][][]f32, *[][][]f32, *[][][]f32),
                  (*[][][]f32, *[][][]f32, *[][][]f32, *[][][]f32, *[][][]f32, *[][][]f32)) =
        let (dh, qh0, qh1, qh2, qh3, qh4, qh5) =
          edge_phase g cells 0 edges h_scale nh la ah h_decay h_source ne la ae
                     slabs cpml_a cpml_b cpml_inv_kappa psi_h ld dh qh0 qh1 qh2 qh3 qh4 qh5
        let dh = inject3 ld dh stp 3
        let (de, qe0, qe1, qe2, qe3, qe4, qe5) =
          edge_phase g cells 1 edges e_scale ne la ae e_decay e_source nh ld dh
                     slabs cpml_a cpml_b cpml_inv_kappa psi_e ld de qe0 qe1 qe2 qe3 qe4 qe5
        let de = inject3 ld de stp 6
        in (de, dh, (qh0, qh1, qh2, qh3, qh4, qh5), (qe0, qe1, qe2, qe3, qe4, qe5))
      -- The initial tiled state, with m tiles (nt or 0).
      let initial (m: i64) = (to_tiled g m P0 ex ey ez, to_tiled g m P0 hx hy hz)
      let none () : *[0][P0][TY * TX][3]f32 = replicate 0 (replicate P0 (replicate (TY * TX) (replicate 3 0)))
      let psi_h = (copy psi_h0, copy psi_h1, copy psi_h2, copy psi_h3, copy psi_h4, copy psi_h5)
      let psi_e = (copy psi_e0, copy psi_e1, copy psi_e2, copy psi_e3, copy psi_e4, copy psi_e5)
      -- Second CPML memory buffers; each step writes the other set.
      let qh = (copy psi_h0, copy psi_h1, copy psi_h2, copy psi_h3, copy psi_h4, copy psi_h5)
      let qe = (copy psi_e0, copy psi_e1, copy psi_e2, copy psi_e3, copy psi_e4, copy psi_e5)
      let dft = (copy dft_re, copy dft_im, copy dft_weight)
      -- T: scratch state for the first step of a pass. Only the cells that
      -- step writes are ever read, so it starts uninitialised and holds just
      -- their items (and a spare row for reads that go unused). A pass holds
      -- two whole states, A and C, plus T.
      let (nu, tslot) = item_slots (nt * P0) first
      let tplace = (true, tslot)
      let scratch () : *store [nu + 1][1] =
        #[scratch] replicate (nu + 1) (replicate 1 (replicate (TY * TX) (replicate 3 0f32)))
      let (ae, ah, psi_h, psi_e, qh, qe, dft) =
        loop (ae: *[][P0][TY * TX][3]f32, ah: *[][P0][TY * TX][3]f32,
              psi_h: (*[][][]f32, *[][][]f32, *[][][]f32, *[][][]f32, *[][][]f32, *[][][]f32),
              psi_e: (*[][][]f32, *[][][]f32, *[][][]f32, *[][][]f32, *[][][]f32, *[][][]f32),
              qh: (*[][][]f32, *[][][]f32, *[][][]f32, *[][][]f32, *[][][]f32, *[][][]f32),
              qe: (*[][][]f32, *[][][]f32, *[][][]f32, *[][][]f32, *[][][]f32, *[][][]f32),
              dft: (*[R]f32, *[R]f32, *[Q]f32))
        = (none (), none (), psi_h, psi_e, qh, qe, dft) for n < nsteps / 2 do
          let s = 2 * n
          -- The pass loop builds its initial state in the first pass, and an
          -- empty one in every later pass. Generated code releases a memory
          -- block only when its variable is reassigned (or at exit), and an
          -- initial loop value stays allocated for the whole loop: either
          -- way a fourth state would stay alive.
          let (fe, fh) = initial (if n == 0 then nt else 0)
          let (ae, ah) = if n == 0 then (inject3 whole (fe :> tiled [nt][P0]) step0 0, fh :> tiled [nt][P0])
                         else (ae :> tiled [nt][P0], ah :> tiled [nt][P0])
          let (ce, ch) = core_pass g zb zt cf e_codes_x e_codes_y e_codes_z e_table_x e_table_y e_table_z cidx ae ah
          let (te, th, qh, qe) = step first (step0 + s) whole ae ah tplace (scratch ()) (scratch ()) psi_h psi_e qh qe
          let dft = observe s (step0 + s) tplace te th dft
          -- Pre-E sources of the following step.
          let te = inject3 tplace te (step0 + s + 1) 0
          let (ce, ch, psi_h, psi_e) = step edge (step0 + s + 1) tplace te th whole ce ch qh qe psi_h psi_e
          let dft = observe (s + 1) (step0 + s + 1) whole ce ch dft
          let ce = if s + 2 >= nsteps then ce else inject3 whole ce (step0 + s + 2) 0
          in (ce, ch, psi_h, psi_e, qh, qe, dft)
      -- An odd step count ends with one plain step everywhere.
      let (ae, ah) = if nsteps >= 2 then (ae :> tiled [nt][P0], ah :> tiled [nt][P0])
                     else let (fe, fh) = initial nt in (inject3 whole (fe :> tiled [nt][P0]) step0 0, fh :> tiled [nt][P0])
      let (ae, ah, psi_h, psi_e, dft) =
        if nsteps % 2 == 0 then (ae, ah, psi_h, psi_e, dft)
        else let s = nsteps - 1
             let whole_scratch () : *tiled [nt][P0] =
               #[scratch] replicate nt (replicate P0 (replicate (TY * TX) (replicate 3 0f32)))
             let (te, th, qh, qe) = step (map i32.i64 (iota (nt * P0 * (TY * TX)))) (step0 + s) whole ae ah
                                         whole (whole_scratch ()) (whole_scratch ()) psi_h psi_e qh qe
             in (te, th, qh, qe, observe s (step0 + s) whole te th dft)
      let (psi_h0, psi_h1, psi_h2, psi_h3, psi_h4, psi_h5) = psi_h
      let (psi_e0, psi_e1, psi_e2, psi_e3, psi_e4, psi_e5) = psi_e
      let (dft_re, dft_im, dft_weight) = dft
      in (untile g ae 0 z0 y0 x0, untile g ae 1 z1 y1 x1, untile g ae 2 z2 y2 x2,
          untile g ah 0 z3 y3 x3, untile g ah 1 z4 y4 x4, untile g ah 2 z5 y5 x5,
          psi_h0, psi_h1, psi_h2, psi_h3, psi_h4, psi_h5,
          psi_e0, psi_e1, psi_e2, psi_e3, psi_e4, psi_e5,
          dft_re, dft_im, dft_weight)
  in (ex, ey, ez, hx, hy, hz,
      psi_h0, psi_h1, psi_h2, psi_h3, psi_h4, psi_h5,
      psi_e0, psi_e1, psi_e2, psi_e3, psi_e4, psi_e5,
      dft_re, dft_im, dft_weight)
