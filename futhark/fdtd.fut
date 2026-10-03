-- Complete multi-step 3D Yee/CPML program for BeamZ's Futhark backend.
--
-- Arrays use BeamZ's [z][y][x] component shapes; storage axis 0 is z, 1 is y and
-- 2 is x. The arithmetic mirrors cuda/src/yee_primitives.cuh: magnetic phases
-- take forward differences, electric phases take backward differences with PEC
-- ghost samples on metallic faces, and CPML recurrences live on packed slabs.
-- Each timestep applies, in order: pre-E sources, H, H sources, E, E sources and
-- the vector DFT accumulation.
--
-- During a run every field component is stored zero-padded to the common
-- (z+1, y+1, x+1) bounding shape, so that one kernel per phase can advance all
-- three components: Futhark fuses only maps of equal size, and separate
-- component kernels would each reread the shared curl inputs. Logical
-- component shapes travel as `dims` and decide every boundary.

type dims = (i64, i64, i64)

def axis_coord (axis: i64) (k: i64) (j: i64) (i: i64) : i64 =
  if axis == 0 then k else if axis == 1 then j else i

def axis_size (axis: i64) ((a, b, c): dims) : i64 =
  if axis == 0 then a else if axis == 1 then b else c

def with_coord (axis: i64) (v: i64) (k: i64) (j: i64) (i: i64) : (i64, i64, i64) =
  if axis == 0 then (v, j, i) else if axis == 1 then (k, v, i) else (k, j, v)

def inside ((a, b, c): dims) (k: i64) (j: i64) (i: i64) : bool =
  k < a && j < b && i < c

-- tabulate_3d decomposes each flat thread index with two 64-bit divisions,
-- which GPUs emulate. Field volumes are far below 2^31 cells, so decompose
-- with 32-bit arithmetic instead.
def tabulate3 't (n0: i64) (n1: i64) (n2: i64) (f: i64 -> i64 -> i64 -> t) : *[n0][n1][n2]t =
  let m2 = i32.i64 n2
  let m12 = i32.i64 (n1 * n2)
  in unflatten_3d (tabulate (n0 * n1 * n2) (\q ->
       let q = i32.i64 q
       let k = q / m12
       let r = q - k * m12
       let j = r / m2
       in f (i64.i32 k) (i64.i32 j) (i64.i32 (r - j * m2))))

def edge (edges: i64) (bit: i64) : bool = (edges & (1 << bit)) != 0

-- Inverse spacing at one output coordinate: isotropic (0), axis-uniform (1) or
-- rectilinear (2) metrics.
def metric (kind: i64) (inv: f32) (m: []f32) (coord: i64) : f32 =
  if kind == 0 then inv else if kind == 1 then #[unsafe] m[0] else #[unsafe] m[coord]

-- Magnetic phase: a missing forward neighbor contributes nothing.
def forward (v: [][][]f32) (n: dims) (axis: i64) (s: f32) (k: i64) (j: i64) (i: i64) : f32 =
  let co = axis_coord axis k j i
  in if co + 1 >= axis_size axis n then 0
     else let (k1, j1, i1) = with_coord axis (co + 1) k j i
          in #[unsafe] (v[k1, j1, i1] - v[k, j, i]) * s

-- Electric phase: metallic faces see a zero ghost sample, open faces none.
def backward (v: [][][]f32) (n: dims) (axis: i64) (edges: i64) (s: f32)
             (k: i64) (j: i64) (i: i64) : f32 =
  let co = axis_coord axis k j i
  let m = axis_size axis n
  in if co == 0 then (if edge edges (2 * axis) then #[unsafe] v[k, j, i] * s else 0)
     else if co == m
     then (if edge edges (2 * axis + 1)
           then let (k1, j1, i1) = with_coord axis (m - 1) k j i
                in #[unsafe] (-v[k1, j1, i1]) * s
           else 0)
     else let (k1, j1, i1) = with_coord axis (co - 1) k j i
          in #[unsafe] (v[k, j, i] - v[k1, j1, i1]) * s

def derivative (phase: i64) (edges: i64) (v: [][][]f32) (n: dims) (axis: i64) (s: f32)
               (k: i64) (j: i64) (i: i64) : f32 =
  if phase == 0 then forward v n axis s k j i else backward v n axis edges s k j i

-- Tangential E and normal H vanish on metallic faces.
def pec (phase: i64) (component: i64) (edges: i64) (n: dims) (k: i64) (j: i64) (i: i64) : bool =
  let normal = 2 - component
  let hit axis =
    let co = axis_coord axis k j i
    in (co == 0 && edge edges (2 * axis)) || (co == axis_size axis n - 1 && edge edges (2 * axis + 1))
  in if phase == 0 then hit normal
     else (normal != 0 && hit 0) || (normal != 1 && hit 1) || (normal != 2 && hit 2)

-- Material coefficients broadcast along any unit axis.
def coefficient [a][b][c] (v: [a][b][c]f32) (k: i64) (j: i64) (i: i64) : f32 =
  #[unsafe] v[if a == 1 then 0 else k, if b == 1 then 0 else j, if c == 1 then 0 else i]

-- Material source coefficient: from a codebook when `codes` packs one 8-bit
-- index per cell (four per word, in logical component order), else dense.
def material (n: dims) (source: [][][]f32) (table: []f32) (codes: []i32)
             (k: i64) (j: i64) (i: i64) : f32 =
  if length codes == 0 then coefficient source k j i
  else let (_, b, c) = n
       let l = (k * b + j) * c + i
       let code = (#[unsafe] codes[l >> 2] >> i32.i64 (8 * (l & 3))) & 0xff
       in #[unsafe] table[i64.i32 code]

def packed (co: i64) (n: i64) (lo: i64) (hi: i64) : i64 =
  if co < lo then co else if co >= n - hi then lo + co - (n - hi) else -1

-- One CPML term: the stretched derivative, or the plain one outside its slab.
def stretched [L] (n: i64) (axis: i64) (lo: i64) (hi: i64)
    (ca: [L]f32) (cb: [L]f32) (ck: [L]f32) (psi: [][][]f32)
    (d: f32) (k: i64) (j: i64) (i: i64) : f32 =
  let p = packed (axis_coord axis k j i) n lo hi
  in if p < 0 then d
     else let (pk, pj, pi) = with_coord axis p k j i
          let next = #[unsafe] f32.fma cb[p] psi[pk, pj, pi] (ca[p] * d)
          in #[unsafe] f32.fma d ck[p] next

-- One field component at one cell; the CPML memory is advanced by `psi_next`.
-- Its curl is f0's derivative along ax0 minus f1's along ax1.
def field_value [L]
    (phase: i64) (component: i64) (edges: i64) (scale: i64 -> i64 -> f32)
    (n: dims) (old: [][][]f32) (decay: [][][]f32) ((source, table, codes): ([][][]f32, []f32, []i32))
    (f0: [][][]f32) (n0: dims) (ax0: i64) (f1: [][][]f32) (n1: dims) (ax1: i64)
    (slabs: [12][2]i64) (ca: [12][L]f32) (cb: [12][L]f32) (ck: [12][L]f32)
    (psi0: [][][]f32) (psi1: [][][]f32) (k: i64) (j: i64) (i: i64) : f32 =
  let t0 = 6 * phase + 2 * component
  let t1 = t0 + 1
  in if !(inside n k j i) || pec phase component edges n k j i then 0
     else
       let d0 = derivative phase edges f0 n0 ax0 (scale ax0 (axis_coord ax0 k j i)) k j i
       let d1 = derivative phase edges f1 n1 ax1 (scale ax1 (axis_coord ax1 k j i)) k j i
       let curl =
         #[unsafe]
         stretched (axis_size ax0 n) ax0 slabs[t0, 0] slabs[t0, 1] ca[t0] cb[t0] ck[t0] psi0 d0 k j i
         - stretched (axis_size ax1 n) ax1 slabs[t1, 0] slabs[t1, 1] ca[t1] cb[t1] ck[t1] psi1 d1 k j i
       let s = material n source table codes k j i
       in f32.fma (if phase == 0 then -s else s) curl (coefficient decay k j i * #[unsafe] old[k, j, i])

-- Advance all three components of one phase in a single kernel. `n` holds the
-- outputs' logical shapes, `m` those of the curl inputs `f`.
def phase_next [P0][P1][P2][L]
    (phase: i64) (edges: i64) (scale: i64 -> i64 -> f32)
    (n: (dims, dims, dims)) (old: ([P0][P1][P2]f32, [P0][P1][P2]f32, [P0][P1][P2]f32))
    (decay: ([][][]f32, [][][]f32, [][][]f32))
    (source: (([][][]f32, []f32, []i32), ([][][]f32, []f32, []i32), ([][][]f32, []f32, []i32)))
    (m: (dims, dims, dims)) (f: ([P0][P1][P2]f32, [P0][P1][P2]f32, [P0][P1][P2]f32))
    (slabs: [12][2]i64) (ca: [12][L]f32) (cb: [12][L]f32) (ck: [12][L]f32)
    (psi: ([][][]f32, [][][]f32, [][][]f32, [][][]f32, [][][]f32, [][][]f32))
    : (*[P0][P1][P2]f32, *[P0][P1][P2]f32, *[P0][P1][P2]f32) =
  -- Three tabulates of one size over shared inputs: Futhark fuses them into
  -- one kernel.
  let value c = field_value phase c edges scale
  in #[unsafe]
     (tabulate3 P0 P1 P2 (value 0 n.0 old.0 decay.0 source.0 f.2 m.2 1 f.1 m.1 0
                                slabs ca cb ck psi.0 psi.1),
      tabulate3 P0 P1 P2 (value 1 n.1 old.1 decay.1 source.1 f.0 m.0 0 f.2 m.2 2
                                slabs ca cb ck psi.2 psi.3),
      tabulate3 P0 P1 P2 (value 2 n.2 old.2 decay.2 source.2 f.1 m.1 2 f.0 m.0 1
                                slabs ca cb ck psi.4 psi.5))

-- Advance one packed CPML recurrence for an output component of shape dims.
def psi_next [pa][pb][pc][L]
    (phase: i64) (edges: i64) (scale: i64 -> i64 -> f32) (dims: dims)
    (term: i64) (src: [][][]f32) (m: dims) (axis: i64)
    (slabs: [12][2]i64) (ca: [12][L]f32) (cb: [12][L]f32)
    (psi: [pa][pb][pc]f32) : *[pa][pb][pc]f32 =
  let n = axis_size axis dims
  let lo = slabs[term, 0]
  let hi = slabs[term, 1]
  in #[unsafe]
     tabulate3 pa pb pc (\pk pj pi ->
       let p = axis_coord axis pk pj pi
       let co = if p < lo then p else n - hi + (p - lo)
       let (k, j, i) = with_coord axis co pk pj pi
       let d = derivative phase edges src m axis (scale axis co) k j i
       in f32.fma cb[term, p] psi[pk, pj, pi] (ca[term, p] * d))

def dims3 [a][b][c] 't (_: [a][b][c]t) : dims = (a, b, c)

-- Copy a component into zero-padded run storage.
def pad [a][b][c] (P0: i64) (P1: i64) (P2: i64) (v: [a][b][c]f32) : *[P0][P1][P2]f32 =
  tabulate3 P0 P1 P2 (\k j i -> if inside (a, b, c) k j i then #[unsafe] v[k, j, i] else 0)

-- The logical component inside run storage.
def crop [P0][P1][P2] (a: i64) (b: i64) (c: i64) (v: [P0][P1][P2]f32) : *[a][b][c]f32 =
  #[unsafe] tabulate3 a b c (\k j i -> v[k, j, i])

-- Add one (timing, component) source group to a field. Cells of other groups,
-- outside the field or on constrained PEC cells are dropped by reduce_by_index;
-- selecting by group id keeps every decision on the device.
def inject [a][b][c][K] (field: *[a][b][c]f32) (step: i64) (group: i32)
    (groups: [K]i32) (target: [K]i32) (amplitude: [K]f32) (offset: [K]i32)
    (length: [K]i32) (waves: []f32) : *[a][b][c]f32 =
  let (targets, values) =
    unzip (map5 (\g tgt amp off len ->
                   if g != group then (-1, 0)
                   else let t = i64.max 0 (i64.min step (i64.i32 len - 1))
                        in (i64.i32 tgt, amp * #[unsafe] waves[i64.i32 off + t]))
                groups target amplitude offset length)
  in unflatten_3d (reduce_by_index (flatten_3d field) (+) 0 targets values)

-- Interpolated sample of one monitored component; neighbors are accumulated
-- in plan order. Plans are neighbor-major ([N][P]) so that threads handling
-- consecutive points read consecutive plan entries.
def gather [n][N][P] (field: [n]f32) (indices: [N][P]i32) (weights: [N][P]f32) (p: i64) : f32 =
  loop acc = 0f32 for q < N do
    let o = i64.i32 indices[q, p]
    in if o >= 0 && o < n then acc + #[unsafe] field[o] * weights[q, p] else acc

def two_pi : f32 = 6.2831853071795864769

def isfinite (x: f32) : bool = !(f32.isnan x || f32.isinf x)

-- Accumulate one timestep of every packed vector DFT monitor.
def accumulate_dft [M][P][N][F][R][Q]
    (step: i64) (time: f32) (dt: f32)
    (ex: []f32) (ey: []f32) (ez: []f32) (hx: []f32) (hy: []f32) (hz: []f32)
    (indices: [M][6][N][P]i32) (weights: [M][6][N][P]f32) (freqs: [M][F]f32)
    (masks: [M][6]f32) (counts: [M][5]i32) (codes: [M][2]i32) (windows: [M][3]f32)
    (re: *[R]f32) (im: *[R]f32) (wsum: *[Q]f32) : (*[R]f32, *[R]f32, *[Q]f32) =
  -- Window, sine and cosine per monitor and frequency; zero when inactive.
  let phases =
    tabulate_2d M F (\m f ->
      let fc = i64.i32 counts[m, 0]
      let interval = i64.max 1 (i64.i32 counts[m, 2])
      let valid = f < fc && fc > 0 && fc <= F
                  && (codes[m, 0] == 0 || codes[m, 0] == 1)
                  && (codes[m, 1] == 0 || codes[m, 1] == 1)
      let start = windows[m, 0]
      let stop = windows[m, 1]
      in if valid && step % interval == 0 && time >= start && time <= stop
         then let window =
                if codes[m, 0] == 1 && isfinite stop && stop > start
                then let tau = f32.min (f32.max ((time - start) / (stop - start)) 0) 1
                     in 0.5 * (1 - f32.cos (two_pi * tau))
                else 1
              let arg = two_pi * freqs[m, f] * time
              in (window, f32.sin arg, f32.cos arg)
         else (0, 0, 0))
  let samples =
    tabulate_3d M 6 P (\m c p ->
      let idx = indices[m, c]
      let w = weights[m, c]
      in if c == 0 then gather ex idx w p
         else if c == 1 then gather ey idx w p
         else if c == 2 then gather ez idx w p
         else if c == 3 then gather hx idx w p
         else if c == 4 then gather hy idx w p
         else gather hz idx w p)
  let updates =
      tabulate (M * 6 * F * P) (\q ->
         let p = q % P
         let f = (q / P) % F
         let c = (q / (P * F)) % 6
         let m = q / (P * F * 6)
         let (window, sn, cs) = phases[m, f]
         let fc = i64.i32 counts[m, 0]
         let pc = i64.i32 counts[m, 1]
         let interval = i64.max 1 (i64.i32 counts[m, 2])
         let value_offset = i64.i32 counts[m, 3]
         let length_unit = windows[m, 2]
         let scale =
           if codes[m, 1] == 1
           then window * (dt * f32.i64 interval * 299792458 / length_unit / f32.sqrt two_pi)
           else window
         let ok = window != 0 && p < pc && masks[m, c] != 0 && pc <= P
                  && value_offset >= 0 && value_offset + 6 * fc * pc <= R
                  && (codes[m, 1] != 1 || (isfinite length_unit && length_unit > 0))
         let sample = samples[m, c, p]
         in if ok then (value_offset + (c * fc + f) * pc + p, scale * sample * cs, scale * sample * sn)
            else (-1, 0, 0))
  let (targets, re_values, im_values) = unzip3 updates
  let weight_updates =
    flatten
      (tabulate_2d M F (\m f ->
         let (window, _, _) = phases[m, f]
         let offset = i64.i32 counts[m, 4]
         in if window != 0 && i64.i32 counts[m, 1] >= 1 && offset >= 0
               && offset + i64.i32 counts[m, 0] <= Q
            then (offset + f, window)
            else (-1, 0)))
  let (weight_targets, weight_values) = unzip weight_updates
  in (reduce_by_index re (+) 0 targets re_values,
      reduce_by_index im (+) 0 targets im_values,
      reduce_by_index wsum (+) 0 weight_targets weight_values)

-- Advance a whole run. Scalars that do not change between invocations are
-- static FFI attributes; clocks arrive as one-element device arrays so that
-- chunked runs never synchronize the host. Source targets and monitor plans
-- index the padded run storage.
entry program
    [z0][y0][x0][z1][y1][x1][z2][y2][x2][z3][y3][x3][z4][y4][x4][z5][y5][x5]
    [L][K][M][P][N][F][R][Q]
    (nsteps: i64) (dt: f32) (inv_resolution: f32) (edges: i64) (metric_kind: i64)
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
  let (ex, ey, ez, hx, hy, hz,
       psi_h0, psi_h1, psi_h2, psi_h3, psi_h4, psi_h5,
       psi_e0, psi_e1, psi_e2, psi_e3, psi_e4, psi_e5,
       dft_re, dft_im, dft_weight) =
    loop (ex, ey, ez, hx, hy, hz,
          psi_h0, psi_h1, psi_h2, psi_h3, psi_h4, psi_h5,
          psi_e0, psi_e1, psi_e2, psi_e3, psi_e4, psi_e5,
          dft_re, dft_im, dft_weight) =
         -- Inputs alias XLA buffers that are also outputs, so the loop works
         -- on private copies and the handler copies results back at the end.
         (inject' (pad P0 P1 P2 ex) step0 0,
          inject' (pad P0 P1 P2 ey) step0 1,
          inject' (pad P0 P1 P2 ez) step0 2,
          pad P0 P1 P2 hx, pad P0 P1 P2 hy, pad P0 P1 P2 hz,
          copy psi_h0, copy psi_h1, copy psi_h2, copy psi_h3, copy psi_h4, copy psi_h5,
          copy psi_e0, copy psi_e1, copy psi_e2, copy psi_e3, copy psi_e4, copy psi_e5,
          copy dft_re, copy dft_im, copy dft_weight)
    for s < nsteps do
      let step = step0 + s
      -- 1. Pre-E sources were applied at the end of the previous iteration
      --    (or before the loop), after that step's observation.
      -- 2. Magnetic phase.
      let (hx', hy', hz') =
        phase_next 0 edges h_scale nh (hx, hy, hz) h_decay h_source ne (ex, ey, ez)
                   slabs cpml_a cpml_b cpml_inv_kappa
                   (psi_h0, psi_h1, psi_h2, psi_h3, psi_h4, psi_h5)
      let psi_h0 = psi_next 0 edges h_scale nh.0 0 ez ne.2 1 slabs cpml_a cpml_b psi_h0
      let psi_h1 = psi_next 0 edges h_scale nh.0 1 ey ne.1 0 slabs cpml_a cpml_b psi_h1
      let psi_h2 = psi_next 0 edges h_scale nh.1 2 ex ne.0 0 slabs cpml_a cpml_b psi_h2
      let psi_h3 = psi_next 0 edges h_scale nh.1 3 ez ne.2 2 slabs cpml_a cpml_b psi_h3
      let psi_h4 = psi_next 0 edges h_scale nh.2 4 ey ne.1 2 slabs cpml_a cpml_b psi_h4
      let psi_h5 = psi_next 0 edges h_scale nh.2 5 ex ne.0 1 slabs cpml_a cpml_b psi_h5
      -- 3. Magnetic sources.
      let hx = inject' hx' step 3
      let hy = inject' hy' step 4
      let hz = inject' hz' step 5
      -- 4. Electric phase.
      let (ex', ey', ez') =
        phase_next 1 edges e_scale ne (ex, ey, ez) e_decay e_source nh (hx, hy, hz)
                   slabs cpml_a cpml_b cpml_inv_kappa
                   (psi_e0, psi_e1, psi_e2, psi_e3, psi_e4, psi_e5)
      let psi_e0 = psi_next 1 edges e_scale ne.0 6 hz nh.2 1 slabs cpml_a cpml_b psi_e0
      let psi_e1 = psi_next 1 edges e_scale ne.0 7 hy nh.1 0 slabs cpml_a cpml_b psi_e1
      let psi_e2 = psi_next 1 edges e_scale ne.1 8 hx nh.0 0 slabs cpml_a cpml_b psi_e2
      let psi_e3 = psi_next 1 edges e_scale ne.1 9 hz nh.2 2 slabs cpml_a cpml_b psi_e3
      let psi_e4 = psi_next 1 edges e_scale ne.2 10 hy nh.1 2 slabs cpml_a cpml_b psi_e4
      let psi_e5 = psi_next 1 edges e_scale ne.2 11 hx nh.0 1 slabs cpml_a cpml_b psi_e5
      -- 5. Electric sources.
      let ex = inject' ex' step 6
      let ey = inject' ey' step 7
      let ez = inject' ez' step 8
      -- 6. Observe the fully constrained end-of-step fields.
      let time = f32.fma (f32.i64 (i64.i32 elapsed_steps[0] + s + 1)) dt time_origin[0]
      let (dft_re, dft_im, dft_weight) =
        if M == 0 then (dft_re, dft_im, dft_weight)
        else accumulate_dft step time dt
               (flatten_3d ex) (flatten_3d ey) (flatten_3d ez)
               (flatten_3d hx) (flatten_3d hy) (flatten_3d hz)
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
  in (crop z0 y0 x0 ex, crop z1 y1 x1 ey, crop z2 y2 x2 ez,
      crop z3 y3 x3 hx, crop z4 y4 x4 hy, crop z5 y5 x5 hz,
      psi_h0, psi_h1, psi_h2, psi_h3, psi_h4, psi_h5,
      psi_e0, psi_e1, psi_e2, psi_e3, psi_e4, psi_e5,
      dft_re, dft_im, dft_weight)
