-- Loeffler 8-point DCT (11 multiplications), ported from
-- https://pepijndevos.nl/2018/07/04/loefflers-discrete-cosine-transform-algorithm-in-futhark.html
-- to Futhark 0.27, plus a separable 3D 8x8x8 transform built on it.
--
-- Scaling: fdct8 is sqrt(8) times the orthonormal DCT-II, so its DC term is the
-- plain sum. idct8 divides by 8, so idct8 (fdct8 x) == x. Output is in
-- natural frequency order; checked against scipy.fft.dct(norm='ortho').

def butterfly (a: f32) (b: f32) : (f32, f32) = (a + b, a - b)

def rotate (s: f32) (c: f32) (a: f32) (b: f32) : (f32, f32) =
  (a * c + b * s, b * c - a * s)

def cs (k: f32) (n: i64) : f32 = k * f32.cos (f32.i64 n * f32.pi / 16)

def fdct8 (a: [8]f32) : [8]f32 =
  let r2 = f32.sqrt 2
  let (s1_0, s1_7) = butterfly a[0] a[7]
  let (s1_1, s1_6) = butterfly a[1] a[6]
  let (s1_2, s1_5) = butterfly a[2] a[5]
  let (s1_3, s1_4) = butterfly a[3] a[4]
  -- even part
  let (s2_0, s2_3) = butterfly s1_0 s1_3
  let (s2_1, s2_2) = butterfly s1_1 s1_2
  let (y0, y4) = butterfly s2_0 s2_1
  let (y2, y6) = rotate (cs r2 2) (cs r2 6) s2_2 s2_3
  -- odd part
  let (s2_4, s2_7) = rotate (cs 1 5) (cs 1 3) s1_4 s1_7
  let (s2_5, s2_6) = rotate (cs 1 7) (cs 1 1) s1_5 s1_6
  let (s3_4, s3_6) = butterfly s2_4 s2_6
  let (s3_7, s3_5) = butterfly s2_7 s2_5
  let (y1, y7) = butterfly s3_7 s3_4
  let y3 = r2 * s3_5
  let y5 = r2 * s3_6
  in [y0, y1, y2, y3, y4, y5, y6, y7]

def idct8 (y: [8]f32) : [8]f32 =
  let r2 = f32.sqrt 2
  -- odd part
  let (s4_7, s4_4) = butterfly y[1] y[7]
  let s4_5 = r2 * y[3]
  let s4_6 = r2 * y[5]
  let (s3_4, s3_6) = butterfly s4_4 s4_6
  let (s3_7, s3_5) = butterfly s4_7 s4_5
  let (s2_4, s2_7) = rotate (-(cs 1 5)) (cs 1 3) s3_4 s3_7
  let (s2_5, s2_6) = rotate (-(cs 1 7)) (cs 1 1) s3_5 s3_6
  -- even part
  let (s3_0, s3_1) = butterfly y[0] y[4]
  let (s3_2, s3_3) = rotate (-(cs r2 2)) (cs r2 6) y[2] y[6]
  let (s2_0, s2_3) = butterfly s3_0 s3_3
  let (s2_1, s2_2) = butterfly s3_1 s3_2
  let (s1_0, s1_7) = butterfly s2_0 s2_7
  let (s1_1, s1_6) = butterfly s2_1 s2_6
  let (s1_2, s1_5) = butterfly s2_2 s2_5
  let (s1_3, s1_4) = butterfly s2_3 s2_4
  in map (/ 8) [s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7]

-- Separable 3D transforms of one 8x8x8 block, indexed [z][y][x].
def along_x f (b: [8][8][8]f32) : [8][8][8]f32 = map (map f) b
def along_y f (b: [8][8][8]f32) : [8][8][8]f32 = map (\p -> transpose (map f (transpose p))) b
def along_z f (b: [8][8][8]f32) : [8][8][8]f32 =
  transpose (map transpose (map (map f) (map transpose (transpose b))))

def fdct3 (b: [8][8][8]f32) : [8][8][8]f32 = b |> along_x fdct8 |> along_y fdct8 |> along_z fdct8
def idct3 (b: [8][8][8]f32) : [8][8][8]f32 = b |> along_z idct8 |> along_y idct8 |> along_x idct8

entry fdct_rows (xs: [][8]f32) : [][8]f32 = map fdct8 xs
entry idct_rows (ys: [][8]f32) : [][8]f32 = map idct8 ys
entry fdct_blocks (bs: [][8][8][8]f32) : [][8][8][8]f32 = map fdct3 bs
entry idct_blocks (bs: [][8][8][8]f32) : [][8][8][8]f32 = map idct3 bs
