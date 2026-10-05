#!/usr/bin/env python3
"""Accuracy of lossy fixed-rate state compression in a Yee FDTD run.

A 64^3 PEC box (a resonator: nothing leaves, so errors never decay) with a
dielectric slab and sphere, driven by a soft Ez pulse at about 20 cells per
vacuum wavelength. Every K = 2 steps all six field components go through a
compress/decompress round trip, as if the state lived compressed in DRAM
between temporal-tiling passes. Errors are measured against the
uncompressed float32 run.

Schemes, at a given rate in bits per value (all fixed-rate except "abstol"):
  fp16, bf16   plain narrow floats (16 bits)
  zfp          ZFP fixed-rate mode, 4^3 blocks (zfpy)
  dct3         8^3 Loeffler-equivalent DCT, block floating point: one f32
               scale per block, coefficient bits by zonal allocation
               b(s) = round(b0 - slope * s), s = u + v + w
  dct2         same with 8x8 DCTs per z plane (fits 2.5D streaming)
  bfp          block floating point without a transform (control)
  round/trunc  quantiser rounding: to nearest, or toward zero (magnitude
               truncation: in an orthonormal basis it cannot add energy)
  abstol       dct3 with one global absolute quantisation step; variable
               rate, reported as its zeroth-order entropy

Needs zfpy (scratch venv). Usage: compress_study.py [--steps N] [--quick]
"""
import argparse
import itertools
import multiprocessing as mp
import numpy as np
import scipy.fft as F

N = 64
K = 2
H, E = ("hx", "hy", "hz"), ("ex", "ey", "ez")


def geometry():
    z, y, x = np.meshgrid(*(np.arange(N),) * 3, indexing="ij")
    eps = np.ones((N, N, N), np.float32)
    eps[x >= 40] = 4.0
    eps[(z - 32) ** 2 + (y - 32) ** 2 + (x - 26) ** 2 <= 36] = 6.0
    src = np.exp(-((z - 32) ** 2 + (y - 32) ** 2 + (x - 14) ** 2) / (2 * 1.5 ** 2)).astype(np.float32)
    return (np.float32(0.5) / eps).astype(np.float32), src


def sh(a, ax, o):
    b = np.zeros_like(a)
    s, d = [slice(None)] * 3, [slice(None)] * 3
    if o > 0:
        s[ax], d[ax] = slice(o, None), slice(None, -o)
    else:
        s[ax], d[ax] = slice(None, o), slice(-o, None)
    b[tuple(d)] = a[tuple(s)]
    return b


def step(f, cb, src, t):
    ch = np.float32(0.5)
    ex, ey, ez, hx, hy, hz = (f[c] for c in E + H)
    d = lambda a, ax: sh(a, ax, 1) - a          # forward difference
    b = lambda a, ax: a - sh(a, ax, -1)          # backward difference
    hx = hx - ch * (d(ez, 1) - d(ey, 0))
    hy = hy - ch * (d(ex, 0) - d(ez, 2))
    hz = hz - ch * (d(ey, 2) - d(ex, 1))
    ex = ex + cb * (b(hz, 1) - b(hy, 0))
    ey = ey + cb * (b(hx, 0) - b(hz, 2))
    ez = ez + cb * (b(hy, 2) - b(hx, 1))
    w = np.float32(np.exp(-((t - 120) / 40.0) ** 2) * np.sin(2 * np.pi * t / 40.0))
    ez = ez + w * src
    return dict(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz)


# --- codecs: array -> (reconstruction, bits per value) ---------------------

def blocks(a, b, dims):
    """[N,N,N] -> [..., b, b(, b)] blocks over the last `dims` axes."""
    n = N // b
    if dims == 3:
        return a.reshape(n, b, n, b, n, b).transpose(0, 2, 4, 1, 3, 5)
    return a.reshape(N, n, b, n, b).transpose(0, 1, 3, 2, 4)


def unblocks(c, b, dims):
    n = N // b
    if dims == 3:
        return c.transpose(0, 3, 1, 4, 2, 5).reshape(N, N, N)
    return c.transpose(0, 1, 3, 2, 4).reshape(N, N, N)


def zonal(dims, b0, slope):
    idx = np.indices((8,) * dims).sum(0)
    bits = np.round(b0 - slope * idx)
    bits[bits < 2] = 0                 # 1 bit would be sign only: drop instead
    return np.minimum(bits, 24)


def alloc_for_rate(dims, rate, slope):
    """Largest b0 whose zonal allocation fits rate * 8^dims - 32 bits."""
    budget = rate * 8 ** dims - 32
    lo, hi = 0.0, 64.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if zonal(dims, mid, slope).sum() <= budget:
            lo = mid
        else:
            hi = mid
    return zonal(dims, lo, slope)


def bfp_quant(c, bits, dims, mode="round"):
    axes = tuple(range(-dims, 0))
    scale = np.abs(c).max(axis=axes, keepdims=True)
    scale = np.where(scale == 0, 1, scale)
    lv = np.where(bits > 0, 2.0 ** (bits - 1) - 1, 0)
    rnd = np.round if mode == "round" else np.trunc   # trunc: |Q(c)| <= |c|
    q = np.where(lv > 0, rnd(c / scale * np.maximum(lv, 1)) / np.maximum(lv, 1) * scale, 0)
    return q.astype(np.float32)


def dct_codec(dims, rate, slope, mode="round"):
    bits = alloc_for_rate(dims, rate, slope)
    axes = tuple(range(-dims, 0))
    real = (bits.sum() + 32) / 8 ** dims

    def f(a):
        c = F.dctn(blocks(a, 8, dims), axes=axes, norm="ortho")
        return unblocks(F.idctn(bfp_quant(c, bits, dims, mode), axes=axes, norm="ortho"), 8, dims).astype(np.float32)
    return f, real


def bfp_codec(rate, mode="round"):
    bits = np.full((8, 8, 8), np.floor((rate * 512 - 32) / 512))
    real = (bits.sum() + 32) / 512

    def f(a):
        return unblocks(bfp_quant(blocks(a, 8, 3), bits, 3, mode), 8, 3)
    return f, real


def zfp_codec(rate):
    import zfpy

    def f(a):
        return zfpy.decompress_numpy(zfpy.compress_numpy(np.ascontiguousarray(a), rate=rate))
    return f, rate


def abstol_codec(tol, stats):
    def f(a):
        c = F.dctn(blocks(a, 8, 3), axes=(-3, -2, -1), norm="ortho")
        q = np.round(c / tol)
        vals, counts = np.unique(q, return_counts=True)
        p = counts / counts.sum()
        stats.append(float(-(p * np.log2(p)).sum()))
        return unblocks(F.idctn(q * tol, axes=(-3, -2, -1), norm="ortho"), 8, 3).astype(np.float32)
    return f, None


def make_codec(spec):
    kind, *args = spec
    if kind == "none":
        return (lambda a: a), 32
    if kind == "fp16":
        return (lambda a: a.astype(np.float16).astype(np.float32)), 16
    if kind == "bf16":
        return (lambda a: (a.view(np.uint32) + 0x8000 & 0xFFFF0000).view(np.float32)), 16
    if kind == "zfp":
        return zfp_codec(*args)
    if kind == "dct3":
        return dct_codec(3, *args)
    if kind == "dct2":
        return dct_codec(2, *args)
    if kind == "bfp":
        return bfp_codec(*args)
    raise ValueError(kind)


def simulate(spec, steps, checkpoints):
    cb, src = geometry()
    f = {c: np.zeros((N, N, N), np.float32) for c in E + H}
    stats = []
    if spec[0] == "abstol":
        codec, rate = abstol_codec(spec[1], stats)
    else:
        codec, rate = make_codec(spec)
    snaps, probe = {}, []
    for t in range(steps):
        f = step(f, cb, src, t)
        if (t + 1) % K == 0:
            f = {c: codec(v) for c, v in f.items()}
        probe.append(f["ez"][32, 32, 48])
        if t + 1 in checkpoints:
            snaps[t + 1] = {c: v.copy() for c, v in f.items()}
    if stats:
        rate = float(np.mean(stats))
    return spec, rate, snaps, np.array(probe)


def rel(a, b):
    num = sum(float(np.sum((a[c] - b[c]) ** 2)) for c in a)
    den = sum(float(np.sum(b[c] ** 2)) for c in a)
    return np.sqrt(num / den)


def run(specs, steps, procs):
    cps = sorted({steps // 4, steps // 2, steps})
    with mp.Pool(procs) as pool:
        res = pool.starmap(simulate, [(s, steps, set(cps)) for s in [("none",)] + specs])
    _, _, ref, pref = res[0]
    out = []
    for spec, rate, snaps, probe in res[1:]:
        errs = [rel(snaps[c], ref[c]) for c in cps]
        perr = np.linalg.norm(probe - pref) / np.linalg.norm(pref)
        en = lambda f: sum(float((v.astype(np.float64) ** 2).sum()) for v in f.values())
        out.append((spec, rate, errs, perr, en(snaps[cps[-1]]) / en(ref[cps[-1]])))
    return cps, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--procs", type=int, default=30)
    a = p.parse_args()
    rates = [16, 12, 8, 6, 4]
    slopes = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    # Pick the best zonal slope per rate on a shorter run.
    modes = ("round", "trunc")
    tune = [(k, r, s, m) for k in ("dct3", "dct2") for r in rates for s in slopes for m in modes]
    _, tres = run(tune, a.steps // 3, a.procs)
    best = {}
    for spec, rate, errs, *_ in tres:
        key = (spec[0], spec[1], spec[3])
        if key not in best or errs[-1] < best[key][1]:
            best[key] = (spec, errs[-1])
    specs = [("fp16",), ("bf16",)]
    specs += [("zfp", r) for r in rates]
    specs += [best[k, r, m][0] for k in ("dct3", "dct2") for m in modes for r in rates]
    specs += [("bfp", r, m) for m in modes for r in rates]
    specs += [("abstol", t) for t in (1e-3, 1e-4, 1e-5, 1e-6)]
    cps, res = run(specs, a.steps, a.procs)
    print("energy = sum of squared fields at the end relative to float32 (> 1: injected)")
    print(f"relative L2 error of all six fields vs float32, compressed every {K} steps; "
          f"probe = Ez time series in the slab")
    print(f"{'scheme':24s} {'bits/val':>8s} " + " ".join(f"{'t=' + str(c):>10s}" for c in cps) + f" {'probe':>10s} {'energy':>8s}")
    for spec, rate, errs, perr, ratio in res:
        name = " ".join(str(x) for x in spec)
        print(f"{name:24s} {rate:8.2f} " + " ".join(f"{e:10.2e}" for e in errs) + f" {perr:10.2e} {ratio:8.3f}")


if __name__ == "__main__":
    main()
