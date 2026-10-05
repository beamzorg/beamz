#!/usr/bin/env python3
"""Time an entry point at two step counts; report GCUPS from the difference.

    bench.py EXE ENTRY NZ NY NX [S1 S2] [--runs R] [--check]
"""
import argparse, os, statistics, subprocess, tempfile

p = argparse.ArgumentParser()
p.add_argument("exe"); p.add_argument("entry")
p.add_argument("nz", type=int); p.add_argument("ny", type=int); p.add_argument("nx", type=int)
p.add_argument("s1", type=int, nargs="?", default=24)
p.add_argument("s2", type=int, nargs="?", default=120)
p.add_argument("--runs", type=int, default=5)
p.add_argument("--extra", nargs="*", default=[])
p.add_argument("--cpu", action="store_true", help="host backend: no NVRTC options")
a = p.parse_args()

def run(steps):
    with tempfile.NamedTemporaryFile("r") as t:
        inp = f"{a.nz}i64 {a.ny}i64 {a.nx}i64 {steps}i64\n"
        subprocess.run([a.exe, "-e", a.entry, "-r", str(a.runs), "-t", t.name,
                        *([] if a.cpu else ["--nvrtc-option=--fmad=false"]), "-b", *a.extra],
                       input=inp.encode(), check=True, stdout=subprocess.DEVNULL)
        return statistics.median(int(x) for x in t.read().split()) / 1e6

t1, t2 = run(a.s1), run(a.s2)
per = (t2 - t1) / (a.s2 - a.s1)
cells = a.nz * a.ny * a.nx
print(f"{os.path.basename(a.exe)} {a.entry} {a.nz}x{a.ny}x{a.nx}: {per*1e3:.3f} ms/step  {cells/per/1e9:.2f} GCUPS")
