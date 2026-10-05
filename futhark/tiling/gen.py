#!/usr/bin/env python3
"""Instantiate a tiling experiment with constants and compile it.

    gen.py SRC BACKEND OUT NAME=VALUE... [PATCH=KERNEL_SUBSTRING]

Every .fut file next to SRC is copied next to OUT with each `def NAME : t = v`
replaced, so constants in imported files are overridden too.
"""
import os, re, subprocess, sys

src, backend, out, *defs = sys.argv[1:]
patch = [d.split("=", 1)[1] for d in defs if d.startswith("PATCH=")]
defs = [d for d in defs if not d.startswith("PATCH=")]
here = os.path.dirname(os.path.abspath(src))
outdir = os.path.dirname(os.path.abspath(out))
hits = {d.split("=")[0]: 0 for d in defs}
for f in os.listdir(here):
    if not f.endswith(".fut"):
        continue
    text = open(os.path.join(here, f)).read()
    for d in defs:
        name, value = d.split("=")
        text, n = re.subn(rf"^def {name} : (\w+) = .*$", rf"def {name} : \1 = {value}", text, flags=re.M)
        hits[name] += n
    target = out + ".fut" if f == os.path.basename(src) else os.path.join(outdir, f)
    open(target, "w").write(text)
assert all(hits.values()), hits
subprocess.run([os.path.join(here, "fc.sh"), backend, out + ".fut", out], check=True)
if patch:
    # Move intra-block results to global memory, then relink the host code.
    subprocess.run([sys.executable, os.path.join(here, "..", "intrablock.py"), out + ".c", patch[0]], check=True)
    subprocess.run(["gcc", out + ".c", "-o", out, "-O3", "-std=c99", "-I/opt/cuda/include",
                    "-L/opt/cuda/lib64", "-lcuda", "-lcudart", "-lnvrtc", "-lm"], check=True)
