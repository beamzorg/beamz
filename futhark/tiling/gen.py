#!/usr/bin/env python3
"""Instantiate a tiling experiment with constants and compile it.

    gen.py SRC BACKEND OUT NAME=VALUE...

Every .fut file next to SRC is copied next to OUT with each `def NAME : t = v`
replaced, so constants in imported files are overridden too.
"""
import os, re, subprocess, sys

src, backend, out, *defs = sys.argv[1:]
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
