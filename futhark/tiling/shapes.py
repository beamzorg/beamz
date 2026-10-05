"""Staggered checkerboard redundancy for non-rectangular tile shapes.
Reuses checker.stagger with its rectangular colour mask swapped out.

    shapes.py [K LEAD]    (default "2 1"; "4 2" takes several minutes)
"""
import inspect, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import checker

src = inspect.getsource(checker.stagger)
src = src.replace("def stagger(shape, TY, TX,", "def stagger_m(shape, BLACK, TY, TX,")
src = src.replace("black = ((jj // TY) + (ii // TX)) % 2 == 0", "black = BLACK(jj, ii)")
src = src.replace("return stuck, diff, count[0] / plain_updates", "return stuck, diff, (count[0] / plain_updates, stats['redundant'] / plain_updates, stats['reach'])")
ns = dict(vars(checker))
exec(src, ns)
stagger_m = ns["stagger_m"]

S = {
 "rect 8x16":        lambda j, i: ((j // 8) + (i // 16)) % 2 == 0,
 "diamond (u,w)/16": lambda j, i: (((i + j) // 16) + ((i - j + 1000) // 16)) % 2 == 0,
 "shear /  8x16":    lambda j, i: ((j // 8) + ((i + j) // 16)) % 2 == 0,
 "shear \\ 8x16":    lambda j, i: ((j // 8) + ((i - j + 1000) // 16)) % 2 == 0,
 "shear / 16x8":     lambda j, i: (((j + i) // 8) + (i // 16)) % 2 == 0,
 "shear \\ 16x8":    lambda j, i: (((j - i + 1000) // 8) + (i // 16)) % 2 == 0,
}
K, lead = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (2, 1)
cfgs = [(1, 1, "H"), (1, 1, "EH"), (2, 1, "EH"), (2, 2, "EH"), (2, 2, "H")] if K == 2 else [(2, 2, "EH"), (3, 2, "EH"), (2, 3, "EH")]
H, E = checker.H, checker.E
for name, f in S.items():
    for depth, halo, hc in cfgs:
        stuck, diff, w = stagger_m((3, 67, 135), f, 8, 16, 12, lead, K, verbose=False,
                                   depth=depth, halo=halo, hcomps={"H": H, "E": E, "EH": H + E}[hc])
        if stuck == 0 and diff == 0:
            print(f"{name:18s} K={K} lead={lead} depth {depth} halo {halo} {hc:2s}: own {w[0]:.4f} redundant {w[1]:.4f} reach {w[2]}", flush=True)
            break
    else:
        print(f"{name:18s} K={K}: stuck with all configs", flush=True)
