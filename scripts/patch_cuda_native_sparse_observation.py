#!/usr/bin/env python3
"""Emit an opt-in copied consumer with direct sparse raw-ID lookup.

Sorted raw IDs, staged values, and chronological DFT arithmetic remain unchanged.
The initial experiment retains the old mask/keys and admits the added map bytes.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one sparse observation anchor: {old[:80]}")
    return source.replace(old, new, 1)


def patch(source):
    source = once(
        source,
        " beamz::cuda::deep::Observation observation;",
        """ beamz::cuda::deep::Observation observation;
 const int32_t* observation_tiles;const int32_t* observation_leaf;
 int observation_ny,observation_nx;""",
    )
    source = once(
        source,
        " const int cell=(z*o.mask_ny+y)*o.mask_nx+x;",
        """#if BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION
 const int tile=(z*p.observation_ny+(y>>3))*p.observation_nx+(x>>4);
 const int slot=__ldg(p.observation_tiles+tile);if(slot<0)return;
 const size_t leaf=size_t(slot)*128*6+(size_t(y&7)*16+(x&15))*6+field;
 const int raw=__ldg(p.observation_leaf+leaf);
 if(raw>=0)o.values[size_t(version-1)*o.key_count+raw]=value;
#else
 const int cell=(z*o.mask_ny+y)*o.mask_nx+x;""",
    )
    source = once(
        source,
        " if(lo<o.key_count&&o.keys[lo]==key)o.values[static_cast<size_t>(version-1)*o.key_count+lo]=value;\n}",
        " if(lo<o.key_count&&o.keys[lo]==key)o.values[static_cast<size_t>(version-1)*o.key_count+lo]=value;\n#endif\n}",
    )
    source = once(
        source,
        " out->observation=w.observation;",
        """#if BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION
 if(w.observation.key_count&&(!w.observation_tiles||!w.observation_leaf||
    w.observation_grid[0]!=w.observation.mask_nz||
    w.observation_grid[1]!=(w.observation.mask_ny+7)/8||
    w.observation_grid[2]!=(w.observation.mask_nx+15)/16))return false;
#endif
 out->observation_tiles=w.observation_tiles;out->observation_leaf=w.observation_leaf;
 out->observation_ny=w.observation_grid[1];out->observation_nx=w.observation_grid[2];
 out->observation=w.observation;""",
    )
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.input.resolve() == a.output.resolve():
        p.error("Output must differ from frozen input")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(patch(a.input.read_text()))


if __name__ == "__main__":
    main()
