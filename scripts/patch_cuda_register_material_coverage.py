"""Add an opt-in host-only global coverage policy to frozen UniformE D2 facts.

The exact whole-column certificate is unchanged. Private bit20 is retained only
when at least three quarters of nonzero original coarse masks admit. Every CUDA
source and kernel object can be reused byte-for-byte from the parent candidate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

COVERAGE_GUARD = r"""
    // RESEARCH_UNIFORM_D2_COVERAGE: avoid mixed class code on sparse coverage.
    size_t active_coarse_tiles=0;
    for(int64_t i=0;i<rows*columns;++i)active_coarse_tiles+=original[i]!=0;
    if(!active_coarse_tiles||uniform_d2_tiles_*4<active_coarse_tiles*3){
      uniform_d2_masks_.assign(original,original+rows*columns);
      uniform_d2_tiles_=0;
    }
"""


def coverage_facts(facts):
    anchor = '    Check(cudaMemcpyAsync(base_inputs_[mask_index].data,uniform_d2_masks_.data(),masks.bytes,cudaMemcpyHostToDevice,stream_.value),"upload private uniform-E facts");'
    if facts.count(anchor) != 1:
        raise ValueError("Expected one frozen UniformE upload anchor")
    return facts.replace(anchor, COVERAGE_GUARD + anchor, 1)


def patch(source):
    source = Path(source)
    root = Path(__file__).resolve().parents[1]
    if source.resolve() == (root / "cuda").resolve():
        raise ValueError("Use an isolated copied candidate")
    target = source / "src/native_extension.cc"
    before = target.read_text()
    if "RESEARCH_UNIFORM_D2_COVERAGE" in before:
        raise ValueError("Coverage policy already present")
    target.write_text(coverage_facts(before))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-source", type=Path, required=True)
    patch(p.parse_args().candidate_source)
