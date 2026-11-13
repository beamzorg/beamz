"""Install the experimental compact edge lowering into a copied source tree."""
import argparse
import shutil
from pathlib import Path


def patch(source):
    scripts = Path(__file__).resolve().parent
    if source.resolve() == (scripts.parent / "cuda").resolve():
        raise ValueError("Use a copied candidate source")
    path = source / "src/mstep.cuh"
    text = path.read_text()
    marker = "template <bool CopyCore>\ncudaError_t LaunchMstepFullEdge("
    if text.count(marker) != 1:
        raise ValueError("Expected one unmodified full-edge launcher")
    text = text.replace(marker, '#include "compact_full_edge.cuh"\n\n' + marker)
    start = text.index(marker)
    insert = text.index("  const int blocks =", start)
    text = text[:insert] + (
        "  if constexpr (!CopyCore)\n"
        "    if (TryMstepCompactEdge(stream, launch, cells, ny, nx)) return cudaGetLastError();\n"
    ) + text[insert:]
    path.write_text(text)
    shutil.copyfile(scripts / "cuda_compact_full_edge.cuh", source / "src/compact_full_edge.cuh")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-source", type=Path, required=True)
    patch(p.parse_args().candidate_source)
