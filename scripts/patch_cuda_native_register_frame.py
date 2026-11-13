"""Copy-only complete7x15x31 register/phase-frame cache candidate."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch(source):
    old = "constexpr int BZ=5,BY=15,BX=31,CZ=6,CY=16,CX=32,Threads=512;"
    assert source.count(old) == 1
    source = source.replace(
        old, "constexpr int BZ=7,BY=15,BX=31,CZ=8,CY=16,CX=32,Threads=512;"
    )
    source = source.replace(
        "4ULL*PAD*sizeof(float)+8*sizeof(size_t)+(sizeof(int))",
        "3ULL*PAD*sizeof(float)+8*sizeof(size_t)+(sizeof(int))",
    )
    old = "constexpr size_t SharedBytes=4ULL*PAD*sizeof(float)+8*sizeof(size_t)+sizeof(int)+2ULL*LowXPoints*sizeof(float);"
    assert source.count(old) == 1
    source = source.replace(
        old,
        "constexpr size_t SharedBytes=3ULL*PAD*sizeof(float)+8*sizeof(size_t)+sizeof(int)+6ULL*LowXPoints*sizeof(float);",
    )
    start = source.index(
        "template<int Depth,bool Fast>\n__device__ __forceinline__ void StageLowX"
    )
    epoch = source.index("__global__ void BeginEpoch", start)
    kernel = source.index(
        "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=", epoch
    )
    end = source.index("inline bool MakeSource(", kernel)
    source = (
        source[:start]
        + source[epoch:kernel]
        + (ROOT / "scripts/cuda_native_register_frame_body.inc").read_text()
        + "\n"
        + source[end:]
    )
    source = source.replace(
        "<<<count,Threads,SharedBytes,stream>>>",
        "<<<count,(Fast?1024:512),SharedBytes,stream>>>",
    )
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.input.resolve() == a.output.resolve():
        p.error("copied output required")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(patch(a.input.read_text()))


if __name__ == "__main__":
    main()
