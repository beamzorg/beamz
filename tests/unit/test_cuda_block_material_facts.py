"""Exercise the actual C++ certificate with independently decoded coefficients."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.cuda_source_layout import CUDA_INCLUDE_FLAGS


@pytest.fixture(scope="module")
def certificate(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("block-certificate")
    source = directory / "facts.cc"
    source.write_text(r"""
#include "native_moving.cc"
#include <fstream>
int main(int argc,char**argv){
 std::ifstream in(argv[1],std::ios::binary);std::ofstream out(argv[2],std::ios::binary);
 auto read=[&](void*p,size_t n){in.read((char*)p,n);if(!in)throw std::runtime_error("truncated test");};
 BeamzBuffer fields[6]{},tables[3]{},codes[3]{};int32_t n[18],depth;read(n,sizeof(n));read(&depth,4);
 for(int c=0;c<6;++c){fields[c].rank=3;fields[c].element_type=kBeamzF32;for(int a=0;a<3;++a)fields[c].dims[a]=n[c*3+a];}
 std::vector<uint32_t>pool[3],words[3];const void*t[3],*w[3];
 for(int c=0;c<3;++c){int32_t header[3];read(header,12);pool[c].resize(header[1]);words[c].resize(header[2]);read(pool[c].data(),pool[c].size()*4);read(words[c].data(),words[c].size()*4);
  tables[c].rank=1;tables[c].element_type=kBeamzF32;tables[c].dims[0]=pool[c].size();codes[c].rank=header[0];codes[c].element_type=kBeamzS32;
  codes[c].dims[0]=header[0]==1?words[c].size():1;codes[c].dims[1]=words[c].size();t[c]=pool[c].data();w[c]=words[c].data();}
 const auto plan=BeamzBuildNativeMovingPlan(fields,depth,{},nullptr,nullptr,12);
 try{const auto facts=BeamzCertifyNativeMovingUniformE(fields,tables,codes,t,w,plan);
  int32_t supported=facts.supported;out.write((char*)&supported,4);out.write((char*)facts.bits.data(),12);out.write((char*)&facts.peak_scratch_bytes,8);out.write((char*)facts.admitted.data(),facts.admitted.size());
 }catch(const std::invalid_argument&){return 7;}
}
""")
    binary = directory / "facts"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            *CUDA_INCLUDE_FLAGS,
            "-I",
            str(root / "scripts"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )
    return binary


def pack(values, rank):
    if rank == 1:
        pool, inverse = np.unique(values, return_inverse=True)
        assert len(pool) <= 256
        padded = np.zeros((len(values) + 3) // 4 * 4, np.uint8)
        padded[: len(values)] = inverse
        return pool, padded.view(np.int32)
    pool, headers = [], []
    for start in range(0, len(values), 32):
        block = values[start : start + 32]
        if np.all(block == block[0]):
            headers.append(~len(pool))
            pool.append(int(block[0]))  # Deliberately duplicate constant aliases.
        else:
            headers.append(len(pool))
            pool.extend(block.tolist())  # Last literal need not have padding.
    return np.array(pool, np.uint32), np.array(headers, np.int32)


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize(
    "mode",
    [
        "uniform",
        "one-ulp",
        "signed-zero",
        "tail",
        "many-values",
        "cap-exact",
        "overflow-dominant",
        "overflow-diverse",
    ],
)
def test_exact_bits_and_full_trajectory(certificate, tmp_path, rank, mode):
    shapes = np.array(
        [
            [33, 34, 36],
            [33, 35, 35],
            [34, 34, 35],
            [34, 35, 35],
            [34, 34, 36],
            [33, 35, 36],
        ],
        np.int32,
    )
    depth = 16
    oversized = {
        "many-values": 257,
        "cap-exact": 4096,
        "overflow-dominant": 8000,
        "overflow-diverse": 8192,
    }
    if rank == 1 and mode in oversized:
        # Every byte code is valid, but the advertised palette exceeds the
        # format's 256-entry support. Reject it before allocating the histogram.
        packet = bytearray(shapes.tobytes() + np.int32(depth).tobytes())
        for shape in shapes[3:]:
            pool = np.float32(2).view(np.uint32) + np.arange(
                oversized[mode], dtype=np.uint32
            )
            words = np.zeros((int(np.prod(shape)) + 3) // 4, np.int32)
            packet.extend(np.array([1, len(pool), len(words)], np.int32).tobytes())
            packet.extend(pool.tobytes())
            packet.extend(words.tobytes())
        source, output = tmp_path / "oversized.bin", tmp_path / "oversized-out.bin"
        source.write_bytes(packet)
        subprocess.run([str(certificate), str(source), str(output)], check=True)
        raw = output.read_bytes()
        assert np.frombuffer(raw, np.int32, count=1)[0] == 0
        assert np.frombuffer(raw, np.uint64, count=1, offset=16)[0] == 0
        assert len(raw) > 24 and not np.frombuffer(raw, np.uint8, offset=24).any()
        return
    # Obtain the geometry from the same header, but compute all box facts in NumPy.
    root = Path(__file__).resolve().parents[2]
    import re

    owned_z = int(
        re.search(
            r"kOwnedZ=(\d+)",
            (root / "cuda/src/temporal/host/native_moving.h").read_text(),
        )[1]
    )
    owned = np.array([owned_z, 15, 31])
    counts = (shapes.max(0) + depth + owned - 1) // owned
    origins = np.indices(counts).reshape(3, -1).T * owned
    expected = np.ones(len(origins), bool)
    bits = []
    packet = bytearray(shapes.tobytes() + np.int32(depth).tobytes())
    for c, shape in enumerate(shapes[3:]):
        common = np.float32(0 if mode == "signed-zero" else 2 * (c + 1)).view(np.uint32)
        values = np.full(int(np.prod(shape)), common, np.uint32)
        if mode in ("one-ulp", "signed-zero"):
            values[(12 * shape[1] + 17) * shape[2] + 20] = common ^ (
                1 if mode == "one-ulp" else 0x80000000
            )
        if mode == "tail":
            values[-1] = common ^ 1
        if mode == "many-values":
            values[1000:1600] += np.arange(600, dtype=np.uint32)
        if mode == "cap-exact":
            values += np.arange(len(values), dtype=np.uint32) % 4096
        if mode == "overflow-dominant":
            values[1000:9000] += np.arange(8000, dtype=np.uint32)
        if mode == "overflow-diverse":
            values += np.arange(len(values), dtype=np.uint32)
        pool, words = pack(values, rank)
        packet.extend(np.array([rank, len(pool), len(words)], np.int32).tobytes())
        packet.extend(pool.tobytes())
        packet.extend(words.tobytes())
        unique, frequency = np.unique(values, return_counts=True)
        # The all-unique fallback stream ends on its last literal candidate.
        selected = (
            values[-1] if mode == "overflow-diverse" else unique[np.argmax(frequency)]
        )
        bits.append(selected)
        grid = values.reshape(shape)
        for i, origin in enumerate(origins):
            lo = np.clip(origin - depth, 0, shape)
            hi = np.clip(origin + owned - 1, 0, shape)
            expected[i] &= np.all(
                grid[tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))]
                == selected
            )
    source, output = tmp_path / "input.bin", tmp_path / "output.bin"
    source.write_bytes(packet)
    subprocess.run([str(certificate), str(source), str(output)], check=True)
    raw = output.read_bytes()
    assert np.frombuffer(raw, np.int32, count=1)[0] == 1
    np.testing.assert_array_equal(
        np.frombuffer(raw, np.uint32, count=3, offset=4), bits
    )
    scratch = int(np.frombuffer(raw, np.uint64, count=1, offset=16)[0])
    # One 2D prefix and two packed XY planes, independent of the Z extent.
    prefix_bound = max(
        4 * ((ny + 1) * (words + 1) + 2 * ny * words)
        for _, ny, nx in shapes[3:]
        for words in [(nx + 31) // 32]
    )
    assert scratch == (65536 if mode == "uniform" else max(65536, prefix_bound))
    np.testing.assert_array_equal(np.frombuffer(raw, np.uint8, offset=24), expected)
    if mode.startswith("overflow"):
        subprocess.run([str(certificate), str(source), str(output)], check=True)
        assert output.read_bytes() == raw


@pytest.mark.parametrize("component", [0, 1, 2])
@pytest.mark.parametrize("overflow", [False, True])
def test_nonfinite_candidate_never_certifies_a_trajectory(
    certificate, tmp_path, component, overflow
):
    shapes = np.full((6, 3), 33, np.int32)
    packet = bytearray(shapes.tobytes() + np.int32(16).tobytes())
    for c in range(3):
        common = np.float32(np.nan if c == component else 2 * (c + 1)).view(np.uint32)
        values = np.full(33**3, common, np.uint32)
        if overflow:
            values[1000:9000] = np.float32(2).view(np.uint32) + np.arange(
                8000, dtype=np.uint32
            )
        pool, words = pack(values, 2)
        packet.extend(np.array([2, len(pool), len(words)], np.int32).tobytes())
        packet.extend(pool.tobytes())
        packet.extend(words.tobytes())
    source, output = tmp_path / "nan.bin", tmp_path / "nan-out.bin"
    source.write_bytes(packet)
    subprocess.run([str(certificate), str(source), str(output)], check=True)
    raw = output.read_bytes()
    assert np.frombuffer(raw, np.int32, count=1)[0] == 0
    assert not np.frombuffer(raw, np.uint8, offset=24).any()


@pytest.mark.parametrize("header", [100000, -100000])
def test_invalid_pool_address_fails_closed(certificate, tmp_path, header):
    shapes = np.full((6, 3), 33, np.int32)
    packet = bytearray(shapes.tobytes() + np.int32(16).tobytes())
    for _ in range(3):
        words = np.full((33**3 + 31) // 32, header, np.int32)
        packet.extend(np.array([2, 1, len(words)], np.int32).tobytes())
        packet.extend(np.float32(2).tobytes())
        packet.extend(words.tobytes())
    source, output = tmp_path / "input.bin", tmp_path / "output.bin"
    source.write_bytes(packet)
    assert subprocess.run([str(certificate), str(source), str(output)]).returncode == 7
