"""Exercise the actual C++ certificate with independently decoded coefficients."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def certificate(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("block-certificate")
    source = directory / "facts.cc"
    source.write_text(r'''
#define BeamzCertifyNativeMovingUniformE LegacyByteCertificate
#include "native_moving.cc"
#undef BeamzCertifyNativeMovingUniformE
#include "cuda_native_block_material_facts.inc"
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
  int32_t supported=facts.supported;out.write((char*)&supported,4);out.write((char*)facts.bits.data(),12);out.write((char*)facts.admitted.data(),facts.admitted.size());
 }catch(const std::invalid_argument&){return 7;}
}
''')
    binary = directory / "facts"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(root / "cuda/src"),
                    "-I", str(root / "scripts"), str(source), "-o", str(binary)], check=True)
    return binary


def pack(values, rank):
    if rank == 1:
        pool, inverse = np.unique(values, return_inverse=True)
        assert len(pool) <= 256
        padded = np.zeros((len(values) + 3) // 4 * 4, np.uint8)
        padded[:len(values)] = inverse
        return pool, padded.view(np.int32)
    pool, headers = [], []
    for start in range(0, len(values), 32):
        block = values[start:start + 32]
        if np.all(block == block[0]):
            headers.append(~len(pool))
            pool.append(int(block[0]))  # Deliberately duplicate constant aliases.
        else:
            headers.append(len(pool))
            pool.extend(block.tolist())  # Last literal need not have padding.
    return np.array(pool, np.uint32), np.array(headers, np.int32)


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("mode", ["uniform", "one-ulp", "signed-zero", "tail", "many-values"])
def test_exact_bits_and_full_trajectory(certificate, tmp_path, rank, mode):
    if rank == 1 and mode == "many-values":
        pytest.skip("More than 256 exact values requires block format")
    shapes = np.array([[33, 34, 36], [33, 35, 35], [34, 34, 35],
                       [34, 35, 35], [34, 34, 36], [33, 35, 36]], np.int32)
    depth = 16
    # Obtain the geometry from the same header, but compute all box facts in NumPy.
    root = Path(__file__).resolve().parents[2]
    import re
    owned_z = int(re.search(r"kOwnedZ=(\d+)", (root / "cuda/src/native_moving.h").read_text())[1])
    owned = np.array([owned_z, 15, 31])
    counts = (shapes.max(0) + depth + owned - 1) // owned
    origins = np.indices(counts).reshape(3, -1).T * owned
    expected = np.ones(len(origins), bool)
    bits = []
    packet = bytearray(shapes.tobytes() + np.int32(depth).tobytes())
    for c, shape in enumerate(shapes[3:]):
        common = np.float32(0 if mode == "signed-zero" else 2).view(np.uint32)
        values = np.full(int(np.prod(shape)), common, np.uint32)
        if mode in ("one-ulp", "signed-zero"):
            values[(12 * shape[1] + 17) * shape[2] + 20] = common ^ (1 if mode == "one-ulp" else 0x80000000)
        if mode == "tail":
            values[-1] = common ^ 1
        if mode == "many-values":
            values[1000:1600] += np.arange(600, dtype=np.uint32)
        pool, words = pack(values, rank)
        packet.extend(np.array([rank, len(pool), len(words)], np.int32).tobytes())
        packet.extend(pool.tobytes()); packet.extend(words.tobytes())
        unique, frequency = np.unique(values, return_counts=True)
        selected = unique[np.argmax(frequency)]
        bits.append(selected)
        grid = values.reshape(shape)
        for i, origin in enumerate(origins):
            lo = np.clip(origin - depth, 0, shape)
            hi = np.clip(origin + owned - 1, 0, shape)
            expected[i] &= np.all(grid[tuple(slice(a, b) for a, b in zip(lo, hi))] == selected)
    source, output = tmp_path / "input.bin", tmp_path / "output.bin"
    source.write_bytes(packet)
    subprocess.run([str(certificate), str(source), str(output)], check=True)
    raw = output.read_bytes()
    assert np.frombuffer(raw, np.int32, count=1)[0] == 1
    np.testing.assert_array_equal(np.frombuffer(raw, np.uint32, count=3, offset=4), bits)
    np.testing.assert_array_equal(np.frombuffer(raw, np.uint8, offset=16), expected)


@pytest.mark.parametrize("header", [100000, -100000])
def test_invalid_pool_address_fails_closed(certificate, tmp_path, header):
    shapes = np.full((6, 3), 33, np.int32)
    packet = bytearray(shapes.tobytes() + np.int32(16).tobytes())
    for _ in range(3):
        words = np.full((33 ** 3 + 31) // 32, header, np.int32)
        packet.extend(np.array([2, 1, len(words)], np.int32).tobytes())
        packet.extend(np.float32(2).tobytes()); packet.extend(words.tobytes())
    source, output = tmp_path / "input.bin", tmp_path / "output.bin"
    source.write_bytes(packet)
    assert subprocess.run([str(certificate), str(source), str(output)]).returncode == 7
