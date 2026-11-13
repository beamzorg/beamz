"""Compare the real C++ moving material certificate to a NumPy box oracle."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts.cuda_material_tile_facts import uniform_boxes


def test_compiled_material_certificate_matches_bits_boxes_and_aliases(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "facts.cc"
    source.write_text(r"""
#include "native_moving.h"
#include <fstream>
#include <cstring>
int main(int argc,char**argv){
 const int add[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};BeamzBuffer f[6]{};
 for(int c=0;c<6;++c){f[c].rank=3;f[c].element_type=kBeamzF32;for(int a=0;a<3;++a)f[c].dims[a]=33+add[c][a];}
 const auto plan=BeamzBuildNativeMovingPlan(f,4,{},nullptr,nullptr,12);
 std::ofstream out(argv[1],std::ios::binary);
 for(int mode=0;mode<4;++mode){std::vector<uint32_t>words[3];float table[3][2]{};BeamzBuffer tables[3]{},codes[3]{};const void*t[3],*w[3];
  for(int c=0;c<3;++c){table[c][0]=mode==0?1.f:mode==1?2.f:0.f;table[c][1]=mode==0?2.f:mode==1?2.f:-0.f;
   tables[c].rank=1;tables[c].element_type=kBeamzF32;tables[c].dims[0]=mode==3?1:2;
   int count=f[3+c].dims[0]*f[3+c].dims[1]*f[3+c].dims[2];words[c].assign((count+3)/4,0);
   for(int i=0;i<count;++i){unsigned code=mode==1?(i&1):mode==3?255:0;if(c==0&&(mode==0||mode==2)){const int changed=(12*f[3].dims[1]+17)*f[3].dims[2]+20;if(i==changed)code=1;}words[c][i>>2]|=code<<(8*(i&3));}
   codes[c].rank=1;codes[c].element_type=kBeamzS32;codes[c].dims[0]=words[c].size();t[c]=table[c];w[c]=words[c].data();
  }
  const auto facts=BeamzCertifyNativeMovingUniformE(f,tables,codes,t,w,plan);if(!facts.supported)return 3;
  out.write((const char*)facts.bits.data(),12);out.write((const char*)facts.admitted.data(),facts.admitted.size());
 }
}
""")
    binary = tmp_path / "facts"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(root / "cuda/src"),
            str(source),
            str(root / "cuda/src/native_moving.cc"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    output = tmp_path / "facts.bin"
    subprocess.run([str(binary), str(output)], check=True)
    shapes = np.array(
        [
            [33, 33, 34],
            [33, 34, 33],
            [34, 33, 33],
            [34, 34, 33],
            [34, 33, 34],
            [33, 34, 34],
        ]
    )
    owned = np.array([5, 15, 31])
    counts = (shapes.max(0) + 4 + owned - 1) // owned
    origins = np.indices(counts).reshape(3, -1).T * owned
    raw = output.read_bytes()
    stride = 12 + len(origins)
    assert len(raw) == 4 * stride
    for mode in range(4):
        expected = np.ones(len(origins), bool)
        bits = []
        for c, shape in enumerate(shapes[3:]):
            count = int(np.prod(shape))
            codes = np.zeros(count, np.uint8)
            table = np.array(
                [1, 2] if mode == 0 else [2, 2] if mode == 1 else [0, -0.0], np.float32
            )
            if mode == 1:
                codes = np.arange(count, dtype=np.uint32).astype(np.uint8) & 1
            if mode == 3:
                codes[:] = 255
                table = np.array([0], np.float32)
            if c == 0 and mode in (0, 2):
                codes[(12 * shape[1] + 17) * shape[2] + 20] = 1
            padded = np.zeros((count + 3) // 4 * 4, np.uint8)
            padded[:count] = codes
            fact = uniform_boxes(padded.view(np.int32), table, shape, origins, depth=4)
            expected &= fact["admitted"]
            bits.append(fact["bits"])
        offset = mode * stride
        assert np.array_equal(
            np.frombuffer(raw, np.uint32, count=3, offset=offset), bits
        )
        assert np.array_equal(
            np.frombuffer(raw, np.uint8, count=len(origins), offset=offset + 12),
            expected,
        )
