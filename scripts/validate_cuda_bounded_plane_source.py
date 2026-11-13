"""Run the actual emitted CPML-plane source callback and host admission in C++."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

from scripts.cuda_trace_phase_cycle import block,sha
from scripts.validate_cuda_source_plane_pair import HARNESS


def validate(candidate):
    candidate=Path(candidate).resolve()
    proof=json.loads((candidate/'bounded-source-proof.json').read_text())
    source=candidate/'source'
    text=(source/'src/asymmetric_native.cuh').read_text()
    if sha(text.encode())!=proof['consumer_sha256']:raise ValueError('Candidate source changed')
    parent=(Path(proof['parent_source'])/'src/asymmetric_native.cuh').read_text()
    if sha(parent.encode())!=proof['parent_consumer_sha256']:raise ValueError('Parent source changed')
    _,_,reference=block(parent,'for(int source=0;source<s.count;++source)')
    _,_,bounded=block(text,'if(x==s.starts[2]&&(Policy!=5||(')
    harness=HARNESS.replace('@ORIGINAL@',reference).replace('@PLANE@','constexpr int Policy=5;\n'+bounded)
    harness=harness.replace('int z=12+random_word()%(nz-24),y=12+random_word()%(ny-24)',
                            'int z=int(random_word()%(nz+6))-3,y=int(random_word()%(ny+6))-3')
    harness=harness.replace('size_t checks=0,errors=0,facts=0,rejected=0,bounds=0;',
                            'size_t checks=0,errors=0,facts=0,rejected=0,bounds=0,changed=0;')
    harness=harness.replace('errors+=std::bit_cast<uint32_t>(a)!=std::bit_cast<uint32_t>(b);',
                            'changed+=std::bit_cast<uint32_t>(a)!=std::bit_cast<uint32_t>(initial);errors+=std::bit_cast<uint32_t>(a)!=std::bit_cast<uint32_t>(b);')
    harness=harness.replace('return errors||!bounds;',
                            'std::printf("changed_by_source=%zu\\n",changed);return errors||!bounds||changed<1000;')
    directory=candidate/'host-source-proof'
    directory.mkdir(exist_ok=False)
    cpp=directory/'bounded-source.cc';binary=directory/'bounded-source'
    cpp.write_text(harness)
    command=['g++','-std=c++20','-O2','-ffp-contract=off','-fno-fast-math','-fsanitize=address,undefined',
             '-I'+str(source/'src'),str(cpp),'-o',str(binary)]
    compiled=subprocess.run(command,capture_output=True,text=True,timeout=60)
    if compiled.returncode:raise RuntimeError(compiled.stderr[-8000:])
    executed=subprocess.run([str(binary)],capture_output=True,text=True,timeout=30)
    (directory/'output.log').write_text(executed.stdout+executed.stderr)
    if executed.returncode:raise ValueError(executed.stdout+executed.stderr)
    result=dict(schema='beamz.bounded-source-emitted-host-proof.v1',passed=True,
                actual_callback=True,ASan_UBSan=True,consumer_sha256=proof['consumer_sha256'],
                source_sha256=sha(cpp.read_bytes()),command=command,stdout=executed.stdout,
                scope='Actual source FP expressions, transverse bounds and host topology mutation checks; no CUDA scheduling claim')
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(validate(a.candidate)))
