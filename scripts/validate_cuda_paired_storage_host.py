"""Audit emitted paired shared imports, independent coordinates and guard spans."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from scripts.cuda_trace_phase_cycle import block


def validate(candidate):
    candidate=Path(candidate).resolve();src=candidate/'source/src';proof=json.loads((candidate/'paired-handoff-compiler-proof.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if not proof.get('paired_handoff') or proof.get('phase_group_stride')!=49536 or proof.get('phase_shared_bytes')!=101376:raise ValueError('Exact paired shared layout required')
    for name,value in proof['helpers'].items():
        if sha(src/name)!=value:raise ValueError('Changed bound helper:'+name)
    c=(src/'asymmetric_native.cuh').read_text();body=(src/'native_phase_rebinding.cuh').read_text();handoff=(src/'native_paired_handoff.cuh').read_text()
    constants='\n'.join(line for line in c.splitlines() if line.startswith('constexpr int BZ=') or line.startswith('constexpr int PAD='))
    geometry=constants+'\n'+block(body,'int PhaseSlot(')[2]+'\n'+block(body,'struct PhasePoint{')[2]+';\nconstexpr int PhaseGhostPoints=PAD-BZ*BY*BX;\n'
    geometry+='\n'.join(block(body,s)[2] for s in ('PhasePoint PhaseGhostLogical(','PhasePoint PhaseGhostPhysicalSlot('))+'\n'+block(c,'unsigned NativeSupportMask(')[2]
    code=Path(__file__).with_name('prove_cuda_paired_storage_host.cc').read_text().replace('// ACTUAL_GEOMETRY',geometry).replace('// ACTUAL_SHARED_IMPORT','template<int Depth,int T,bool Electric>\n'+block(handoff,'void PairedScatterGhost(')[2])
    if proof.get('phase_threads')==256:code=re.sub(r'\b512\b','256',code)
    out=candidate/'host-paired-storage-proof';out.mkdir(exist_ok=False);rows=[]
    for name in ('actual','wrong-parent-Z-slot','wrong-family-source'):
        folder=out/name;folder.mkdir();text=code
        if name=='wrong-parent-Z-slot':text=text.replace('(-n-1+BZ)&(CZ-1)','(-n-2+BZ)&(CZ-1)')
        if name=='wrong-family-source':text=text.replace('v.ex=peer[source]','v.ex=peer[PAD+source]')
        cc=folder/'proof.cc';cc.write_text(text);binary=folder/'proof'
        built=subprocess.run(['g++','-std=c++20','-O2','-fsanitize=address,undefined',str(cc),'-o',str(binary)],capture_output=True,text=True);(folder/'build.log').write_text(built.stdout+built.stderr)
        if built.returncode:raise RuntimeError('Paired shared provider compile failed:'+str(folder/'build.log'))
        run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=180);(folder/'run.log').write_text(run.stdout+run.stderr)
        if (run.returncode!=0 if name=='actual' else run.returncode==0):raise RuntimeError('Unexpected paired provider audit:'+name+run.stdout+run.stderr)
        rows.append(dict(name=name,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr[-1500:]))
    result=dict(passed=True,consumer_sha256=proof['consumer_sha256'],helpers=proof['helpers'],rows=rows,actual_local_geometry_and_guards=True)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.candidate)))
