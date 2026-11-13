"""Execute emitted waveform staging and split-round source callbacks on CPU."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

from scripts.cuda_trace_phase_cycle import block,sha
from scripts.validate_cuda_source_plane_pair import HARNESS


def validate(candidate):
    candidate=Path(candidate).resolve();source=candidate/'source'
    proof=json.loads((candidate/'source-wave-cache-proof.json').read_text())
    text=(source/'src/asymmetric_native.cuh').read_text()
    if sha(text.encode())!=proof['consumer_sha256']:raise ValueError('Candidate changed')
    parent=(Path(proof['parent_source'])/'src/asymmetric_native.cuh').read_text()
    _,_,reference=block(parent,'for(int source=0;source<s.count;++source)')
    _,_,bounded=block(text,'if(x==s.starts[2]&&(Policy!=5||(')
    _,_,stage=block(text,'void StageSourceWaves(')
    harness=HARNESS.replace('struct Params{int macro_step;};',
                            'struct Params{int macro_step;Source sources[6];};using Parameters=Params;')
    before='template<bool PlanePair=true>float candidate('
    harness=harness.replace(before,stage+'\n'+before)
    setup='''constexpr int Policy=5;constexpr bool CachedWaves=true;
 Parameters bindings=p;bindings.sources[2]=s;bindings.sources[4]=s;
 float wave_values[4];StageSourceWaves(bindings,wave_values,version);
 const float*source_waves=wave_values;const int field=(version&1)?2:4;
'''
    harness=harness.replace('@ORIGINAL@',reference).replace('@PLANE@',setup+bounded)
    harness=harness.replace('int z=12+random_word()%(nz-24),y=12+random_word()%(ny-24)',
                            'int z=int(random_word()%(nz+6))-3,y=int(random_word()%(ny+6))-3')
    # Different H/E clocks, waveform lengths and values must not be conflated.
    extra='''
 size_t independent_checks=0;
 for(int hstep:{-9,0,8,17})for(int estep:{-3,3,21})for(int macro:{0,5,19})for(int version=1;version<=35;++version){
  float hvalues[10],evalues[22];for(int i=0;i<10;++i)hvalues[i]=float(2*i+1);for(int i=0;i<22;++i)evalues[i]=float(-3*i-2);
  Parameters p{macro};p.sources[2]=Source{2,1,1,1,5,nullptr,&hstep,nullptr,hvalues};p.sources[4]=Source{2,1,1,1,11,nullptr,&estep,nullptr,evalues};
  float values[4];StageSourceWaves(p,values,version);
  int ht=max(0,min(hstep+macro+version-1,4)),et=max(0,min(estep+macro+version-1,10));
  errors+=values[0]!=hvalues[ht]||values[1]!=hvalues[5+ht]||values[2]!=evalues[et]||values[3]!=evalues[11+et];++independent_checks;
 }
 std::printf("independent_wave_checks=%zu\\n",independent_checks);
'''
    harness=harness.replace('return errors||!bounds;',extra+'return errors||!bounds;')
    directory=candidate/'host-wave-proof';directory.mkdir(exist_ok=False)
    rows=[]
    for label,code,expected in [('actual',harness,0),('wrong-field',harness.replace('p.sources[2+2*family]','p.sources[2]'),1)]:
        cpp=directory/(label+'.cc');binary=directory/label;cpp.write_text(code)
        cmd=['g++','-std=c++20','-O2','-ffp-contract=off','-fno-fast-math','-fsanitize=address,undefined',
             '-I'+str(source/'src'),str(cpp),'-o',str(binary)]
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
        if r.returncode:raise RuntimeError(r.stderr[-8000:])
        r=subprocess.run([str(binary)],capture_output=True,text=True,timeout=30)
        if bool(r.returncode)!=bool(expected):raise ValueError(label+' '+r.stdout+r.stderr)
        rows.append(dict(label=label,command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,source_sha256=sha(cpp.read_bytes())))
    result=dict(schema='beamz.emitted-source-wave-host-proof.v1',passed=True,
                consumer_sha256=proof['consumer_sha256'],ASan_UBSan=True,records=rows,
                scope='Actual waveform staging, separate H/E clocks and source FP arithmetic; negative field-alias control; no CUDA visibility claim')
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True)
    a=p.parse_args();print(json.dumps(validate(a.candidate)))
