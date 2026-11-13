"""Check causal, counter-free retirement using the canonical cell/version model."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types
from scripts.build_cuda_partial_plane_pipeline import once


def model():
    path=Path(__file__).with_name('prove_cuda_partial_plane_pipeline.py');original=path.read_text();source=original
    source=once(source,"disposal=False,closed_epochs=","disposal=False,distal=True,closed_epochs=")
    source=once(source,"if not state['remaining'] and state['ready']:dispose(key)","if not distal and not state['remaining'] and state['ready']:dispose(key)")
    source=once(source,"if not state['remaining'] or negative=='early-packet-disposal':dispose(key)","if (not distal and not state['remaining']) or negative=='early-packet-disposal':dispose(key)")
    source=once(source,"            acknowledge(t,t.n)\n            t.state='E'",'''            acknowledge(t,t.n)
            if distal and t.n<depth:
                q=tuple(x-1 for x in t.q)
                if negative=='axial-disposer':q=(t.q[0]-1,t.q[1],t.q[2])
                if q in tasks and t.n<=tasks[q].last:dispose((owner(q),t.n))
            t.state='E' ''')
    source=once(source,"        elif t.state=='H':execute_math(t,0);t.state='imports'",'''        elif t.state=='H':
            if negative=='before-distal-import' and t.n<depth:
                q=tuple(x-1 for x in t.q)
                if q in tasks and t.n<=tasks[q].last:dispose((owner(q),t.n))
            execute_math(t,0);t.state='imports' ''')
    source=once(source,"            t.state='H'\n        elif t.state=='H':",'''            if negative=='late-startup-disposer' and version:
                q=tuple(x-1 for x in t.q)
                if q in tasks and version<=tasks[q].last:dispose((owner(q),version))
            t.state='H'
        elif t.state=='H':''')
    module=types.ModuleType('beamz_distal_disposal_model');module.__file__=str(path);sys.modules[module.__name__]=module;exec(compile(source,str(path)+'[distal-retirement]', 'exec'),module.__dict__)
    return module.simulate,dict(base_model_sha256=hashlib.sha256(original.encode()).hexdigest(),distal_model_sha256=hashlib.sha256(source.encode()).hexdigest())


def report():
    simulate,ancestry=model();cases=[]
    for minimum,owned,depth in [((11,13,15),(3,5,7),4),((13,17,21),(7,15,31),16),((4,4,11),(3,3,7),4)]:
        for workers in (1,4,82):
            for schedule in ('random','producer-fast','consumer-fast'):
                cases.append(simulate(minimum=minimum,owned=owned,depth=depth,workers=workers,seed=37,schedule=schedule,disposal=True,closed_epochs=4,closed_flags='homogeneous'))
    negatives={}
    for name in ('axial-disposer','before-distal-import','late-startup-disposer'):
        for geometry in [dict(minimum=(4,4,11),owned=(3,3,7),depth=4),dict(minimum=(15,17,47),owned=(7,15,31),depth=16)]:
            try:simulate(**geometry,workers=82,seed=37,schedule='consumer-fast',disposal=True,negative=name,closed_epochs=4,closed_flags='homogeneous',closed_allocation='no-reuse')
            except AssertionError as e:negatives[name]=str(e);break
        else:raise AssertionError('Ineffective causal-disposal negative:'+name)
    return dict(passed=True,schema='beamz.causal-distal-disposal.v1',**ancestry,cases=cases,negative_controls=negatives,
                proof='During active distal-reader fullstep n, every ancestor first<=distal.first<=n. Producer last>=n implies all8last>=n. Every n readiness in this8task box follows actual imports; 3parent readiness therefore joins all7readers before distal importjoin. Future late-startup reads use first-1<n. CTA completion orders disposal before owner reuse.',
                scope='Canonical six-field/version reads with poison-after-disposal; no CUDA visibility or performance claim')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=report();a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=True,cases=len(r['cases']),reads=sum(c['field_reads'] for c in r['cases']),disposed=sum(c['disposed_packets'] for c in r['cases']),negatives=r['negative_controls'])))
