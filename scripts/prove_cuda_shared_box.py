"""Recheck traced shared-field and CPML handoffs under7x7x31 ownership.

Rebinds the existing randomized warp-interleaving oracle to the new owner map.
The dependency graph comes from the actual typed Yee/CPML operator. Negative
controls remove barriers/imports or alias the two field families.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import inspect,itertools,json
from pathlib import Path
from scripts import prove_cuda_traced_lifetime_shared_fields as original
from scripts.prove_cuda_stationary_psi_cache import prove,prove_owner_boundary_classes
from scripts.cuda_trace_phase_cycle import sha


@dataclass(frozen=True)
class Owner:
    shape:tuple[int,int,int]=(8,8,32)
    threads:int=512
    def point(self,tid,j):return tid//256+2*j,(tid//32)%8,tid%32
    def tid(self,p):return (p[0]%2)*256+p[1]*32+p[2]


def rebound():
    namespace=dict(vars(original));namespace['Owner']=Owner
    for fun in (original.simulate,original.prove_hazards):
        text=inspect.getsource(fun)
        text=text.replace('% 16','% 8').replace('// 15','// 7').replace('% 15','% 7')
        text=text.replace('* 15','* 7').replace('tid < 105','tid < 49')
        text=text.replace('list(range(32))','list(range(16))')
        exec(compile(text,'<same-shared-field-oracle-new-owner>', 'exec'),namespace)
    return namespace['simulate'],namespace['prove_hazards']


def run(quick=False):
    simulate,hazards=rebound()
    g=Owner();mapping={g.point(t,j):(t,j) for t in range(512) for j in range(4)}
    assert len(mapping)==2048 and all(g.tid(p)==t for p,(t,j) in mapping.items())
    cases=[simulate(first=f,steps=d,seed=s) for f,d,s in itertools.product((1,4,13),(1,2) if quick else (1,2,8,16),(7,29))]
    negatives=[]
    for name in original.NEGATIVES:
        try:simulate(first=1,steps=2,seed=7,negative=name)
        except original.Hazard as error:negatives.append(dict(mutation=name,rejected=True,witness=str(error)))
        else:raise AssertionError('Unsafe schedule escaped: '+name)
    packed=[prove(n,d,owned=(7,7,31),mode='single-axis',load_policy='first-owned')
            for n,d in (((32,35,37),4),((64,96,128),16),((65,97,129),16))]
    return dict(schema='beamz.shared-box-version-hazard-proof.v1',owner_shape=[8,8,32],threads=512,
                source_sha256=sha(Path(__file__).read_bytes()),
                base_oracle_sha256=sha(Path(original.__file__).read_bytes()),
                positive_cases=cases,negative_controls=negatives,same_phase_hazards=hazards(),
                total_field_and_clock_checks=sum(c['actual_field_and_clock_checks'] for c in cases),
                packed_CPML_handoffs=packed,owner_boundaries=prove_owner_boundary_classes((7,7,31)),
                scope='CPU dataflow/version/hazard proof; actual numerical CUDA and sanitizer checks remain separate')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--quick',action='store_true');a=p.parse_args();r=run(a.quick);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ('total_field_and_clock_checks','threads')}))
