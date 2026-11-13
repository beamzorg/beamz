"""Check native warp ownership, versioned reads and separate-family hazards."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path

from scripts.cuda_space_time_ir import FIELDS,yee_operator


def prove(threads=1024,negative=None):
    cache=(8,16,32);pad=4096;slots=pad//threads;step=threads//512
    points=[(tid//512+j*step,(tid//32)&15,tid&31) for tid in range(threads) for j in range(slots)]
    assert len(points)==len(set(points))==pad
    origin=(35,45,62)
    def physical(q,n):
        low=-n-1
        return tuple(o+low+((x-low)&(c-1)) for o,x,c in zip(origin,q,cache))
    def index(q):return (q[0]*16+q[1])*32+q[2]
    checked=0;reads=[set(),set()];writes=[set(),set()]
    hbase=0 if negative=='alias-families' else 3*pad
    op=yee_operator(cpml_axes=(0,1,2))
    for n in (1,2,4,7,8,15,16,31,32,35):
        head=tuple((-n-1)&(c-1) for c in cache)
        for q in points:
            for field in range(3):
                writes[0].add(hbase+field*pad+index(q));writes[1].add(field*pad+index(q))
            if any(x==h for x,h in zip(q,head)):continue
            p=physical(q,n)
            for update in op.updates:
                f=update.output.field
                if f not in FIELDS:continue
                phase=int(f.startswith('E'))
                for read in update.reads:
                    if read.field not in FIELDS:continue
                    source=tuple((x+d)&(c-1) for x,d,c in zip(q,read.offset,cache))
                    version=n-1+read.version
                    if negative=='wrong-old-E-version' and not phase and read.field.startswith('E'):version=n
                    want=tuple(x+d for x,d in zip(p,read.offset))
                    assert physical(source,version)==want,(phase,q,n,read,source,physical(source,version),want)
                    sf=FIELDS.index(read.field);is_self=not any(read.offset)
                    # The actual frozen callbacks keep self values private.
                    if sf//3!=phase and not is_self:
                        address=(hbase if phase else 0)+(sf%3)*pad+index(source)
                        assert 0<=address<6*pad
                        reads[phase].add(address)
                    if sf//3==phase:assert read.field==f and is_self
                    checked+=1
    # Within each half-step, no warp writes a shared bank that another warp
    # can still read. F1 publishes all H before E; F2 retires H readers before
    # next-step H stores. Private self-values have one native thread owner.
    assert not (reads[0]&writes[0]),'H stores race old E readers'
    assert not (reads[1]&writes[1]),'E stores race current H readers'
    assert writes[0].isdisjoint(writes[1]),'Family homes alias'
    return dict(threads=threads,slots=slots,point_owners=pad,versioned_reads=checked,
                H_shared_reads=len(reads[0]),E_shared_reads=len(reads[1]),
                H_shared_writes=len(writes[0]),E_shared_writes=len(writes[1]),passed=True)


def report(candidate):
    candidate=Path(candidate).resolve();proof=json.loads((candidate/'native-family-banks-compiler-proof.json').read_text())
    c=(candidate/'source/src/asymmetric_native.cuh').read_text()
    if hashlib.sha256(c.encode()).hexdigest()!=proof['consumer_sha256']:raise ValueError('Actual family bank source changed')
    for edit in proof['bindings']:
        phase=edit['new_phase'];retire=edit['new_retire']
        if 'zs==hz||sy==hy||sx==hx' not in phase or 'frame[(3+c)*PAD+Slot(zs,sy,sx)]=h[c][j];' not in phase or phase.count('__syncthreads()')!=1:
            raise ValueError('Actual H provider phase binding changed')
        if 'frame[c*PAD+Slot(zs,sy,sx)]=e[c][j];' not in retire or retire.count('__syncthreads()')!=1:
            raise ValueError('Actual E provider retirement binding changed')
        if c.count(phase)!=2 or c.count(retire)!=2:raise ValueError('Actual numerical bodies no longer bind the proved phases')
    cases=[prove(t) for t in (512,1024)];negatives={}
    for name in ('alias-families','wrong-old-E-version'):
        try:prove(1024,name)
        except AssertionError as e:negatives[name]=str(e)
        else:raise AssertionError('Ineffective bank negative:'+name)
    return dict(schema='beamz.native-family-phase-legality.v1',passed=True,consumer_sha256=proof['consumer_sha256'],cases=cases,negative_controls=negatives,
                proof_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Native512/1024point maps, exact physical versioned field operands over cache wraps, private self ownership and disjoint shared read/write families; actual phase text bound. No cross-CTA CUDA visibility or GPU numerical/timing claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=report(a.candidate);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=True,versioned_reads=sum(c['versioned_reads'] for c in r['cases']),negative_controls=len(r['negative_controls']))))
