"""Quantify per-surface-region export lifetimes in physics-closed cohorts."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path

from scripts.cuda_policy_closed_cohorts import compile_cohorts

REGION_POINTS={1:420,2:180,3:30,4:84,5:14,6:6,7:1}


def analyze(plan,epochs=4):
    physical=compile_cohorts(plan,epochs,'earliest',omit_war=True);mapping=physical['cohort_of'];nz,ny,nx=plan['counts'];depth=plan['depth'];n=len(mapping)
    counts=Counter();maximum=0
    for t in range(n):
        z,y,x=t//(ny*nx),(t//nx)%ny,t%nx;counts[mapping[t],z+y+x]+=1;maximum=max(maximum,counts[mapping[t],z+y+x])
    rows=[];copied_points=0;reserved_bytes=0;checks=0
    for mask,points in REGION_POINTS.items():
        ends={}
        for t in range(n):
            z,y,x=t//(ny*nx),(t//nx)%ny,t%nx;last=mapping[t]
            for d in range(1,8):
                if d&~mask:continue
                dz,dy,dx=d&1,(d>>1)&1,(d>>2)&1
                if z+dz<nz and y+dy<ny and x+dx<nx:last=max(last,mapping[t+dz*ny*nx+dy*nx+dx])
            if last>mapping[t]:ends[t]=last
        busy=[];available=[];slots={};next_slot=0
        for c,cohort in enumerate(physical['cohorts']):
            while busy and busy[0][0]<c:
                _,slot=heapq.heappop(busy);heapq.heappush(available,slot)
            for t in cohort['tiles']:
                if t not in ends:continue
                if available:slot=heapq.heappop(available)
                else:slot=next_slot;next_slot+=1
                slots[t]=slot;heapq.heappush(busy,(ends[t],slot))
        # Independent interval exclusion audit on every actual color.
        last_end={}
        for t in (t for c in physical['cohorts'] for t in c['tiles'] if t in slots):
            slot=slots[t];assert last_end.get(slot,-1)<mapping[t];last_end[slot]=ends[t];checks+=1
        used=len(ends)*points;copied_points+=used;capacity=next_slot*(depth-1)*6*points*4;reserved_bytes+=capacity
        rows.append(dict(high_axis_mask=mask,points_per_packet=points,exported_fragments=len(ends),peak_interval_colors=next_slot,reserved_bytes=capacity,copied_points=used))
    # Check all735 native surface points against the region partition.
    seen=Counter()
    for z in range(7):
        for y in range(15):
            for x in range(31):
                mask=(z==6)|((y==14)<<1)|((x==30)<<2)
                if mask:seen[mask]+=1
    assert dict(seen)==REGION_POINTS
    internal_owners=epochs*maximum;internal_bytes=internal_owners*depth*6*735*4
    return dict(schema='beamz.export-fragment-lifetime-study.v1',passed=True,physics_cohorts=physical['cohort_count'],tiles=n,region_rows=rows,
                internal_full_packet_owners=internal_owners,internal_full_packet_bytes=internal_bytes,export_fragment_bytes=reserved_bytes,
                total_trail_bytes=internal_bytes+reserved_bytes,interval_exclusion_checks=checks,
                extra_copy_surface_fraction=copied_points/(n*735),
                scope='Actual pre-material geometry/source/monitor policy plan; exact7surface reader-equivalence regions and export interval colors. Hypothesis only: no generated CUDA, FP or speed claim;128Bpool alignment padding notincluded.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=analyze(json.loads(a.plan.read_text()));r['input_sha256']=hashlib.sha256(a.plan.read_bytes()).hexdigest();r['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='region_rows'}))
