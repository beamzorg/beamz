"""Summarize only completed, hash-checked complete-physics comparisons."""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(area):
    area=Path(area).resolve();contract=json.loads((area/'control-contract.json').read_text())
    rows=[]
    for path in sorted(area.glob('*/prefix-comparison.json')):
        result=json.loads(path.read_text());plan=json.loads((path.parent/'plan.json').read_text())
        for worker in plan['workers']:
            assert digest(worker['frozen'])==worker['frozen_sha256']
        benchmark=path.parent/'workers/benchmark_native.py'
        fixture_path=Path(plan['fixture_prefix']);fixture=json.loads(fixture_path.read_text())
        assert result['control_identity_verified']
        assert digest(fixture_path)==plan['fixture_hashes'][str(fixture_path)]
        assert fixture['attributes']['boundary_code']==12*256 and fixture['attributes']['cpml_enabled']==1
        assert len(fixture['state_leaves'])==30
        control,candidate,shas,names=[],[],set(),set()
        for pair in result['pairs']:
            control.append(pair['baseline_s']);candidate.append(pair['candidate_s'])
            for side in ('baseline','candidate'):
                report_path=Path(pair[side]['path']);value=json.loads(report_path.read_text())
                assert digest(report_path)==pair[side]['sha256']
                assert value['status']=='ok' and value['host_inputs_unchanged']
                assert value['worker_sha256']==digest(benchmark)
                assert value['parity']['bitwise_exact'] and len(value['parity']['leaves'])==30
                assert value['repeat_parity']['bitwise_exact'] and len(value['repeat_parity']['leaves'])==30
                for parity in ('parity','repeat_parity'):
                    assert all(leaf['passed'] and leaf['finite'] and leaf['bitwise_exact']
                               for leaf in value[parity]['leaves'])
                assert value['module_sha256']==digest(value['module'])
                if side=='baseline':assert value['module_sha256']==contract['control_sha256']
                else:
                    shas.add(value['module_sha256']);names.add(Path(value['module']).parents[1].name)
        assert len(shas)==len(names)==1
        shape=[min(leaf['shape'][axis] for leaf in fixture['state_leaves'] if leaf['mapping']['kind']=='field') for axis in range(3)]
        attrs=fixture['attributes'];full=attrs['nsteps']>256
        rows.append(dict(experiment=path.parent.name,candidate=next(iter(names)),candidate_sha256=next(iter(shas)),
                         N_zyx='x'.join(map(str,shape)),nsteps=attrs['nsteps'],duration_fs=attrs['nsteps']*attrs['dt']*1e15,
                         control_s=statistics.median(control),candidate_s=statistics.median(candidate),
                         paired_speedup=result['median_paired_speedup'],
                         paired_range_min=min(p['speedup'] for p in result['pairs']),
                         paired_range_max=max(p['speedup'] for p in result['pairs']),
                         fresh_process_pairs=result['fresh_process_pairs'],
                         all30_first_repeat_bitwise=True,inputs_preserved=True,
                         full_duration=full,fixture=str(fixture_path),comparison=str(path)))
    builds=[]
    for candidate in sorted(area.iterdir()):
        log=candidate/'build-driver.log'
        if not candidate.is_dir() or not log.exists():continue
        frozen=candidate/'frozen-execution-manifest.json'
        if frozen.exists():
            proof=json.loads(frozen.read_text())
            assert digest(proof['module'])==proof['sha256']
            builds.append(dict(candidate=candidate.name,status='admitted',**proof))
        else:
            messages=log.read_text().splitlines()
            failure=next((line for line in messages if line.startswith('ValueError:')),None)
            builds.append(dict(candidate=candidate.name,status='rejected',reason=failure or '\n'.join(messages[-10:])))
    report=dict(schema='beamz.systolic-trace-study.v1',recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                control=contract,rows=rows,builds=builds,scope='Only completed matched comparisons; short screens are not full-duration results; pair ranges are not confidence intervals')
    (area/'comparisons.json').write_text(json.dumps(report,indent=2)+'\n')
    with (area/'comparisons.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]) if rows else []);writer.writeheader();writer.writerows(rows)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--area',type=Path,required=True)
    a=p.parse_args();r=summarize(a.area)
    print(json.dumps(dict(completed_comparisons=len(r['rows']),compiled=len(r['builds']),
                         admitted=sum(b['status']=='admitted' for b in r['builds']),
                         full_duration=sum(row['full_duration'] for row in r['rows']))))
