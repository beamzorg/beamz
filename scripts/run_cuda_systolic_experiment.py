"""Run one immutable native candidate against an explicit frozen control.

This is opt-in GPU work. A process-group deadline and campaign lock encompass
preparation, validation and fresh-process comparison. Control/candidate hashes
are checked before execution. No installed module is replaced.
"""
from __future__ import annotations
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.cache/cuda-native-autoresearch-2026-11-11-1607/native-only/bin/python'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--fixture', type=Path)
    p.add_argument('--suffix', default='screen')
    p.add_argument('--full-validation', action='store_true')
    p.add_argument('--abba', action='store_true')
    p.add_argument('--samples', type=int, default=3)
    a = p.parse_args()
    candidate = a.candidate.resolve()
    contract = json.loads(a.contract.read_text())
    if a.abba and contract.get('ABBA') is False:
        raise ValueError('This campaign uses fast A/B iteration; ABBA is disabled by its contract')
    frozen = json.loads((candidate/'frozen-execution-manifest.json').read_text())
    module = Path(frozen['module'])
    if digest(module) != frozen['sha256'] or digest(contract['control_module']) != contract['control_sha256']:
        raise ValueError('Frozen candidate/control module hash mismatch')
    deadline = datetime.datetime.fromisoformat(contract['gpu_deadline_utc'].replace('Z', '+00:00')).timestamp()
    out = candidate.parent/(candidate.name+'-'+a.suffix)
    out.mkdir(exist_ok=False)
    records = []

    def save():
        (out/'campaign-processes.json').write_text(json.dumps(
            dict(candidate=frozen, control=contract, records=records), indent=2)+'\n')

    def run(label, command, limit):
        remaining = deadline-time.time()
        if remaining < 30:
            raise RuntimeError('GPU deadline reached; retain time for report')
        row = dict(label=label, command=list(map(str, command)),
                   start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        records.append(row)
        save()
        start = time.monotonic()
        with (out/(label+'.campaign.log')).open('w') as log:
            child = subprocess.Popen(row['command'], cwd=ROOT, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     env=dict(os.environ, PYTHONPATH=str(ROOT)))
            row['pid'] = child.pid
            save()
            try:
                row['returncode'] = child.wait(timeout=min(limit, remaining))
            except subprocess.TimeoutExpired:
                row['timed_out'] = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                row['returncode'] = child.returncode
        row['wall_s'] = time.monotonic()-start
        save()
        print(json.dumps({k:row[k] for k in ('label', 'returncode', 'wall_s')}), flush=True)
        if row['returncode']:
            raise RuntimeError('Candidate stage failed: '+label)

    with (candidate.parent/'gpu.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run('prepare', ['python', 'scripts/run_cuda_plane_experiment.py', '--stage', 'prepare',
                        '--baseline', contract['control_module'], '--expected-control-sha256',
                        contract['control_sha256'], '--candidate', module, '--output', out], 60)
        if a.fixture:
            plan = json.loads((out/'plan.json').read_text())
            plan['fixture_prefix'] = str(a.fixture.resolve())
            plan['fixture_full'] = str(a.fixture.resolve())
            plan['fixture_hashes'] = {str(a.fixture.resolve()):digest(a.fixture)}
            plan['scope'] = 'Matched supplied fixture, complete FP32 CPML12 source/DFT physics; all30 first/repeat outputs'
            (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
        if a.full_validation:
            run('validate', [PYTHON, 'scripts/run_cuda_plane_experiment.py', '--stage', 'validate',
                             '--candidate', module, '--output', out], 900)
        else:
            for depth, steps in ((4, 7), (16, 35)):
                run('gate'+str(depth), [PYTHON, out/'workers/validate_cuda_native_moving.py',
                     '--module', module, '--fixture', ROOT/'.cache/cuda-native-autoresearch-2026-11-11-1607/fixtures/small-mie-seeded.json',
                     '--depth', depth, '--steps', steps, '--oracle-kind', 'program',
                     '--output', out/('gate'+str(depth)+'.json')], 180)
        command = [PYTHON, 'scripts/run_cuda_plane_experiment.py', '--stage', 'benchmark',
                   '--candidate', module, '--output', out, '--warmups', '0' if a.fixture else '1',
                   '--samples', a.samples]
        if not a.abba:
            command += ['--screen']
        run('benchmark', command, 2000)


if __name__ == '__main__':
    main()
