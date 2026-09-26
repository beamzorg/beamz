"""Owned-pod task controller: preserve the largest probe, validate, fit a tail case."""
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/workspace/evidence')
OUT = ROOT / 'sweep'
PARENT = 5756
PRIMARY_DEADLINE = dt.datetime.fromisoformat('2026-09-26T19:00:00+00:00').timestamp()
TAIL_DEADLINE = dt.datetime.fromisoformat('2026-09-26T19:05:00+00:00').timestamp()
paused = False


def log(**data):
    print(json.dumps(dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), **data)), flush=True)


def state(pid):
    try:
        rest = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return rest[0], int(rest[49]) if len(rest) > 49 else None
    except FileNotFoundError:
        return 'X', None


def cmd(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return ''


def record(name, status, **extra):
    rows = json.loads((OUT/'summary.json').read_text())
    datafile = OUT / f'{name}.json'
    row = dict(name=name, status=status, **extra)
    if datafile.exists():
        d = json.loads(datafile.read_text())
        assert d['final_state_finite'] and d['monitor_weight_min'] == 256
        assert d['native_sha256'] == '6753641e1a682751ad00c10abda53bd4cd33fd3569ccf048baea018bb94be00d'
        row.update(shape=d['shape'], resolution_nm=d['resolution_nm'], gcups=d['kernel_gcups'],
                   wall_s=d['worker_measurement_wall_s'], max_peak_live_bytes=max(x['stats']['peak_bytes_in_use'] for x in d['device_memory']))
    rows = [r for r in rows if r['name'] != name] + [row]
    tmp = OUT / 'summary.tmp'
    tmp.write_text(json.dumps(rows, indent=2)+'\n')
    tmp.replace(OUT/'summary.json')
    log(stage='recorded', **row)


try:
    assert 'benchmark_h100_capacity.py' in cmd(PARENT)
    child = monitor = None
    while time.time() < PRIMARY_DEADLINE:
        if state(PARENT)[0] in ('X', 'Z'):
            raise RuntimeError('Primary runner exited before largest case')
        children = Path(f'/proc/{PARENT}/task/{PARENT}/children').read_text().split()
        for pid in map(int, children):
            line = cmd(pid)
            if 'balanced-896.json' in line and 'benchmark_modal_stepping.py' in line:
                child = pid
            if line.startswith('nvidia-smi '):
                monitor = pid
        if child:
            break
        time.sleep(2)
    if not child:
        raise RuntimeError('Largest case did not start before cutoff')
    os.kill(PARENT, signal.SIGSTOP)
    paused = True
    log(stage='hold_parent', parent=PARENT, child=child, monitor=monitor,
        reason='Finish largest probe before selecting remaining cases; wall cutoff remains 19:00 UTC')
    deadline_kill = False
    while state(child)[0] not in ('Z', 'X'):
        if time.time() >= PRIMARY_DEADLINE:
            os.killpg(child, signal.SIGTERM)
            time.sleep(10)
            if state(child)[0] not in ('Z', 'X'):
                os.killpg(child, signal.SIGKILL)
            deadline_kill = True
            break
        time.sleep(2)
    exit_status = state(child)[1]
    if monitor and cmd(monitor).startswith('nvidia-smi '):
        os.kill(monitor, signal.SIGTERM)
    success = (OUT/'balanced-896.json').exists() and not deadline_kill
    record('balanced-896', 'ok' if success else ('timeout' if deadline_kill else 'failed'),
           recovered_existing_result=True, wait_status=exit_status, controlled_cutoff='19:00 UTC')
    os.kill(PARENT, signal.SIGTERM)
    os.kill(PARENT, signal.SIGCONT)
    paused = False
    log(stage='release_for_validation')
    # The already queued validator takes the same exclusive GPU lock.
    while time.time() < TAIL_DEADLINE:
        if (ROOT/'validation/done').exists():
            break
        if state(6711)[0] in ('X', 'Z'):
            break
        time.sleep(2)
    parity = ROOT/'validation/comparison-80.json'
    if not parity.exists() or not all(r['passed'] for r in json.loads(parity.read_text())['comparisons']):
        raise RuntimeError('No passing 80 nm propagated parity gate; skip optional tail')
    # Use measured setup cost, never observed throughput, to select a tail size.
    baseline = json.loads((OUT/'planar-128.json').read_text())
    remaining = TAIL_DEADLINE - time.time()
    candidates = [n for n in (160,192,224) if baseline['worker_measurement_wall_s']*(n/128)**3*1.20+45 < remaining]
    if candidates:
        n = max(candidates)
        name = f'planar-{n}'
        log(stage='tail_selected', name=name, remaining_s=remaining,
            estimated_s=baseline['worker_measurement_wall_s']*(n/128)**3*1.20)
        env = dict(os.environ, PYTHONPATH='/workspace/beamz-h100', LD_LIBRARY_PATH='',
                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.95',
                   NCCL_NVLS_ENABLE='0', BEAMZ_CUDA_CPML_PSI_PRECISION='fp32', NUMPY_MADVISE_HUGEPAGE='0')
        env.pop('CUDA_VISIBLE_DEVICES', None)
        command = ['/workspace/venv/bin/python','scripts/benchmark_modal_stepping.py','--shape',str(n),str(8*n),str(64*n),'--devices','8','--backend','cuda_streamed','--host-setup','--frequencies','101','--timesteps','256','--samples','5','--resolution-nm','80','--output',str(OUT/f'{name}.json')]
        with open('/workspace/gpu.lock','w') as lock, (OUT/f'{name}.log').open('w') as output, (OUT/f'{name}.nvml.csv').open('w') as telemetry:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            mon = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem','--format=csv,nounits','-lms','500'], stdout=telemetry, stderr=subprocess.STDOUT)
            proc = subprocess.Popen(command, cwd='/workspace/beamz-h100', env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=max(1, TAIL_DEADLINE-time.time()-15))
                status = 'ok' if code == 0 else 'failed'
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait()
                status = 'timeout'
            finally:
                mon.terminate()
                mon.wait(timeout=10)
            record(name,status,command=command,adaptive_tail=True)
    else:
        log(stage='skip_tail',remaining_s=remaining)
    (OUT/'sweep.done').touch()
    log(stage='done')
finally:
    if paused:
        os.kill(PARENT,signal.SIGCONT)
