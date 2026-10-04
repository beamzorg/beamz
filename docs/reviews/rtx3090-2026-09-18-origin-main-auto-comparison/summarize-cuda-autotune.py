import json,statistics,hashlib,shutil,tarfile
from pathlib import Path
root=Path.cwd();out=root/'docs/reviews/rtx3090-2026-09-18-origin-main-auto-comparison';rows=json.loads((out/'summary.json').read_text());assert len(rows)==12
summary={};telemetry=[]
for case in ('narrow','wide','irregular'):
 baseline=json.loads((out/f'{case}-main-1.json').read_text());values={}
 for variant in ('main','branch_auto'):
  records=[r for r in rows if r['case']==case and r['variant']==variant];assert len(records)==2
  values[variant]=statistics.mean(r['median_gcups'] for r in records)
  for r in records:
   tag=f"{case}-{variant}-{r['repeat']}";d=json.loads((out/(tag+'.json')).read_text())
   for key in ('initial_state','coefficients','sources','workload','cpml_dtypes'):assert d[key]==baseline[key],(tag,key)
   telemetry.extend(json.loads((out/(tag+'.telemetry.json')).read_text()))
   if variant=='branch_auto':
    assert d['tuning']['validated'] and d['tuning']['rejected']=={}
    assert d['tuning']['cache_hit']==(r['repeat']==2)
    assert len(d['tuning']['samples_s'])==6
    if r['repeat']==1:
     values['choice']=d['tuning']['choice'];values['calibration_s']=d['tuning']['calibration_s']
     values['numerical_failures_vs_main']=[{k:x[k] for k in ('index','shape','mismatch_count','max_abs_error','reference_scale')} for x in d['final_state_validation'] if not x['tolerance_pass']]
 values['change_percent']=100*(values['branch_auto']/values['main']-1);summary[case]=values
record={'cases':summary,'max_temperature_c':max(t['temperature_c'] for t in telemetry),'min_free_mib':min(t['free_mib'] for t in telemetry),'power_limits_w':sorted({t['limit_w'] for t in telemetry})}
(out/'analysis.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
lines=['# Automatic CUDA layout and shell selection — 2026-09-18','',
'Implemented automatic first-use calibration plus a persistent lookup cache. The selector measures the three cyclic storage orders and two existing CPML shell tiles. It retains canonical storage unless a candidate reduces median runtime by at least 3% and wins both timing halves. Precision, physical shape, CPML thickness, source timing and monitor sampling remain unchanged.', '',
'## Fresh automatic-branch versus origin/main comparison','',
'Compared fetched main `c5fe0d8833ac6539241ae9d3c98445e5a7305786` against the local branch with automatic selection enabled and no manual layout/kernel overrides. Each shape ran main/automatic/automatic/main in fresh processes, with four warmups and nine timed 256-step executions per process. Values below average the two process medians. Identical inputs: FP32, lossless material, CPML12, one mode source, two compact mode monitors, three frequencies. All conversion work is timed; setup/calibration/compilation are excluded.', '',
'| Physical domain (z,y,x) | Main GCUPS | Automatic branch GCUPS | Change | Chosen axes / shell | Calibration cost |',
'|---|---:|---:|---:|---|---:|']
for case,shape in [('narrow','1024×256×64'),('wide','128×256×512'),('irregular','257×193×341 (smooth)')]:
 d=summary[case];choice=d['choice'];axes=''.join(map(str,choice['axes']))
 lines.append(f"| {shape} | {d['main']:.3f} | {d['branch_auto']:.3f} | {d['change_percent']:+.1f}% | {axes} / {choice['shell_tile']} | {d['calibration_s']:.1f} s |")
lines+=['','The second automatic process hit the persistent cache in all three cases; it did not repeat calibration. Initial states, coefficient arrays and compiled source arrays matched main exactly for every run. All six candidate layouts/tiles in every calibration passed bitwise complete-state validation against the branch canonical baseline.', '',
'## Validation and limits','',
'- 139 CPU tests passed across tuning policy, cache/backend contracts, runtime contracts and simulation architecture.',
'- Two seeded hardware tests passed for both GPU/default and CPU setup. Each checks all six candidates, persistent lookup and caller-state preservation.',
'- A preceding same-process matrix verified selected-versus-default complete-state equality and isolated the disabled policy from the automatically selected cached plan.',
'- Branch-versus-main still has the earlier small CPML tolerance failures in the binary-material cases. Field/monitor comparisons pass. Automatic selection adds no numerical difference relative to the branch baseline, but this is not proof of full equivalence to main.',
'- The automatic scope is one RTX3090, >=8,388,608 cells, 32–256 steps, uniform 3D lossless diagonal materials, CPML12 and native-compatible slab sources/DFT monitors. Other workloads keep their existing dispatch. This is not a guarantee of beating main for arbitrary domains or schedules.',
'- First-use calibration costs about half a minute for these cases and is additional startup work. It does not pay back within a single short run. The exact-workload cache favors repeated simulations; changed physical requests recalibrate.',
'- GPU/driver/power limit, exact request and solver implementation changes invalidate cached measurements. Manual CUDA overrides take precedence. Cache writes are atomic; unwritable caches do not prevent execution.', '',
f"Peak observed GPU temperature was {record['max_temperature_c']:.0f} C; minimum free GPU memory was {record['min_free_mib']:.0f} MiB. Power remained 370 W. GPU benchmark processes ran sequentially. Nothing was pushed; native CUDA kernels and FP32 defaults were unchanged by this work.", '',
'## Use','',
'Ordinary `sim.compile(num_steps=256, backend="cuda_streamed")` selects automatically when eligible. Inspect `beamz.simulation.cuda.tuning.tuning_report(program)` for the choice, measurements, cache hit and calibration cost. `BEAMZ_CUDA_AUTOTUNE=off` disables tuning; `refresh` recalibrates after clearing the program cache or restarting. `BEAMZ_CUDA_TUNING_CACHE` overrides the persistent cache directory.', '',
'[Fresh comparison artifacts](rtx3090-2026-09-18-origin-main-auto-comparison/) contain raw timings, input fingerprints, numerical comparison details, telemetry and solver/source provenance. [Initial selector validation](rtx3090-2026-09-18-autotune/) contains the six-candidate profiles and cache/disable checks.']
(root/'docs/reviews/cuda-autotuning-2026-09-18.md').write_text('\n'.join(lines)+'\n')
for f in ['scripts/benchmark_cuda_revision.py','scripts/benchmark_cuda_realistic.py','.cache/perf/run-origin-main-auto-comparison.py','.cache/perf/summarize-cuda-autotune.py','.cache/perf/cuda-autotune-cpu-final.log','.cache/perf/cuda-autotune-hardware-tests.log']:
 shutil.copy2(root/f,out/Path(f).name)
with tarfile.open(out/'branch-solver-source.tar.gz','w:gz') as a:
 for p in sorted([*root.glob('beamz/**/*.py'),*root.glob('cuda/src/*')]):
  if p.is_file():a.add(p,arcname=str(p.relative_to(root)))
meta=json.loads((out/'provenance.json').read_text());h=hashlib.sha256()
for p in sorted([*root.glob('beamz/**/*.py'),*root.glob('cuda/src/*')]):
 if p.is_file():h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
assert h.hexdigest()==meta['branch_source_sha256'],'Solver sources changed during final comparison'
