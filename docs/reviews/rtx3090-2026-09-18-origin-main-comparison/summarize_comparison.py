import json,statistics
from pathlib import Path
root=Path.cwd();out=root/'docs/reviews/rtx3090-2026-09-18-origin-main-comparison'
rows=json.loads((out/'summary.json').read_text());assert len(rows)==14
agg={};numeric={};telemetry=[]
for case in ('narrow','wide','irregular'):
 baseline=json.loads((out/f'{case}-main-1.json').read_text())
 for variant in ('main','branch_default','branch_layout'):
  records=[r for r in rows if r['case']==case and r['variant']==variant]
  if not records:continue
  assert len(records)==2
  speeds=[r['median_gcups'] for r in records]
  key=case+'-'+variant
  agg[key]={'mean_process_median_gcups':statistics.mean(speeds),'process_medians_gcups':speeds}
  for r in records:
   tag=f"{case}-{variant}-{r['repeat']}"
   d=json.loads((out/(tag+'.json')).read_text())
   assert all(d[k]==baseline[k] for k in ('workload','initial_state','coefficients','sources','cpml_dtypes')),tag
   states=d['final_state_validation']
   numeric[tag]={'exact':all(x['exact'] for x in states),'failed_leaves':[{k:x[k] for k in ('index','shape','mismatch_count','max_abs_error','reference_scale')} for x in states if not x.get('tolerance_pass',True)],'max_field_error_over_leaf_peak':max(x['max_abs_error']/max(x['reference_scale'],1e-30) for x in states[:6]),'max_cpml_error_over_leaf_peak':max(x['max_abs_error']/max(x['reference_scale'],1e-30) for x in states[6:18]),'max_dft_error_over_leaf_peak':max(x['max_abs_error']/max(x['reference_scale'],1e-30) for x in states[25:27])}
   telemetry.extend(json.loads((out/(tag+'.telemetry.json')).read_text()))
 for variant in ('branch_default','branch_layout'):
  k=case+'-'+variant
  if k in agg:agg[k]['change_vs_main_percent']=100*(agg[k]['mean_process_median_gcups']/agg[case+'-main']['mean_process_median_gcups']-1)
# Layout must preserve branch results exactly in both repeats.
branch_hashes=[]
for variant in ('branch_default','branch_layout'):
 for repeat in (1,2):
  d=json.loads((out/f'narrow-{variant}-{repeat}.json').read_text())
  branch_hashes.append([x['sha256'] for x in d['final_state_validation']])
assert all(h==branch_hashes[0] for h in branch_hashes)
result={'performance':agg,'numerical_comparison':numeric,'identical_inputs_all_runs':True,'narrow_branch_layout_exact':True,'max_temperature_c':max(t['temperature_c'] for t in telemetry),'min_free_mib':min(t['free_mib'] for t in telemetry),'power_limits_w':sorted({t['limit_w'] for t in telemetry})}
(out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
lines=['# Matched origin/main versus local CUDA branch — 2026-09-18','',
'Compared fetched `origin/main` commit `c5fe0d8833ac6539241ae9d3c98445e5a7305786` with the local working tree based on `b0a5ff32`. These are fresh matched measurements, not a comparison against historical documentation.', '',
'Both use RTX3090, CUDA 13.3.73, SM86 Release builds with fast math off, the same Python/JAX environment, FP32 fields and auxiliary state, lossless material, CPML12 on all faces, one solved mode source, two compact mode monitors and three frequencies. All domains contain approximately 16.8 million logical cells. No padding or lossy workloads were used.', '',
'Each fresh process runs 256 steps, four warmups and nine synchronized timed executions. Cases use forward/reverse process order: main/default/layout/layout/default/main for narrow, main/default/default/main otherwise. The reported value is the mean of two process medians. Conversion costs are included; setup, compilation, mode solving, validation and host result hashing are excluded. These are warm compiled-executable rates, not public API or cold-start rates. No confidence interval is inferred from two process repeats.', '',
'| Physical domain (z,y,x) | Material | Main default | Branch default | Change | Branch opt-in layout | Change vs main |',
'|---|---|---:|---:|---:|---:|---:|']
for case,shape,material in [('narrow','1024×256×64','binary'),('wide','128×256×512','binary'),('irregular','257×193×341','smooth')]:
 m=agg[case+'-main'];b=agg[case+'-branch_default'];l=agg.get(case+'-branch_layout')
 lines.append(f"| {shape} | {material} | {m['mean_process_median_gcups']:.3f} | {b['mean_process_median_gcups']:.3f} | {b['change_vs_main_percent']:+.1f}% | "+(f"{l['mean_process_median_gcups']:.3f} | {l['change_vs_main_percent']:+.1f}% |" if l else '— | — |'))
lines+=['','All throughput values are logical GCUPS. The opt-in narrow variant uses `BEAMZ_CUDA_STORAGE_AXES=120` and `BEAMZ_CUDA_CPML_CORE_FUSION=0`. Defaults have all `BEAMZ_CUDA_*` overrides removed. The main checkout and native build are isolated; the branch extension was never replaced.', '',
'## Numerical comparison', '',
'Every run has identical initial-state, compiled-coefficient and compiled-source fingerprints for its physical case. Every final state is finite. All four narrow branch runs (default and cyclic storage) have identical complete-state hashes.', '',
'Branch versus main is not bitwise identical. The existing hardware-test tolerance was evaluated without relaxation: rtol=3e-5 and atol=max(3e-6, 1e-6 times reference leaf peak). Some CPML entries exceed it. This comparison therefore does not establish full numerical equivalence to main. The branch contains earlier changes to floating-point operation ordering; their contribution is plausible, but was not isolated in this study.', '',
'| Case / branch configuration (first repeat) | Max field error / leaf peak | Max CPML error / leaf peak | Entries failing tolerance |',
'|---|---:|---:|---:|']
for case,variant in [('narrow','branch_default'),('narrow','branch_layout'),('wide','branch_default'),('irregular','branch_default')]:
 n=numeric[f'{case}-{variant}-1']
 lines.append(f"| {case} / {variant} | {n['max_field_error_over_leaf_peak']:.3g} | {n['max_cpml_error_over_leaf_peak']:.3g} | {sum(x['mismatch_count'] for x in n['failed_leaves'])} |")
lines+=['','## Provenance and operational limits','',
f"The GPU power limit remained {result['power_limits_w']} W; peak observed temperature was {result['max_temperature_c']:.0f} C and minimum free GPU memory was {result['min_free_mib']:.0f} MiB. GPU processes ran sequentially with a 2-GiB headroom and 85-C stop guard. No solver changes, pushes or default changes were made for this comparison.", '',
'[Raw samples, full-state validation, input fingerprints, compiler flags, telemetry and archived benchmark/source files](rtx3090-2026-09-18-origin-main-comparison/). `analysis.json` records all process medians and every tolerance failure. An initial diagnostic run stopped at the first tolerance failure; its log is retained and its timing is excluded from the table. The completed runs record failures rather than silently relaxing tolerances.', '',
'No >=9-GCUPS guarantee, arbitrary-domain guarantee, or speedup outside these three workloads is established.']
(root/'docs/reviews/cuda-origin-main-comparison-2026-09-18.md').write_text('\n'.join(lines)+'\n')
