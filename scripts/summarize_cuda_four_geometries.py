"""Validate and summarize a matched four-geometry CUDA campaign."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_cosine_cuda_results import archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--area', type=Path, required=True)
    parser.add_argument('--candidate-suffix', default='selected')
    parser.add_argument('--main-commit', required=True)
    args = parser.parse_args()
    area = args.area.resolve()
    rows = []
    for case in ('mie', 'cosine', 'ring', 'mmi'):
        main_dir, candidate_dir = area / (case + '-main'), area / (case + '-' + args.candidate_suffix)
        m = json.loads((main_dir / 'results.json').read_text())
        c = json.loads((candidate_dir / 'results.json').read_text())
        previous_dir = area / (case + '-bounded')
        previous = json.loads((previous_dir / 'results.json').read_text())
        for key in ('grid_shape', 'steps', 'frequencies', 'source_sha256', 'cpml_cells_per_face'):
            assert m[key] == c[key], (case, key)
        assert c['temporal_active'] and not c['fallback_reason']
        assert previous['input_fingerprints'] == c['input_fingerprints'], case
        parity = {name: archive(previous_dir / filename, candidate_dir / filename)
                  for name, filename in [('state', 'state'), ('observables', 'arrays.npz')]}
        assert all(value['all_bitwise'] for value in parity.values()), case
        accuracy = {name: archive(main_dir / filename, candidate_dir / filename)
                    for name, filename in [('state', 'state'), ('observables', 'arrays.npz')]}
        assert all(value['all_finite'] for value in accuracy.values()), case
        observable_error = max(x['relative_l2'] for x in accuracy['observables']['leaves'].values())
        flux_error = max((x['peak_scaled_max_error'] for name, x in accuracy['observables']['leaves'].items()
                          if name.endswith('.flux')), default=0)
        assert observable_error < 5e-5 and flux_error < 1e-6, (case, observable_error, flux_error)
        def metrics(data):
            cold_memory = data['cold_notebook_memory']
            return dict(warm_engine_s=data['runs'][1]['engine_s'], cold_engine_s=data['runs'][0]['engine_s'],
                cold_wall_s=data['total_cold_wall_s'], setup_decode_s=data['cold_public_s']-data['runs'][0]['engine_s'],
                host_gb=max(cold_memory['peak_host_rss_bytes'], cold_memory['host_maxrss_bytes'])/1e9,
                gpu_gb=cold_memory['peak_gpu_process_bytes']/1e9,
                all_runs_host_gb=data['memory']['peak_host_rss_bytes']/1e9,
                all_runs_gpu_gb=data['memory']['peak_gpu_process_bytes']/1e9)
        left, right = metrics(m), metrics(c)
        row = dict(case=case, shape=c['grid_shape'], steps=c['steps'], frequencies=c['frequencies'],
            main=left, candidate=right, speedup=left['warm_engine_s']/right['warm_engine_s'],
            previous_speedup=previous['runs'][1]['engine_s']/right['warm_engine_s'],
            module=c['native_module'], module_sha256=c['native_module_sha256'],
            max_observable_relative_l2=observable_error, max_flux_peak_scaled_error=flux_error,
            all134_inputs_unchanged=True, all30_states_unchanged=True,
            main_directory=str(main_dir), candidate_directory=str(candidate_dir))
        (area / (case+'-selected-comparison.json')).write_text(json.dumps(dict(parity=parity, accuracy=accuracy), indent=2)+'\n')
        rows.append(row)
        print(case, round(row['speedup'], 4), 'host GB', round(left['host_gb'], 3), round(right['host_gb'], 3), flush=True)
    assert len({row['module_sha256'] for row in rows}) == 1
    result = dict(main_commit=args.main_commit, gpu='RTX 3090', power_limit_w=250,
        field_precision='FP32', cpml_cells_per_face=12, warm_repeats=1,
        memory_scope='Peak process RSS/NVML during fresh cold execution, before validation export; decimal GB',
        all134_inputs_unchanged=True, all30_states_unchanged=True, per_geometry_solver_selectors=False, rows=rows)
    (area/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    data = json.dumps(rows)
    html = '''<!doctype html><html><head><meta charset="utf-8"><style>
:root{font-family:var(--font-sans)}body{margin:0;color:var(--foreground)}button{font:inherit;border:1px solid var(--border);color:var(--muted-foreground);background:var(--secondary);border-radius:var(--radius);padding:7px 10px;cursor:pointer}button.active{color:var(--primary-foreground);background:var(--primary)}nav{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 18px}.row{margin:0 0 20px}.line{display:flex;justify-content:space-between;gap:12px;align-items:baseline}.label{font-weight:600}.detail,small{color:var(--muted-foreground);font-size:12px}.numbers{font-variant-numeric:tabular-nums;font-size:13px}.bars{display:grid;gap:4px;margin-top:7px}.bar{height:10px;border-radius:3px;background:var(--chart-1)}.bar.main{background:var(--muted-foreground);opacity:.45}.legend{display:flex;gap:16px;font-size:12px;margin:0 0 15px}.dot{display:inline-block;width:9px;height:9px;background:var(--chart-1);margin-right:5px}.dot.main{background:var(--muted-foreground);opacity:.5}footer{font-size:12px;line-height:1.5;color:var(--muted-foreground);border-top:1px solid var(--border);padding-top:12px}@media(max-width:450px){.line{display:block}.numbers{margin-top:3px}}
</style></head><body><nav id="nav"></nav><div class="legend"><span><i class="dot main"></i>origin/main CUDA</span><span><i class="dot"></i>Shared CUDA candidate</span></div><div id="chart"></div><footer>RTX 3090 · 250 W · FP32 · 12 CPML cells per face. One warm repeat per fresh process. Cold total includes preparation, simulation and decoding. Memory peaks precede validation export. Same candidate module for every geometry; regular grids and fixed horizons.</footer><script>
const rows=ROWS;
const names={mie:'Mie cube',cosine:'Cosine crossing',ring:'Ring resonator',mmi:'1×4 MMI'};
const metrics=[['warm_engine_s','Core solver','s'],['cold_wall_s','Cold total','s'],['setup_decode_s','Setup + decoding','s'],['host_gb','Host peak','GB'],['gpu_gb','GPU peak','GB']];
function show(key){const metric=metrics.find(x=>x[0]===key);document.querySelectorAll('button').forEach(b=>b.classList.toggle('active',b.dataset.key===key));document.getElementById('chart').innerHTML=rows.map(r=>{let a=r.main[key],b=r.candidate[key],scale=Math.max(a,b),gain=a/b;return `<div class="row"><div class="line"><span class="label">${names[r.case]}</span><span class="numbers">${a.toFixed(2)} → ${b.toFixed(2)} ${metric[2]} · <b>${gain.toFixed(2)}×</b></span></div><span class="detail">${r.shape.join(' × ')} cells · ${r.steps.toLocaleString()} steps</span><div class="bars"><div class="bar main" style="width:${100*a/scale}%"></div><div class="bar" style="width:${100*b/scale}%"></div></div></div>`}).join('')}
const nav=document.getElementById('nav');metrics.forEach(m=>{const b=document.createElement('button');b.textContent=m[1];b.dataset.key=m[0];b.onclick=()=>show(m[0]);nav.appendChild(b)});show('warm_engine_s');
</script></body></html>'''.replace('ROWS', data)
    (area/'report.html').write_text(html)


if __name__ == '__main__':
    main()
