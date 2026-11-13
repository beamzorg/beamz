#!/usr/bin/env python3
"""Render recorded CUDA nanoparticle results as a standalone HTML chart.

Only successful measured cases are plotted. No extrapolation or cross-engine
speed claims are inferred. The HTML works offline without a server or assets.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def normalize_report(report):
    if report.get("schema") == "beamz.cuda-mie-final-comparison.v1":
        report = {
            **report,
            "cases": [r["case"] for r in report["results"]],
            "series": [
                {"id": "streamed", "label": "CUDA streamed"},
                {"id": "control_mstep", "label": "Mstep control"},
                {"id": "candidate_mstep", "label": "Optimized mstep"},
            ],
            "results": [{**r, "series": r["curve"]} for r in report["results"]],
        }
    return report


def render(report):
    report = normalize_report(report)
    # Keep the inspectable provenance in the companion JSON; the interactive
    # page needs only measured plotting fields, not full worker metadata/logs.
    chart = {"series": report.get("series"), "results": []}
    metrics = (
        "median_warm_executable_s",
        "median_warm_public_s",
        "scene_to_first_public_result_s",
    )
    for row in report.get("results", ()):
        chart["results"].append(
            {
                "status": row["status"],
                "schedule": row["schedule"],
                "series": row.get("series"),
                "case": {"resolution_nm": row["case"]["resolution_nm"]},
                "schedule_decision": {
                    "core_fraction": (row.get("schedule_decision") or {}).get(
                        "core_fraction"
                    )
                },
                **{key: row.get(key) for key in metrics},
            }
        )
    data = json.dumps(chart).replace("<", "\\u003c")
    cases = report.get("cases", ())
    case = cases[0] if cases else {}
    boundary = (
        f"{case.get('pml_cells', '?')} CPML cells per face at every resolution "
        "(physical thickness varies with grid spacing)"
        if case.get("pml_policy") == "fixed_cells"
        else f"{case.get('realized_pml_nm', '?')} nm CPML per face"
    )
    description = (
        f"{case.get('cube_nm_including_pml', '?')} nm cube including {boundary}; "
        f"{case.get('radius_nm', '?')} nm radius n={case.get('particle_index', '?')} sphere; "
        f"{case.get('wavelength_nm', '?')} nm finite-aperture illumination; "
        f"{case.get('monitor_faces', '?')} DFT faces; "
        f"{case.get('actual_duration_fs', 0):.2f} fs physical duration "
        f"({'fixed-step control' if case.get('duration_policy') == 'fixed_steps' else 'fixed physical-time sweep'}). "
        "Explicit workload assumptions, not an exact FDTDX reproduction or an analytic Mie accuracy result. "
        "No points are inferred for unmeasured resolutions. See JSON for physical duration, "
        "per-case PML rounding, hardware, source hashes, samples and validation."
    )
    context = ""
    if report.get("schema") == "beamz.cuda-mie-final-comparison.v1":
        first = report["results"][0]
        telemetry = str(first.get("gpu_before") or "").splitlines()
        device = telemetry[-1].split(",")[0] if len(telemetry) > 1 else "CUDA device"
        device = device.replace("NVIDIA GeForce ", "")
        samples = len(first.get("warm_executable_samples_s", ()))
        context = (
            "<p>"
            + html.escape(
                f"{device} · {report['power_limit_w']} W · FP32. One fresh process per point; "
                f"{samples} solve samples, no confidence interval."
            )
            + "</p>"
        )
    return (
        """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{font-family:var(--font-sans,system-ui);color:var(--foreground,#ddd)}
body{margin:0}p{color:var(--muted-foreground,#999);font-size:13px;line-height:1.5}
label,table{font-size:13px}select{font:inherit;color:inherit;background:var(--card,#222);border:1px solid var(--border,#555);padding:6px;border-radius:5px}
svg{width:100%;height:310px;overflow:visible}svg text{fill:var(--muted-foreground,#999);font-size:12px}
table{border-collapse:collapse;width:100%;margin-top:10px}th,td{text-align:left;padding:7px 5px;border-bottom:1px solid var(--border,#444)}
.streamed{color:var(--chart-1,#65a9ff)}.mstep{color:var(--chart-2,#f2ad58)}
</style></head><body>
__CONTEXT__
<label>Runtime metric <select id="metric"><option value="median_warm_executable_s">Warm executable</option><option value="median_warm_public_s">Warm public call</option><option value="scene_to_first_public_result_s">Scene to first public result</option></select></label>
<svg id="chart" viewBox="0 0 700 310" role="img" aria-label="Measured CUDA runtimes versus grid resolution"></svg>
<p><span id="legend"></span> · Both axes logarithmic. Lower is faster. Lines connect measurements only.</p>
<table><thead><tr id="columns"></tr></thead><tbody id="rows"></tbody></table>
<p>__DESCRIPTION__</p>
<script>
const report=__DATA__;
const records=report.results.filter(r=>r.status==='ok');
const groups=(report.series||[{id:'streamed',label:'Streamed'},{id:'mstep',label:'Mstep'}]).map((g,i)=>({...g,color:`var(--chart-${i+1},${['#65a9ff','#f2ad58','#62c58d'][i%3]})`}));
const seriesKey=r=>r.series||r.schedule;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
document.getElementById('legend').innerHTML=groups.map(g=>`<span style="color:${g.color}">● ${esc(g.label)}</span>`).join(' &nbsp; ');
document.getElementById('columns').innerHTML='<th>Resolution</th>'+groups.map(g=>`<th>${esc(g.label)}</th>`).join('')+'<th>Core eligible</th>';
function draw(){
 const metric=document.getElementById('metric').value;
 const rows=records.filter(r=>Number.isFinite(r[metric])&&r[metric]>0);
 const values=rows.map(r=>r[metric]); if(!values.length){document.getElementById('chart').innerHTML='<text x="65" y="50">This metric was not measured.</text>';document.getElementById('rows').innerHTML='';return;}
 const resolutions=[...new Set(records.map(r=>r.case.resolution_nm))].sort((a,b)=>b-a);
 const maxR=Math.max(...resolutions),minR=Math.min(...resolutions);
 const hi=maxR===minR?maxR*1.2:maxR,lo=maxR===minR?minR/1.2:minR;
 const ymin=Math.floor(Math.log10(Math.min(...values))),ymax=Math.max(ymin+1,Math.ceil(Math.log10(Math.max(...values))));
 const x=r=>65+600*Math.log(hi/r)/Math.log(hi/lo),y=t=>265-235*(Math.log10(t)-ymin)/(ymax-ymin);
 let svg='';
 for(let p=ymin;p<=ymax;p++){const v=10**p;svg+=`<line x1="65" x2="665" y1="${y(v)}" y2="${y(v)}" stroke="var(--border,#444)"/><text x="56" y="${y(v)+4}" text-anchor="end">${v>=1?v.toFixed(0):v.toPrecision(1)} s</text>`;}
 for(const r of [...new Set([5,4,2.5,2,1,...resolutions])].filter(r=>r<=hi&&r>=lo)){svg+=`<line x1="${x(r)}" x2="${x(r)}" y1="30" y2="265" stroke="var(--border,#444)"/><text x="${x(r)}" y="285" text-anchor="middle">${r}</text>`;}
 for(const group of groups){
  const points=rows.filter(r=>seriesKey(r)===group.id).sort((a,b)=>b.case.resolution_nm-a.case.resolution_nm);
  svg+=`<polyline fill="none" stroke="${group.color}" stroke-width="3" points="${points.map(r=>x(r.case.resolution_nm)+','+y(r[metric])).join(' ')}"/>`;
  for(const r of points){const px=x(r.case.resolution_nm),py=y(r[metric]);svg+=`<circle cx="${px}" cy="${py}" r="5" fill="${group.color}"><title>${esc(group.label)} ${r.case.resolution_nm}nm: ${r[metric].toFixed(4)}s</title></circle>`;}
 }
 svg+='<text x="365" y="307" text-anchor="middle">Resolution (nm)</text>';
 document.getElementById('chart').innerHTML=svg;
 document.getElementById('rows').innerHTML=resolutions.map(dx=>{
  const values=groups.map(g=>records.find(r=>seriesKey(r)===g.id&&r.case.resolution_nm===dx));
  const b=records.find(r=>r.schedule==='mstep'&&r.case.resolution_nm===dx);
  const time=r=>r&&Number.isFinite(r[metric])?r[metric].toFixed(3)+' s':'—';
  return `<tr><td>${dx} nm</td>${values.map(r=>`<td>${time(r)}</td>`).join('')}<td>${b&&Number.isFinite(b.schedule_decision?.core_fraction)?(100*b.schedule_decision.core_fraction).toFixed(1)+'%':'—'}</td></tr>`;
 }).join('');
}
document.getElementById('metric').addEventListener('change',draw);draw();
</script></body></html>""".replace("__DATA__", data)
        .replace("__DESCRIPTION__", html.escape(description))
        .replace("__CONTEXT__", context)
    )


def export_png(report, path):
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullLocator

    report = normalize_report(report)
    groups = report.get(
        "series",
        [{"id": "streamed", "label": "Streamed"}, {"id": "mstep", "label": "Mstep"}],
    )
    colors = ("#4d91db", "#e5a142", "#319668")
    figure, axis = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    resolutions = set()
    measured_times = []
    for index, group in enumerate(groups):
        rows = sorted(
            (
                r
                for r in report["results"]
                if r["status"] == "ok" and r.get("series", r["schedule"]) == group["id"]
            ),
            key=lambda r: r["case"]["resolution_nm"],
            reverse=True,
        )
        x = [r["case"]["resolution_nm"] for r in rows]
        y = [r["median_warm_executable_s"] for r in rows]
        measured_times.extend(y)
        resolutions.update(x)
        axis.plot(x, y, "o-", color=colors[index % len(colors)], label=group["label"])
    ticks = sorted(resolutions, reverse=True)
    if not measured_times or min(measured_times) <= 0:
        raise ValueError("PNG export requires positive measured solve times")
    lower = math.floor(math.log10(min(measured_times)))
    upper = max(lower + 1, math.ceil(math.log10(max(measured_times))))
    axis.set(
        xscale="log",
        yscale="log",
        xlabel="Resolution (nm)",
        ylabel="Synchronized solve time (s)",
    )
    axis.set_xticks(ticks, [f"{value:g}" for value in ticks])
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_yticks([m * 10**p for p in range(lower, upper + 1) for m in (1, 2, 5)])
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_ylim(10**lower, 10**upper)
    axis.invert_xaxis()
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    first = next(r for r in report["results"] if r["status"] == "ok")
    case = first["case"]
    boundary = (
        f"{case['pml_cells']} CPML cells/face"
        if case.get("pml_policy") == "fixed_cells"
        else f"{case['realized_pml_nm']:g} nm CPML/face"
    )
    duration = (
        f"{case['timesteps']} steps"
        if case.get("duration_policy") == "fixed_steps"
        else f"{case['actual_duration_fs']:.2f} fs"
    )
    figure.suptitle(
        f"BeamZ · {case['cube_nm_including_pml']:g} nm cube · {boundary} · {duration}"
    )
    telemetry = str(first.get("gpu_before") or "").splitlines()
    device = telemetry[-1].split(",")[0] if len(telemetry) > 1 else "CUDA device"
    device = device.replace("NVIDIA GeForce ", "")
    power = f" · {report['power_limit_w']} W" if "power_limit_w" in report else ""
    samples = len(first.get("warm_executable_samples_s", ()))
    sampling = (
        f"one process/point, median of {samples} solves; no CI"
        if samples
        else "recorded solve medians"
    )
    figure.supxlabel(
        f"{device}{power} · FP32 · {sampling}",
        fontsize=9,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    if (
        report.get("workload") != "cuda_mie_like_nanoparticle"
        and report.get("schema") != "beamz.cuda-mie-final-comparison.v1"
    ):
        parser.error("Expected a recorded CUDA nanoparticle benchmark")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(report))
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        export_png(report, args.png)
    print(html.escape(str(args.output)))


if __name__ == "__main__":
    main()
