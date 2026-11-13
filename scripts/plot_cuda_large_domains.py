#!/usr/bin/env python3
"""Build a source-backed local HTML chart from explicit larger-domain runs.

Inputs are raw field-probe JSON stdout files from balanced fresh processes.
Complete-runtime results remain a separate panel: a field-only kernel speedup
must never become a claim about the complete CPML/source/DFT engine.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path


def source(path):
    raw = path.read_bytes()
    return json.loads(raw), {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def field_rows(paths, control="spatial"):
    groups = defaultdict(list)
    sources = []
    geometry = None
    for path in paths:
        row, provenance = source(path)
        assert row["mismatches"] == row["bad_trail_reads"] == 0
        assert row["repeat_mismatches"] == 0
        assert all(row[k] for k in (
            "repeat_checked", "all_guards_checked", "trail_guards_checked",
            "tiled_control", "streamed_readback",
        ))
        current = tuple(row["tile_zyx"]), row["steps"]
        if geometry is None:
            geometry = current
        if current != geometry:
            raise ValueError("Use one explicit tile/depth for each chart curve")
        key = "matched_d2_ms" if control == "d2" else "tiled_ms"
        if not row.get(key):
            raise ValueError(f"Missing validated timing samples for {control}: {path}")
        tiled = statistics.median(row[key])
        moving = statistics.median(row["moving_ms"])
        if min(tiled, moving) <= 0:
            raise ValueError("Timings must be positive")
        groups[row["size"]].append((tiled, moving, tiled / moving))
        sources.append(provenance)
    result = []
    for n, samples in sorted(groups.items()):
        ratios = [x[2] for x in samples]
        result.append({
            "size": n, "million_yee_cells": n**3 / 1e6,
            "tile_zyx": geometry[0], "steps": geometry[1],
            "fresh_processes": len(samples),
            "control": control,
            "control_ms": statistics.median(x[0] for x in samples),
            "temporal_ms": statistics.median(x[1] for x in samples),
            "paired_speedup": statistics.median(ratios),
            "paired_speedup_min": min(ratios),
            "paired_speedup_max": max(ratios),
            "paired_process_speedups": ratios,
        })
    return result, sources


def complete_rows(root):
    rows, sources = [], []
    for path in sorted(root.glob("n*-steps*/summary.json")):
        case, provenance = source(path)
        results = {x["variant"]: x for x in case["results"]}
        assert all(x["all30_exact"] and x["all30_bitwise"]
                   for x in results.values())
        rows.append({
            "size": case["shape_zyx"][0], "steps": case["steps"],
            "ffi_s": results["ffi-d2"]["median_wall_s"],
            "native_s": results["native-d2"]["median_wall_s"],
            "generic_d2_s": results["generic-d2"]["median_wall_s"],
            "generic_d4_s": results["generic-d4"]["median_wall_s"],
        })
        sources.append(provenance)
    return rows, sources


def render(fields, complete, sources, label, tuning=None):
    esc = lambda value: html.escape(str(value), quote=True)  # noqa: E731
    control_name = "Rolling D2" if fields[0]["control"] == "d2" else "Spatial"
    engine_panel = ""
    if tuning:
        tuning_max = max(r["median_s"] for r in tuning["rows"]) * 1.05
        names = {"default": "16×32 · whole column", "8x32-seg64": "8×32 · segment 64",
                 "default-seg64": "16×32 · segment 64"}
        bars = "".join(
            f'<div class="timing-row"><span>{names[r["variant"]]}</span>'
            '<span class="track" aria-hidden="true">'
            f'<span class="timing-bar" style="width:{r["median_s"]/tuning_max*100:.3f}%"></span>'
            f'</span><span>{r["median_s"]:.3f} s</span></div>'
            for r in tuning["rows"]
        )
        best = max(r["median_paired_speedup"] for r in tuning["rows"])
        engine_panel = (
            '<section><h2>Complete 512³ simulation: '
            f'{(best-1)*100:.1f}% more throughput</h2>'
            '<p>Same FP32 packet, twelve CPML cells, mode source, monitors and 512 updates. '
            'Existing D2 launch settings; lower time is better.</p>'
            f'{bars}<p class="muted">Three fresh processes per setting, rotated order, '
            'two timed samples each. All 30 outputs match byte for byte. '
            'This gain comes from column segmentation; the deeper cache remains experimental.</p></section>'
        )
    maximum = max(2, *(r["paired_speedup_max"] for r in fields)) * 1.08
    left, right, top, bottom = 47, 606, 18, 205
    width = right - left
    first, last = fields[0]["size"], fields[-1]["size"]
    x = lambda i: left + width * (fields[i]["size"] - first) / max(1, last - first)  # noqa: E731
    y = lambda v: bottom - (bottom - top) * v / maximum  # noqa: E731
    svg = []
    for tick in range(int(maximum) + 1):
        svg.append(f'<line x1="{left}" x2="{right}" y1="{y(tick):.2f}" '
                   f'y2="{y(tick):.2f}" class="grid"/>'
                   f'<text x="38" y="{y(tick)+4:.2f}" text-anchor="end">{tick}×</text>')
    svg.append(f'<line x1="{left}" x2="{right}" y1="{y(1):.2f}" '
               f'y2="{y(1):.2f}" class="reference"/>')
    points = " ".join(f"{x(i):.2f},{y(r['paired_speedup']):.2f}"
                      for i, r in enumerate(fields))
    svg.append(f'<polyline points="{points}" class="curve"/>')
    for i, row in enumerate(fields):
        px, py = x(i), y(row["paired_speedup"])
        anchor = "start" if i == 0 else "end" if i == len(fields)-1 else "middle"
        label_x = px + 8 if i == 0 else px - 8 if i == len(fields)-1 else px
        tooltip = (f"{row['size']}³: {row['paired_speedup']:.3f}×; "
                   f"control {row['control_ms']:.2f}ms, temporal {row['temporal_ms']:.2f}ms; "
                   f"{row['fresh_processes']} fresh processes")
        svg.append(
            f'<g><title>{esc(tooltip)}</title>'
            f'<line x1="{px:.2f}" x2="{px:.2f}" '
            f'y1="{y(row["paired_speedup_min"]):.2f}" '
            f'y2="{y(row["paired_speedup_max"]):.2f}" class="range"/>'
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4"/>'
            f'<text x="{label_x:.2f}" y="{py-12:.2f}" text-anchor="{anchor}" '
            f'class="point-label">{row["paired_speedup"]:.2f}×</text>'
            f'<text x="{px:.2f}" y="226" text-anchor="middle">{row["size"]}³</text>'
            f'<text x="{px:.2f}" y="243" text-anchor="middle" class="sub">'
            f'{row["million_yee_cells"]:.1f}M cells</text></g>'
        )
    table = "".join(
        f'<tr><th>{r["size"]}³</th><td>{r["control_ms"]:.1f} ms</td>'
        f'<td>{r["temporal_ms"]:.1f} ms</td><td>{r["paired_speedup"]:.2f}×</td></tr>'
        for r in fields
    )
    whole = "".join(
        f'<tr><th>{r["size"]}³</th><td>{r["ffi_s"]:.3f} s</td>'
        f'<td>{r["native_s"]:.3f} s</td>'
        f'<td>{r["generic_d4_s"]/r["generic_d2_s"]:.2f}× slower</td></tr>'
        for r in complete
    )
    provenance = json.dumps({"field_rows": fields, "complete_rows": complete, "engine_tuning": tuning,
                             "sources": sources}, indent=2).replace("<", "\\u003c")
    return f'''<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{{font-family:var(--font-sans,system-ui);color:var(--foreground,#e5e7eb);margin:0}}
h2{{font-size:17px;margin:0 0 8px}}p{{font-size:13px;line-height:1.5;margin:6px 0 12px}}
.muted,.sub{{color:var(--muted-foreground,#98a2b3)}}
section{{margin:0 0 24px}}svg{{width:100%;height:265px;overflow:visible}}
svg text{{fill:var(--muted-foreground,#98a2b3);font-size:12px}}
svg .sub{{font-size:11px}}svg .point-label{{fill:var(--foreground,#e5e7eb);font-weight:600}}
.grid{{stroke:var(--border,#303844);stroke-width:1}}
.reference{{stroke:var(--muted-foreground,#98a2b3);stroke-dasharray:4 4}}
.curve{{stroke:var(--chart-2,#47c4a2);stroke-width:2.5;fill:none}}
.range{{stroke:var(--chart-2,#47c4a2);stroke-width:5;opacity:.5}}
circle{{fill:var(--chart-2,#47c4a2)}}
table{{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}}
th,td{{padding:7px 5px;border-bottom:1px solid var(--border,#303844);text-align:right}}
th:first-child{{text-align:left}}thead th{{font-weight:500;color:var(--muted-foreground,#98a2b3)}}
details{{font-size:12px}}pre{{max-height:280px;overflow:auto;white-space:pre-wrap;font-size:11px}}
.timing-row{{display:grid;grid-template-columns:minmax(110px,180px) 1fr 62px;gap:10px;align-items:center;
font-size:12px;font-variant-numeric:tabular-nums;margin:12px 0}}
.timing-row>span:last-child{{text-align:right}}.track{{height:12px;background:var(--secondary,#252a31)}}
.timing-bar{{display:block;height:12px;background:var(--chart-1,#75a8ef)}}
</style></head><body>
{engine_panel}
<section><h2>Large-domain temporal blocking</h2>
<p>{esc(label)} · local RTX 3090 · FP32 · {fields[0]['steps']} full Yee updates.
Field-only kernel experiment; excludes CPML, sources, materials and DFT monitors.</p>
<svg viewBox="0 0 650 255" role="img" aria-label="Paired speedup versus the selected matched CUDA control">
{''.join(svg)}</svg>
<p class="muted">Speedup versus {esc(control_name)} CUDA. Points: median paired process speedup;
vertical ranges: observed process minimum–maximum, not confidence intervals.</p>
<table><thead><tr><th>Domain</th><th>{esc(control_name)}</th><th>Temporal</th><th>Speedup</th></tr></thead>
<tbody>{table}</tbody></table></section>
<section><h2>Complete engine control, measured separately</h2>
<p>Binary guide · CPML 12 · mode source · two monitors × three frequencies · 512 updates.
The existing overlapping D4 path is distinct from the new circular-cache kernel.</p>
<table><thead><tr><th>Domain</th><th>JAX → CUDA D2</th><th>Native CUDA D2</th>
<th>Existing D4 / D2</th></tr></thead><tbody>{whole}</tbody></table>
<p class="muted">All 30 output leaves match the reference byte for byte.
Initial screens: one fresh process per variant. Setup and compilation excluded.</p></section>
<details><summary>Inspectable rows and source hashes</summary><pre>{esc(provenance)}</pre></details>
<script type="application/json" id="benchmark-data">{provenance}</script>
</body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=Path, nargs="+", required=True)
    parser.add_argument("--field-label", default="Circular physical cache, owned 7×7×15")
    parser.add_argument("--control", choices=("spatial", "d2"), default="d2")
    parser.add_argument("--complete-root", type=Path, required=True)
    parser.add_argument("--engine-confirmation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fields, sources = field_rows(args.field, args.control)
    complete, more_sources = complete_rows(args.complete_root)
    tuning = None
    if args.engine_confirmation:
        tuning, provenance = source(args.engine_confirmation)
        assert tuning["all30_bitwise_all_runs"]
        assert all(r["fresh_processes"] == 3 for r in tuning["rows"])
        more_sources.append(provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(fields, complete, sources + more_sources,
                                  args.field_label, tuning))
    args.output.with_suffix(".json").write_text(json.dumps({
        "field_rows": fields, "complete_rows": complete, "engine_tuning": tuning,
        "sources": sources + more_sources,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
