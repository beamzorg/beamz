#!/usr/bin/env python3
"""Render three source-backed HTML panels from the frozen CUDA study summaries.

Only the Python standard library is required. No timing is performed. The HTML
contains reviewed rows, source paths and SHA256 hashes for independent inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import statistics
from pathlib import Path

DEFAULT_ROOT = Path(".cache/cuda-native-autoresearch-2026-11-11-1607")


def read_source(root, relative):
    path = root / relative
    raw = path.read_bytes()
    return json.loads(raw), {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def prepare_rows(shared, tuning, deep):
    """Keep each experiment's timing/ownership scope separate."""
    same_core = []
    assert shared["all_30_exact"]
    for backend in ("jax", "native"):
        values = shared["process_medians_s"][backend]
        same_core.append(
            {
                "case": "mie272-full",
                "backend": backend,
                "milliseconds": statistics.median(values) * 1000,
                "process_medians_s": values,
            }
        )
    best = []
    for resolution, case in tuning:
        assert case["all_30_exact"]
        small = resolution == 5
        speedup = case["best_speedup_ratio" if small else "segment64_speedup"]
        best.append(
            {
                "resolution_nm": resolution,
                "shape_zyx": [136 if small else 272] * 3,
                "tile": "8x16" if small else "16x32",
                "segment": "64",
                "speedup": speedup,
                "throughput_gain_percent": (speedup - 1) * 100,
                "process_medians_s": case["process_medians_s"],
            }
        )
    assert deep["all_30_exact"]
    depth_four = [
        {
            "variant": variant,
            "seconds": statistics.median(deep["process_medians_s"][variant]),
            "process_medians_s": deep["process_medians_s"][variant],
        }
        for variant in ("v6", "v8", "v10")
    ]
    assert len(same_core) == 2 and len(best) == 2 and len(depth_four) == 3
    return {"same_core": same_core, "best_tuning": best, "depth_four": depth_four}


def esc(value):
    return html.escape(str(value), quote=True)


def bar(label, value, maximum, display, style="solid"):
    if not 0 <= value <= maximum:
        raise ValueError("Chart value outside its explicit zero-based scale")
    return (
        f'<div class="bar-row"><span class="row-label">{esc(label)}</span>'
        '<span class="track" aria-hidden="true">'
        f'<span class="mark {style}" style="width:{value / maximum * 100:.5f}%"></span>'
        f'</span><span class="value">{esc(display)}</span></div>'
    )


def ruler(labels):
    return (
        '<div class="bar-row ruler"><span></span><span class="tick-row">'
        + "".join(f"<span>{esc(label)}</span>" for label in labels)
        + "</span><span></span></div>"
    )


def render(rows, sources):
    core_accessible = "; ".join(
        f"{row['case']} {row['backend']}: {row['milliseconds']:.1f} milliseconds"
        for row in rows["same_core"]
    )
    tuning_accessible = "; ".join(
        f"{row['resolution_nm']:g} nanometers: {row['throughput_gain_percent']:.1f} percent throughput gain"
        for row in rows["best_tuning"]
    )
    depth_accessible = "; ".join(
        f"{row['variant']}: {row['seconds']:.3f} seconds" for row in rows["depth_four"]
    )
    core = ""
    labels = {"mie272-full": "272³ · 6,560 updates · 30 fs"}
    for case in ("mie272-full",):
        core += f'<div class="group"><p class="group-title">{esc(labels[case])}</p>'
        for row in (x for x in rows["same_core"] if x["case"] == case):
            core += bar(
                "JAX FFI" if row["backend"] == "jax" else "Native",
                row["milliseconds"],
                20000,
                f"{row['milliseconds'] / 1000:.3f} s",
                "outline" if row["backend"] == "jax" else "solid",
            )
        core += "</div>"
    core += ruler(["0", "5", "10", "15", "20 s"])
    tuning = ""
    for row in rows["best_tuning"]:
        title = f"{row['resolution_nm']:g} nm · {row['shape_zyx'][0]}³"
        tuning += f'<div class="group"><p class="group-title">{esc(title)}</p>'
        tuning += bar(
            f"{row['tile'].replace('x', '×')} / {row['segment']}",
            row["throughput_gain_percent"],
            20,
            f"+{row['throughput_gain_percent']:.1f}%",
            "solid",
        )
        tuning += f'<p class="setting">{row["speedup"]:.3f}× throughput versus 16×32 whole column</p></div>'
    tuning += ruler(["0", "5", "10", "15", "20%"])
    names = {"v6": "v6 · generic", "v8": "v8 · typed", "v10": "v10 · 2 frames"}
    depth = '<div class="depth-bars">'
    for row in rows["depth_four"]:
        depth += bar(
            names[row["variant"]], row["seconds"], 3, f"{row['seconds']:.3f} s"
        )
    depth += "</div>" + ruler(["0", "1", "2", "3 s"])
    provenance = json.dumps({"sources": sources, "reviewed_rows": rows}, indent=2)
    evidence = provenance.replace("</", "<\\/")
    source_list = "".join(
        f'<li><code>{esc(source["path"])}</code><br><code class="hash">SHA256 {esc(source["sha256"])}</code></li>'
        for source in sources
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BeamZ CUDA study · measured execution and scheduling</title>
<style>
:root{{--study-ink:var(--foreground,#ddd);--study-muted:var(--muted-foreground,#aaa);--study-rule:var(--border,#555);--study-mark:var(--chart-1,#4d91db);}}
*{{box-sizing:border-box}}body{{margin:0;background:transparent;color:var(--study-ink);font:14px/1.45 var(--font-sans,system-ui),sans-serif}}main{{max-width:1240px;margin:0;padding:0}}h1{{font-size:25px;line-height:1.2;margin:0 0 8px;font-weight:650}}.intro{{margin:0 0 24px;color:var(--study-muted)}}.panels{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:26px}}section{{min-width:0;border-top:1px solid var(--study-rule);padding-top:16px}}section:first-child .plot{{height:140px}}h2{{font-size:18px;line-height:1.3;margin:0 0 6px;font-weight:650}}.subtitle{{margin:0;color:var(--study-muted);min-height:43px}}.plot{{height:190px;display:flex;flex-direction:column;justify-content:space-between;padding-top:18px;padding-bottom:3px}}.group-title{{margin:0 0 9px;font-size:13px;font-weight:600}}.bar-row{{display:grid;grid-template-columns:92px minmax(0,1fr)67px;gap:9px;align-items:center;min-height:29px}}.row-label{{font-size:12px;line-height:1.2}}.track{{display:block;height:18px;border-left:1px solid var(--study-rule);position:relative;min-width:0}}.mark{{display:block;height:14px;position:absolute;left:0;top:2px;background:var(--study-mark);border:1.5px solid var(--study-mark)}}.mark.outline{{background:transparent}}.value{{font-size:12px;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}.ruler{{margin-top:8px;min-height:23px;color:var(--study-muted);align-items:start}}.tick-row{{display:flex;justify-content:space-between;border-top:1px solid var(--study-rule);padding-top:5px;font-size:10px;white-space:nowrap}}.setting{{margin:8px 0 0;color:var(--study-muted);font-size:11px}}.depth-bars{{display:grid;gap:28px;margin-top:10px}}.note{{font-size:12px;line-height:1.5;color:var(--study-muted);margin:16px 0 0}}.note strong{{color:var(--study-ink)}}details{{margin-top:28px;border-top:1px solid var(--study-rule);padding-top:14px}}summary{{cursor:pointer;font-weight:600}}ul{{padding-left:18px}}li{{margin:12px 0}}code{{font:11px/1.4 ui-monospace,monospace;overflow-wrap:anywhere}}.hash{{color:var(--study-muted)}}pre{{font:11px/1.5 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;margin-top:18px}}.footer{{font-size:12px;color:var(--study-muted);margin:24px 0 0}}
@media(max-width:980px){{.panels{{grid-template-columns:1fr;gap:28px}}.subtitle{{min-height:0}}.plot{{height:190px}}.bar-row{{grid-template-columns:112px minmax(0,1fr)76px}}.row-label,.value{{font-size:13px}}.setting{{font-size:12px}}.tick-row{{font-size:11px}}.note{{max-width:780px}}}}
@media(max-width:480px){{main{{padding:0}}h1{{font-size:23px}}.bar-row{{grid-template-columns:93px minmax(0,1fr)69px;gap:7px}}.row-label,.value{{font-size:12px}}}}
</style></head><body><main>
<header><p class="intro">RTX 3090 · 250 W · FP32 · CPML12 · complete source and DFT work</p></header>
<div class="panels">
<section aria-labelledby="native-title"><h2 id="native-title">Full-run native / JAX latency</h2><p class="subtitle">Same compiled CUDA core · lower is better</p><div class="plot" role="img" aria-label="{esc(core_accessible)}">{core}</div><p class="note">Same compiled CUDA core, input state and schedule. Median of <strong>two fresh-process medians</strong> per backend; two samples after one warmup. Preserving copies and all 26 chunks are included. Process ranges are inspectable below; no confidence interval is inferred.</p></section>
<section aria-labelledby="tuning-title"><h2 id="tuning-title">Full-run D2 tuning</h2><p class="subtitle">Throughput gain over whole column · higher is better</p><div class="plot" role="img" aria-label="{esc(tuning_accessible)}">{tuning}</div><p class="note"><strong>Complete 30 fs scattering pulse.</strong> Same corrected FFI core, two fresh processes per setting in balanced order. One warmup and two samples each. The short 256-step gain at 2.5 nm did not survive the full pulse.</p></section>
<section aria-labelledby="depth-title"><h2 id="depth-title">D4 tracing variants</h2><p class="subtitle">272³ · 256 updates · lower is better</p><div class="plot" role="img" aria-label="{esc(depth_accessible)}">{depth}</div><p class="note">Two fresh processes per variant, symmetric order; two samples after one warmup. <strong>All remain slower than D2</strong> in separate same-binary screens. D2 timing is intentionally omitted here to preserve comparison scope.</p></section>
</div>
<p class="footer">Every run shown passes exact finite array equality on all 30 state leaves. Preparation, compilation and downloads are outside warm timing. The panels describe separate experiments; compare values within each panel.</p>
<details><summary>Evidence, exact process medians and source hashes</summary><ul>{source_list}</ul><pre>{esc(provenance)}</pre></details>
<script type="application/json" id="cuda-study-evidence">{evidence}</script>
</main></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run_root / "report/cuda-native-study.html"
    shared, source_a = read_source(
        args.run_root, "full30fs-large-corrected/summary.json"
    )
    small, source_b = read_source(
        args.run_root, "full30fs-5nm-tuning-corrected/summary.json"
    )
    large, source_d = read_source(
        args.run_root, "full30fs-tuning-corrected/summary.json"
    )
    deep, source_c = read_source(args.run_root, "deep-balanced-results/summary.json")
    rows = prepare_rows(shared, [(5, small), (2.5, large)], deep)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(rows, [source_a, source_b, source_d, source_c]))
    print(output)


if __name__ == "__main__":
    main()
