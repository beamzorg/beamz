#!/usr/bin/env python3
"""Render two validated same-device FDTD campaigns into inspectable artifacts."""

from __future__ import annotations

import argparse
import base64
import html
import json
import statistics
from pathlib import Path


def campaign(path, *, completed_only=False):
    result = json.loads(path.read_text())
    rows = []
    for case in result["cases"]:
        if case["status"] != "passed":
            if completed_only:
                continue
            raise ValueError("Refusing to chart an incomplete/failed case")
        canonical = case["validations"]["beamz"]["result"]["actual_work"]["canonical"]
        row = {
            "resolution_nm": canonical["dx_m"] * 1e9,
            "shape": canonical["shape_zyx"],
            "steps": canonical["steps"],
        }
        for engine in ("beamz", "fdtdx"):
            trials = [t["result"] for t in case["trials"] if t["engine"] == engine]
            row[engine] = {
                "warm_solve_s": statistics.geometric_mean(
                    statistics.median(t["timing"]["warm_solve_samples_s"])
                    for t in trials
                ),
                "process_medians_s": [
                    statistics.median(t["timing"]["warm_solve_samples_s"])
                    for t in trials
                ],
                "public_s": statistics.geometric_mean(
                    statistics.median(t["timing"]["warm_public_samples_s"])
                    for t in trials
                ),
                "adapter_cold_s": statistics.geometric_mean(
                    t["timing"]["cold_scene_to_first_s"] for t in trials
                ),
                "device_global_peak_gib": max(
                    max(t["memory"]["device_global"]["peak_used_bytes"]) for t in trials
                )
                / 2**30,
            }
        row["fdtdx_over_beamz"] = (
            row["fdtdx"]["warm_solve_s"] / row["beamz"]["warm_solve_s"]
        )
        measurements = case["gate"]["result"]["tolerances_and_measurements"][
            "measurements"
        ]
        row["numerical"] = {
            "max_dft_relative_l2": max(
                v["relative_l2"] for k, v in measurements.items() if "phasor" in k
            ),
            "final_field_relative_l2": {
                k: measurements[k]["relative_l2"] for k in ("E_xyz", "H_si_xyz")
            },
            "final_field_signal_normalized_l2": {
                k: measurements[k].get("error_l2_relative_to_signal_witness")
                for k in ("E_xyz", "H_si_xyz")
            },
        }
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--remote", type=Path)
    parser.add_argument("--completed-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = {"RTX 3090": campaign(args.local, completed_only=args.completed_only)}
    if args.remote:
        data["RTX 4090"] = campaign(args.remote, completed_only=args.completed_only)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(data),
        figsize=(8 if len(data) == 1 else 11, 4.6),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    axes = axes[0]
    colors = {"beamz": "#2563eb", "fdtdx": "#e67e22"}
    for ax, (gpu, rows) in zip(axes, data.items(), strict=True):
        x = [r["resolution_nm"] for r in rows]
        for engine, label in [("beamz", "BeamZ CUDA"), ("fdtdx", "FDTDX 0.6.3")]:
            y = [r[engine]["warm_solve_s"] for r in rows]
            ax.plot(x, y, "o-", color=colors[engine], label=label, linewidth=2)
            ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xticks(x, [f"{v:g}" for v in x])
        ax.grid(alpha=0.22, which="both")
        ax.set_title(gpu)
        ax.set_xlabel("Resolution (nm)")
        ax.legend(frameon=False)
    axes[0].set_ylabel("Warm solve time (seconds, log scale)")
    fig.suptitle("Same 680 nm Mie scene · CPML12 · FP32 · 30 fs", fontsize=13)
    image_path = args.output_dir / "comparison.png"
    fig.savefig(image_path, dpi=180)
    encoded = base64.b64encode(image_path.read_bytes()).decode()
    trs = []
    for gpu, rows in data.items():
        for r in rows:
            trs.append(
                f"<tr><td>{html.escape(gpu)}</td><td>{r['resolution_nm']:g}</td><td>{r['shape'][0]}³</td><td>{r['beamz']['warm_solve_s']:.2f} s</td><td>{r['fdtdx']['warm_solve_s']:.2f} s</td><td>{r['fdtdx_over_beamz']:.2f}×</td></tr>"
            )
    page = """<!doctype html><meta charset="utf-8"><title>BeamZ versus FDTDX</title>
<style>html{background:#fff;color-scheme:light}body{font:15px system-ui,sans-serif;max-width:1080px;margin:24px auto;color:#182433;padding:0 20px}h1{font-size:25px;margin-bottom:6px}p{line-height:1.5;color:#405268}img{width:100%}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:8px 12px;border-bottom:1px solid #e3e8ed}th:first-child,td:first-child{text-align:left}.note{font-size:13px;background:#f1f5f9;padding:13px;border-radius:8px}b{color:#182433}</style>
<h1>BeamZ CUDA versus released FDTDX 0.6.3</h1>
<p>Matched common material, source, CPML, and detector data. Two fresh processes per engine; each point is the geometric mean of process medians from three synchronized warm solves.</p>"""
    page += f'<img alt="Validated same-device solve time comparison" src="data:image/png;base64,{encoded}">'
    if args.completed_only:
        page += "<p><b>Stopped at user request.</b> Only completed timing cases are shown. The 2 nm case passed numerical validation; timing was stopped. No RTX4090 simulation ran and no RunPod pod was created.</p>"
    page += (
        "<table><thead><tr><th>GPU</th><th>nm</th><th>Grid</th><th>BeamZ</th><th>FDTDX</th><th>FDTDX / BeamZ</th></tr></thead><tbody>"
        + "".join(trs)
        + "</tbody></table>"
    )
    page += """<p class="note"><b>Interpretation:</b> Ratios above 1 favor BeamZ. The RTX3090 used its existing 250 W limit. FDTDX0.6.3 is the released JAX implementation, not the announced PyTorch rewrite. Strong-signal fields and full-run detector phasors passed their numerical checks. Weak late-time fields differ by 13–25% relative to their small residual norms, requiring a signal-normalized error floor; this benchmark does not establish 1% relative accuracy of residual ringdown. Compilation/setup are excluded from these warm timings.</p>"""
    (args.output_dir / "comparison.html").write_text(page)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
