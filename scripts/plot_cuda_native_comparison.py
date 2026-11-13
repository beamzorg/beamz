#!/usr/bin/env python3
"""Plot source-backed balanced native CUDA trials, with observed process ranges."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

FIXTURES = ("historical", "cube256", "large")
LABELS = ("160×320×384", "256³", "256×512×512")
WORKFLOWS = ("cuda_mstep", "cuda_streamed")
COLORS = ("#4d91db", "#e5a142")


def render(report):
    groups = {(g["fixture"], g["prepare_backend"]): g for g in report["groups"]}
    parts = []

    def x(gain):
        return 190 + gain * 2400

    for tick in (0, 5, 10, 15, 20):
        px = x(tick / 100)
        parts.append(
            f'<line x1="{px}" x2="{px}" y1="22" y2="275" class="grid"/>'
            f'<text x="{px}" y="297" text-anchor="middle">+{tick}%</text>'
        )
    for index, (fixture, label) in enumerate(zip(FIXTURES, LABELS, strict=True)):
        center = 65 + 85 * index
        parts.append(f'<text x="174" y="{center + 4}" text-anchor="end">{label}</text>')
        for workflow_index, workflow in enumerate(WORKFLOWS):
            group = groups[fixture, workflow]
            y = center - 12 + 25 * workflow_index
            gain = group["speedup"] - 1
            low, high = (v - 1 for v in group["paired_ratio_range"])
            rates = group["balanced_medians_gcups"]
            tooltip = html.escape(
                f"{workflow} prepared first: {rates['control']:.3f} → "
                f"{rates['candidate']:.3f} GCUPS; two processes per variant"
            )
            color = f"var(--chart-{workflow_index + 1},{COLORS[workflow_index]})"
            parts.append(
                f'<rect x="190" y="{y - 9}" width="{x(gain) - 190}" height="18" '
                f'fill="{color}" rx="2"><title>{tooltip}</title></rect>'
                f'<path d="M{x(low)},{y}H{x(high)} M{x(low)},{y - 4}V{y + 4} '
                f'M{x(high)},{y - 4}V{y + 4}" class="range"/>'
                f'<text x="{x(high) + 10}" y="{y + 4}">+{100 * gain:.1f}%</text>'
            )
    return """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{margin:0;font-family:var(--font-sans,system-ui);color:var(--foreground,#ddd)}
svg{width:100%;height:320px}svg text{fill:var(--foreground,#ddd);font-size:13px}
.grid{stroke:var(--border,#444)}.range{stroke:var(--foreground,#ddd);fill:none;stroke-width:1.5}
p{font-size:13px;line-height:1.5;color:var(--muted-foreground,#999);margin:6px 0}strong{color:var(--foreground,#ddd)}
.one{color:var(--chart-1,#4d91db)}.two{color:var(--chart-2,#e5a142)}
</style></head><body><p><strong>CUDA mstep throughput gain over equally tuned control</strong></p>
<p><span class="one">● Mstep prepared first</span> &nbsp; <span class="two">● Streamed prepared first</span></p>
<svg viewBox="0 0 720 320" role="img" aria-label="Balanced native CUDA throughput gains">__SVG__</svg>
<p>RTX 3090 · unchanged 250 W limit · 12 CPML cells per face · 256 steps.
Both builds tuned over whole/32/64-plane segments. All 24 fresh-process trials
match the complete reference state exactly. Two processes per variant and case;
whiskers show the observed paired-process range, not a confidence interval.</p>
</body></html>""".replace("__SVG__", "".join(parts))


def export_png(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {(g["fixture"], g["prepare_backend"]): g for g in report["groups"]}
    figure, axis = plt.subplots(figsize=(8, 4.2), layout="constrained")
    for offset, workflow in enumerate(WORKFLOWS):
        selected = [groups[fixture, workflow] for fixture in FIXTURES]
        gains = [100 * (g["speedup"] - 1) for g in selected]
        errors = [
            [
                gain - 100 * (g["paired_ratio_range"][0] - 1)
                for gain, g in zip(gains, selected, strict=True)
            ],
            [
                100 * (g["paired_ratio_range"][1] - 1) - gain
                for gain, g in zip(gains, selected, strict=True)
            ],
        ]
        positions = [i + (offset - 0.5) * 0.28 for i in range(3)]
        axis.barh(
            positions,
            gains,
            height=0.25,
            color=COLORS[offset],
            xerr=errors,
            label=("Mstep prepared first", "Streamed prepared first")[offset],
            capsize=3,
        )
    axis.set(
        yticks=range(3),
        yticklabels=LABELS,
        xlim=(0, 20),
        xlabel="Cell throughput increase over equally tuned control (%)",
    )
    axis.invert_yaxis()
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=False)
    figure.suptitle("BeamZ CUDA · RTX 3090 · 250 W · CPML12")
    figure.supxlabel(
        "Whiskers: observed paired-process range, not a confidence interval", fontsize=9
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(report))
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        export_png(report, args.png)


if __name__ == "__main__":
    main()
