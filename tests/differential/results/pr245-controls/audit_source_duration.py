"""Audit duration-dependent broadband profile drives from retained source samples.

This diagnoses temporal subbands, not the combined spatial injection: distinct
mode-profile coefficients can cancel much of their individual late-time drive.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from beamz import LIGHT_SPEED
from beamz.devices.sources.time import (
    analytic_subband_waveforms,
    chebyshev_frequency_nodes,
)

HERE = Path(__file__).resolve().parent


def main():
    rows = sorted(
        (
            r
            for r in json.loads((HERE / "runs.json").read_text())
            if r["run"].startswith("ring-fixed-")
        ),
        key=lambda r: r["run_time_ps"],
    )
    report = {
        "method": "Replay the source compiler's analytic subband partition on retained in-phase/quadrature samples and the original three Chebyshev profile frequencies.",
        "limitation": "Subband amplitudes are not the net injected field or power; spatial profile coefficients and cancellations are not included.",
        "runs": [],
        "adjacent_prefix_comparisons": [],
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    previous = None
    for row in rows:
        raw = np.load(Path(row["raw_artifact_path"]) / "monitor_data.npz")
        time = raw["source_time_s"]
        pulse = raw["source_signal"].astype(float) + 1j * raw[
            "source_quadrature"
        ].astype(float)
        _, bands = analytic_subband_waveforms(
            pulse,
            dt=time[1] - time[0],
            profile_frequencies=chebyshev_frequency_nodes(
                LIGHT_SPEED / 1.55e-6, np.ptp(raw["frequencies_hz"]), 3
            ),
        )
        peaks = np.max(abs(bands), axis=1)
        envelope = np.max(abs(bands) / peaks[:, None], axis=0)
        drive = np.concatenate([bands.real, bands.imag], axis=0)
        drive_peaks = np.max(abs(drive), axis=1)
        normalized_drive = abs(drive) / drive_peaks[:, None]
        report["runs"].append(
            {
                "run": row["run"],
                "duration_ps": row["run_time_ps"],
                "terminal_subband_envelope_over_own_peak": (
                    abs(bands[:, -1]) / peaks
                ).tolist(),
                "terminal_largest_real_or_quadrature_drive_over_own_peak": float(
                    normalized_drive[:, -1].max()
                ),
            }
        )
        if previous is not None:
            old_name, old_time, old_bands = previous
            n = min(len(old_time), len(time))
            delta = abs(old_bands[:, :n] - bands[:, :n]) / max(peaks)
            report["adjacent_prefix_comparisons"].append(
                {
                    "runs": [old_name, row["run"]],
                    "maximum_common_prefix_difference_over_larger_run_peak": float(
                        delta.max()
                    ),
                    "maximum_first_2ps_difference_over_larger_run_peak": float(
                        delta[:, time[:n] < 2e-12].max()
                    ),
                }
            )
        previous = row["run"], time, bands
        label = f"{row['run_time_ps']:g}-ps record"
        # Bin maxima keep the carrier envelope readable without discarding peaks.
        bins = np.array_split(np.arange(len(time)), 800)
        times = np.array([time[b].mean() for b in bins]) * 1e12
        maxima = np.array([envelope[b].max() for b in bins])
        axes[0].semilogy(times, maxima, label=label)
        axes[1].semilogy(times - time[-1] * 1e12, maxima, label=label)
    for ax in axes:
        ax.axhline(1e-6, color="#444444", ls="--", label="Source-off criterion")
        ax.set(ylabel="Largest profile envelope / its own peak", ylim=(1e-7, 2))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set(
        xlabel="Time (ps)", title="Profile drives depend on the full FFT record"
    )
    axes[1].set(
        xlabel="Time relative to record end (ps)",
        xlim=(-2, 0),
        title="Periodic tails approach the record boundary",
    )
    fig.suptitle(
        "Temporal profile audit · not combined spatial injection or injected power"
    )
    fig.savefig(HERE / "source_duration_audit.png", dpi=170)
    (HERE / "source_duration_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
