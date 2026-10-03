"""Collect retained local simulation evidence into compact review artifacts.

Run from the checkout root after completing the experiments:
    python tests/differential/results/pr245-main-refresh/collect.py \
        validation-artifacts/pr245-main-refresh
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent
    spectra = destination / "spectra"
    spectra.mkdir(exist_ok=True)
    records = []
    for summary in sorted(args.raw.glob("*/summary.json")):
        record = json.loads(summary.read_text())
        if "performance" not in record:
            continue
        run_id = summary.parent.name
        raw_file = summary.parent / "monitor_data.npz"
        actual_hash = hashlib.sha256(raw_file.read_bytes()).hexdigest()
        if actual_hash != record["monitor_data_sha256"]:
            raise ValueError(f"Retained field hash differs: {raw_file}")
        with np.load(raw_file) as raw:
            arrays = {
                key: raw[key]
                for key in raw.files
                if key.startswith(("S_", "diagnostic_", "flux_", "grid_"))
                or key
                in {
                    "frequencies_hz",
                    "valid_mask",
                    "incident_power",
                    "guided_output_power",
                    "power_sum",
                    "loss_estimate",
                }
            }
            incident = raw["incident_power"]
            # Port o1 points out of the domain: its minus wave is incident,
            # and its plus wave is reflected back toward the source.
            reflected = raw["diagnostic_o1__P_plus"]
            record["maximum_input_reflection"] = float(np.max(reflected / incident))
            record["source_record_hashes"] = {
                key: hashlib.sha256(
                    np.ascontiguousarray(raw[key]).tobytes()
                ).hexdigest()
                for key in ("source_time_s", "source_signal", "source_quadrature")
            }
        compact = spectra / f"{run_id}.npz"
        np.savez_compressed(compact, **arrays)
        center = int(np.argmin(abs(np.asarray(record["wavelengths_um"]) - 1.55)))
        record["power_at_1550nm"] = {
            key: values[center] for key, values in record["powers"].items()
        }
        record["sampled_wavelength_um"] = record["wavelengths_um"][center]
        record["field_decay_passed"] = record["termination"]["field_decay"] <= 1e-5
        if "selected_output_max" in record:
            record["selected_output_passed"] = record["selected_output_max"] <= 1.02
        record["run_id"] = run_id
        record["raw_directory"] = str(summary.parent.resolve())
        record["compact_spectra_sha256"] = hashlib.sha256(
            compact.read_bytes()
        ).hexdigest()
        patch = summary.parent / "working-tree.patch"
        if patch.stat().st_size:
            patch_dir = destination / "source-patches"
            patch_dir.mkdir(exist_ok=True)
            (patch_dir / f"{run_id}.patch.gz").write_bytes(
                gzip.compress(patch.read_bytes(), mtime=0)
            )
        records.append(record)
    (destination / "runs.json").write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n"
    )
    attempts = []
    superseded_path = args.raw / "superseded-runs.json"
    superseded = {
        record["run_id"]: record
        for record in (
            json.loads(superseded_path.read_text()) if superseded_path.exists() else []
        )
    }
    for campaign in (
        "campaign.json",
        "extra-campaign.json",
        "headroom-campaign.json",
        "final-headroom-campaign.json",
    ):
        path = args.raw / campaign
        if not path.exists():
            continue
        for attempt in json.loads(path.read_text()):
            if attempt["name"] in superseded:
                attempt["stopped"] = superseded[attempt["name"]]["reason"]
                attempt["stop_note"] = superseded[attempt["name"]]["note"]
            directory = args.raw / attempt["name"]
            config = directory / "config.json"
            if config.exists():
                attempt["configuration"] = json.loads(config.read_text())
            attempt["summary_available"] = (directory / "summary.json").exists()
            patch = directory / "working-tree.patch"
            if patch.exists() and patch.stat().st_size:
                patch_dir = destination / "source-patches"
                patch_dir.mkdir(exist_ok=True)
                (patch_dir / f"{attempt['name']}.patch.gz").write_bytes(
                    gzip.compress(patch.read_bytes(), mtime=0)
                )
            attempts.append(attempt)
    (destination / "attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
    print(f"Collected {len(records)} runs with verified raw field hashes.")
    for record in records:
        channel = "o3" if record["device"] == "mmi2x2" else "conversion"
        power = record["power_at_1550nm"].get(channel)
        print(
            record["run_id"],
            f"power={power}",
            f"decay={record['termination']['field_decay']:.3g}",
            f"runtime={record['performance']['runtime_s']:.2f}s",
            f"host_RSS={record['peak_host_rss_mib'] / 1024:.2f}GiB",
        )


if __name__ == "__main__":
    main()
