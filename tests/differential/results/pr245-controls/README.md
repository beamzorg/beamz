# PR #245 controlled experiments

Fresh RTX 3090 runs using JAX 0.9.0/CUDA 12, Python 3.11.15, at code revision
`9641d493`. These experiments use the pinned silicon/silica indices, 220-nm core,
source pulse, three source frequency profiles, five mode candidates, and
Farjadpour diagonal smoothing. The TM0 control additionally uses nitride upper
cladding. Each run adds an exact 1550-nm sample to the original five frequencies.

![Straight-guide controls](straight_controls.png)

| Guide / launched mode | Maximum transmission error | Maximum reflection | Maximum output-plane power spread |
|---|---:|---:|---:|
| 500 nm / TE0 | 0.7404% | 0.2759% | 0.4654 percentage points |
| 1200 nm / TE0 | 0.1421% | 0.3164% | 0.1247 percentage points |
| 1200 nm / TE1 | 0.2604% | 0.3098% | 0.1895 percentage points |
| 405 nm / TM0, nitride top | 0.7881% | 0.2093% | 0.0701 percentage points |

All four meet the investigation's proposed 1% transmission, reflection, and
monitor-distance controls. The maximum modal-versus-integrated-flux discrepancy
is 0.1261% of incident power. These are independently meshed 12-µm straight
guides, not full-device-grid controls; they rule out a gross normalization error
in these simple geometries but do not establish validity of device-local modal
extraction. The three output monitors are alternative measurements of one guide
and must not be added together as physical outputs.

`runs.json` preserves complete scalar and spectral summaries, exact options,
code revisions, raw artifact paths, and SHA-256 hashes. The named NPZ files
retain complex S parameters, forward/backward modal amplitudes, flux checks,
projection residuals, conditioning, effective indices, and realized grid edges.
Full raw monitor fields and geometry/field plots remain at the recorded local
artifact paths. A preliminary duplicate TE0 run is excluded from this evidence.

Reproduce one control from the repository root:

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --no-sync python \
  -m scripts.investigate_passive_soi straight_te1 --exact-center \
  --output validation-artifacts/pr245/te1-new
```

Replace `straight_te1` with `straight_te0`, `straight_wide_te0`, or `straight_tm0`.
Use a new output directory for every experiment. The `--no-sync` flag preserves
an explicitly installed CUDA-enabled JAX environment.
Regenerate the figure using `python tests/differential/results/pr245-controls/plot_results.py`.

AI-assisted implementation, experiments, and analysis: OpenAI Codex.
