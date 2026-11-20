# CUDA optimization evidence

[PR #334 validation summary](pr334-post-merge-validation-2026-10-17.md) covers
notebook accuracy, RTX 3090 performance, known failures and the modal-projection
follow-up. It links immutable historical reports and raw data for the Futhark,
schedule/startup and H100 studies. The [CUDA component](https://github.com/beamzorg/beamz/blob/feat/cuda-mstep/cuda/README.md)
documents the supported API and selection limits.

Keep new generated measurements under `.cache/` or as CI artifacts rather than
adding raw datasets, logs and derived copies here.

Earlier studies remain historical: [main-merge validation](cuda-merge-validation-2026-09-24.md),
[random domains](cuda-random-final-comparison-2026-09-18.md), and
[layout prediction](cuda-layout-prediction-2026-09-18.md). Their experimental
settings are not current defaults.
