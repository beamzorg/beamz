# CUDA optimization evidence

[PR #334 validation summary](pr334-post-merge-validation-2026-10-17.md) covers
notebook accuracy, RTX 3090 performance, known failures and the modal-projection
follow-up. It links immutable historical reports and raw data for the Futhark,
schedule/startup and H100 studies. The [CUDA component](https://github.com/beamzorg/beamz/blob/feat/cuda-mstep/cuda/README.md)
documents the supported API and selection limits.

Keep new generated measurements under `.cache/` or as CI artifacts rather than
adding raw datasets, logs and derived copies here.

Earlier RTX 3090 studies, raw measurements, logs, and their analysis scripts are
preserved in the [immutable pre-cleanup snapshot](https://github.com/beamzorg/beamz/tree/dabecdc231f82194ecb8f9f352d5654be12a8c0c/docs/reviews).
The [layout prediction study](https://github.com/beamzorg/beamz/blob/dabecdc231f82194ecb8f9f352d5654be12a8c0c/docs/reviews/cuda-layout-prediction-2026-09-18.md)
and its [replay script](https://github.com/beamzorg/beamz/blob/dabecdc231f82194ecb8f9f352d5654be12a8c0c/scripts/analyze_cuda_layout_prediction.py)
record the historical method and limits. Check out that commit to reproduce the
analysis against its original data and code. These experimental settings are
not current defaults.
