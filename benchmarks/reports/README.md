# Benchmark reports

These are historical measurements, not tests or performance guarantees. Each report records
its workload, code revision, hardware, limitations and commands. Small CSV summaries sit
beside the reports.

- [Curved-interface smoothing](CURVED_INTERFACE_SMOOTHING.md): full-tensor Farjadpour
  smoothing reaches the accuracy of the default diagonal smoothing on a 2× coarser mesh in a
  3D SiN bend: 13.2× faster with 5.8× less GPU memory. Includes a setup-only CPML memory fix
  that cuts memory for every method and lets fine full-tensor runs fit on an 8 GB GPU.
- [FIT evaluation](FIT_EVALUATION.md): for photonics a separate Finite Integration Technique
  backend is not justified, because its gains transfer to Yee FDTD. For RF/microstrip a gated follow-up
  stage is proposed: ports, S-parameters, graded meshes, conductor loss and conformal PEC,
  each checked against a matched Yee control.
