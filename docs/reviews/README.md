# CUDA optimization evidence

[Four-geometry shared CUDA campaign, Nov14](cuda-four-geometries-2026-11-14.md)
compares one candidate against main `387cc8f` on the RTX3090/250W:
**Mie 2.310×, cosine 1.878×, ring 1.606×, MMI 1.811×** warm core speed.
All 134 inputs and all 30 states remain identical to the preceding candidate;
153 native checks and a scoped positive memcheck pass. Host/GPU memory fall
for the three photonic examples. Mie GPU memory falls, but host peak remains
8.7% above main and setup is 2.3 s longer. The 2–3×/memory/setup target is not
met across all four cases. Experimental candidate; defaults unchanged.


[Two-hour trace-driven CUDA scheduling study, Nov13 evening](cuda-systolic-tracing-2026-11-13.md)
confirms fresh full-solve A–B–B–A gains over the preceding temporal winner:
**512³ 104.825→103.642s (1.011417×)**,
**564³ 147.636→146.217s (1.009700×)**.
The selected bounded-source/terminal-reuse/cache-exit composition preserves all 30
first/repeat outputs bytewise. Fifteen resource-admitted variants pass coupled
physics; seven builds are resource-rejected. 181 CPU tests, four selected-launch
CUDA diagnostic scopes and positive exact-kernel launch witnesses pass.
The rotated ownership compiler has a CPU proof; executable column schedules
lose or tie. No further 2× is established. Local RTX3090/250W, FP32/CPML12,
opt-in source/module; production defaults unchanged.

[Five-hour CUDA tracing and scheduling study, Nov13 afternoon](cuda-trace-five-hour-2026-11-13-afternoon.md)
confirms further full-solve gains over the starting temporal backend:
**512³ 111.918→108.224s (1.034132×)** and
**564³ 158.324→152.654s (1.037139×)** in final fresh A/B
confirmations. Separate two-pair ABBA stages establish the source/profile-cache
and geometry increments. All 30 first/repeat outputs match bytewise; ten scoped
sanitizer runs pass. Persistent full-step/half-step pipelines and the proved
three-parent reduction remain slower. Another 2× was not achieved. Local
RTX3090/250W, FP32/CPML12, opt-in module; defaults unchanged. Raw/generated
artifacts remain under `.cache/`.

[Two-hour CUDA tracing continuation, Nov13](cuda-tracing-continuation-2026-11-13.md)
confirms fresh full-solve A–B–B–A gains over V12: **512³ 116.220→111.940s
(1.038232×)** and **564³ 166.111→158.263s (1.049585×)**. Every first/repeat
matches all30 leaves bytewise;289 positive/16 rejection checks,80 source-plane
guards and positive memory/sync/race scopes pass. The measured winner combines
axis specialization, certified source-plane tracing and trail-load caching.
Shared/private ψ caches and the correct compiled cube lose or tie. CPU studies
add ordered axis-layout equivalence, a field-bank alias theorem and a concrete
native memory plan. Local RTX3090/250W, FP32/CPML12, defaults unchanged.

[Five-hour temporal CUDA and tracing continuation, Nov13](cuda-five-hour-study-2026-11-13.md)
confirms full512³ Mie execution **124.584→116.079 s**, 1.073262× throughput over
the strongest prior temporal control, with two fresh processes per build in
A–B–B–A order and all30 first/repeat leaves bytewise exact. This is **2.381×**
the fresh pinned-main default-CUDA control. Trace-proven source facts and
per-policy storage enable1024-thread source kernels; source-positive memory,
sync and race checks pass. Full564³ confirms182.222→166.083s (1.097178×); the smaller272³ check ties/loses
against tunedD2. The study closes the
preceding NoIO1024 race gap, and documents rejected larger-cache variants plus
a resource-only shared-CPML-state prototype. All workloads retain FP32, CPML12,
sources and DFT; local RTX3090 only, production defaults unchanged.

[Two-hour lifetime-aware tracing continuation, Nov12–13](cuda-plane-retirement-study-2026-11-12.md)
confirms full 512³ Mie execution 155.604→138.112 s in four fresh A–B–B–A processes:
1.12665× throughput, with all 30 first/repeat leaves bytewise exact. The winner
combines support facts and NoIO uniform-E specialization; larger plane-retiring
tiles are correct but slower. This is approximately 1.99× the earlier pinned-main
default CUDA measurement. It includes cold costs, scoped sanitizer evidence,
an interactive comparison and the next compiler constraints. Defaults are unchanged.
The late dedicated 1024-thread NoIO variant reaches 131.634 s in one fresh
process (1.1821× the previous temporal median), but broad race coverage remains
incomplete after a timeout; it is a provisional research build.

[Direct stock-main CUDA comparison, Nov12](cuda-autoresearch-four-hour-2026-11-11.md#direct-stock-main-comparison-2026-11-12)
records untouched main `d5282752`:275.31s default streamed CUDA,195.67s explicit
default Mstep,155.69s new temporal CUDA on the same full512³ Mie workload. Gains
are1.77×/1.26× respectively, with all30 first/repeat leaves bytewise exact.
These are single fresh-process comparisons; raw evidence is in
`.cache/main-vs-temporal-2026-11-12/`.

[Four-hour tracing-driven CUDA study](cuda-autoresearch-four-hour-2026-11-11.md)
records the complete512³,30fs Mie result:174.51s starting D2,170.70s improved
D2,155.62s traced temporal blocking, with all30 state leaves bytewise exact.
The several-fold target remains unmet. It includes replicated guide gains,
cold costs, rejected experiments, validation gaps, an interactive chart and
the next E-plane retirement architecture. Production defaults are unchanged.

[Larger domains and circular-cache CUDA](cuda-large-domain-study-2026-11-11.md)
records the local 384³/512³/640³ field-only crossover, paper-derived cache
geometry, and complete CPML12/source/DFT controls through 512³. The accompanying
[first-principles audit](cuda-large-domain-first-principles-2026-11-11.md) separates
requested-byte models and dependency proofs from measured runtime. The
[integration plan](../../cuda/CIRCULAR_RUNTIME_PLAN.md) specifies the bridge to
the complete native engine.

[Two-hour native CUDA and temporal scheduling study](cuda-native-autoresearch-2026-11-11.md)
records the JAX-independent prepared-input runner, matched shared-core trials,
exact D4 physics, rejected performance candidates, safe workspace reduction and
the next moving-subdomain architecture.

[CUDA-native runtime and deep temporal blocking plan](cuda-native-runtime-plan-2026-11-11.md)
sets the next architecture: JAX-independent forward execution, generated
space-time schedules, integrated CPML/source/DFT work, and measured promotion gates.

[Released FDTDX comparison](fdtdx-comparison-2026-11-11.md) documents the
matched CPML12 Mie recipe, numerical admission criteria and completed local
RTX3090 measurements. RunPod was stopped without provisioning a GPU.
Raw evidence remains under `.cache/`.

[2026-11-11 tracing and CPML12 optimization](cuda-autoresearch-2026-11-11.md)
records the accepted native defaults, two packed-material fixes, balanced
11–16% throughput gains and the complete 680 nm nanoparticle sweep. It links
the tracing architecture, validation and local artifacts under `.cache/`.

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
