# P0→P1→P2 supervised acceptance (2026-11-17)

GPT-6.1 Sol supervised Luna single-task deliveries, reviewed each before assembly, and obtained root gates after P0 and P1. See supervision_log.md. Work was staged separately; historical files and all 14 frozen protocol files remain unchanged.

## Actual runs

| Condition | Evaluation | Accuracy | Macro-F1 | Training seconds |
|---|---|---:|---:|---:|
| BANKING77 8-shot independent seed42 | Validation | .769000 | .764026 | 30.11 |
| CLINC150 8-shot independent seed42 | Validation | .876667 | .873246 | 61.76 |
| CLINC150 8-shot independent seed42 | Explicit final Test | .861778 | .858408 | 62.40 |
| BANKING77 full pool | Validation | 0.865000 | 0.856228 | 548.26 |

P1 actual 8-shot Validation results (sample std, ddof=1):

| Dataset | Mode | Accuracy Mean±Std | Macro-F1 Mean±Std |
|---|---|---:|---:|
| banking77 | independent | 0.769000 ± 0.011000 | 0.762413 ± 0.013372 |
| banking77 | nested | 0.768667 ± 0.005686 | 0.761914 ± 0.009545 |
| clinc150 | independent | 0.865667 ± 0.009701 | 0.861661 ± 0.010280 |
| clinc150 | nested | 0.864667 ± 0.012342 | 0.859947 ± 0.013251 |

P1 covers 12 configurations: two datasets × two modes × seeds 42/43/44. Ten new runs and two P0 reuses after exact manifest/settings/version checks. Sampling/training seeds vary together; BANKING's train/validation partition also changes. Do not interpret this as training-only variance. Full CLINC and 4/16-shot training remain unrun.

## Verification

- Final 01 and 02 each ran top to bottom in separate fresh Python processes and separate isolated projects with no results directory at start. Method: IPython InteractiveShell cell execution and capture, not nbclient/Jupyter kernel execution. Final notebooks contain the real captured outputs and execution counts.
- 01 executed 3 cells with zero errors and generated 14 content-matching audit files. 02 executed 5 cells with zero errors, trained 8,996 full-pool samples, and never created/read results/protocol. Test metrics are null.
- BANKING pool/validation/test = 8996/1000/3080. CLINC = 14995/3000/4500. Original JSON SHA, 150-name mapping, source row order, OOS audit and full protected splits passed.
- Reversed/non-contiguous labels map to int64 0..C-1 for both datasets. Subset-only evaluation batches pass. Hash, seed/K, duplicate-ID and label-swap corruptions are rejected; legal seed43/K4 passes. Nested 4⊂8⊂16 ID sets pass; old independent row order matches frozen manifests.
- Missing-package tests mock detection before imports; no installed package was removed. Test without authorization is rejected. CLINC bootstrap tests use mocked transport with real cached bytes and reject corrupt SHA; network download availability was not tested.
- Final-Test diagnostics use P0's complete saved predictions without retraining: 150×150 matrix and support total 4500; synthetic metrics match sklearn. Validation/unauthorized diagnostics are suppressed; empty errors retain CSV headers. Top 10 actual groups are displayed in the captured P1 evidence.
- Shared protocol functions are identical in both setup cells. All functions are inline in setup. Optional grid/multiseed branches are controlled and disabled by default; only default False branches were exercised in P2. P1 separately exercised the 8-shot multi-seed scope. The 4/16 sampling invariants were checked, but their training remains unrun.
- 14 new-schema per-run JSONs contain all required fields and null unmeasured scores. Historical JSONs retain their old schema. Summary links resolve to all 12 individual JSONs.

Training settings remained batch16/epoch1/iterations5/lr2e-5. Timers exclude model loading and diagnostic writing, include Trainer initialization; phase CUDA peaks are absolute allocated memory including resident allocations.

## Handoff

Acceptance artifacts and temporary harnesses remain in the shared supervised_repair workspace. Delivered notebooks have no dependency on those scripts. No external service messages, credentials, package removal or full CLINC run occurred. Root installs only the manifest-listed files after source-hash recheck.
