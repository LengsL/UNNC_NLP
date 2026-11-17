# Supervision log

Each Luna assignment was limited to one function group. The supervisor reviewed each delivered text before notebook assembly and execution. No worker wrote to the source project.

| Stage/task | Review | Evidence and decision |
|---|---|---|
| P0 training/evaluation function | Returned once | First draft encoded the selected train labels instead of the full pool space, hardcoded settings and an independent filename. Requested a shared encoder, constants, mode-aware naming and rejection of noninteger predictions. |
| P0 revised training/evaluation | Accepted | Both datasets remap reversed non-contiguous labels to int64 0..C-1. Subset-only evaluation batches pass. Test without explicit authorization fails. Real BANKING validation, CLINC validation and CLINC final-test runs saved complete predictions/metrics. |
| P0 dependency/protocol integration | Accepted by supervisor | Mock missing-package checks precede third-party imports. Seed42/K8 frozen manifests match row for row. Seed43/K4 succeeds. Changed seed/K/hash is detected. Original 14 protocol SHA256 values remain unchanged. |
| P0 gate | Passed by root | Root reviewed the actual metrics and the stated new-process IPython execution method, then authorized P1. |
| P1 protocol sampling group | Returned for strengthening | Independent branch preserves legacy pandas sampling. Nested selects deterministic per-class prefixes. Requested explicit original source ID uniqueness and full selected text/label/hash restoration to prevent label swaps escaping a hash-only check. |
| P1 revised protocol group | Accepted | Runtime tests proved unchanged legacy row order, deterministic 4⊂8⊂16 nested ID sets, valid new seed/K, rejection of invalid nested budgets and mutated metadata, duplicate IDs and label swaps. None budget reports fewshot=null while returning a pool copy. |
| P1 final-test diagnostics | Returned once, then accepted | Removed an unsolicited filename suffix to retain consistent keys, required training-protocol validation before model creation, and asserted confusion counts match prediction rows. The 3-class fixture matches sklearn confusion and per-class metrics, empty-error CSV headers persist, and validation/unauthorized requests create no diagnostics. Real CLINC final-test predictions produced a 150×150 matrix, per-class report and top error groups. |
| P1 seed summary helper | Accepted | Runtime fixture proves independent/nested separation, mean and sample std (ddof=1), retained None-budget groups, and rejection of insufficient/duplicate/different seeds or test results. Full configuration-level JSON references are retained. |
| P1 12-configuration analysis | Accepted | Actual 8-shot validation covers both datasets × both modes × seeds 42/43/44. Ten new runs and two manifest/config/version-verified P0 reuses. Four groups each contain three seeds; original 14 protocol SHA256 values remain unchanged. |
| P1 gate | Passed by root | Root presented four Mean±Std groups and authorized P2. |
| P2 01 cell group | Accepted | All helper definitions/imports consolidated in setup; frozen mismatch prints before asserting; summary identifies mode/budget/pool copies. Fresh empty-results run passed 3 cells with zero errors and 14 matching files. |
| P2 02 cell group | Accepted | All definitions consolidated in setup; default BANKING full pool/Validation, optional flags False; grid propagates mode. AST/default/shared-protocol checks passed before actual execution. |
| P2 bootstrap/export checks | Accepted | Mock transport of real cached CLINC bytes verifies URL/SHA bootstrap; corrupt bytes rejected. Synthetic differing CSV filename printed before failure; frozen content remains untouched. |

| P2 empty-results default | Accepted | Final 02 passed 5 cells with zero errors, 8,996 samples, Validation only. All 14 new-schema JSONs pass required-field/null checks; all frozen SHA256 values remain unchanged. |
