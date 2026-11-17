# COMP4153 CW1 — development workspace

Run notebooks independently from the project root or notebooks/. All functions are inline. Data/model downloads may require network access; caches can be prepared for offline use. CLINC original JSON is read locally if available, otherwise downloaded in memory and checked by SHA256.

- 01 audits both datasets and displays fixed protocol fingerprints, statistics, overlaps and label mapping in cells. No protocol files are exported.
- 02 runs SetFit on BANKING77 only; 03 runs SetFit on CLINC150 only. Default Full Pool and Validation evaluation; optional analyses remain disabled. Explicit Test evaluation requires EVALUATION_SPLIT="test" and FINAL_TEST=True.
- Metrics, training settings, versions, predictions, errors and authorized Test diagnostics are displayed in notebook cells and retained in memory. No result CSV/JSON files are exported. Trainer work directories are temporary; checkpoints remain disabled.
- Existing genuine experiment outputs are retained. The new display code has not been trained again. A concurrently saved 03 run is retained, including its batch_size=64 setting and genuine CLINC150 Validation outputs; these must be considered when comparing training settings across methods. Run each final notebook from top to bottom and retain outputs before submission.

## Final submission

Submit Name_ID.pdf separately from Name_ID_source.zip. The ZIP must contain at least 12 executed, independent notebooks: five existing methods plus the proposed method, each on BANKING77 and CLINC150. Use Name_ID_MethodName_DatasetName.ipynb. More existing methods require additional notebooks. 01 is a development audit and is not one of the 12 experiments. FastFit/Method3–5/Proposed notebooks must contain actual implementations and results before inclusion; they are not fabricated here.

Do not package datasets, weights, caches, separate .py files, README, requirements.txt, environment.yml or other source files. This development workspace may retain documentation and the original data cache; these are excluded from the submission ZIP. The written paper requires at least one appendix output page per method–dataset experiment (minimum 12).

Historical acceptance documents describe the earlier file-export workflow. The current notebook-only workflow supersedes their output-path and frozen-file requirements; data/sampling/training rules remain unchanged. Earlier generated CSV/JSON evidence is archived outside this project before removal. Cite external data/models/code and include the required AI usage declaration.
