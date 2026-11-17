# P0 supervised acceptance

Supervisor reviewed the Luna training function twice. The first draft was returned for pool-space label encoding, shared encoding helper, constants, mode-aware naming, and strict integer prediction validation. The revised function passed review.

Verified in a fresh Python process using IPython cell execution and the local public-data/model caches:

- Both seed-42 independent 8-shot manifests match the frozen 01 CSVs row for row.
- BANKING split sizes: pool 8,996, validation 1,000, test 3,080; 77 labels; 616 training examples.
- CLINC split sizes: pool 14,995, validation 3,000, test 4,500; 150 labels; 1,200 training examples. Original JSON SHA256, ID row order, converted-label mapping and OOS comparison passed.
- Legal seed 43/K4 succeeded. Mutating the sampling seed, budget or selected hash failed validation as intended.
- Reversed, non-contiguous synthetic labels remapped to 0..C-1 with int64 training and actual arrays; evaluation batches containing only a subset of classes succeeded.
- Mocked missing torch, setfit and sentence_transformers produced actionable current-interpreter pip instructions before downstream imports. No package was uninstalled or installed.
- Test requests without explicit final-test authorization failed before model creation.
- 01 ran all cells and retained execution outputs. All 14 original protocol file SHA256 values remain unchanged.

Actual SetFit runs (batch 16, epoch 1, iterations 5, learning rate 2e-5, seed 42, independent 8-shot):

| Dataset | Evaluation | Accuracy | Macro-F1 | Training seconds |
|---|---|---:|---:|---:|
| BANKING77 | Validation | .769000 | .764026 | 30.1107 |
| CLINC150 | Validation | .876667 | .873246 | 61.7562 |
| CLINC150 | Explicit final Test | .861778 | .858408 | 62.4001 |

New metrics and prediction filenames include dataset, method, budget, sampling mode, seed and evaluation split. Historical pool/test results were retained. P0 did not train full CLINC or repeat the historical full BANKING test run. Training results are supported by JSON metrics, complete predictions, error CSVs, and the captured execution result; the temporary harness is not a notebook dependency.
