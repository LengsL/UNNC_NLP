# Phase-one engineering acceptance

Accepted after the final independent-kernel run on 2026-11-11.

- 01 and 02 ran in separate fresh Jupyter kernels using Anaconda py3. CUDA forward and backward computation passed.
- BANKING77 removed 7 training overlaps. Final pool/validation/test are 8996/1000/3080 with no normalized-text intersection. Fixed 8-shot sample is 77 x 8 = 616.
- CLINC retained original domain validation/test. Removed 3 train-validation and 2 train-test rows, including 4 label conflicts. Pool/validation/test are 14995/3000/4500; no normalized-text intersection. Verified 150-ID mapping and 150 x 8 sample. Extra 100 converted null-label rows are marked unconfirmed and excluded.
- Regenerated 14 protocol files exactly match the saved version.
- BANKING77 pilot validation Accuracy = 0.769 and Macro-F1 = 0.7640263650627385 with 231 readable errors. Runtime, GPU peaks, training settings and library versions are saved in metrics.
- Five candidate methods have primary references and evidence-level notes. AncSetFit code contents and MICL implementation completeness need resolving before implementation. Four additional local pilots and CLINC training are next-phase work.
- Data, model weights, caches, credentials and historical backups are excluded from the upload.
