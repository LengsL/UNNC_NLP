# COMP4153 CW1 · Few-shot intent classification

Use the `py3` environment and start Jupyter in this directory or `notebooks/`.

1. Run [01 · Dataset exploration](notebooks/01_dataset_exploration.ipynb) to review the dataset statistics and verify the fixed split and 8-shot manifests in `results/protocol/`.
2. Run [02 · BANKING77 SetFit pilot](notebooks/02_setfit_banking77_pilot.ipynb) to train on the fixed sample and evaluate on validation. The pilot saves metrics in `results/metrics/` and errors in `results/predictions/`.
3. Use [the baseline screening matrix](references/baseline_matrix.md) and [the fixed-split instructions](references/protocol_usage.md) when testing other methods.

These are development notebooks. For Moodle, prepare the required independently runnable method–dataset notebooks with their executed outputs and embedded setup and data-loading logic.

## Environment and first run

Activate Anaconda `py3` and launch `python -m jupyterlab` in this directory. The verified interpreter is `C:/APP/Anaconda/envs/py3/python.exe`. `requirements.txt` records the tested versions. Recreating its CUDA PyTorch build requires the matching official PyTorch wheel index.

Run 01 first. It downloads the original CLINC JSON into `data/clinc150/` if absent and verifies its SHA256. Hugging Face Datasets downloads BANKING77 and the converted CLINC audit data. Run 02 next to load the model and train. Initial downloads require internet access. Once cached, `HF_HUB_OFFLINE=1` and `HF_DATASETS_CACHE` pointing to the prepared cache support offline verification and avoid Hub retries. These are runtime environment options, not changes to model settings.

Phase one covers data integrity, fixed sampling, one BANKING77 SetFit pilot and five-method preliminary screening. Other methods and the CLINC SetFit experiment are next-phase tasks. The complete coursework requires five existing methods plus the proposed approach on both datasets, an academic paper and appendix evidence. The duplicate policy follows the teacher's reply by retaining official protected splits and removing overlapping training samples.
