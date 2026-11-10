# Reuse the fixed splits

Run `notebooks/01_dataset_exploration.ipynb` to prepare or verify the seed-42 manifests in `results/protocol/`. Start Jupyter in the project or `notebooks/` directory.

The first code cell of `notebooks/02_setfit_banking77_pilot.ipynb` loads BANKING77 and restores the saved sample order by `source_id`. It verifies labels, normalized-text hashes, disjoint splits and the 8-shot counts. `fewshot_train` is the training dataset and `bank_val` is the validation frame. The pilot reads the official test manifest for the overlap audit.

For CLINC150, use the original `data/clinc150/data_full.json` with the `clinc150_*.csv` manifests. Resolve `clinc_train_<i>`, `clinc_validation_<i>`, and `clinc_test_<i>` from `train`, `val`, and `test` respectively. Use the verified `clinc150_id_to_name.json` for integer labels. Preserve manifest order and check labels and text hashes before fitting each baseline.
