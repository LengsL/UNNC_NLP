# Inline protocol and notebook-only results

01/02/03 share identical build_protocol code, normalization, original-row Source IDs and split/shot rules. 01 keeps the original seed-42 independent 8-shot manifest fingerprints inside its final cell; it no longer exports or reads 14 audit files. New seeds/K remain valid in experiment notebooks.

BANKING77 defaults to the cleaned full pool with fixed stratified validation; CLINC150 uses 14,995 cleaned training rows, official 3,000 Validation and 4,500 Test rows, and 150 intent names. CLINC IDs index original in-domain JSON rows, never cleaned-row positions. OOS and unconfirmed converted extras are excluded as before.

Metrics and tables are displayed and retained in memory, not CSV/JSON files. Each run returns summary with its scalar metrics, settings, versions, protocol metadata, predictions and errors. Repeated-seed summaries use run_ids instead of references to nonexistent files. Final Test diagnostics are gated by explicit authorization. Full Pool stays the main condition; low-resource analyses stay optional.

Final ZIP contains only executed method–dataset notebooks; this document, 01 and cached source data remain development resources.
