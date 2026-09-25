# Jev Risk Classification

Reproducible research code comparing TypeSafe Jev with Random Forest and XGBoost for binary transaction-risk classification. The source of truth is the Python package under `src/jev_risk`; notebooks call that package for exploration and results.

## Research protocol

- Target: `Risk_Incident` (`0` = no incident; `1` = incident).
- Supplied dataset: 10,000 rows; 1,448 positive transactions (14.48%); transaction IDs are unique.
- `Risk_Type`, `Incident_Severity`, and `Error_Code` each perfectly encode the target in this file. They are excluded from all model inputs and Jev state. `Currency` is constant. High-cardinality identifiers and `Counterparty` are also excluded from the primary feature set.
- The primary features are transaction type, amount, category, payment method, system latency, login frequency, failed attempts, IP region, and calendar components derived from `Date`.
- The CSV uses `NA` as a valid `IP_Region` code (37 rows) and the literal `None` as a negative-row placeholder in the post-event columns. The loader preserves these tokens and treats only empty CSV cells as missing. Numeric median and categorical most-frequent imputers are fitted only on training data.
- One seeded, stratified 80/20 split is shared by the prevalence baseline, Random Forest, XGBoost, and Jev. The split manifest stores the sample IDs, labels, and source CSV SHA-256.
- RF and XGBoost use training-only preprocessing and five-fold stratified CV selected by average precision. `--no-tune` is available for a faster initial smoke run.
- Jev uses TypeSafe `Choice` with `risk_incident` / `no_risk_incident` criteria. `y_prob` is Jev's positive-option probability; `confidence` is stored separately. A small pilot must succeed before full-set inference is enabled.
- Selective metrics describe accepted cases only; rejected cases count toward human-review count and reduce coverage. Latency is per-row local inference for classical models and hosted round-trip time for Jev.

## Setup

Python dependencies are managed with **uv** (no `pip` install steps):

```bash
uv sync --extra dev
```

On macOS, XGBoost also needs the OpenMP runtime:

```bash
brew install libomp
```

The project environment includes the `jev-risk` command and JupyterLab. Launch notebooks with:

```bash
uv run jupyter lab
```

Select the project environment's Python kernel. Notebooks do not install dependencies; they use the package installed by `uv sync`.

## Run the research pipeline

Commands default to the supplied `accounting_dataset.csv` and write generated artifacts under `results/`.

```bash
# Audit schema, class distribution, missingness, duplicates, and leakage
uv run jev-risk audit

# Persist the fixed train/test sample assignments
uv run jev-risk split

# Train the dummy baseline, RF, and XGBoost with five-fold training-only tuning
uv run jev-risk train

# Optional faster local smoke run without hyperparameter search
uv run jev-risk train --no-tune
```

### TypeSafe API key

When you have the key, add it to a **project-root `.env` file** using this exact variable name:

```dotenv
TYPESAFE_API_KEY=your_typesafe_api_key_here
```

To create the ignored local file from the template:

```bash
cp .env.example .env
```

Then replace the placeholder value in `.env`. `.env` is excluded by `.gitignore`; do not put the key in notebooks, prediction files, or source code. The key is not needed to audit data, split, train classical models, or evaluate existing predictions.

Run a five-row schema/response pilot first (the selection includes both target classes):

```bash
uv run jev-risk jev-pilot --sample-size 5
```

Review `results/predictions/jev_pilot.csv` and `results/metadata/jev_pilot.json`. After a successful pilot, run Jev for all held-out rows:

```bash
uv run jev-risk jev
```

The pilot and full run use the default TypeSafe model alias `jev-latest`. Full inference checks that the successful pilot used the same dataset fingerprint, split seed, prompt version, and model alias, and that every response uses the concrete model version recorded by the pilot. If `jev-latest` moves to a new version during a run, use `--model` with a pinned version for both the pilot and full run. API responses are cached by transaction ID and input-state fingerprint in `results/cache/jev.jsonl`; rerunning resumes completed rows **without issuing new requests for cached rows**. Jev prints the cache count immediately, indicates when the first uncached request starts, and reports progress every 25 rows. A full run issues one request per uncached held-out row (2,000 rows total), so it can take several minutes. `results/predictions/jev.csv` is exported only when the full test set has completed; an incomplete rerun invalidates any previous full export. The SDK's retry behavior handles transient API errors; persistent failures stop the run without fabricating a prediction.

### Evaluate and export results

```bash
uv run jev-risk evaluate
```

Evaluation never makes a TypeSafe API request. If Jev has started but `jev.csv` is not yet complete, this command reports the partial cache and asks you to resume `uv run jev-risk jev`. The full comparison also verifies that Jev's exported dataset fingerprint matches local training; after updating this code, rerun `uv run jev-risk jev` to regenerate any older export without that fingerprint (completed matching rows are reused from cache). To evaluate only the local models while Jev is still running, opt in explicitly:

```bash
uv run jev-risk evaluate --local-only
```

The command reads available prediction CSVs, verifies identical test IDs and labels, and writes:

- `results/metrics/model_comparison.csv` — accuracy, precision, recall, F1, ROC-AUC, average precision, Brier score, and latency.
- `results/metrics/latency_summary.csv` — mean, median, and P95 latency.
- `results/metrics/bootstrap_intervals.csv` — seeded percentile intervals for model metrics and paired model differences from shared bootstrap samples.
- `results/metrics/confidence_thresholds.csv` — Jev coverage and metrics among accepted cases, when Jev predictions are available.
- `results/figures/` — confusion matrices, ROC, precision-recall, calibration, and (when Jev confidence is present) confidence/coverage plots.

`average_precision` is used as the reported PR-AUC summary. Accuracy should be interpreted alongside the 14.48% positive prevalence and the dummy baseline. The Jev latency measures a hosted API round trip, not just model compute.

### Interpreting the current Jev result

The TypeSafe Python SDK integration follows its documented `TypeSafeClient.system_one(state=..., questions={...: Choice(...)})` flow. Jev's selected `choice` supplies `y_pred`, `probabilities["risk_incident"]` supplies `y_prob`, and `confidence` remains a separate selective-classification score. The saved response cache, exported probabilities, sample IDs, and ground-truth labels agree; this benchmark does not have an inverted label or probability mapping.

The present test set is 14.5% positive, while Jev predicts the positive class for 71.9% of rows. Its ROC-AUC is 0.492 and average precision is 0.151 (near the 0.145 positive-class prevalence). RF and XGBoost also have ROC-AUC values close to chance (0.503 and 0.515). Jev's high recall (0.700) comes with many false positives and a Brier score of 0.459. Accuracy alone therefore understates the distinction between class imbalance, discrimination, and decision threshold.

The question currently asks whether each row *should* be flagged but does not define observable incident criteria. The CSV's incident type, severity, and error code are post-event labels, so they cannot be used to supply those criteria without target leakage. The available pre-event attributes may not explain the labels well enough for any model to generalize. Neither Jev nor the local baselines can infer an unknown label-generation policy from one row's attributes. Do not change the question wording, add leakage fields, or choose a new probability threshold on the held-out test set to improve reported scores. A revised Jev question needs explicit, independently justified risk criteria and a separately versioned experiment; changing the prompt invalidates the cached Jev responses and requires fresh paid inference.

## Repository map

```text
src/jev_risk/       Data validation, shared features/split, models, Jev, metrics, CLI
tests/              Unit and offline integration tests
notebooks/          EDA, split inspection, training, Jev pilot, evaluation
results/            Generated predictions, models, metadata, tables, and figures
accounting_dataset.csv  Supplied research dataset
```

Run the offline test suite with:

```bash
uv run pytest
```
