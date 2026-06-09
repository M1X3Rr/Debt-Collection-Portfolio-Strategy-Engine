# Tabular benchmark sandbox

Runs **Ridge**, **CatBoost**, and **LightGBM** on the **same** train/valid/test files as the main app (`config.json` → `data.train_path`, `valid_path`, `test_path`). Default mode matches the neural network **feature set** and **target scaling** (`split.fit_preprocessors`, `get_target_series`).

## Install

From the project root (with the main `requirements.txt` already installed):

```bash
pip install -r benchmark/requirements.txt
```

## Run

```bash
python -m benchmark.run --model all
python -m benchmark.run --model ridge --extended
python -m benchmark.run --model catboost --objective huber --stack
python -m benchmark.run --compare-nn
```

Use `--config path/to/config.json` if not in the cwd.

## Prediction point and leakage

- **Target**: `Paid Value` or `Paid Value / Case Value` when `target_transform.mode` is `paid_ratio`, then log-scaled like training.
- **Excluded from features** (see `columns.exclude_features` in `config.json`): e.g. `#`, `Case ID`, `Payment date`, `Import date`, `Import Name`, `Date End`.  
  - **Payment date** can leak outcome timing; keep excluded for “at import” style prediction.  
  - **Import date** / **Date End** are excluded as raw inputs because **weekly action rates** (`*_per_week`) already encode case duration from those dates in `split.prepare_splits`.
- **Name** is not in the default feature list; do not add it unless you have a clear use (e.g. linkage only outside this benchmark).

## Parity vs `--extended`

| Mode | Features |
|------|-----------|
| Default | Same as `PortfolioDataset`: categoricals + scaled numerics (Case Value, DPD, Debtor Age, actions, weekly rates). |
| `--extended` | Adds `days_active`, `import_batch_id`, and `behavior_cluster` (KMeans fit on **train** numerics+actions only). |

## Splits and grouped evaluation

Benchmarks **do not re-split** data. If you ran `split.py` with `split_by_import: true`, `(Client, Import date)` groups stay in one fold; using the saved `train.xlsx` / `valid.xlsx` / `test.xlsx` preserves that structure.

## Outputs

`--output-dir` (default `benchmark/outputs`) receives JSON metrics and optional CSV summaries.
