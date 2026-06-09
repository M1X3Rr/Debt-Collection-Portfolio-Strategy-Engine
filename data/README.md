# Data directory

This public repository ships **synthetic demo data only**. Real client portfolios, trained models, and company-specific exports are intentionally excluded.

| Path | Purpose |
|------|---------|
| `train/train.xlsx` | Synthetic training split |
| `valid/valid.xlsx` | Synthetic validation split |
| `test/test.xlsx` | Synthetic test split |
| `sample_portfolio.xlsx` | Small upload example for the Prediction tab |

Regenerate all demo files:

```bash
python scripts/generate_sample_data.py
```

After replacing data, retrain models from the **Training** tab (admin) or run your training pipeline so `models/` matches the new splits.
