# User guide — Debt Collection Portfolio Strategy Engine

## Introduction

The application predicts **expected paid value (BRUT)** for a portfolio you upload, using **gradient boosting models** (CatBoost + LightGBM blend) trained on historical collection data. It simulates **Low**, **Medium**, **High**, and **Historical** effort scenarios and shows results in a bar chart.

**BRUT** is the total predicted **Paid Value** (cash collected). Action costs can be configured for reference; the main chart focuses on gross predicted paid value.

The in-app help viewer loads this document from `user_guide.md`.

---

## Login modes

| Mode | Access |
|------|--------|
| **User** | Prediction tab, Training Insights (read-only), portfolio upload and simulation |
| **Admin** | Same as User, plus **Training** and **Testing** tabs to retrain and evaluate GBM models |

Credentials are stored in `users.json` (copy from `users.json.example` on first setup).

---

## Prediction tab

### 1. Upload portfolio data

- Use **Upload Portfolio Data** (Excel `.xlsx` / `.xls` or CSV).
- **Clear Data** resets the upload so you can load another file.
- Use your own sanitized portfolio export. The public demo includes synthetic sample files in `data/`; only train on real data if you have the right to use it.

**You do not need `*_per_week` columns in the file.** The app computes them from:

- `Import date` and `Date End` (case active period)
- Total actions: `Calls`, `Letters`, `SMS`, `Emails`

Only rows with **valid dates** and **positive duration** are used when averaging weekly rates.

### 2. Effort settings

| Control | Purpose |
|---------|---------|
| **Baseline** | **Default** — average weekly actions from the **uploaded** portfolio (per client when simulating). **Training Average** — global training-data baselines. **Custom** — you enter weekly-style values in the action fields. |
| **Calls / Letters / SMS / Emails** | Shown for Custom baseline, or read-only averages after upload when Baseline = Default. |
| **Change % (+/-)** | Spread between Low and High around Medium (e.g. 20% → Low ≈ −20%, High ≈ +20% from medium weekly rates). |
| **Target Paid Value** | Optional portfolio total; the app may derive an effective % change to approach that target on Medium. |

**Action costs** (Calls, Letters, SMS, Emails) are used for cost context; the bar chart shows predicted paid value, not net after costs.

### 3. Run strategy simulation

Click **Run Strategy Simulation**. For each **client** in the file (or one group if only one client), the app:

1. Builds **per-client** weekly baselines from uploaded cases.
2. Derives **Low / Medium / High** action totals per case (from weekly rates × typical case length for that client).
3. Runs the **paid-value GBM** with those action levels.
4. Runs a **Historical** scenario using training-derived suggested actions (may differ from the label averages).

### 4. Reading the chart

Four bars:

| Bar | Meaning |
|-----|---------|
| **Low** | Predicted total paid if every case used reduced effort. |
| **Medium** | Predicted total paid at baseline effort. |
| **High** | Predicted total paid at increased effort. |
| **Historical** | Predicted total paid using **suggested** actions from training efficiency analysis. |

**Labels under the x-axis**

- **Low / Medium / High**: actions per week (C=Calls, L=Letters, S=SMS, E=Emails), **rounded up** to whole numbers when fractional.
- **Historical**: **actual average weekly actions** from your uploaded slice (may show decimals, e.g. C:2.4).

**Above each predicted bar** (Low, Medium, High only):

- Top line: total predicted EUR.
- Second line: **recovery rate %** = predicted total ÷ sum of **Case Value** for the current filter.

**Historical** shows only the EUR total (no recovery % on that bar).

Use **Client** and **Import** filters to narrow the chart; totals and recovery % update for the filtered cases.

### 5. Important expectations

- **More effort does not always mean higher predicted paid.** The model learned from historical data where heavy contact often correlates with **harder** cases. The terminal may log `Non-monotonic totals observed` — that is informational, not a broken sum.
- **Historical label vs Historical height:** the label shows your upload’s average weekly actions; the bar height uses **suggested** action counts from training, which can differ.
- Compare predictions to actual **Paid Value** in the upload (if present) or to known collections for the period — totals around **75–85% of case value** are common depending on portfolio quality.

Results can be exported to Excel (`predictions_output_YYYY-MM-DD.xlsx`).

---

## Training Insights tab

- Statistics from **`data/train/train.xlsx`** (paid value, actions, weekly rates, effort buckets).
- Use **Refresh** after retraining or replacing training data.
- Helps interpret what the model saw during training vs what you simulate on a new upload.

---

## Training tab (Admin)

Retrains the **GBM stack** (not the default path for end users):

1. KMeans behavior clusters (optional extended features)
2. CatBoost median paid model
3. LightGBM paid model for high **Case Value** rows
4. CatBoost multi-output model for weekly actions

Progress is shown in the status area (no per-epoch loss curve for GBM).

Options:

- Train on existing splits in `data/train`, `data/valid`, `data/test`
- Or **generate a new split** from a raw file (filtering, weekly rates, train/valid/test split)

---

## Testing tab (Admin)

Evaluate the trained models on a file you select, or run:

```bash
python test.py
```

Metrics include MAE/RMSE on paid value and business tolerance bands by case-value buckets.

---

## Comparing predictions with / without cap (developers)

Paid predictions can be **capped per case** at `Case Value × ratio` (default 80%, aligned with training).

**Quick toggle (code only, not in GUI):**

1. Open `prediction_cap.py`
2. Set `USE_PAID_PREDICTION_CAP = False` to disable capping
3. Restart the app and re-run simulation

**Config** (`config.json` → `prediction_cap`):

```json
"prediction_cap": {
  "enabled": true,
  "per_case_max_paid_to_case_ratio": 0.8,
  "also_cap_at_case_value": false
}
```

Capping applies only when both `USE_PAID_PREDICTION_CAP` is `True` and `enabled` is `true`.

---

## Portfolio file checklist

Required for simulation (minimum):

- `Client`, `Location`, `Product` (as configured)
- `Case Value`, `DPD`, `Debtor Age`
- `Calls`, `Letters`, `SMS`, `Emails`
- `Import date`, `Date End`

Optional but useful:

- `Paid Value` — for comparing outcomes; dropped during prediction so the model does not “cheat”
- `Import Name` — for import filter in the chart

See **README.md** for full column list and validation rules.
