import os
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

from gbm_inference import load_paid_bundle, predict_paid_value_blended
from train import load_config


def load_table_test(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".csv"):
        try:
            return pd.read_csv(path, encoding="utf-8")
        except (UnicodeDecodeError, UnicodeError):
            return pd.read_csv(path, encoding="cp1252")
    return pd.read_excel(path)


def load_artifacts() -> Tuple[dict, pd.DataFrame, object, object, object, object]:
    """
    Load configuration, test data, preprocessing artifacts, and GBM bundle.
    """
    config = load_config()
    data_cfg = config["data"]
    artifacts_dir = data_cfg["artifacts_dir"]

    test_df = load_table_test(data_cfg["test_path"])

    encoder = joblib.load(os.path.join(artifacts_dir, "encoder.pkl"))
    scaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))
    target_scaler = joblib.load(os.path.join(artifacts_dir, "target_scaler.pkl"))

    bundle = load_paid_bundle(config)
    return config, test_df, encoder, scaler, target_scaler, bundle


def _case_value_bucket(
    cv: float,
    q50: float,
    q75: float,
    q90: float,
) -> str:
    if cv <= q50:
        return "≤p50"
    if cv <= q75:
        return "p50–p75"
    if cv <= q90:
        return "p75–p90"
    return ">p90"


def _business_metric_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    case_values: np.ndarray,
    routing: dict,
    tolerance_pct: float,
) -> None:
    qmap = routing.get("case_value_quantiles_train", {})
    q50 = float(qmap.get("p50", np.nan))
    q75 = float(qmap.get("p75", np.nan))
    q90 = float(qmap.get("p90", np.nan))
    if not np.isfinite(q50):
        print("Business metrics: no train quantiles in blend_routing; skipping buckets.")
        return

    thr = float(routing.get("case_value_threshold", 0.0))
    high_mask = case_values >= thr
    print(
        f"Routing diagnostics: {100.0 * high_mask.mean():.1f}% of test rows used extended-LightGBM path (CV >= {thr:.2f})."
    )

    tol = tolerance_pct / 100.0
    eps = 1e-9
    rel_err = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps)
    within = rel_err <= tol

    buckets: Dict[str, List[bool]] = {
        "≤p50": [],
        "p50–p75": [],
        "p75–p90": [],
        ">p90": [],
    }
    for cv, ok in zip(case_values, within):
        b = _case_value_bucket(float(cv), q50, q75, q90)
        buckets[b].append(bool(ok))

    print(f"\nWithin ±{tolerance_pct}% of actual (by train Case Value bucket):")
    for name, flags in buckets.items():
        if not flags:
            print(f"  {name}: (no rows)")
            continue
        print(f"  {name}: {100.0 * np.mean(flags):.1f}%  (n={len(flags)})")


def evaluate_on_test() -> None:
    """
    Evaluate GBM blended models on the test dataset: MAE/RMSE + business metrics + plot.
    """
    config, test_df, encoder, scaler, target_scaler, bundle = load_artifacts()

    cols_cfg = config["columns"]
    data_cfg = config["data"]
    plot_dir = "data/plots"
    target_col = cols_cfg["target_column"]
    gbm_cfg = config.get("gbm", {})
    paid_blend = gbm_cfg.get("paid_blend", {})
    tolerance_pct = float(paid_blend.get("relative_error_tolerance_pct", 20))

    manifest, cat_model, lgb_booster, kmeans, kmeans_cols, routing = bundle

    if target_col not in test_df.columns:
        raise ValueError(f"Test data missing target column '{target_col}'.")

    y_true = pd.to_numeric(test_df[target_col], errors="coerce").fillna(0.0).values
    y_pred = predict_paid_value_blended(
        test_df,
        config,
        encoder,
        scaler,
        target_scaler,
        manifest,
        cat_model,
        lgb_booster,
        kmeans,
        kmeans_cols,
        routing,
    )

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5

    print(f"Test MAE:  {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    cv_col = str(routing.get("case_value_column", "Case Value"))
    if cv_col in test_df.columns:
        case_values = pd.to_numeric(test_df[cv_col], errors="coerce").fillna(0.0).values
    else:
        case_values = np.zeros(len(test_df), dtype=float)
    _business_metric_table(y_true, y_pred, case_values, routing, tolerance_pct)

    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "actual_vs_predicted.png")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
    plt.xlabel("Actual Paid Value")
    plt.ylabel("Predicted Paid Value")
    plt.title("Actual vs Predicted Paid Value (Test Set, GBM)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Scatter plot saved to: {plot_path}")


if __name__ == "__main__":
    evaluate_on_test()
