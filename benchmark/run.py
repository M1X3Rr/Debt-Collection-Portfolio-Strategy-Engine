"""
CLI: benchmark Ridge, CatBoost, LightGBM on saved train/valid/test splits.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.features import (  # noqa: E402
    build_ridge_matrix,
    build_tree_frames,
    prepare_benchmark_data,
    targets_scaled,
)
from benchmark.metrics import inverse_targets, pinball_loss, regression_metrics  # noqa: E402
from benchmark.models_tabular import (  # noqa: E402
    fit_predict_catboost,
    fit_predict_lightgbm,
    fit_predict_quantile_triplet,
    fit_predict_ridge,
)
from benchmark.optional_nn_compare import print_nn_compare  # noqa: E402
from benchmark.stacking import stack_predictions  # noqa: E402


def _ensure_output_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _report(
    name: str,
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    target_scaler,
) -> Dict[str, Any]:
    y_true = inverse_targets(y_true_scaled, target_scaler)
    y_pred = inverse_targets(y_pred_scaled, target_scaler)
    m = regression_metrics(y_true, y_pred)
    m["model"] = name
    print(f"{name} test MAE:  {m['mae']:.6f}")
    print(f"{name} test RMSE: {m['rmse']:.6f}")
    return m


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    data = prepare_benchmark_data(
        config_path=args.config,
        extended=args.extended,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
    )
    y_tr, y_va, y_te = targets_scaled(data)

    tr_df, va_df, te_df = build_tree_frames(data)
    cat_cols = [c for c in data.categorical_features if c in tr_df.columns]
    X_tr, X_va, X_te, _ = build_ridge_matrix(
        data.train_df, data.valid_df, data.test_df, data
    )

    results: Dict[str, Any] = {
        "config": args.config,
        "extended": args.extended,
        "objective": args.objective,
    }
    out_dir = _ensure_output_dir(args.output_dir)
    all_metrics: List[Dict[str, Any]] = []

    if args.compare_nn:
        print_nn_compare(args.config, output_dir=out_dir)

    if args.quantile:
        backend = args.quantile_backend
        qmodels = fit_predict_quantile_triplet(
            backend,
            tr_df,
            va_df,
            te_df,
            cat_cols,
            y_tr,
            y_va,
            random_state=args.random_state,
        )
        y_true_orig = inverse_targets(y_te, data.target_scaler)
        for key, (pred_va, pred_te, _) in qmodels.items():
            y_pred_orig = inverse_targets(pred_te, data.target_scaler)
            m = regression_metrics(y_true_orig, y_pred_orig)
            m["model"] = f"{backend}_{key}"
            if key == "q0.5":
                print(f"{backend} {key} test MAE:  {m['mae']:.6f}")
                print(f"{backend} {key} test RMSE: {m['rmse']:.6f}")
            all_metrics.append(m)
        p1_te = inverse_targets(qmodels["q0.1"][1], data.target_scaler)
        p9_te = inverse_targets(qmodels["q0.9"][1], data.target_scaler)
        pin_1 = pinball_loss(y_true_orig, p1_te, 0.1)
        pin_9 = pinball_loss(y_true_orig, p9_te, 0.9)
        results["pinball_q0.1"] = pin_1
        results["pinball_q0.9"] = pin_9
        print(f"Pinball loss (alpha=0.1): {pin_1:.6f}")
        print(f"Pinball loss (alpha=0.9): {pin_9:.6f}")
        results["metrics"] = all_metrics
        with open(out_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return results

    models = []
    if args.model == "all":
        models = ["ridge", "catboost", "lightgbm"]
    else:
        models = [args.model]

    if args.stack:
        pred_r_va, pred_r_te, ridge_m = fit_predict_ridge(
            X_tr,
            y_tr,
            X_va,
            X_te,
            use_huber=args.ridge_huber,
            random_state=args.random_state,
        )
        if args.stack_tree == "catboost":
            pred_t_va, pred_t_te, tree_m = fit_predict_catboost(
                tr_df,
                va_df,
                te_df,
                cat_cols,
                y_tr,
                y_va,
                objective=args.objective,
                iterations=args.iterations,
                early_stopping_rounds=args.early_stopping,
                random_state=args.random_state,
            )
        else:
            pred_t_va, pred_t_te, tree_m = fit_predict_lightgbm(
                tr_df,
                va_df,
                te_df,
                cat_cols,
                y_tr,
                y_va,
                objective=args.objective,
                iterations=args.iterations,
                early_stopping_rounds=args.early_stopping,
                random_state=args.random_state,
            )
        pred_st_te, meta = stack_predictions(
            pred_r_va,
            pred_r_te,
            pred_t_va,
            pred_t_te,
            y_va,
        )
        m = _report("stack_ridge_" + args.stack_tree, y_te, pred_st_te, data.target_scaler)
        all_metrics.append(m)
        results["metrics"] = all_metrics
        with open(out_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return results

    for mname in models:
        if mname == "ridge":
            _, pred_te, _ = fit_predict_ridge(
                X_tr,
                y_tr,
                X_va,
                X_te,
                use_huber=args.ridge_huber,
                random_state=args.random_state,
            )
            all_metrics.append(_report("ridge", y_te, pred_te, data.target_scaler))
        elif mname == "catboost":
            _, pred_te, _ = fit_predict_catboost(
                tr_df,
                va_df,
                te_df,
                cat_cols,
                y_tr,
                y_va,
                objective=args.objective,
                iterations=args.iterations,
                early_stopping_rounds=args.early_stopping,
                random_state=args.random_state,
            )
            all_metrics.append(_report("catboost", y_te, pred_te, data.target_scaler))
        elif mname == "lightgbm":
            _, pred_te, _ = fit_predict_lightgbm(
                tr_df,
                va_df,
                te_df,
                cat_cols,
                y_tr,
                y_va,
                objective=args.objective,
                iterations=args.iterations,
                early_stopping_rounds=args.early_stopping,
                random_state=args.random_state,
            )
            all_metrics.append(_report("lightgbm", y_te, pred_te, data.target_scaler))

    results["metrics"] = all_metrics
    with open(out_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tabular benchmark vs NN parity splits.")
    p.add_argument("--config", default="config.json", help="Path to config.json")
    p.add_argument(
        "--model",
        choices=["ridge", "catboost", "lightgbm", "all"],
        default="all",
    )
    p.add_argument("--extended", action="store_true", help="Add days_active, import_batch_id, behavior_cluster")
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument(
        "--objective",
        choices=["mse", "mae", "huber", "quantile"],
        default="mse",
        help="Tree objective (Ridge ignores except --ridge-huber)",
    )
    p.add_argument(
        "--quantile",
        action="store_true",
        help="Train quantile models at 0.1 / 0.5 / 0.9 (uses --quantile-backend)",
    )
    p.add_argument(
        "--quantile-backend",
        choices=["catboost", "lightgbm"],
        default="lightgbm",
    )
    p.add_argument(
        "--stack",
        action="store_true",
        help="Stack Ridge + tree (see --stack-tree); meta fit on validation",
    )
    p.add_argument(
        "--stack-tree",
        choices=["catboost", "lightgbm"],
        default="catboost",
    )
    p.add_argument("--ridge-huber", action="store_true", help="Use HuberRegressor instead of Ridge")
    p.add_argument("--compare-nn", action="store_true", help="Print NN test metrics if model.pth exists")
    p.add_argument("--output-dir", default="benchmark/outputs")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--early-stopping", type=int, default=50)
    p.add_argument("--random-state", type=int, default=42)
    return p


def main() -> None:
    args = build_parser().parse_args()
    os.chdir(_ROOT)
    run_benchmark(args)


if __name__ == "__main__":
    main()
