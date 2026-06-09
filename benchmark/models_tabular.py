"""Ridge, CatBoost, LightGBM trainers with shared objective mapping."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge

try:
    from catboost import CatBoostRegressor, Pool
except ImportError:
    CatBoostRegressor = None  # type: ignore
    Pool = None  # type: ignore

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore


def _mark_categories(df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in categorical_columns:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


def fit_predict_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray,
    use_huber: bool = False,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    if use_huber:
        model = HuberRegressor(max_iter=200)
    else:
        model = Ridge(alpha=1.0, random_state=random_state)
    model.fit(X_train, y_train)
    return model.predict(X_valid), model.predict(X_test), model


def catboost_loss(objective: str) -> str:
    o = objective.lower()
    if o == "mse" or o == "rmse":
        return "RMSE"
    if o == "mae":
        return "MAE"
    if o == "huber":
        return "Huber"
    if o == "quantile":
        return "Quantile:alpha=0.5"
    return "RMSE"


def lgbm_params(objective: str, quantile_alpha: float = 0.5) -> Dict[str, Any]:
    o = objective.lower()
    if o == "mse" or o == "rmse":
        return {"objective": "regression", "metric": "rmse"}
    if o == "mae":
        return {"objective": "mae", "metric": "mae"}
    if o == "huber":
        return {"objective": "huber", "metric": "rmse", "alpha": 0.9}
    if o == "quantile":
        return {"objective": "quantile", "metric": "quantile", "alpha": quantile_alpha}
    return {"objective": "regression", "metric": "rmse"}


def fit_predict_catboost(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_features: List[str],
    y_train: np.ndarray,
    y_valid: np.ndarray,
    objective: str = "mse",
    iterations: int = 2000,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
    quantile_alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    if CatBoostRegressor is None:
        raise ImportError("catboost is not installed. pip install catboost")

    if objective.lower() == "quantile":
        loss = f"Quantile:alpha={quantile_alpha}"
    else:
        loss = catboost_loss(objective)

    cat_features = [c for c in categorical_features if c in train_df.columns]
    train_pool = Pool(train_df, y_train, cat_features=cat_features)
    valid_pool = Pool(valid_df, y_valid, cat_features=cat_features)
    test_pool = Pool(test_df, cat_features=cat_features)

    model = CatBoostRegressor(
        loss_function=loss,
        iterations=iterations,
        random_seed=random_state,
        verbose=False,
        early_stopping_rounds=early_stopping_rounds,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model.predict(valid_df), model.predict(test_df), model


def fit_predict_lightgbm(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_features: List[str],
    y_train: np.ndarray,
    y_valid: np.ndarray,
    objective: str = "mse",
    iterations: int = 2000,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
    quantile_alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    if lgb is None:
        raise ImportError("lightgbm is not installed. pip install lightgbm")

    tr = _mark_categories(train_df, categorical_features)
    va = _mark_categories(valid_df, categorical_features)
    te = _mark_categories(test_df, categorical_features)

    cat_cols = [c for c in categorical_features if c in tr.columns]
    params = {
        "verbosity": -1,
        "seed": random_state,
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
    }
    p_extra = lgbm_params(objective, quantile_alpha)
    params.update(p_extra)

    cat_feat = cat_cols if cat_cols else None
    dtr = lgb.Dataset(tr, label=y_train, categorical_feature=cat_feat, free_raw_data=False)
    dva = lgb.Dataset(va, label=y_valid, categorical_feature=cat_feat, reference=dtr, free_raw_data=False)

    model = lgb.train(
        params,
        dtr,
        num_boost_round=iterations,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    num_iter = model.best_iteration
    if num_iter is None:
        num_iter = iterations
    pred_va = model.predict(va, num_iteration=num_iter)
    pred_te = model.predict(te, num_iteration=num_iter)
    return pred_va, pred_te, model


def fit_predict_quantile_triplet(
    backend: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_features: List[str],
    y_train: np.ndarray,
    y_valid: np.ndarray,
    random_state: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, Any]]:
    """Train three quantile models (0.1, 0.5, 0.9) for interval-style outputs."""
    alphas = [0.1, 0.5, 0.9]
    out: Dict[str, Tuple[np.ndarray, np.ndarray, Any]] = {}
    for a in alphas:
        if backend == "catboost":
            loss = f"Quantile:alpha={a}"
            if CatBoostRegressor is None:
                raise ImportError("catboost is not installed")
            cat_features = [c for c in categorical_features if c in train_df.columns]
            train_pool = Pool(train_df, y_train, cat_features=cat_features)
            valid_pool = Pool(valid_df, y_valid, cat_features=cat_features)
            model = CatBoostRegressor(
                loss_function=loss,
                iterations=2000,
                random_seed=random_state,
                verbose=False,
                early_stopping_rounds=50,
            )
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            key = f"q{a}"
            out[key] = (model.predict(valid_df), model.predict(test_df), model)
        else:
            if lgb is None:
                raise ImportError("lightgbm is not installed")
            tr = _mark_categories(train_df, categorical_features)
            va = _mark_categories(valid_df, categorical_features)
            te = _mark_categories(test_df, categorical_features)
            cat_cols = [c for c in categorical_features if c in tr.columns]
            params = {
                "objective": "quantile",
                "alpha": a,
                "metric": "quantile",
                "verbosity": -1,
                "seed": random_state,
                "num_leaves": 63,
                "learning_rate": 0.05,
            }
            cat_feat = cat_cols if cat_cols else None
            dtr = lgb.Dataset(tr, label=y_train, categorical_feature=cat_feat, free_raw_data=False)
            dva = lgb.Dataset(va, label=y_valid, categorical_feature=cat_feat, reference=dtr, free_raw_data=False)
            model = lgb.train(
                params,
                dtr,
                num_boost_round=2000,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            num_iter = model.best_iteration
            if num_iter is None:
                num_iter = 2000
            pred_va = model.predict(va, num_iteration=num_iter)
            pred_te = model.predict(te, num_iteration=num_iter)
            key = f"q{a}"
            out[key] = (pred_va, pred_te, model)
    return out
