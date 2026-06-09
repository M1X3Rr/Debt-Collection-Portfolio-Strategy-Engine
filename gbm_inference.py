"""
Shared GBM feature construction and blended paid-value inference (production).
Aligns with benchmark parity + extended frames; uses split.py preprocessors.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from split import get_target_series


def feature_column_lists(config: dict) -> Tuple[List[str], List[str]]:
    cols_cfg = config["columns"]
    exclude = set(cols_cfg.get("exclude_features", []))
    cats = [c for c in cols_cfg["categorical_features"] if c not in exclude]
    nums = [
        c
        for c in (
            cols_cfg["numerical_features"]
            + cols_cfg.get("action_features", [])
            + cols_cfg.get("weekly_action_features", [])
        )
        if c not in exclude
    ]
    return cats, nums


def apply_product_placeholder(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Mirror split._normalize_product_placeholder logic for inference rows."""
    cols_cfg = config.get("columns", {})
    product_col = cols_cfg.get("product_column", "Product")
    name_col = cols_cfg.get("debtor_name_column", "Name")
    placeholder = cols_cfg.get("product_missing_placeholder", "Unknown_Product")
    if product_col not in df.columns:
        return df
    out = df.copy()
    prod = out[product_col]
    prod_str = prod.astype(str).str.strip()
    missing_mask = prod.isna() | (prod_str == "") | (prod_str.str.lower() == "nan")
    if name_col in out.columns:
        name_str = out[name_col].astype(str).str.strip()
        missing_mask = missing_mask | (prod_str == name_str)
    if missing_mask.any():
        out.loc[missing_mask, product_col] = placeholder
    return out


def days_active_series(df: pd.DataFrame, config: dict) -> pd.Series:
    cols_cfg = config["columns"]
    date_cfg = cols_cfg.get("date_features", {})
    imp = date_cfg.get("import_date", "Import date")
    end = date_cfg.get("end_date", "Date End")
    if imp not in df.columns or end not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    a = pd.to_datetime(df[imp], errors="coerce")
    b = pd.to_datetime(df[end], errors="coerce")
    days = (b - a).dt.days
    return days.fillna(0).clip(lower=0).astype(float)


def import_batch_id_series(df: pd.DataFrame, config: dict) -> pd.Series:
    date_cfg = config["columns"].get("date_features", {})
    imp = date_cfg.get("import_date", "Import date")
    client_col = "Client"
    if imp not in df.columns or client_col not in df.columns:
        return pd.Series(0, index=df.index, dtype=int)
    key = (
        df[client_col].astype(str).str.strip()
        + "|"
        + pd.to_datetime(df[imp], errors="coerce").astype(str)
    )
    return pd.factorize(key)[0].astype(int)


def kmeans_behavior_block_columns(config: dict) -> List[str]:
    cols_cfg = config["columns"]
    return [
        c
        for c in (
            cols_cfg["numerical_features"]
            + cols_cfg.get("action_features", [])
            + cols_cfg.get("weekly_action_features", [])
        )
        if c not in cols_cfg.get("exclude_features", [])
    ]


def fit_behavior_kmeans(train_df: pd.DataFrame, config: dict, n_clusters: int, random_state: int) -> Tuple[KMeans, List[str]]:
    block_cols = [c for c in kmeans_behavior_block_columns(config) if c in train_df.columns]
    if not block_cols:
        raise ValueError("No numeric columns available for behavior KMeans.")
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    tr_raw = train_df[block_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    km.fit(tr_raw)
    return km, block_cols


def assign_behavior_cluster(df: pd.DataFrame, kmeans: KMeans, block_cols: List[str]) -> np.ndarray:
    full = np.zeros((len(df), len(block_cols)), dtype=float)
    for j, c in enumerate(block_cols):
        if c in df.columns:
            full[:, j] = pd.to_numeric(df[c], errors="coerce").fillna(0).values
    return kmeans.predict(full)


def add_extended_columns(
    df: pd.DataFrame,
    config: dict,
    kmeans: Optional[KMeans] = None,
    kmeans_block_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    out = df.copy()
    out["days_active"] = days_active_series(out, config)
    out["import_batch_id"] = import_batch_id_series(out, config)
    if kmeans is not None and kmeans_block_cols is not None:
        out["behavior_cluster"] = assign_behavior_cluster(out, kmeans, kmeans_block_cols)
    else:
        out["behavior_cluster"] = 0
    return out


EXTENDED_COLS = ["days_active", "import_batch_id", "behavior_cluster"]


def build_parity_tree_frame(df: pd.DataFrame, config: dict, categorical_features: List[str], numeric_features: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in categorical_features:
        out[c] = df[c].astype(str) if c in df.columns else "Unknown"
    nums = list(numeric_features)
    for c in nums:
        out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0) if c in df.columns else 0.0
    return out


def build_extended_tree_frame(
    df: pd.DataFrame,
    config: dict,
    categorical_features: List[str],
    numeric_features: List[str],
    kmeans: KMeans,
    kmeans_block_cols: List[str],
) -> pd.DataFrame:
    work = add_extended_columns(df, config, kmeans, kmeans_block_cols)
    base = build_parity_tree_frame(work, config, categorical_features, numeric_features)
    for c in EXTENDED_COLS:
        base[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)
    return base


def catboost_cat_features(df: pd.DataFrame, categorical_features: List[str]) -> List[str]:
    return [c for c in categorical_features if c in df.columns]


def mark_lgbm_categories(df: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in categorical_features:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


def scaled_target_vector(df: pd.DataFrame, config: dict, target_scaler) -> np.ndarray:
    ts = get_target_series(df, config).astype(np.float64).values.reshape(-1, 1)
    ts = np.nan_to_num(ts, nan=0.0)
    return target_scaler.transform(ts).astype(np.float64).ravel()


def inverse_paid_prediction(scaled: np.ndarray, target_scaler, config: dict, case_values: Optional[np.ndarray] = None) -> np.ndarray:
    """Inverse target scaler; if paid_ratio mode, multiply by case value."""
    arr = np.asarray(scaled, dtype=np.float64).reshape(-1, 1)
    core = target_scaler.inverse_transform(arr).reshape(-1)
    tt_cfg = config.get("target_transform", {}) if isinstance(config, dict) else {}
    tt_mode = str(tt_cfg.get("mode", "paid_value")).lower()
    if tt_mode == "paid_ratio" and case_values is not None:
        cv = np.maximum(np.asarray(case_values, dtype=np.float64), 0.0)
        return np.maximum(core * cv, 0.0)
    return np.maximum(core, 0.0)


# Artifact filenames
MANIFEST_NAME = "model_manifest.json"
CATBOOST_PAID_NAME = "catboost_paid.cbm"
LIGHTGBM_PAID_NAME = "lightgbm_paid.txt"
BLEND_ROUTING_NAME = "blend_routing.json"
BEHAVIOR_KMEANS_NAME = "behavior_kmeans.pkl"
ACTION_GBM_NAME = "action_gbm.cbm"


def artifacts_dir_from_config(config: dict) -> str:
    return config["data"]["artifacts_dir"]


def load_manifest(config: dict) -> dict:
    path = os.path.join(artifacts_dir_from_config(config), MANIFEST_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"GBM manifest not found at {path}. Run training (train.py) after split."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_blend_routing(config: dict) -> dict:
    path = os.path.join(artifacts_dir_from_config(config), BLEND_ROUTING_NAME)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_paid_bundle(config: dict):
    from catboost import CatBoostRegressor
    import lightgbm as lgb

    art = artifacts_dir_from_config(config)
    manifest = load_manifest(config)
    cat_path = os.path.join(art, manifest["paid"]["catboost_path"])
    lgb_path = os.path.join(art, manifest["paid"]["lightgbm_path"])
    cat_model = CatBoostRegressor()
    cat_model.load_model(cat_path)
    lgb_booster = lgb.Booster(model_file=lgb_path)
    kmeans_path = os.path.join(art, manifest["paid"]["behavior_kmeans_path"])
    kmeans = joblib.load(kmeans_path)
    kmeans_cols = manifest["paid"]["behavior_kmeans_columns"]
    routing = load_blend_routing(config)
    return manifest, cat_model, lgb_booster, kmeans, kmeans_cols, routing


def predict_paid_value_blended(
    df: pd.DataFrame,
    config: dict,
    encoder,
    scaler,
    target_scaler,
    manifest: dict,
    cat_model,
    lgb_booster,
    kmeans: KMeans,
    kmeans_cols: List[str],
    routing: dict,
) -> np.ndarray:
    """
    Vectorized blended prediction: CatBoost q50 on parity; LightGBM extended when Case Value >= threshold.
    """
    work = apply_product_placeholder(df, config)
    cats = manifest["paid"]["categorical_features"]
    nums = manifest["paid"]["parity_numeric_features"]

    parity = build_parity_tree_frame(work, config, cats, nums)
    cat_feat = catboost_cat_features(parity, cats)
    pred_cat = cat_model.predict(parity)

    ext_df = build_extended_tree_frame(work, config, cats, nums, kmeans, kmeans_cols)
    ext_lgb = mark_lgbm_categories(ext_df, cats)
    cat_cols_lgb = [c for c in cats if c in ext_lgb.columns]
    num_iter = getattr(lgb_booster, "best_iteration", None)
    if num_iter is None:
        num_iter = lgb_booster.num_trees()
    pred_lgb = lgb_booster.predict(ext_lgb, num_iteration=num_iter)

    cv_col = routing.get("case_value_column", "Case Value")
    thr = float(routing["case_value_threshold"])
    if cv_col not in work.columns:
        cv = np.zeros(len(work), dtype=float)
    else:
        cv = pd.to_numeric(work[cv_col], errors="coerce").fillna(0.0).values
    high = cv >= thr
    pred_scaled = np.where(high, pred_lgb, pred_cat).astype(np.float64)
    return inverse_paid_prediction(pred_scaled, target_scaler, config, case_values=cv)


def predict_paid_value_blended_from_bundle(df: pd.DataFrame, config: dict, encoder, scaler, target_scaler, bundle) -> np.ndarray:
    manifest, cat_model, lgb_booster, kmeans, kmeans_cols, routing = bundle
    return predict_paid_value_blended(
        df, config, encoder, scaler, target_scaler,
        manifest, cat_model, lgb_booster, kmeans, kmeans_cols, routing,
    )


def row_dict_to_parity_frame(
    debtor_data: dict,
    config: dict,
    cats: List[str],
    nums: List[str],
) -> pd.DataFrame:
    return build_parity_tree_frame(pd.DataFrame([debtor_data]), config, cats, nums)


def case_value_quantiles_train(train_df: pd.DataFrame, col: str, quantiles: List[float]) -> Dict[str, float]:
    if col not in train_df.columns:
        return {f"p{int(q * 100)}": 0.0 for q in quantiles}
    s = pd.to_numeric(train_df[col], errors="coerce").dropna()
    return {f"p{int(q * 100)}": float(s.quantile(q)) for q in quantiles}


def load_action_gbm(config: dict):
    from catboost import CatBoostRegressor

    art = artifacts_dir_from_config(config)
    manifest = load_manifest(config)
    ap = manifest.get("action", {})
    rel = ap.get("model_path")
    if not rel:
        raise FileNotFoundError("No action GBM in manifest.")
    path = os.path.join(art, rel)
    model = CatBoostRegressor()
    model.load_model(path)
    return model, ap


def predict_weekly_actions_gbm(df: pd.DataFrame, config: dict, encoder, scaler) -> pd.DataFrame:
    """MultiRMSE CatBoost: same cat+num prep as legacy action NN path."""
    from catboost import Pool

    cols_cfg = config["columns"]
    weekly_cfg: List[str] = cols_cfg.get("weekly_action_features", [])
    try:
        manifest = load_manifest(config)
        ap = manifest.get("action", {})
        if not ap.get("model_path"):
            return pd.DataFrame(0.0, index=df.index, columns=weekly_cfg)
        model, ap = load_action_gbm(config)
    except FileNotFoundError:
        return pd.DataFrame(0.0, index=df.index, columns=weekly_cfg)
    weekly_feats: List[str] = ap["weekly_targets"]
    cols_cfg = config["columns"]
    work = apply_product_placeholder(df.copy(), config)

    cat_features = [
        c for c in cols_cfg["categorical_features"] if c not in cols_cfg.get("exclude_features", [])
    ]
    if cat_features:
        for col in cat_features:
            if col not in work.columns:
                work[col] = "Unknown"
            work[col] = work[col].fillna("Unknown").astype(str)

    if hasattr(scaler, "feature_names_in_"):
        numeric_cols = list(scaler.feature_names_in_)
    else:
        numeric_cols = list(
            cols_cfg["numerical_features"]
            + cols_cfg.get("action_features", [])
            + weekly_feats
        )
    for col in numeric_cols:
        if col not in work.columns:
            work[col] = 0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)

    feat_frame = build_parity_tree_frame(work, config, cat_features, numeric_cols)
    cat_feat = catboost_cat_features(feat_frame, cat_features)
    pool = Pool(feat_frame, cat_features=cat_feat)
    preds = model.predict(pool)
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)
    preds = np.maximum(preds, 0.0)
    if preds.shape[1] != len(weekly_feats):
        if preds.shape[1] > len(weekly_feats):
            preds = preds[:, : len(weekly_feats)]
        else:
            pad = np.zeros((preds.shape[0], len(weekly_feats) - preds.shape[1]))
            preds = np.hstack([preds, pad])
    return pd.DataFrame(preds, index=df.index, columns=weekly_feats)
