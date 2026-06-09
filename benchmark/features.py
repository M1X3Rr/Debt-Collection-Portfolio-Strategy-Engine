"""
Load train/valid/test splits and build feature matrices aligned with PortfolioDataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from split import (  # noqa: E402
    fit_preprocessors,
    get_target_series,
    load_config,
    load_table,
)


def _resolve_path(config_path: str, p: str) -> str:
    cfg_dir = Path(config_path).resolve().parent
    path = Path(p)
    if not path.is_absolute():
        path = cfg_dir / path
    return str(path)


def load_splits(config: dict, config_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_cfg = config["data"]
    train_path = _resolve_path(config_path, data_cfg["train_path"])
    valid_path = _resolve_path(config_path, data_cfg["valid_path"])
    test_path = _resolve_path(config_path, data_cfg["test_path"])
    return load_table(train_path), load_table(valid_path), load_table(test_path)


def _feature_column_lists(config: dict) -> Tuple[List[str], List[str]]:
    cols_cfg = config["columns"]
    exclude = set(cols_cfg.get("exclude_features", []))
    cats = [c for c in cols_cfg["categorical_features"] if c not in exclude]
    weekly = cols_cfg.get("weekly_action_features", [])
    nums = [
        c
        for c in (
            cols_cfg["numerical_features"]
            + cols_cfg.get("action_features", [])
            + weekly
        )
        if c not in exclude
    ]
    return cats, nums


def _scaled_target_array(df: pd.DataFrame, config: dict, target_scaler) -> np.ndarray:
    ts = get_target_series(df, config).astype(np.float64).values.reshape(-1, 1)
    ts = np.nan_to_num(ts, nan=0.0)
    return target_scaler.transform(ts).astype(np.float64).ravel()


@dataclass
class BenchmarkData:
    config: dict
    config_path: str
    encoder: object
    scaler: object
    target_scaler: object
    categorical_features: List[str]
    numeric_features: List[str]
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame
    extended_cols: List[str]


def _days_active_series(df: pd.DataFrame, config: dict) -> pd.Series:
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


def _import_batch_id_series(df: pd.DataFrame, config: dict) -> pd.Series:
    cols_cfg = config["columns"]
    date_cfg = cols_cfg.get("date_features", {})
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


def _kmeans_cluster(
    train_num: np.ndarray,
    valid_num: np.ndarray,
    test_num: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    tr = km.fit_predict(train_num)
    va = km.predict(valid_num)
    te = km.predict(test_num)
    return tr, va, te


def prepare_benchmark_data(
    config_path: str = "config.json",
    extended: bool = False,
    n_clusters: int = 8,
    random_state: int = 42,
) -> BenchmarkData:
    config = load_config(config_path)
    train_df, valid_df, test_df = load_splits(config, config_path)
    encoder, scaler, target_scaler = fit_preprocessors(train_df, config)

    cats, nums = _feature_column_lists(config)
    if hasattr(scaler, "feature_names_in_"):
        nums = list(scaler.feature_names_in_)

    extended_cols: List[str] = []
    if extended:
        train_df = train_df.copy()
        valid_df = valid_df.copy()
        test_df = test_df.copy()
        for name, series_fn in [
            ("days_active", lambda d: _days_active_series(d, config)),
            ("import_batch_id", lambda d: _import_batch_id_series(d, config)),
        ]:
            train_df[name] = series_fn(train_df)
            valid_df[name] = series_fn(valid_df)
            test_df[name] = series_fn(test_df)
            extended_cols.append(name)

        # KMeans on train-only numeric block (parity numerics as raw floats before scaling)
        num_block_cols = [
            c
            for c in (
                config["columns"]["numerical_features"]
                + config["columns"].get("action_features", [])
                + config["columns"].get("weekly_action_features", [])
            )
            if c in train_df.columns
        ]
        if num_block_cols:
            tr_raw = train_df[num_block_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
            va_raw = valid_df[num_block_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
            te_raw = test_df[num_block_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
            tr_c, va_c, te_c = _kmeans_cluster(tr_raw, va_raw, te_raw, n_clusters, random_state)
            cname = "behavior_cluster"
            train_df[cname] = tr_c
            valid_df[cname] = va_c
            test_df[cname] = te_c
            extended_cols.append(cname)

    return BenchmarkData(
        config=config,
        config_path=config_path,
        encoder=encoder,
        scaler=scaler,
        target_scaler=target_scaler,
        categorical_features=cats,
        numeric_features=nums,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        extended_cols=extended_cols,
    )


def scaled_numeric_matrix(df: pd.DataFrame, scaler, numeric_features: List[str]) -> np.ndarray:
    cols = list(numeric_features)
    work = df
    for col in cols:
        if col not in work.columns:
            work = work.copy()
            work[col] = 0
    return scaler.transform(work[cols]).astype(np.float64)


def build_tree_frames(
    data: BenchmarkData,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """DataFrames for CatBoost/LightGBM: string categoricals + float numerics (+ extended)."""
    cats = data.categorical_features
    nums = list(data.numeric_features)
    ext = data.extended_cols

    def _one(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in cats:
            out[c] = df[c].astype(str) if c in df.columns else "Unknown"
        for c in nums:
            out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0) if c in df.columns else 0.0
        for c in ext:
            out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        return out

    return _one(data.train_df), _one(data.valid_df), _one(data.test_df)


def tree_categorical_indices(df: pd.DataFrame, categorical_features: List[str]) -> List[int]:
    cols = list(df.columns)
    return [cols.index(c) for c in categorical_features if c in cols]


def build_ridge_matrix(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data: BenchmarkData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """One-hot categoricals (train categories) + scaled numerics + extended as numeric."""
    cats = data.categorical_features
    nums = list(data.numeric_features)
    ext = data.extended_cols

    def _dummies_fit(train: pd.DataFrame, other: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        d_tr = pd.get_dummies(train[cats].astype(str), prefix=cats, dummy_na=False) if cats else pd.DataFrame(index=train.index)
        d_o = pd.get_dummies(other[cats].astype(str), prefix=cats, dummy_na=False) if cats else pd.DataFrame(index=other.index)
        d_o = d_o.reindex(columns=d_tr.columns, fill_value=0)
        return d_tr, d_o

    d_train, d_valid = _dummies_fit(train_df, valid_df)
    _, d_test = _dummies_fit(train_df, test_df)

    num_tr = scaled_numeric_matrix(train_df, data.scaler, nums)
    num_va = scaled_numeric_matrix(valid_df, data.scaler, nums)
    num_te = scaled_numeric_matrix(test_df, data.scaler, nums)

    parts_tr = [d_train.values, num_tr]
    parts_va = [d_valid.values, num_va]
    parts_te = [d_test.values, num_te]
    feature_names = list(d_train.columns) + nums

    for c in ext:
        v_tr = pd.to_numeric(train_df[c], errors="coerce").fillna(0).values.reshape(-1, 1)
        v_va = pd.to_numeric(valid_df[c], errors="coerce").fillna(0).values.reshape(-1, 1)
        v_te = pd.to_numeric(test_df[c], errors="coerce").fillna(0).values.reshape(-1, 1)
        parts_tr.append(v_tr)
        parts_va.append(v_va)
        parts_te.append(v_te)
        feature_names.append(c)

    X_train = np.hstack(parts_tr)
    X_valid = np.hstack(parts_va)
    X_test = np.hstack(parts_te)
    return X_train, X_valid, X_test, feature_names


def targets_scaled(data: BenchmarkData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_tr = _scaled_target_array(data.train_df, data.config, data.target_scaler)
    y_va = _scaled_target_array(data.valid_df, data.config, data.target_scaler)
    y_te = _scaled_target_array(data.test_df, data.config, data.target_scaler)
    return y_tr, y_va, y_te
