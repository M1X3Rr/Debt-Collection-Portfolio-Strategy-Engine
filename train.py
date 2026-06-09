import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Import LogStandardScaler and helpers so joblib can unpickle the scaler
# artifacts and training can align with the configured target transform.
from split import LogStandardScaler, get_target_series  # noqa: F401

from catboost import CatBoostRegressor, Pool
import lightgbm as lgb

from gbm_inference import (
    ACTION_GBM_NAME,
    BEHAVIOR_KMEANS_NAME,
    BLEND_ROUTING_NAME,
    CATBOOST_PAID_NAME,
    EXTENDED_COLS,
    LIGHTGBM_PAID_NAME,
    MANIFEST_NAME,
    apply_product_placeholder,
    build_extended_tree_frame,
    build_parity_tree_frame,
    catboost_cat_features,
    feature_column_lists,
    fit_behavior_kmeans,
    mark_lgbm_categories,
    scaled_target_vector,
)


class ImportGroupBatchSampler:
    """
    Yields batches so that each batch is one (Client, Import date) group,
    or a chunk of a large group. This lets the model see each import
    separately and learn different behaviours per client/import.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        import_col: str,
        client_col: str = "Client",
        batch_size: int = 1024,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        group_key = (
            self.df[client_col].astype(str)
            + "\n"
            + pd.to_datetime(self.df[import_col], errors="coerce").astype(str)
        )
        self.group_to_indices: List[List[int]] = []
        for _, idx in self.df.groupby(group_key).groups.items():
            indices = idx.tolist()
            # If group is larger than batch_size, split into chunks
            for start in range(0, len(indices), batch_size):
                chunk = indices[start : start + batch_size]
                self.group_to_indices.append(chunk)
        self.n_batches = len(self.group_to_indices)

    def __iter__(self):
        order = list(range(self.n_batches))
        if self.shuffle:
            np.random.shuffle(order)
        for i in order:
            yield self.group_to_indices[i]

    def __len__(self) -> int:
        return self.n_batches


def load_config(config_path: str = "config.json") -> dict:
    """
    Load project configuration from a JSON file.

    Parameters
    ----------
    config_path : str
        Path to the configuration JSON file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_table(path: str) -> pd.DataFrame:
    """
    Generic loader for train/valid/test files (CSV or Excel).

    Parameters
    ----------
    path : str
        Path to the data file. Supports `.csv`, `.xlsx`, `.xls`.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    lower = path.lower()
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    
    if lower.endswith(".csv"):
        try:
            return pd.read_csv(path, encoding="utf-8")
        except (UnicodeDecodeError, UnicodeError):
            return pd.read_csv(path, encoding="cp1252")
            
    raise ValueError(f"Unsupported file format for {path}. Use .csv, .xlsx or .xls.")

class PortfolioDataset(Dataset):
    """
    Torch Dataset for the debt collection portfolio data.

    It expects pre-split CSV data and uses encoder/scaler to transform
    categorical and numerical columns into model-ready tensors.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: dict,
        encoder,
        scaler,
        target_scaler=None,
    ) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Source DataFrame (train/valid/test).
        config : dict
            Project configuration.
        encoder :
            Fitted OrdinalEncoder for categorical features.
        scaler :
            Fitted StandardScaler for numerical + action features.
        target_scaler :
            Optional StandardScaler for the target (Paid Value).
        """
        super().__init__()
        self.config = config
        cols_cfg = config["columns"]
        self.categorical_features: List[str] = [
            c
            for c in cols_cfg["categorical_features"]
            if c not in cols_cfg.get("exclude_features", [])
        ]
        # Include weekly action features if available
        weekly_action_features = cols_cfg.get("weekly_action_features", [])
        self.numeric_like_features: List[str] = [
            c
            for c in (cols_cfg["numerical_features"] + cols_cfg["action_features"] + weekly_action_features)
            if c not in cols_cfg.get("exclude_features", [])
        ]
        self.target_column: str = cols_cfg["target_column"]

        self.encoder = encoder
        self.scaler = scaler
        self.target_scaler = target_scaler

        # Extract and transform features.
        if self.categorical_features:
            cat_arr = encoder.transform(df[self.categorical_features])
            # Map unknown/negative codes (e.g. -1) to 0 to keep indices in range
            # for the embedding layers.
            cat_arr = np.where(cat_arr < 0, 0, cat_arr)
        else:
            cat_arr = np.zeros((len(df), 0), dtype=np.int64)

        if self.numeric_like_features:
            # Align with scaler training schema when available (e.g. legacy features).
            if hasattr(scaler, "feature_names_in_"):
                self.numeric_like_features = list(scaler.feature_names_in_)
            for col in self.numeric_like_features:
                if col not in df.columns:
                    df[col] = 0
            num_arr = scaler.transform(df[self.numeric_like_features])
        else:
            num_arr = np.zeros((len(df), 0), dtype=np.float32)

        self.categorical_data = torch.as_tensor(cat_arr, dtype=torch.long)
        self.numeric_data = torch.as_tensor(num_arr, dtype=torch.float32)

        # Target: derive from configuration (raw Paid Value or Paid/Case
        # ratio) and scale if a target_scaler is provided so the model
        # predicts in a normalized space, then invert for reporting.
        target_series = get_target_series(df, config=self.config).astype(np.float32)
        y = target_series.values.reshape(-1, 1)
        if self.target_scaler is not None:
            y = self.target_scaler.transform(y)
        self.targets = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (categorical_indices, numeric_features, target)
        """
        return (
            self.categorical_data[idx],
            self.numeric_data[idx],
            self.targets[idx],
        )


class ActionDataset(Dataset):
    """
    Dataset for training the weekly action‑rate model.

    Inputs:
        - Same categorical features as the main model.
        - Numeric features aligned with the shared scaler
          (numerical + action + weekly features). For the
          action model we only rely on the non‑action features;
          any missing columns are filled with 0 so the scaler
          schema stays consistent.

    Targets:
        - Weekly action features (e.g. Calls_per_week, ...).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: dict,
        encoder,
        scaler,
    ) -> None:
        super().__init__()
        cols_cfg = config["columns"]
        self.categorical_features: List[str] = [
            c
            for c in cols_cfg["categorical_features"]
            if c not in cols_cfg.get("exclude_features", [])
        ]
        weekly_action_features_cfg: List[str] = cols_cfg.get("weekly_action_features", [])

        self.encoder = encoder
        self.scaler = scaler

        # Categorical inputs.
        if self.categorical_features:
            cat_arr = encoder.transform(df[self.categorical_features])
            cat_arr = np.where(cat_arr < 0, 0, cat_arr)
        else:
            cat_arr = np.zeros((len(df), 0), dtype=np.int64)

        # Numeric inputs aligned to the scaler schema.
        if hasattr(scaler, "feature_names_in_"):
            numeric_cols = list(scaler.feature_names_in_)
        else:
            numeric_cols = list(
                cols_cfg["numerical_features"]
                + cols_cfg.get("action_features", [])
                + weekly_action_features_cfg
            )
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0
        num_arr = scaler.transform(df[numeric_cols])

        self.categorical_data = torch.as_tensor(cat_arr, dtype=torch.long)
        self.numeric_data = torch.as_tensor(num_arr, dtype=torch.float32)

        # Targets: use only weekly features that actually exist in the data.
        weekly_present = [
            c for c in weekly_action_features_cfg if c in df.columns
        ]
        if weekly_present:
            y = df[weekly_present].fillna(0).values.astype(np.float32)
        else:
            # No weekly features present – fall back to zeros to avoid crashes.
            y = np.zeros((len(df), 0), dtype=np.float32)
        self.weekly_action_features = weekly_present
        self.targets = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.categorical_data[idx],
            self.numeric_data[idx],
            self.targets[idx],
        )


class PortfolioModel(nn.Module):
    """
    Feedforward neural network with entity embeddings for categorical features.

    #NOTE: Each categorical feature gets its own embedding table. The resulting
    embeddings are concatenated with the numerical features and passed through
    a standard MLP for regression on the Paid Value target.
    """

    def __init__(
        self,
        categorical_cardinalities: Dict[str, int],
        numeric_input_dim: int,
        config: dict,
        output_dim: int = 1,
    ) -> None:
        """
        Initialize the model architecture.

        Parameters
        ----------
        categorical_cardinalities : Dict[str, int]
            Mapping from categorical feature name to number of unique categories.
        numeric_input_dim : int
            Dimension of the numeric input (scaled numerical + action features).
        config : dict
            Project configuration with embedding_dims and hidden_layers settings.
        """
        super().__init__()
        model_cfg = config["model"]
        emb_dims_cfg = model_cfg.get("embedding_dims", {})

        self.categorical_features = list(categorical_cardinalities.keys())

        # Build embedding layers for each categorical column.
        self.emb_layers = nn.ModuleDict()
        total_emb_dim = 0
        for col, cardinality in categorical_cardinalities.items():
            # Derive embedding dimension either from config or a rule-of-thumb.
            emb_dim = emb_dims_cfg.get(col)
            if emb_dim is None:
                # Rule-of-thumb: min(50, size//2) for unseen columns.
                emb_dim = int(min(50, max(2, cardinality // 2)))
            self.emb_layers[col] = nn.Embedding(
                num_embeddings=max(cardinality + 1, 2),  # reserve index for unknown
                embedding_dim=emb_dim,
            )
            total_emb_dim += emb_dim

        input_dim = total_emb_dim + numeric_input_dim

        # Define MLP layers.
        layers: List[nn.Module] = []
        hidden_layers: List[int] = model_cfg.get("hidden_layers", [64, 32])
        dropout = float(model_cfg.get("dropout", 0.1))

        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        # Final regression head (supports multi-output regression).
        layers.append(nn.Linear(prev_dim, int(output_dim)))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters
        ----------
        x_cat : torch.Tensor
            Tensor of shape (batch_size, num_categorical_features) with
            integer-encoded categories.
        x_num : torch.Tensor
            Tensor of shape (batch_size, num_numeric_features) with
            scaled numerical + action features.

        Returns
        -------
        torch.Tensor
            Predicted Paid Value of shape (batch_size, 1).
        """
        emb_list = []
        # Map columns by fixed order.
        for i, col in enumerate(self.categorical_features):
            emb = self.emb_layers[col](x_cat[:, i])
            emb_list.append(emb)

        if emb_list:
            x_emb = torch.cat(emb_list, dim=1)
            x = torch.cat([x_emb, x_num], dim=1) if x_num.numel() > 0 else x_emb
        else:
            x = x_num

        return self.mlp(x)


def get_categorical_cardinalities(
    df: pd.DataFrame,
    config: dict,
    encoder,
) -> Dict[str, int]:
    """
    Infer the cardinality of each categorical feature from the encoder.

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame.
    config : dict
        Configuration dictionary.
    encoder :
        Fitted OrdinalEncoder.

    Returns
    -------
    Dict[str, int]
        Mapping from feature name to cardinality.
    """
    cols_cfg = config["columns"]
    cat_features = [
        c
        for c in cols_cfg["categorical_features"]
        if c not in cols_cfg.get("exclude_features", [])
    ]

    cardinalities: Dict[str, int] = {}
    if not cat_features:
        return cardinalities

    # OrdinalEncoder.categories_ is list aligned with the columns passed to fit.
    for col, cats in zip(cat_features, encoder.categories_):
        cardinalities[col] = len(cats)
    return cardinalities


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
) -> float:
    """
    Train the model for a single epoch.

    Parameters
    ----------
    model : nn.Module
        The neural network model.
    loader : DataLoader
        DataLoader for the training set.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    criterion :
        Loss function (e.g., MSELoss).
    device : torch.device
        Computation device (CPU or CUDA).

    Returns
    -------
    float
        Average training loss over the epoch.
    """
    model.train()
    running_loss = 0.0
    n_samples = 0

    import math as _math  # type: ignore

    for x_cat, x_num, y in loader:
        x_cat = x_cat.to(device)
        x_num = x_num.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(x_cat, x_num)
        loss = criterion(preds, y)
        loss.backward()

        # Clip gradients to prevent exploding updates that destabilize training.
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        except Exception:
            pass

        # Skip parameter update if loss has become non‑finite.
        if _math.isfinite(loss.item()):
            optimizer.step()

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    return running_loss / max(n_samples, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
) -> float:
    """
    Evaluate the model on a validation or test set.

    Parameters
    ----------
    model : nn.Module
        The neural network model.
    loader : DataLoader
        DataLoader for the evaluation set.
    criterion :
        Loss function (e.g., MSELoss).
    device : torch.device
        Computation device (CPU or CUDA).

    Returns
    -------
    float
        Average loss over the dataset.
    """
    model.eval()
    running_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for x_cat, x_num, y in loader:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            y = y.to(device)

            preds = model(x_cat, x_num)
            loss = criterion(preds, y)

            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

    return running_loss / max(n_samples, 1)


def _report(progress_callback: Optional[Callable[[str, float], None]], phase: str, frac: float) -> None:
    if progress_callback:
        progress_callback(phase, frac)
    print(f"[GBM {phase}] ({100.0 * frac:.0f}%)")


def train_all_gbm(
    config: dict,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    metrics_callback: Optional[Callable[[str, List[float], List[float]], None]] = None,
) -> None:
    """
    Train CatBoost median (parity), LightGBM (extended + routing), and optional
    multi-output CatBoost for weekly action rates. Writes manifest + blend routing.
    """
    data_cfg = config["data"]
    cols_cfg = config["columns"]
    train_cfg = config["training"]
    gbm_cfg = config.get("gbm", {})
    paid_blend = gbm_cfg.get("paid_blend", {})
    rs = int(train_cfg.get("random_state", gbm_cfg.get("behavior_kmeans", {}).get("random_state", 42)))

    train_df = load_table(data_cfg["train_path"])
    valid_df = load_table(data_cfg["valid_path"])
    train_df = apply_product_placeholder(train_df, config)
    valid_df = apply_product_placeholder(valid_df, config)

    artifacts_dir = data_cfg["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)
    scaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))
    target_scaler = joblib.load(os.path.join(artifacts_dir, "target_scaler.pkl"))

    cats, nums = feature_column_lists(config)
    if hasattr(scaler, "feature_names_in_"):
        nums = list(scaler.feature_names_in_)

    _report(progress_callback, "behavior_kmeans", 0.05)
    kcfg = gbm_cfg.get("behavior_kmeans", {})
    kmeans, kmeans_cols = fit_behavior_kmeans(
        train_df,
        config,
        int(kcfg.get("n_clusters", 8)),
        int(kcfg.get("random_state", rs)),
    )
    joblib.dump(kmeans, os.path.join(artifacts_dir, BEHAVIOR_KMEANS_NAME))

    parity_tr = build_parity_tree_frame(train_df, config, cats, nums)
    parity_va = build_parity_tree_frame(valid_df, config, cats, nums)
    y_tr = scaled_target_vector(train_df, config, target_scaler)
    y_va = scaled_target_vector(valid_df, config, target_scaler)
    cat_feat = catboost_cat_features(parity_tr, cats)

    _report(progress_callback, "catboost_paid", 0.2)
    ccfg = gbm_cfg.get("catboost_paid", {})
    train_pool = Pool(parity_tr, y_tr, cat_features=cat_feat)
    val_pool = Pool(parity_va, y_va, cat_features=cat_feat)
    cat_paid = CatBoostRegressor(
        loss_function=str(ccfg.get("loss_function", "Quantile:alpha=0.5")),
        iterations=int(ccfg.get("iterations", 2000)),
        random_seed=rs,
        verbose=bool(ccfg.get("verbose", False)),
        early_stopping_rounds=int(ccfg.get("early_stopping_rounds", 50)),
    )
    cat_paid.fit(train_pool, eval_set=val_pool, use_best_model=True)
    if metrics_callback:
        try:
            evals = cat_paid.get_evals_result() or {}
            train_vals = evals.get("learn", {}).get("RMSE") or evals.get("learn", {}).get("Quantile:alpha=0.5") or []
            val_vals = evals.get("validation", {}).get("RMSE") or evals.get("validation", {}).get("Quantile:alpha=0.5") or []
            if train_vals or val_vals:
                metrics_callback(
                    "catboost_paid",
                    [float(v) for v in train_vals],
                    [float(v) for v in val_vals],
                )
        except Exception:
            pass
    cat_path = os.path.join(artifacts_dir, CATBOOST_PAID_NAME)
    cat_paid.save_model(cat_path)
    print(f"Saved CatBoost paid model to {cat_path}")

    _report(progress_callback, "lightgbm_paid", 0.45)
    ext_tr = build_extended_tree_frame(train_df, config, cats, nums, kmeans, kmeans_cols)
    ext_va = build_extended_tree_frame(valid_df, config, cats, nums, kmeans, kmeans_cols)
    ext_tr_lgb = mark_lgbm_categories(ext_tr, cats)
    ext_va_lgb = mark_lgbm_categories(ext_va, cats)
    cat_cols_lgb = [c for c in cats if c in ext_tr_lgb.columns]
    cat_feat_lgb = cat_cols_lgb if cat_cols_lgb else None
    lcfg = gbm_cfg.get("lightgbm_paid", {})
    lgb_params = {
        "objective": str(lcfg.get("objective", "regression")),
        "metric": str(lcfg.get("metric", "rmse")),
        "verbosity": -1,
        "seed": rs,
        "learning_rate": float(lcfg.get("learning_rate", 0.05)),
        "num_leaves": int(lcfg.get("num_leaves", 63)),
        "feature_fraction": float(lcfg.get("feature_fraction", 0.9)),
        "bagging_fraction": float(lcfg.get("bagging_fraction", 0.8)),
        "bagging_freq": int(lcfg.get("bagging_freq", 1)),
    }
    dtr = lgb.Dataset(ext_tr_lgb, label=y_tr, categorical_feature=cat_feat_lgb, free_raw_data=False)
    dva = lgb.Dataset(ext_va_lgb, label=y_va, categorical_feature=cat_feat_lgb, reference=dtr, free_raw_data=False)
    lgb_eval_result: Dict = {}
    lgb_paid = lgb.train(
        lgb_params,
        dtr,
        num_boost_round=int(lcfg.get("iterations", 2000)),
        valid_sets=[dva],
        callbacks=[
            lgb.early_stopping(int(lcfg.get("early_stopping_rounds", 50)), verbose=False),
            lgb.record_evaluation(lgb_eval_result),
        ],
    )
    if metrics_callback:
        try:
            val_block = lgb_eval_result.get("valid_0", {})
            metric_name = str(lcfg.get("metric", "rmse"))
            val_vals = val_block.get(metric_name) or val_block.get("rmse") or []
            train_vals: List[float] = []
            if val_vals:
                metrics_callback(
                    "lightgbm_paid",
                    train_vals,
                    [float(v) for v in val_vals],
                )
        except Exception:
            pass
    lgb_path = os.path.join(artifacts_dir, LIGHTGBM_PAID_NAME)
    lgb_paid.save_model(lgb_path)
    print(f"Saved LightGBM paid model to {lgb_path}")

    cv_col = str(paid_blend.get("case_value_column", "Case Value"))
    p_high = float(paid_blend.get("high_value_percentile", 90)) / 100.0
    if cv_col not in train_df.columns:
        threshold = 0.0
        qmap = {}
    else:
        cv_num = pd.to_numeric(train_df[cv_col], errors="coerce").dropna()
        threshold = float(cv_num.quantile(p_high)) if len(cv_num) else 0.0
        eval_q = paid_blend.get("evaluation_quantiles", [0.5, 0.75, 0.9, 0.99])
        qmap = {f"p{int(q * 100)}": float(cv_num.quantile(q)) for q in eval_q if len(cv_num)}

    routing = {
        "case_value_column": cv_col,
        "high_value_percentile": paid_blend.get("high_value_percentile", 90),
        "case_value_threshold": threshold,
        "case_value_quantiles_train": qmap,
        "relative_error_tolerance_pct": paid_blend.get("relative_error_tolerance_pct", 20),
    }
    blend_path = os.path.join(artifacts_dir, BLEND_ROUTING_NAME)
    with open(blend_path, "w", encoding="utf-8") as f:
        json.dump(routing, f, indent=2)
    print(f"Saved blend routing to {blend_path} (threshold={threshold:.4f})")

    weekly_feats = [c for c in cols_cfg.get("weekly_action_features", []) if c in train_df.columns]
    action_block: Dict = {"model_path": None, "weekly_targets": weekly_feats, "backend": "none"}
    if weekly_feats:
        action_block["backend"] = "catboost_multi"
        action_block["model_path"] = ACTION_GBM_NAME
        _report(progress_callback, "action_gbm", 0.7)
        feat_tr = parity_tr.copy()
        feat_va = parity_va.copy()
        Y_tr = train_df[weekly_feats].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
        Y_va = valid_df[weekly_feats].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
        acfg = gbm_cfg.get("action_gbm", {})
        apool_tr = Pool(feat_tr, Y_tr, cat_features=cat_feat)
        apool_va = Pool(feat_va, Y_va, cat_features=cat_feat)
        action_model = CatBoostRegressor(
            loss_function=str(acfg.get("loss_function", "MultiRMSE")),
            iterations=int(acfg.get("iterations", 2000)),
            random_seed=rs,
            verbose=False,
            early_stopping_rounds=int(acfg.get("early_stopping_rounds", 50)),
        )
        action_model.fit(apool_tr, eval_set=apool_va, use_best_model=True)
        action_path = os.path.join(artifacts_dir, ACTION_GBM_NAME)
        action_model.save_model(action_path)
        print(f"Saved action GBM to {action_path}")
    else:
        print("No weekly action columns in train data; skipping action GBM.")
        action_block["weekly_targets"] = list(cols_cfg.get("weekly_action_features", []))

    manifest = {
        "version": 2,
        "backend": "gbm_blended",
        "paid": {
            "catboost_path": CATBOOST_PAID_NAME,
            "lightgbm_path": LIGHTGBM_PAID_NAME,
            "behavior_kmeans_path": BEHAVIOR_KMEANS_NAME,
            "behavior_kmeans_columns": kmeans_cols,
            "categorical_features": cats,
            "parity_numeric_features": nums,
            "extended_features": list(EXTENDED_COLS),
        },
        "action": action_block,
    }

    man_path = os.path.join(artifacts_dir, MANIFEST_NAME)
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {man_path}")
    _report(progress_callback, "done", 1.0)


def main() -> None:
    """
    Train gradient boosting models (production): blended paid value + optional weekly action model.
    """
    config = load_config()
    train_all_gbm(config, progress_callback=None)


if __name__ == "__main__":
    main()

# Main training module for the debt collection strategy engine