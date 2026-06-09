# Main prediction module for the debt collection strategy engine
import itertools
import os
from typing import Callable, Dict, List, Optional, Tuple
from zipfile import BadZipFile

import joblib
import numpy as np
import pandas as pd

from train import load_config, load_table
from split import LogStandardScaler, case_duration_weeks_series, compute_weekly_rates  # noqa: F401

from gbm_inference import (
    load_paid_bundle,
    predict_paid_value_blended_from_bundle,
    predict_weekly_actions_gbm,
)
from prediction_cap import apply_paid_prediction_cap, is_prediction_cap_active


# Default action grids to use if not defined in config
DEFAULT_ACTIONS_GRID = {
    "Calls": [0, 5, 10, 15, 20, 25, 30],
    "Letters": [0, 1, 2, 3, 4],
    "SMS": [0, 5, 10, 15, 20]
}

_INSIGHTS_CACHE: Dict[Tuple[str, float, float], Dict] = {}

# ~4.33 weeks; used for long-active client profiling from training data.
WEEKS_ONE_MONTH = 30.0 / 7.0
MIN_CASES_FOR_LONG_ACTIVE_FLAG = 5


def load_artifacts_for_inference():
    """
    Load configuration and preprocessing artifacts (encoder, scalers).

    Returns
    -------
    Tuple[dict, object, object, object]
        (config, encoder, scaler, target_scaler)
    """
    config = load_config()
    data_cfg = config["data"]
    artifacts_dir = data_cfg["artifacts_dir"]

    encoder = joblib.load(os.path.join(artifacts_dir, "encoder.pkl"))
    scaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))
    target_scaler = joblib.load(os.path.join(artifacts_dir, "target_scaler.pkl"))

    return config, encoder, scaler, target_scaler


def predict_paid_values_for_dataframe(
    df: pd.DataFrame,
    config: dict,
    encoder,
    scaler,
    target_scaler,
    bundle=None,
) -> np.ndarray:
    """Blended GBM paid predictions in Paid Value (or ratio-rescaled) units."""
    if bundle is None:
        bundle = load_paid_bundle(config)
    preds = predict_paid_value_blended_from_bundle(
        df, config, encoder, scaler, target_scaler, bundle
    )
    cv_col = (
        config.get("gbm", {})
        .get("paid_blend", {})
        .get("case_value_column", "Case Value")
    )
    if cv_col in df.columns:
        case_values = pd.to_numeric(df[cv_col], errors="coerce").fillna(0.0).values
        preds = apply_paid_prediction_cap(preds, case_values, config)
    return preds


def predict_weekly_actions(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict weekly average actions per channel for each row in df using Model A.

    Returns a DataFrame with the same index as df and columns equal to
    config["columns"]["weekly_action_features"], containing non‑negative
    predicted rates.
    """
    config, encoder, scaler, _ = load_artifacts_for_inference()
    return predict_weekly_actions_gbm(df, config, encoder, scaler)


def debtor_row_to_features(
    debtor_data: Dict,
    config: dict,
    encoder,
    scaler,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a single debtor dictionary into encoded categorical and scaled numeric arrays.

    #NOTE: This function expects that debtor_data already includes values for
    all non-action (portfolio) features. Action features will be varied elsewhere.

    Parameters
    ----------
    debtor_data : Dict
        Mapping of column name to value for one debtor.
    config : dict
        Project configuration.
    encoder :
        Fitted OrdinalEncoder.
    scaler :
        Fitted StandardScaler.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (categorical_encoded[1, num_cat], numeric_scaled[1, num_num])
    """
    cols_cfg = config["columns"]
    categorical_features = [
        c
        for c in cols_cfg["categorical_features"]
        if c not in cols_cfg.get("exclude_features", [])
    ]
    numeric_like_features = [
        c
        for c in (
            cols_cfg["numerical_features"] + cols_cfg["action_features"]
        )
        if c not in cols_cfg.get("exclude_features", [])
    ]

    # Construct a temporary DataFrame with a single row.
    df = pd.DataFrame([debtor_data])

    # Normalize Product when it was filled with debtor Name.
    product_col = cols_cfg.get("product_column", "Product")
    name_col = cols_cfg.get("debtor_name_column", "Name")
    placeholder = cols_cfg.get("product_missing_placeholder", "Unknown_Product")
    if product_col in df.columns:
        prod = df[product_col]
        prod_str = prod.astype(str).str.strip()
        missing_mask = prod.isna() | (prod_str == "") | (prod_str.str.lower() == "nan")
        if name_col in df.columns:
            name_str = df[name_col].astype(str).str.strip()
            missing_mask = missing_mask | (prod_str == name_str)
        if missing_mask.any():
            df.loc[missing_mask, product_col] = placeholder

    if categorical_features:
        for col in categorical_features:
            if col not in df.columns:
                df[col] = "Unknown"
            df[col] = df[col].fillna("Unknown").astype(str)
        cat_arr = encoder.transform(df[categorical_features])
        # Align with training behaviour: route any unknown category codes (<0)
        # to index 0 so they are valid embedding indices.
        cat_arr = np.where(cat_arr < 0, 0, cat_arr)
    else:
        cat_arr = np.zeros((1, 0), dtype=np.int64)

    if numeric_like_features:
        # Align with scaler training schema when available (e.g. legacy features).
        if hasattr(scaler, "feature_names_in_"):
            numeric_like_features = list(scaler.feature_names_in_)
        for col in numeric_like_features:
            if col not in df.columns:
                df[col] = 0
        mean_map = {}
        if hasattr(scaler, "mean_") and hasattr(scaler, "feature_names_in_"):
            mean_map = dict(zip(scaler.feature_names_in_, scaler.mean_))
        for col in numeric_like_features:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().any():
                df[col] = df[col].fillna(mean_map.get(col, 0))
        num_arr = scaler.transform(df[numeric_like_features])
    else:
        num_arr = np.zeros((1, 0), dtype=np.float32)

    return cat_arr.astype(np.int64), num_arr.astype(np.float32)


def generate_action_scenarios(
    base_debtor_data: Dict,
    config: dict,
) -> List[Dict]:
    """
    Generate a list of debtor dicts with different action combinations.

    Action ranges are read from `config["actions_grid"]`. For each combination
    of (Calls, Letters, SMS, ...) the corresponding keys in the debtor dictionary
    are updated.

    Parameters
    ----------
    base_debtor_data : Dict
        Base debtor information (portfolio-level features).
    config : dict
        Project configuration containing actions_grid.

    Returns
    -------
    List[Dict]
        List of debtor dictionaries, one for each action scenario.
    """
    actions_grid = config.get("actions_grid", {})
    if not actions_grid:
        # Use default grids if not defined in config
        actions_grid = DEFAULT_ACTIONS_GRID
        print("Using default actions_grid since none defined in config.")

    action_keys = list(actions_grid.keys())
    action_values_list = [actions_grid[k] for k in action_keys]

    scenarios: List[Dict] = []
    for combo in itertools.product(*action_values_list):
        scenario = dict(base_debtor_data)  # shallow copy base features
        for key, val in zip(action_keys, combo):
            scenario[key] = val
        scenarios.append(scenario)

    return scenarios


def recommend_strategy(
    debtor_data: Dict,
) -> Dict:
    """
    Recommend the optimal action strategy for a single debtor.

    This function:
    1. Loads the trained model and preprocessing artifacts.
    2. Generates a grid of action scenarios based on config["actions_grid"].
    3. Scores each scenario with the model.
    4. Returns the scenario (action combination) with the highest predicted Paid Value.

    Parameters
    ----------
    debtor_data : Dict
        Base debtor information, including portfolio-level features (e.g.,
        Client, Location, Case Value, DPD, Debtor Age). Action features like
        Calls, SMS, Letters will be overwritten by scenario values.

    Returns
    -------
    Dict
        A dictionary containing:
        - "best_actions": dict of the best action combination.
        - "best_predicted_paid_value": float with the predicted Paid Value.
        - "all_scenarios": list of dicts with "actions" and "predicted_paid_value".
    """
    config, encoder, scaler, target_scaler = load_artifacts_for_inference()
    bundle = load_paid_bundle(config)

    cols_cfg = config["columns"]
    target_col = cols_cfg["target_column"]

    # Generate all action scenarios.
    scenarios = generate_action_scenarios(debtor_data, config)

    rows_for_pred: List[Dict] = []
    for scenario in scenarios:
        scenario = dict(scenario)
        scenario.pop(target_col, None)
        rows_for_pred.append(scenario)
    pred_df = pd.DataFrame(rows_for_pred)
    preds_arr = predict_paid_values_for_dataframe(
        pred_df, config, encoder, scaler, target_scaler, bundle=bundle
    )

    results = []
    best_pred = -float("inf")
    best_actions = None

    for scenario, pred in zip(scenarios, preds_arr):
        pred = float(pred)

        # Extract only the action subset for reporting.
        actions_only = {k: scenario[k] for k in config["columns"]["action_features"]}
        results.append(
            {
                "actions": actions_only,
                "predicted_paid_value": float(pred),
            }
        )

        if pred > best_pred:
            best_pred = pred
            best_actions = actions_only

    return {
        "best_actions": best_actions,
        "best_predicted_paid_value": float(best_pred),
        "all_scenarios": results,
    }


def recommend_portfolio_strategy(
    portfolio_df: pd.DataFrame,
    config: Optional[dict] = None,
    action_costs: Optional[Dict[str, float]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    Find optimal actions for an entire portfolio, maximizing Net Value (Predicted - Costs).

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        DataFrame containing portfolio cases with features.
    config : dict, optional
        Project configuration. Loaded from config.json if not provided.
    action_costs : Dict[str, float], optional
        Cost per action. Uses config["action_costs"] if not provided.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional columns:
        - Optimal_Calls, Optimal_Letters, Optimal_SMS: Recommended actions
        - Optimal_Predicted_Value: Predicted paid value for optimal actions
        - Optimal_Action_Cost: Total cost of optimal actions
        - Optimal_Net_Value: Predicted - Cost
        - Optimal_ROI: (Predicted - Cost) / Cost
    """
    if config is None:
        config = load_config()

    loaded_config, encoder, scaler, target_scaler = load_artifacts_for_inference()
    bundle = load_paid_bundle(loaded_config)

    cols_cfg = loaded_config["columns"]
    action_features = cols_cfg.get("action_features", ["Calls", "Letters", "SMS"])
    weekly_action_features = cols_cfg.get("weekly_action_features", [])
    actions_grid = loaded_config.get("actions_grid", DEFAULT_ACTIONS_GRID)
    date_cfg = cols_cfg.get("date_features", {})
    import_col = date_cfg.get("import_date", "Import date")
    end_col = date_cfg.get("end_date", "End Date")

    # Ensure weekly rate columns exist if the model expects them
    portfolio_df = portfolio_df.copy()

    # Normalize Product when it was filled with debtor Name.
    product_col = cols_cfg.get("product_column", "Product")
    name_col = cols_cfg.get("debtor_name_column", "Name")
    placeholder = cols_cfg.get("product_missing_placeholder", "Unknown_Product")
    if product_col in portfolio_df.columns:
        prod = portfolio_df[product_col]
        prod_str = prod.astype(str).str.strip()
        missing_mask = prod.isna() | (prod_str == "") | (prod_str.str.lower() == "nan")
        if name_col in portfolio_df.columns:
            name_str = portfolio_df[name_col].astype(str).str.strip()
            missing_mask = missing_mask | (prod_str == name_str)
        if missing_mask.any():
            portfolio_df.loc[missing_mask, product_col] = placeholder

    if weekly_action_features and import_col in portfolio_df.columns and end_col in portfolio_df.columns:
        portfolio_df = compute_weekly_rates(portfolio_df, loaded_config)

    if action_costs is None:
        action_costs = loaded_config.get("action_costs", {
            "Calls": 0.2,
            "Letters": 1.22,
            "SMS": 0.03
        })

    # Generate all possible action combinations
    action_keys = list(actions_grid.keys())
    action_values_list = [actions_grid[k] for k in action_keys]
    all_combinations = list(itertools.product(*action_values_list))

    results = []
    total_rows = len(portfolio_df)

    # Pre-compute case durations (weeks) for each row if weekly features are needed
    case_durations = {}
    if weekly_action_features and import_col in portfolio_df.columns and end_col in portfolio_df.columns:
        import_dates = pd.to_datetime(portfolio_df[import_col], errors="coerce")
        end_dates = pd.to_datetime(portfolio_df[end_col], errors="coerce")
        days_diff = (end_dates - import_dates).dt.days
        weeks = (days_diff / 7).clip(lower=1).fillna(1)
        case_durations = weeks.to_dict()

    for idx, row in portfolio_df.iterrows():
        if progress_callback:
            progress_callback(idx + 1, total_rows)
        debtor_data = row.to_dict()
        
        # Get case duration in weeks for this row (if available)
        case_weeks = case_durations.get(idx, 1.0) if case_durations else 1.0
        
        best_net_value = -float("inf")
        best_actions = None
        best_predicted = 0
        best_cost = 0

        for combo in all_combinations:
            # Set actions for this combination
            for key, val in zip(action_keys, combo):
                debtor_data[key] = val
            
            # Update weekly rates if model expects them
            if weekly_action_features:
                for action in action_features:
                    if action in debtor_data:
                        weekly_col = f"{action}_per_week"
                        if weekly_col in weekly_action_features:
                            debtor_data[weekly_col] = debtor_data[action] / case_weeks

            pred_rows = pd.DataFrame([debtor_data])
            predicted = float(
                predict_paid_values_for_dataframe(
                    pred_rows,
                    loaded_config,
                    encoder,
                    scaler,
                    target_scaler,
                    bundle=bundle,
                )[0]
            )
            predicted = max(0.0, predicted)

            # Calculate cost for this combination
            total_cost = sum(
                action_costs.get(key, 0) * val
                for key, val in zip(action_keys, combo)
            )

            # Calculate net value
            net_value = predicted - total_cost

            if net_value > best_net_value:
                best_net_value = net_value
                best_actions = dict(zip(action_keys, combo))
                best_predicted = predicted
                best_cost = total_cost

        results.append({
            "idx": idx,
            **{f"Optimal_{k}": v for k, v in best_actions.items()},
            "Optimal_Predicted_Value": best_predicted,
            "Optimal_Action_Cost": best_cost,
            "Optimal_Net_Value": best_net_value,
            "Optimal_ROI": (best_predicted - best_cost) / best_cost if best_cost > 0 else 0,
        })

    # Merge results back to original dataframe
    results_df = pd.DataFrame(results).set_index("idx")
    return portfolio_df.join(results_df)


def get_training_data_insights(
    train_path: Optional[str] = None,
    include_training_df: bool = False,
) -> Dict:
    """
    Compute statistics from training data for displaying insights.

    Parameters
    ----------
    train_path : str, optional
        Path to training data. Uses config if not provided.

    Returns
    -------
    Dict
        Dictionary containing:
        - action_stats: Mean/median/std for each action
        - paid_value_stats: Mean/median/std for paid value
        - effort_buckets: Stats by Low/Medium/High effort
    """
    config = load_config()
    data_cfg = config["data"]
    cols_cfg = config["columns"]

    if train_path is None:
        train_path = data_cfg["train_path"]
    train_path = os.path.abspath(train_path)
    train_mtime = os.path.getmtime(train_path) if os.path.exists(train_path) else -1.0
    config_mtime = os.path.getmtime("config.json") if os.path.exists("config.json") else -1.0
    cache_key = (train_path, float(train_mtime), float(config_mtime))
    cached = _INSIGHTS_CACHE.get(cache_key)
    if cached is not None:
        result = dict(cached)
        if not include_training_df:
            result.pop("training_df", None)
        return result

    # Load training data using the same helper as the training script.
    try:
        df = load_table(train_path)
    except Exception as exc:
        # If the file is not a valid Excel zip (BadZipFile / "not a zip file"),
        # try to interpret it as CSV – this can happen when the data was saved
        # as CSV but given an .xlsx extension.
        msg = str(exc)
        if isinstance(exc, BadZipFile) or "zip file" in msg.lower():
            try:
                # Use Python engine with automatic delimiter detection,
                # since the file may be a CSV with ';' or other separators
                # but saved with an .xlsx extension. Fall back to a very
                # permissive encoding (latin1) so that odd bytes never
                # break the loader for insights.
                try:
                    df = pd.read_csv(
                        train_path,
                        encoding="utf-8",
                        sep=None,
                        engine="python",
                    )
                except (UnicodeDecodeError, UnicodeError):
                    df = pd.read_csv(
                        train_path,
                        encoding="latin1",
                        sep=None,
                        engine="python",
                    )
            except Exception as exc2:
                raise RuntimeError(
                    f"Failed to load training data from '{train_path}' as Excel (BadZipFile) and CSV: {exc2}"
                ) from exc2
        else:
            # For non-BadZipFile errors, surface a clear message.
            raise RuntimeError(f"Failed to load training data from '{train_path}': {exc}") from exc

    action_features = cols_cfg.get("action_features", ["Calls", "Letters", "SMS"])
    weekly_action_features = cols_cfg.get("weekly_action_features", [])
    date_cfg = cols_cfg.get("date_features", {})
    import_col = date_cfg.get("import_date", "Import date")
    end_col = date_cfg.get("end_date", "End Date")
    target_col = cols_cfg["target_column"]

    # Ensure weekly rate columns exist (if configured) by recomputing them
    # when necessary. We reuse the same helper as in the data-preparation
    # pipeline so that definitions stay consistent.
    missing_weekly: List[str] = []
    if weekly_action_features:
        missing_weekly = [c for c in weekly_action_features if c not in df.columns]
        # Always recompute weekly rates from Import/End dates so stale *_per_week
        # columns in the training file cannot skew baselines or suggested effort.
        if import_col in df.columns and end_col in df.columns:
            df = compute_weekly_rates(df, config)

    # Per-case action statistics (totals over the entire case)
    action_stats: Dict[str, Dict[str, float]] = {}
    for action in action_features:
        if action in df.columns:
            series = pd.to_numeric(df[action], errors="coerce")
            action_stats[action] = {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
            }

    # Per-week action statistics (based on *_per_week features), when available
    action_stats_per_week: Dict[str, Dict[str, float]] = {}
    if weekly_action_features:
        for action in action_features:
            weekly_col = f"{action}_per_week"
            if weekly_col in df.columns:
                series = pd.to_numeric(df[weekly_col], errors="coerce")
                action_stats_per_week[action] = {
                    "mean": float(series.mean(skipna=True)),
                    "median": float(series.median(skipna=True)),
                    "std": float(series.std(skipna=True)),
                    "min": float(series.min(skipna=True)),
                    "max": float(series.max(skipna=True)),
                }

    # Paid value statistics (optional — prediction and strategy simulation
    # must not depend on having Paid Value in the dataset).
    if target_col in df.columns:
        paid_value_series = pd.to_numeric(df[target_col], errors="coerce")
        paid_value_stats = {
            "mean": float(paid_value_series.mean()),
            "median": float(paid_value_series.median()),
            "std": float(paid_value_series.std()),
            "min": float(paid_value_series.min()),
            "max": float(paid_value_series.max()),
        }
    else:
        paid_value_stats = {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    # Paid Value / Case Value ratio (training calibration for portfolio-level cap)
    paid_to_case_ratio = {}
    case_value_col = "Case Value"
    if case_value_col in df.columns and target_col in df.columns:
        cv = pd.to_numeric(df[case_value_col], errors="coerce").fillna(0)
        pv = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
        mask = cv > 0
        if mask.any():
            ratio = pv.loc[mask] / cv.loc[mask]
            ratio = ratio.clip(upper=1.0)  # training may have filtered, but clamp for safety
            paid_to_case_ratio = {
                "mean": float(ratio.mean()),
                "median": float(ratio.median()),
                "p90": float(ratio.quantile(0.90)),
            }
    else:
        paid_to_case_ratio = {"mean": 0.3, "median": 0.2, "p90": 0.6}  # fallback

    # Derive total actions per case and per week (when weekly features exist).
    df["total_actions"] = sum(
        df[action] for action in action_features if action in df.columns
    )

    total_actions_per_week = None
    weekly_cols = [f"{a}_per_week" for a in action_features if f"{a}_per_week" in df.columns]
    if weekly_cols:
        total_actions_per_week = df[weekly_cols].sum(axis=1)
        df["total_actions_per_week"] = total_actions_per_week

    # Use per-week total actions for effort buckets when possible; otherwise
    # fall back to total actions per case.
    bucket_source_col = "total_actions_per_week" if total_actions_per_week is not None else "total_actions"
    q33 = df[bucket_source_col].quantile(0.33)
    q66 = df[bucket_source_col].quantile(0.66)

    def categorize_effort(total: float) -> str:
        if total <= q33:
            return "Low"
        elif total <= q66:
            return "Medium"
        return "High"

    df["effort_bucket"] = df[bucket_source_col].apply(categorize_effort)

    effort_buckets = {}
    for bucket in ["Low", "Medium", "High"]:
        bucket_df = df[df["effort_bucket"] == bucket]
        if len(bucket_df) > 0:
            # Average actions over entire case (totals).
            avg_actions_per_case = {
                action: float(bucket_df[action].mean())
                for action in action_features
                if action in bucket_df.columns
            }

            # Average paid value for the bucket is optional; prediction and
            # strategy simulation must not require a Paid Value column.
            if target_col in bucket_df.columns:
                bucket_paid_series = pd.to_numeric(bucket_df[target_col], errors="coerce")
                avg_paid_value = float(bucket_paid_series.mean())
            else:
                avg_paid_value = 0.0

            # Average actions per week, when weekly features are available.
            avg_actions_per_week = {}
            if weekly_cols:
                for action in action_features:
                    weekly_col = f"{action}_per_week"
                    if weekly_col in bucket_df.columns:
                        series = pd.to_numeric(bucket_df[weekly_col], errors="coerce")
                        avg_actions_per_week[action] = float(series.mean())

            effort_buckets[bucket] = {
                "count": len(bucket_df),
                "avg_paid_value": avg_paid_value,
                # Backwards-compatible per-case totals used by existing UI.
                "avg_actions": avg_actions_per_case,
                # New per-week view that the UI can prefer when present.
                "avg_actions_per_week": avg_actions_per_week,
            }

    # Approximate average case duration in weeks, useful for converting
    # between per-week actions and total actions.
    avg_weeks_open = None
    weeks_series = case_duration_weeks_series(df, config)
    if weeks_series.notna().any():
        avg_weeks_open = float(weeks_series.mean(skipna=True))

    # Actions that have never been used historically (always zero) across
    # the entire portfolio. These are useful for global caps so we don't
    # recommend actions that were never actually applied (e.g. Letters
    # always 0 for all clients).
    never_used_actions = [
        action
        for action, stats in action_stats.items()
        if stats["max"] == 0.0
    ]

    # Per-client \"never used\" actions. This lets us treat cases where a
    # specific client has never used a given action differently from
    # clients that do use it.
    never_used_actions_by_client: Dict[str, List[str]] = {}
    if "Client" in df.columns:
        for client, group in df.groupby("Client"):
            client_never_used: List[str] = []
            for action in action_features:
                if action in group.columns:
                    series = pd.to_numeric(group[action], errors="coerce")
                    if series.max() == 0.0:
                        client_never_used.append(action)
            if client_never_used:
                client_key = str(client).strip().casefold()
                never_used_actions_by_client[client_key] = client_never_used

    suggested_effort_by_client: Dict[str, Dict[str, int]] = {}
    if "Client" in df.columns:
        for client, group in df.groupby("Client"):
            if len(group) == 0:
                continue
            group_working = group.copy()
            group_working["total_actions_client"] = sum(
                group_working[action] for action in action_features if action in group_working.columns
            )
            if weekly_cols:
                group_working["total_actions_per_week_client"] = group_working[weekly_cols].sum(axis=1)
                client_bucket_source = "total_actions_per_week_client"
            else:
                client_bucket_source = "total_actions_client"
            gq33 = group_working[client_bucket_source].quantile(0.33)
            gq66 = group_working[client_bucket_source].quantile(0.66)

            def _bucket_for_client(total: float) -> str:
                if total <= gq33:
                    return "Low"
                if total <= gq66:
                    return "Medium"
                return "High"

            group_working["effort_bucket_client"] = group_working[client_bucket_source].apply(_bucket_for_client)
            best_score = -float("inf")
            best_actions: Dict[str, int] = {}
            for bucket in ["Low", "Medium", "High"]:
                bucket_df = group_working[group_working["effort_bucket_client"] == bucket]
                if bucket_df.empty:
                    continue
                avg_total_actions = 0.0
                avg_actions: Dict[str, float] = {}
                for action in action_features:
                    if action in bucket_df.columns:
                        series = pd.to_numeric(bucket_df[action], errors="coerce")
                        mean_val = float(series.mean()) if series.notna().any() else 0.0
                        avg_actions[action] = mean_val
                        avg_total_actions += mean_val
                if target_col in bucket_df.columns:
                    avg_paid = float(pd.to_numeric(bucket_df[target_col], errors="coerce").mean())
                else:
                    avg_paid = 0.0
                score = (avg_paid / avg_total_actions) if avg_total_actions > 0 else avg_paid
                if score > best_score:
                    best_score = score
                    _w = case_duration_weeks_series(group_working, config)
                    if _w.notna().any():
                        client_weeks = max(float(_w.mean(skipna=True)), 1.0 / 7.0)
                    elif avg_weeks_open:
                        client_weeks = max(float(avg_weeks_open), 1.0 / 7.0)
                    else:
                        client_weeks = 1.0
                    best_actions = {}
                    for action in action_features:
                        weekly_col = f"{action}_per_week"
                        if weekly_col in bucket_df.columns:
                            mean_weekly = float(
                                pd.to_numeric(bucket_df[weekly_col], errors="coerce").mean()
                            )
                            best_actions[action] = int(round(mean_weekly * client_weeks))
                        else:
                            best_actions[action] = int(
                                round(avg_actions.get(action, 0.0))
                            )
            if best_actions:
                client_key = str(client).strip().casefold()
                suggested_effort_by_client[client_key] = best_actions

    client_duration_stats: Dict[str, Dict[str, float]] = {}
    clients_long_active_history: Dict[str, bool] = {}
    if "Client" in df.columns:
        for client, group in df.groupby("Client"):
            client_weeks = case_duration_weeks_series(group, config)
            valid_weeks = client_weeks.dropna()
            n_valid = int(len(valid_weeks))
            if n_valid < MIN_CASES_FOR_LONG_ACTIVE_FLAG:
                continue
            median_weeks = float(valid_weeks.median())
            pct_over_one_month = float((valid_weeks > WEEKS_ONE_MONTH).mean())
            client_key = str(client).strip().casefold()
            client_duration_stats[client_key] = {
                "median_weeks": median_weeks,
                "pct_over_one_month": pct_over_one_month,
                "n_cases": float(n_valid),
            }
            clients_long_active_history[client_key] = (
                median_weeks > WEEKS_ONE_MONTH and pct_over_one_month >= 0.5
            )

    result = {
        "action_stats": action_stats,
        "action_stats_per_week": action_stats_per_week,
        "paid_value_stats": paid_value_stats,
        "paid_to_case_ratio": paid_to_case_ratio,
        "effort_buckets": effort_buckets,
        "total_samples": len(df),
        "avg_weeks_open": avg_weeks_open,
        "never_used_actions": never_used_actions,
        "never_used_actions_by_client": never_used_actions_by_client,
        "suggested_effort_by_client": suggested_effort_by_client,
        "client_duration_stats": client_duration_stats,
        "clients_long_active_history": clients_long_active_history,
    }
    if include_training_df:
        result["training_df"] = df
    _INSIGHTS_CACHE[cache_key] = dict(result)
    return dict(result)


def example_usage() -> None:
    """
    Example standalone usage of `recommend_strategy`.

    Demonstrates how to use the prediction functions with sample data.
    """
    # Example 1: Single debtor recommendation
    debtor_data = {
        "Client": "ClientA",
        "Location": "CityX",
        "Case Value": 1000.0,
        "DPD": 45,
        "Debtor Age": 35,
        "Calls": 0,
        "Letters": 0,
        "SMS": 0,
    }

    print("=== Single Debtor Strategy ===")
    result = recommend_strategy(debtor_data)
    print("Best Action Combination:", result["best_actions"])
    print("Best Predicted Paid Value:", result["best_predicted_paid_value"])

    # Example 2: Training data insights
    print("\n=== Training Data Insights ===")
    try:
        insights = get_training_data_insights()
        print(f"Total training samples: {insights['total_samples']}")
        print(f"Avg Paid Value: {insights['paid_value_stats']['mean']:.2f}")
        for bucket, stats in insights["effort_buckets"].items():
            print(f"  {bucket} Effort: {stats['count']} cases, "
                  f"Avg Paid: {stats['avg_paid_value']:.2f}")
    except Exception as e:
        print(f"Could not load training insights: {e}")


if __name__ == "__main__":
    example_usage()
