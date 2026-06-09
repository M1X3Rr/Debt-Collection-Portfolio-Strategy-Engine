"""
Monthly payout schedule transforms for the Predictions chart drill-through.

Portfolio active length (horizon) comes from configurable per-client settings in
config.json → portfolio_lengths, not from average case duration in the data.
"""
import math
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

MAX_PAYOUT_MONTHS = 60
DAYS_PER_MONTH = 30.4375

EFFORT_TO_PRED_COL = {
    "Low": "Pred_Low",
    "Medium": "Pred_Medium",
    "High": "Pred_High",
}

DEFAULT_PORTFOLIO_LENGTHS: Dict[str, Any] = {
    "no_limit_months": MAX_PAYOUT_MONTHS,
    "default_months": 12,
    "by_client": {
        "Client_A Finance": None,
        "Client_B Bank": None,
        "Client_B Bank - Secondary": None,
        "Client_C Credit": None,
        "Client_A - Early": 1,
        "Client_A - Standard": 36,
        "Client_D Retail": 6,
        "Client_E Bank": None,
        "Client_D - Mortgage": 6,
        "Client_F Digital": None,
        "Client_G Collections": None,
        "DirectMarketing": None,
        "Utility Corp East": 4,
        "Utility Corp West": 4,
        "Utility Distribution": 4,
        "ISP Provider A": 4,
        "Telco Mobile": 5,
        "Telco Fixed": 6,
        "Telco Early": {"days": 8},
        "Telco Primary": 2,
        "ISP Provider B": 3,
        "Telco Secondary": 6,
        "Telco Secondary - Extended": 12,
    },
}


def _normalized_client_key(value: object) -> str:
    """Case-insensitive key; hyphens/extra spaces match (e.g. 'Client_A - Early' == 'Client_A Early')."""
    s = str(value).strip().casefold()
    s = re.sub(r"\s*-\s*", " ", s)
    s = re.sub(r"[-_]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def portfolio_lengths_config(config: dict) -> dict:
    """Merged portfolio_lengths section with built-in defaults."""
    raw = config.get("portfolio_lengths") or {}
    by_client = dict(DEFAULT_PORTFOLIO_LENGTHS["by_client"])
    by_client.update(raw.get("by_client") or {})
    return {
        "no_limit_months": raw.get(
            "no_limit_months", DEFAULT_PORTFOLIO_LENGTHS["no_limit_months"]
        ),
        "default_months": raw.get(
            "default_months", DEFAULT_PORTFOLIO_LENGTHS["default_months"]
        ),
        "by_client": by_client,
    }


def _lookup_raw_client_length(client_name: str, pl_cfg: dict) -> Any:
    key = _normalized_client_key(client_name)
    for name, value in (pl_cfg.get("by_client") or {}).items():
        if _normalized_client_key(name) == key:
            return value
    return pl_cfg.get("default_months", 12)


def portfolio_length_to_months(value: Any, pl_cfg: dict) -> int:
    """
    Convert a portfolio length entry to a chart horizon in whole months.

    None / \"no limit\" uses no_limit_months (capped at MAX_PAYOUT_MONTHS).
    """
    no_limit = int(pl_cfg.get("no_limit_months", MAX_PAYOUT_MONTHS))
    default_months = pl_cfg.get("default_months", 12)

    if value is None:
        return min(max(1, no_limit), MAX_PAYOUT_MONTHS)

    if isinstance(value, dict):
        if "days" in value:
            days = float(value["days"])
            return min(max(1, int(math.ceil(days / DAYS_PER_MONTH))), MAX_PAYOUT_MONTHS)
        if "months" in value:
            value = value["months"]

    try:
        months = float(value)
    except (TypeError, ValueError):
        months = float(default_months)

    return min(max(1, int(math.ceil(months))), MAX_PAYOUT_MONTHS)


def client_portfolio_months(client_name: str, config: dict) -> int:
    pl_cfg = portfolio_lengths_config(config)
    raw = _lookup_raw_client_length(client_name, pl_cfg)
    return portfolio_length_to_months(raw, pl_cfg)


def format_portfolio_length_for_ui(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict) and "days" in value:
        days = value["days"]
        if float(days) == int(days):
            return f"{int(days)}d"
        return f"{days}d"
    if isinstance(value, (int, float)):
        if float(value) == int(value):
            return str(int(value))
        return str(value)
    return str(value)


def parse_portfolio_length_input(text: str) -> Optional[Union[int, float, Dict[str, float]]]:
    """
    Parse a settings field value.

    Blank / \"no limit\" → None (unlimited horizon).
    \"8d\" / \"8 days\" → {\"days\": 8}
    Otherwise → months as a number.
    """
    raw = text.strip()
    if not raw:
        return None
    lowered = raw.casefold()
    if lowered in ("no limit", "none", "unlimited", "-", "n/a"):
        return None

    days_match = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*(?:d|days?)", lowered)
    if days_match:
        days = float(days_match.group(1).replace(",", "."))
        return {"days": days}

    normalized = raw.replace(",", ".")
    try:
        months = float(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid portfolio length: {text!r}") from exc
    if months <= 0:
        return None
    if months == int(months):
        return int(months)
    return months


def resolve_payout_months_for_segment(
    df: pd.DataFrame,
    config: dict,
    training_median_weeks: Optional[float] = None,
    fallback_avg_weeks: Optional[float] = None,
) -> int:
    """
    Chart horizon in months for the filtered portfolio slice.

    Uses the longest configured client portfolio length among clients present
    in df. training_median_weeks and fallback_avg_weeks are ignored (kept for
    call-site compatibility).
    """
    del training_median_weeks, fallback_avg_weeks

    pl_cfg = portfolio_lengths_config(config)
    client_col = config.get("columns", {}).get("categorical_features", ["Client"])[0]

    if df is None or df.empty or client_col not in df.columns:
        return portfolio_length_to_months(pl_cfg.get("default_months", 12), pl_cfg)

    clients = df[client_col].dropna().astype(str).str.strip().unique()
    if len(clients) == 0:
        return portfolio_length_to_months(pl_cfg.get("default_months", 12), pl_cfg)

    return max(client_portfolio_months(client, config) for client in clients)


def _front_loaded_weights(n_months: int) -> np.ndarray:
    """Month shares when case timing is unknown (front-loaded collection curve)."""
    n = max(1, int(n_months))
    raw = np.exp(-0.35 * np.arange(n, dtype=float))
    return raw / raw.sum()


def _case_payout_month_index(
    import_ts: Optional[pd.Timestamp],
    payment_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
    client_months: int,
) -> Optional[int]:
    """
    Month index (0-based) within the client's portfolio horizon for this case.

    Prefers Payment date, then Date End, both measured from Import date.
    """
    n = max(1, int(client_months))
    if import_ts is not None and pd.notna(import_ts):
        if payment_ts is not None and pd.notna(payment_ts):
            days = (payment_ts - import_ts).days
            if days >= 0:
                return min(int(days // DAYS_PER_MONTH), n - 1)
        if end_ts is not None and pd.notna(end_ts):
            days = (end_ts - import_ts).days
            if days > 0:
                idx = int(math.ceil(days / DAYS_PER_MONTH)) - 1
                return min(max(0, idx), n - 1)
    return None


def _schedule_case_prediction(
    pred: float,
    client_months: int,
    import_ts: Optional[pd.Timestamp],
    payment_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
) -> Dict[int, float]:
    """Map a single case prediction to month indices (sums to pred)."""
    if pred <= 0:
        return {}
    n = max(1, int(client_months))
    month_idx = _case_payout_month_index(import_ts, payment_ts, end_ts, n)
    if month_idx is not None:
        return {month_idx: float(pred)}
    weights = _front_loaded_weights(n)
    return {i: float(pred * weights[i]) for i in range(n)}


def compute_monthly_breakdown(
    df: pd.DataFrame,
    effort_level: str,
    config: dict,
    payout_months: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build month-by-month predicted payout and remaining portfolio balance.

    Each case's predicted value at the selected effort level is placed in the
    month when it is expected to be collected, based on Payment date (or Date End)
    relative to Import date, capped to the client's configured portfolio length.
    Cases without timing use a front-loaded curve across that horizon.

    remaining_total[m] is the balance at the *start* of month m+1; month 1 starts
    at the full portfolio predicted total for the selected effort level.
    """
    pred_col = EFFORT_TO_PRED_COL.get(effort_level, "Pred_Medium")
    if pred_col not in df.columns:
        raise ValueError(f"Missing prediction column '{pred_col}' for effort '{effort_level}'")

    work = df.copy()
    work["_pred"] = pd.to_numeric(work[pred_col], errors="coerce").fillna(0.0)
    total_pred = float(work["_pred"].sum())

    cols_cfg = config.get("columns", {})
    client_col = cols_cfg.get("categorical_features", ["Client"])[0]
    date_cfg = cols_cfg.get("date_features", {})
    import_col = date_cfg.get("import_date", "Import date")
    end_col = date_cfg.get("end_date", "Date End")
    payment_col = "Payment date"

    imports = (
        pd.to_datetime(work[import_col], errors="coerce")
        if import_col in work.columns
        else None
    )
    payments = (
        pd.to_datetime(work[payment_col], errors="coerce")
        if payment_col in work.columns
        else None
    )
    ends = (
        pd.to_datetime(work[end_col], errors="coerce")
        if end_col in work.columns
        else None
    )

    segment_months = payout_months or resolve_payout_months_for_segment(work, config)
    monthly_by_index: Dict[int, float] = {}

    for pos, row in work.iterrows():
        pred = float(row["_pred"])
        if pred <= 0:
            continue

        if client_col in work.columns:
            client_name = str(row[client_col]).strip()
            client_months = client_portfolio_months(client_name, config)
        else:
            client_months = segment_months

        imp = imports.loc[pos] if imports is not None else pd.NaT
        pay = payments.loc[pos] if payments is not None else pd.NaT
        end = ends.loc[pos] if ends is not None else pd.NaT

        for idx, amount in _schedule_case_prediction(
            pred, client_months, imp, pay, end
        ).items():
            monthly_by_index[idx] = monthly_by_index.get(idx, 0.0) + amount

    if not monthly_by_index:
        n_months = max(1, segment_months)
        monthly_payout = np.zeros(n_months, dtype=float)
    else:
        n_months = min(max(max(monthly_by_index) + 1, 1), MAX_PAYOUT_MONTHS)
        monthly_payout = np.array(
            [monthly_by_index.get(m, 0.0) for m in range(n_months)],
            dtype=float,
        )
        if total_pred > 0:
            monthly_payout[-1] += total_pred - float(monthly_payout.sum())

    remaining = np.zeros(n_months, dtype=float)
    balance = total_pred
    for m in range(n_months):
        remaining[m] = balance
        balance = max(balance - monthly_payout[m], 0.0)

    return pd.DataFrame(
        {
            "month": np.arange(1, n_months + 1, dtype=int),
            "monthly_payout": monthly_payout,
            "remaining_total": remaining,
            "total_predicted": total_pred,
        }
    )


def list_portfolio_length_clients(config: dict) -> List[str]:
    """Sorted client names for the settings UI (configured + defaults)."""
    pl_cfg = portfolio_lengths_config(config)
    return sorted(pl_cfg.get("by_client", {}).keys(), key=lambda s: s.casefold())
