"""
Paid-value prediction cap for portfolio simulation experiments.

Toggle USE_PAID_PREDICTION_CAP below to compare model output with or without
per-case capping. Not exposed in the GUI.

Ratio and other limits are read from config.json → prediction_cap.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

# ---------------------------------------------------------------------------
# Flip this flag to disable all prediction capping (raw model output only).
# ---------------------------------------------------------------------------
USE_PAID_PREDICTION_CAP = True


def is_prediction_cap_active(config: dict) -> bool:
    """True when both the code switch and config allow capping."""
    if not USE_PAID_PREDICTION_CAP:
        return False
    cap_cfg = config.get("prediction_cap") or {}
    return bool(cap_cfg.get("enabled", True))


def apply_paid_prediction_cap(
    preds: np.ndarray,
    case_values: np.ndarray,
    config: dict,
) -> np.ndarray:
    """
    Optionally cap each predicted paid value.

    - Floor at 0
    - Ceiling at case_value * per_case_max_paid_to_case_ratio (default 0.8)
    - Optional hard ceiling at case_value when also_cap_at_case_value is true
    """
    out = np.asarray(preds, dtype=np.float64).copy()
    out = np.where(np.isfinite(out), out, 0.0)
    out = np.maximum(out, 0.0)

    if not is_prediction_cap_active(config):
        return out

    cap_cfg: Dict[str, Any] = config.get("prediction_cap") or {}
    ratio = float(cap_cfg.get("per_case_max_paid_to_case_ratio", 0.8))
    also_at_cv = bool(cap_cfg.get("also_cap_at_case_value", False))

    cv = np.maximum(np.asarray(case_values, dtype=np.float64), 0.0)
    max_paid = cv * ratio
    if also_at_cv:
        max_paid = np.minimum(max_paid, cv)

    return np.minimum(out, max_paid)
