"""Metrics in original target space (Paid Value or ratio per config)."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    err = y_true - y_pred
    return float(np.mean(np.maximum(alpha * err, (alpha - 1) * err)))


def regression_metrics(
    y_true_orig: np.ndarray,
    y_pred_orig: np.ndarray,
) -> Dict[str, float]:
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse = mean_squared_error(y_true_orig, y_pred_orig) ** 0.5
    return {"mae": float(mae), "rmse": float(rmse)}


def inverse_targets(
    y_scaled: np.ndarray,
    target_scaler,
) -> np.ndarray:
    arr = np.asarray(y_scaled).reshape(-1, 1)
    return target_scaler.inverse_transform(arr).reshape(-1)
