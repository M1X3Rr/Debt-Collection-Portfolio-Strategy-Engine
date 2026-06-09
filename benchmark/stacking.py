"""Stack Ridge + tree: meta Ridge on validation predictions (no test leakage)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def stack_predictions(
    pred_r_valid: np.ndarray,
    pred_r_test: np.ndarray,
    pred_t_valid: np.ndarray,
    pred_t_test: np.ndarray,
    y_valid: np.ndarray,
) -> tuple[np.ndarray, Ridge]:
    meta = Ridge(alpha=1e-3, random_state=42)
    meta.fit(
        np.column_stack([pred_r_valid, pred_t_valid]),
        y_valid,
    )
    pred_test = meta.predict(np.column_stack([pred_r_test, pred_t_test]))
    return pred_test, meta
