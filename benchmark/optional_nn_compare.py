"""Optional NN test metrics (same as test.py) when checkpoint exists."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from split import load_config  # noqa: E402


def _load_nn_artifacts(config_path: str) -> Optional[
    Tuple[dict, Any, Any, Any, Any, dict]
]:
    import torch

    config = load_config(config_path)
    data_cfg = config["data"]
    artifacts_dir = data_cfg["artifacts_dir"]
    cfg_dir = Path(config_path).resolve().parent
    art = cfg_dir / artifacts_dir
    ckpt = art / "model.pth"
    if not ckpt.is_file():
        return None

    import joblib

    test_path = data_cfg["test_path"]
    if not Path(test_path).is_absolute():
        test_path = str(cfg_dir / test_path)
    import pandas as pd

    lower = test_path.lower()
    if lower.endswith(".csv"):
        try:
            test_df = pd.read_csv(test_path, encoding="utf-8")
        except (UnicodeDecodeError, UnicodeError):
            test_df = pd.read_csv(test_path, encoding="cp1252")
    else:
        test_df = pd.read_excel(test_path)

    encoder = joblib.load(str(art / "encoder.pkl"))
    scaler = joblib.load(str(art / "scaler.pkl"))
    target_scaler = joblib.load(str(art / "target_scaler.pkl"))
    try:
        checkpoint = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(ckpt), map_location="cpu")
    return config, test_df, encoder, scaler, target_scaler, checkpoint


def nn_test_metrics(config_path: str = "config.json") -> Optional[Dict[str, float]]:
    import torch
    from train import PortfolioDataset, PortfolioModel

    loaded = _load_nn_artifacts(config_path)
    if loaded is None:
        return None
    config, test_df, encoder, scaler, target_scaler, checkpoint = loaded

    dataset = PortfolioDataset(test_df, config, encoder, scaler, target_scaler)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        drop_last=False,
    )

    cardinalities = checkpoint["cardinalities"]
    numeric_input_dim = checkpoint["numeric_input_dim"]
    model = PortfolioModel(cardinalities, numeric_input_dim, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds: list = []
    all_targets: list = []

    with torch.no_grad():
        for x_cat, x_num, y in loader:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            y = y.to(device)
            preds = model(x_cat, x_num)
            all_preds.append(preds.cpu().numpy().reshape(-1))
            all_targets.append(y.cpu().numpy().reshape(-1))

    y_true_scaled = np.concatenate(all_targets)
    y_pred_scaled = np.concatenate(all_preds)
    y_true = target_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).reshape(-1)
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {"mae": float(mae), "rmse": float(rmse), "model": "neural_net"}


def print_nn_compare(config_path: str, output_dir: Optional[Path] = None) -> None:
    m = nn_test_metrics(config_path)
    if m is None:
        print("NN compare: no checkpoint at models/model.pth (skipped).")
        return
    print(f"NN test MAE:  {m['mae']:.6f}")
    print(f"NN test RMSE: {m['rmse']:.6f}")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "nn_compare.json", "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
