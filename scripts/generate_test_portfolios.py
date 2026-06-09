"""
Generate 10 test portfolios (worst to best) for model evaluation.

Portfolios vary in Case Value, Paid Value, DPD, Debtor Age, Client, Location,
Product, and action counts. Use with run_portfolio_tests.py to compute errors
and devise mitigation plans for data, calculations, or algorithms.

Usage:
    python scripts/generate_test_portfolios.py
    # Writes data/test_portfolios/portfolio_01_worst.xlsx ... portfolio_10_best.xlsx
"""

import json
import os
import random
from datetime import datetime, timedelta

import pandas as pd


def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_portfolio(
    n_cases: int,
    profile: str,
    seed: int,
    config: dict,
) -> pd.DataFrame:
    """
    Generate a single portfolio with the given profile (worst -> best).
    profile: "worst", "bad", ..., "best" (10 levels).
    """
    rng = random.Random(seed)
    cols_cfg = config["columns"]
    cat_features = cols_cfg.get("categorical_features", ["Client", "Location", "Product"])
    action_features = cols_cfg.get("action_features", ["Calls", "Letters", "SMS", "Emails"])
    clients = [f"Client_{i}" for i in range(1, 6)]
    locations = ["North", "South", "East", "West"]
    products = ["Product_A", "Product_B", "Product_C"]

    # Map profile index 0..9 to scaling factors (0 = worst, 9 = best)
    try:
        level = int(profile) if isinstance(profile, str) and profile.isdigit() else {"worst": 0, "best": 9}.get(profile, 4)
    except (ValueError, TypeError):
        level = 4
    if level < 0:
        level = 0
    if level > 9:
        level = 9
    t = level / 9.0  # 0 .. 1

    rows = []
    base_date = datetime(2023, 1, 1)
    for i in range(n_cases):
        # Case Value: worse portfolios have lower values or more variance
        case_value = round(rng.uniform(500 + t * 2000, 5000 + t * 15000), 2)
        # DPD: worse = higher DPD
        dpd = int(rng.uniform(30 * (1 - t) + 10, 180 * (1 - t) + 90 * t))
        # Debtor Age (years): arbitrary range
        debtor_age = round(rng.uniform(25, 65), 1)
        # Paid Value: worse = lower recovery rate
        recovery_rate = 0.05 + 0.45 * t + rng.uniform(-0.05, 0.1)
        recovery_rate = max(0, min(1, recovery_rate))
        paid_value = round(case_value * recovery_rate, 2)
        # Actions: worse = fewer
        calls = int(rng.uniform(0, 5 + 15 * t))
        letters = int(rng.uniform(0, 1 + 3 * t))
        sms = int(rng.uniform(0, 5 + 20 * t))
        emails = int(rng.uniform(0, 5 + 15 * t))
        weeks_open = max(1, int(rng.uniform(4, 52)))
        import_dt = base_date + timedelta(days=rng.randint(0, 200))
        end_dt = import_dt + timedelta(weeks=weeks_open)
        row = {
            "#": i + 1,
            "Case ID": f"CID-{profile}-{i+1:04d}",
            "Client": rng.choice(clients),
            "Location": rng.choice(locations),
            "Product": rng.choice(products),
            "Case Value": case_value,
            "DPD": dpd,
            "Debtor Age": debtor_age,
            "Calls": calls,
            "Letters": letters,
            "SMS": sms,
            "Emails": emails,
            "Paid Value": paid_value,
            "Payment date": end_dt.strftime("%Y-%m-%d"),
            "Import date": import_dt.strftime("%Y-%m-%d"),
            "Date End": end_dt.strftime("%Y-%m-%d"),
            "Import Name": f"Import_{profile}",
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def main():
    config = load_config()
    out_dir = os.path.join("data", "test_portfolios")
    os.makedirs(out_dir, exist_ok=True)
    labels = [
        "worst", "very_bad", "bad", "below_avg", "below_avg2",
        "avg", "above_avg", "good", "very_good", "best",
    ]
    n_cases_per_portfolio = 50
    for idx, label in enumerate(labels):
        df = generate_portfolio(n_cases_per_portfolio, label, seed=42 + idx, config=config)
        path = os.path.join(out_dir, f"portfolio_{idx+1:02d}_{label}.xlsx")
        df.to_excel(path, index=False)
        print(f"Wrote {path} ({len(df)} cases)")
    print(f"Done. {len(labels)} portfolios in {out_dir}")


if __name__ == "__main__":
    main()
