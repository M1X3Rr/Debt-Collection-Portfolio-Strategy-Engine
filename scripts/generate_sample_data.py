"""
Generate synthetic training/validation/test data for the public demo repository.

All client names, locations, and products are fictional. No real portfolio data
is included. Run from the project root:

    python scripts/generate_sample_data.py
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timedelta

import pandas as pd


CLIENTS = [
    "Client_A Finance",
    "Client_B Bank",
    "Client_C Credit",
    "Client_D Retail",
    "Client_E Bank",
    "Utility Corp East",
    "Telco Primary",
    "Telco Secondary",
    "ISP Provider A",
    "DirectMarketing",
]

LOCATIONS = [
    "North Region",
    "South Region",
    "East Region",
    "West Region",
    "Central District",
    "Metro Area",
]

PRODUCTS = ["Product_A", "Product_B", "Product_C", "Product_D"]

NAMES = [
    "Alex Morgan",
    "Jamie Carter",
    "Taylor Reed",
    "Jordan Brooks",
    "Casey Riley",
    "Riley Quinn",
    "Morgan Ellis",
    "Cameron Lee",
    "Sydney Blake",
    "Peyton Avery",
]


def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_rows(n_cases: int, seed: int, import_label: str) -> list[dict]:
    rng = random.Random(seed)
    base_date = datetime(2022, 1, 1)
    rows = []
    for i in range(n_cases):
        case_value = round(rng.uniform(200, 25000), 2)
        dpd = int(rng.uniform(15, 240))
        debtor_age = round(rng.uniform(22, 72), 1)
        recovery_rate = max(0.02, min(0.95, rng.gauss(0.35, 0.15)))
        paid_value = round(case_value * recovery_rate, 2)
        calls = int(rng.uniform(0, 40))
        letters = int(rng.uniform(0, 5))
        sms = int(rng.uniform(0, 30))
        emails = int(rng.uniform(0, 25))
        weeks_open = max(2, int(rng.uniform(4, 78)))
        import_dt = base_date + timedelta(days=rng.randint(0, 700))
        end_dt = import_dt + timedelta(weeks=weeks_open)
        payment_dt = end_dt + timedelta(days=rng.randint(0, 21))
        rows.append(
            {
                "#": i + 1,
                "Case ID": f"DEMO-{import_label}-{i + 1:05d}",
                "Name": rng.choice(NAMES),
                "Client": rng.choice(CLIENTS),
                "Location": rng.choice(LOCATIONS),
                "Product": rng.choice(PRODUCTS),
                "Case Value": case_value,
                "DPD": dpd,
                "Debtor Age": debtor_age,
                "Calls": calls,
                "Letters": letters,
                "SMS": sms,
                "Emails": emails,
                "Paid Value": paid_value,
                "Payment date": payment_dt.strftime("%Y-%m-%d"),
                "Import date": import_dt.strftime("%Y-%m-%d"),
                "Date End": end_dt.strftime("%Y-%m-%d"),
                "Import Name": import_label,
            }
        )
    return rows


def main() -> None:
    config = load_config()
    random_state = int(config.get("training", {}).get("random_state", 42))
    rng = random.Random(random_state)

    all_rows = []
    for idx in range(12):
        all_rows.extend(generate_rows(120, seed=random_state + idx, import_label=f"Import_{idx + 1:02d}"))

    df = pd.DataFrame(all_rows)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    n = len(df)
    n_test = int(n * 0.30)
    n_valid = int(n * 0.15)
    n_train = n - n_test - n_valid

    train_df = df.iloc[:n_train].copy()
    valid_df = df.iloc[n_train : n_train + n_valid].copy()
    test_df = df.iloc[n_train + n_valid :].copy()

    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/valid", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)

    train_path = config["data"]["train_path"]
    valid_path = config["data"]["valid_path"]
    test_path = config["data"]["test_path"]
    sample_path = config["data"]["input_path"]

    train_df.to_excel(train_path, index=False)
    valid_df.to_excel(valid_path, index=False)
    test_df.to_excel(test_path, index=False)
    test_df.head(80).to_excel(sample_path, index=False)

    print(f"Wrote {len(train_df)} train rows -> {train_path}")
    print(f"Wrote {len(valid_df)} valid rows -> {valid_path}")
    print(f"Wrote {len(test_df)} test rows -> {test_path}")
    print(f"Wrote sample upload file -> {sample_path}")


if __name__ == "__main__":
    main()
