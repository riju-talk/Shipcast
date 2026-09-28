"""Shared config, SQL loading and metrics for the ShipCast pipeline."""

import json
import random
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = ROOT / "sql"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42
HORIZON = 14
# Rolling-origin backtest: 4 cutoffs, 21 days apart; the last one leaves exactly 14 days of data.
CUTOFFS = pd.to_datetime(["2018-06-05", "2018-06-26", "2018-07-17", "2018-08-07"])
# Spine starts 2017-01-01; training origins start a month later so lag28/mean28 are fully populated.
TRAIN_START = pd.Timestamp("2017-02-01")

# Macro-regions used by the capacity LP (Task C).
REGIONS = {
    "North": ["AC", "AM", "AP", "PA", "RO", "RR", "TO"],
    "Northeast": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Center-West": ["DF", "GO", "MS", "MT"],
    "Southeast": ["ES", "MG", "RJ", "SP"],
    "South": ["PR", "RS", "SC"],
}
STATE_TO_REGION = {s: r for r, states in REGIONS.items() for s in states}


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_sql(name: str, table: str) -> pd.DataFrame:
    """Execute a SQL script from sql/ (run from the repo root) and return the table it creates."""
    con = duckdb.connect()
    con.execute(f"SET file_search_path = '{ROOT.as_posix()}'")
    con.execute((SQL_DIR / name).read_text())
    return con.execute(f"SELECT * FROM {table}").df()


def load_daily_volume() -> pd.DataFrame:
    df = run_sql("01_daily_volume.sql", "daily_volume")
    df["d"] = pd.to_datetime(df["d"])
    return df.sort_values(["state", "d"]).reset_index(drop=True)


def load_orders_b() -> pd.DataFrame:
    df = run_sql("02_orders_b.sql", "orders_b")
    df["purchase_ts"] = pd.to_datetime(df["purchase_ts"])
    return df


def wape(actual, forecast) -> float:
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    return float(np.abs(forecast - actual).sum() / actual.sum())


def bias(actual, forecast) -> float:
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    return float((forecast - actual).sum() / actual.sum())


def score_backtest(preds: pd.DataFrame, model: str) -> dict:
    """preds: columns cutoff, state, d, horizon, actual, forecast."""
    per_cutoff = [
        {"cutoff": c.strftime("%Y-%m-%d"), "wape": wape(g.actual, g.forecast),
         "bias": bias(g.actual, g.forecast), "n": len(g)}
        for c, g in preds.groupby("cutoff")
    ]
    by_h = preds.groupby("horizon").apply(lambda g: wape(g.actual, g.forecast), include_groups=False)
    return {
        "model": model,
        "wape": wape(preds.actual, preds.forecast),
        "bias": bias(preds.actual, preds.forecast),
        "mean_cutoff_wape": float(np.mean([p["wape"] for p in per_cutoff])),
        "per_cutoff": per_cutoff,
        "wape_by_horizon": {int(h): float(v) for h, v in by_h.items()},
    }


def save_json(obj, name: str) -> None:
    (RESULTS / name).write_text(json.dumps(obj, indent=2))
    print(f"saved results/{name}")
