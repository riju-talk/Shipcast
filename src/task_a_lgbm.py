"""Task A: one global LightGBM with direct multi-horizon prediction (horizon is a feature).

Each training row is (state, origin d, horizon h) -> y(state, d + h), with features taken
from the SQL table at the origin d plus calendar flags of the target date (known in advance).
For cutoff c only rows with d + h <= c are used, so no target ever falls after the cutoff.
The forecast for c + h uses the origin row d = c.

Two models per cutoff:
  - Tweedie point forecast (scored with WAPE / bias)
  - Quantile alpha=0.9 (P90) forecast, consumed by the capacity LP in Task C
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from common import (CUTOFFS, HORIZON, RESULTS, SEED, TRAIN_START, load_daily_volume,
                    save_json, score_backtest, seed_everything)

seed_everything()

ORIGIN_FEATURES = ["y", "lag1", "lag7", "lag14", "lag28", "mean7", "mean28", "std7", "std28",
                   "dow", "month"]
FEATURES = ORIGIN_FEATURES + ["horizon", "snaive", "target_dow", "target_is_holiday",
                              "target_is_black_friday", "state"]

vol = load_daily_volume()
states = sorted(vol.state.unique())
vol["state"] = pd.Categorical(vol.state, categories=states)


def stack_horizons(vol: pd.DataFrame) -> pd.DataFrame:
    """Long table of (state, origin, horizon) rows with target and target-date calendar features."""
    g = vol.groupby("state", observed=True)
    frames = []
    for h in range(1, HORIZON + 1):
        f = vol[["state", "d"] + ORIGIN_FEATURES].copy()
        f["horizon"] = h
        f["target_d"] = f.d + pd.Timedelta(days=h)
        f["target"] = g.y.shift(-h)
        # Same weekday in the last observed week: y(d + h - 7*ceil(h/7)), always <= origin
        f["snaive"] = g.y.shift(7 * int(np.ceil(h / 7)) - h)
        f["target_dow"] = f.target_d.dt.dayofweek + 1
        f["target_is_holiday"] = g.is_holiday.shift(-h)
        f["target_is_black_friday"] = g.is_black_friday.shift(-h)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


panel = stack_horizons(vol)
# Calendar flags for target dates beyond the data end: none of them is a holiday
panel[["target_is_holiday", "target_is_black_friday"]] = (
    panel[["target_is_holiday", "target_is_black_friday"]].fillna(0))

params = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              random_state=SEED, deterministic=True, force_row_wise=True, verbose=-1)

actual = vol.set_index(["state", "d"]).y
point_rows, p90_rows = [], []
for c in CUTOFFS:
    train = panel[(panel.d >= TRAIN_START) & (panel.target_d <= c)]
    assert train.target_d.max() <= c and train.target.notna().all()
    test = panel[panel.d == c].copy()
    assert len(test) == len(states) * HORIZON

    point = LGBMRegressor(objective="tweedie", tweedie_variance_power=1.2, **params)
    point.fit(train[FEATURES], train.target)
    p90 = LGBMRegressor(objective="quantile", alpha=0.9, **params)
    p90.fit(train[FEATURES], train.target)

    test["actual"] = actual.loc[list(zip(test.state, test.target_d))].to_numpy()
    base = test[["state", "target_d", "horizon", "actual"]].rename(columns={"target_d": "d"})
    base.insert(0, "cutoff", c)
    point_rows.append(base.assign(forecast=np.clip(point.predict(test[FEATURES]), 0, None)))
    p90_rows.append(base.assign(forecast=np.clip(p90.predict(test[FEATURES]), 0, None)))
    print(f"  cutoff {c.date()}: {len(train):,} training rows")

preds = pd.concat(point_rows, ignore_index=True)
preds_p90 = pd.concat(p90_rows, ignore_index=True)
preds.to_csv(RESULTS / "preds_lgbm.csv", index=False)
preds_p90.to_csv(RESULTS / "preds_lgbm_p90.csv", index=False)

res = score_backtest(preds, "lightgbm_global")
res["p90_coverage"] = float((preds_p90.actual <= preds_p90.forecast).mean())
res["feature_importance_gain"] = dict(sorted(
    zip(FEATURES, point.booster_.feature_importance("gain").round(1).tolist()),
    key=lambda kv: -kv[1]))
for p in res["per_cutoff"]:
    print(f"  {p['cutoff']}: WAPE={p['wape']:.4f} bias={p['bias']:+.4f}")
print(f"LightGBM WAPE={res['wape']:.4f} bias={res['bias']:+.4f}  P90 coverage={res['p90_coverage']:.3f}")
save_json(res, "lgbm.json")
