"""Task B: delivery-time promise with quantile LightGBM (purchase-time features only).

Split by purchase date: train 2017-01-01..2018-03-31, test 2018-04-01..2018-06-30
(a held-out quarter whose orders had time to be delivered before the data snapshot).
The promise is ceil(predicted quantile) days after purchase and is compared with Olist's
own promise on the same orders (late rate = share delivered after the promised day).

Required models: P50 and P90. The headline is the pre-registered P90 model vs Olist, so no
choice is made on the test set. Two diagnostics are also saved:
  - validation rule: fit on 2017, pick the lowest alpha whose 2018-Q1 late rate <= Olist's,
    refit, score on test (shows how late-rate calibration transfers across quarters);
  - the full alpha sweep on test (trade-off curve; its "matched" point is test-selected).
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from common import RESULTS, SEED, load_orders_b, save_json, seed_everything

seed_everything()

TRAIN_END, VAL_START = pd.Timestamp("2018-04-01"), pd.Timestamp("2018-01-01")
TEST_END = pd.Timestamp("2018-07-01")
ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
NUMERIC = ["distance_km", "freight", "weight_g", "n_items", "purchase_dow"]
CATEGORICAL = ["customer_state", "seller_state"]
FEATURES = NUMERIC + CATEGORICAL

df = load_orders_b()
for col in CATEGORICAL:
    df[col] = pd.Categorical(df[col], categories=sorted(df[col].unique()))

train = df[df.purchase_ts < TRAIN_END]
test = df[(df.purchase_ts >= TRAIN_END) & (df.purchase_ts < TEST_END)]
fit_part, val_part = train[train.purchase_ts < VAL_START], train[train.purchase_ts >= VAL_START]
print(f"orders: train {len(train):,} (fit {len(fit_part):,} / val {len(val_part):,}), test {len(test):,}")


def quantile_model(alpha: float, data: pd.DataFrame) -> LGBMRegressor:
    m = LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=500, learning_rate=0.05,
                      num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
                      random_state=SEED, deterministic=True, force_row_wise=True, verbose=-1)
    return m.fit(data[FEATURES], data.actual_days)


def evaluate(promise: np.ndarray, actual: pd.Series) -> dict:
    return {"late_rate": float((actual.to_numpy() > promise).mean()),
            "avg_promise_days": float(promise.mean()),
            "median_promise_days": float(np.median(promise))}


def olist(part: pd.DataFrame) -> dict:
    return evaluate(part.olist_promise_days.to_numpy(), part.actual_days)


# ---- 1. choose alpha on validation (2018-Q1) --------------------------------------------
olist_val = olist(val_part)
val_sweep = {}
for a in ALPHAS:
    val_sweep[a] = evaluate(np.ceil(quantile_model(a, fit_part).predict(val_part[FEATURES])),
                            val_part.actual_days)
eligible = [a for a in ALPHAS if val_sweep[a]["late_rate"] <= olist_val["late_rate"]]
chosen = min(eligible) if eligible else max(ALPHAS)
print(f"validation: Olist late {olist_val['late_rate']:.3f}; chosen alpha={chosen}")

# ---- 2. refit on the full training period, score every alpha on the test quarter ----------
olist_test = olist(test)
sweep, test_preds = [], {}
for a in ALPHAS:
    promise = np.ceil(quantile_model(a, train).predict(test[FEATURES])).clip(min=1)
    test_preds[a] = promise
    r = {"alpha": a, **evaluate(promise, test.actual_days)}
    sweep.append(r)
    print(f"  alpha={a:.2f}: late={r['late_rate']:.3f} avg promise={r['avg_promise_days']:.2f}")
print(f"Olist test: late={olist_test['late_rate']:.3f} avg promise={olist_test['avg_promise_days']:.2f}")

by_alpha = {r["alpha"]: r for r in sweep}


def vs_olist(alpha: float, note: str) -> dict:
    r = by_alpha[alpha]
    saved = olist_test["avg_promise_days"] - r["avg_promise_days"]
    return {"alpha": alpha, "note": note,
            "model_late_rate": r["late_rate"], "olist_late_rate": olist_test["late_rate"],
            "model_avg_promise_days": r["avg_promise_days"],
            "olist_avg_promise_days": olist_test["avg_promise_days"],
            "promise_reduction_days": float(saved),
            "promise_reduction_pct": float(saved / olist_test["avg_promise_days"])}


matched = min([a for a in ALPHAS if by_alpha[a]["late_rate"] <= olist_test["late_rate"]], default=max(ALPHAS))

# Late rate per month of the test quarter (May 2018 = truckers' strike)
monthly = []
for m, g in test.assign(p=test_preds[0.9]).groupby(test.purchase_ts.dt.to_period("M")):
    monthly.append({"month": str(m), "olist_late_rate": float(g.late.mean()),
                    "model_late_rate": float((g.actual_days > g.p).mean()),
                    "olist_avg_promise": float(g.olist_promise_days.mean()),
                    "model_avg_promise": float(g.p.mean()), "n": len(g)})

res = {
    "split": {"train": "2017-01-01..2018-03-31", "validation_for_alpha": "2018-01-01..2018-03-31",
              "test": "2018-04-01..2018-06-30", "n_train": len(train), "n_test": len(test)},
    "features": FEATURES,
    "actual_days_test": {"mean": float(test.actual_days.mean()), "median": float(test.actual_days.median())},
    "olist": {"validation": olist_val, "test": olist_test},
    "p50": by_alpha[0.5],
    "p90": by_alpha[0.9],
    "p50_mae_days": float(np.abs(test_preds[0.5] - test.actual_days.to_numpy()).mean()),
    "p90_coverage_test": 1 - by_alpha[0.9]["late_rate"],
    "headline": vs_olist(0.9, "pre-registered P90 model, no selection on test"),
    "matched_late_rate": vs_olist(matched, "lowest alpha with test late rate <= Olist's (selected on test)"),
    "validation_rule": vs_olist(chosen, "alpha chosen on 2018-Q1 validation to match Olist's late rate there"),
    "sweep_test": sweep,
    "sweep_validation": [{"alpha": a, **v} for a, v in val_sweep.items()],
    "monthly_test": monthly,
}
h = res["headline"]
print(f"HEADLINE P90: late {h['model_late_rate']:.3f} vs Olist {h['olist_late_rate']:.3f}; "
      f"promise {h['model_avg_promise_days']:.2f} vs {h['olist_avg_promise_days']:.2f} days "
      f"({h['promise_reduction_days']:.2f} d / {h['promise_reduction_pct']:.1%} shorter)")
m = res["matched_late_rate"]
print(f"matched alpha={matched}: late {m['model_late_rate']:.3f}, {m['promise_reduction_days']:.2f} d shorter")
save_json(res, "promise.json")
