"""Consolidate every task's JSON into results/results.json (the single source for README numbers)."""

import json

from common import RESULTS, save_json


def load(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


naive, lgbm, lstm = load("naive.json"), load("lgbm.json"), load("lstm.json")
promise, capacity = load("promise.json"), load("capacity.json")

forecast = {
    m["model"]: {"wape": m["wape"], "bias": m["bias"],
                 "per_cutoff_wape": {p["cutoff"]: p["wape"] for p in m["per_cutoff"]}}
    for m in (naive, lgbm, lstm) if m
}
out = {
    "task_a_forecast": {
        "models": forecast,
        "lgbm_vs_naive_wape_reduction_pct": 1 - lgbm["wape"] / naive["wape"],
        "lstm_vs_naive_wape_reduction_pct": 1 - lstm["wape"] / naive["wape"] if lstm else None,
        "lgbm_p90_coverage": lgbm["p90_coverage"],
    },
    "task_b_promise": {k: promise[k] for k in
                       ("headline", "matched_late_rate", "validation_rule", "olist", "p50", "p90",
                        "p50_mae_days", "split")},
    "task_c_capacity": capacity["headline"] if capacity else None,
}
save_json(out, "results.json")

a, b = out["task_a_forecast"], out["task_b_promise"]["headline"]
print(f"WAPE naive {naive['wape']:.1%} -> LightGBM {lgbm['wape']:.1%} "
      f"({a['lgbm_vs_naive_wape_reduction_pct']:.0%} lower); LSTM {lstm['wape']:.1%}")
print(f"P90 promise {b['model_avg_promise_days']:.1f} d vs Olist {b['olist_avg_promise_days']:.1f} d "
      f"({b['promise_reduction_pct']:.1%} shorter); late {b['model_late_rate']:.1%} vs {b['olist_late_rate']:.1%}")
