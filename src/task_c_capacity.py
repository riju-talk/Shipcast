"""Task C (optional): split a fixed daily capacity across the 5 Brazilian macro-regions.

For each backtest cutoff, capacity C is fixed for the next 14 days at the trailing 28-day
mean of total daily orders (a network sized to recent typical demand), and scaled by
0.9 / 1.0 / 1.1 for sensitivity. Both policies only use information available at the cutoff.

  - history:  x_r = C * (region share of orders in the 28 days up to the cutoff)
  - LP:       OR-Tools GLOP, per day. Region demand is summarised by two forecast levels from
              Task A (LightGBM point forecast and P90, summed over the region's states).
              Units up to the point forecast are "likely needed" (weight 1.0); units between the
              point forecast and P90 are "possibly needed" (weight 0.3). The LP minimises the
              weighted shortfall, a piecewise-linear proxy for expected unmet demand:
                  min  sum_r 1.0*s1_r + 0.3*s2_r
                  s.t. sum_r x_r <= C
                       x_r + s1_r >= point_r
                       x_r + s1_r + s2_r >= p90_r
                       x, s1, s2 >= 0
              A tiny tie-break term spreads cuts proportionally within a tier, and any capacity
              left above the regions' P90 is shared out in proportion to P90.

Both are scored on ACTUAL demand: unmet = sum over regions and days of max(0, actual - x).
"""

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from common import CUTOFFS, REGIONS, RESULTS, STATE_TO_REGION, load_daily_volume, save_json

W_LIKELY, W_POSSIBLE = 1.0, 0.3
CAPACITY_FACTORS = [0.9, 1.0, 1.1]
regions = list(REGIONS)


def to_region(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return (df.assign(region=df.state.map(STATE_TO_REGION))
              .groupby(["cutoff", "d", "region"])[value].sum().unstack("region")[regions])


def solve_lp(point: np.ndarray, p90: np.ndarray, cap: float) -> np.ndarray:
    p90 = np.maximum(p90, point)
    solver = pywraplp.Solver.CreateSolver("GLOP")
    n = len(regions)
    x = [solver.NumVar(0, solver.infinity(), f"x{i}") for i in range(n)]
    s1 = [solver.NumVar(0, solver.infinity(), f"s1_{i}") for i in range(n)]
    s2 = [solver.NumVar(0, solver.infinity(), f"s2_{i}") for i in range(n)]
    z1, z2 = solver.NumVar(0, 1, "z1"), solver.NumVar(0, 1, "z2")   # max shortfall fraction per tier
    solver.Add(sum(x) <= cap)
    for i in range(n):
        solver.Add(x[i] + s1[i] >= point[i])
        solver.Add(x[i] + s1[i] + s2[i] >= p90[i])
        solver.Add(s1[i] <= z1 * point[i])
        solver.Add(s2[i] <= z2 * (p90[i] - point[i]))
    eps = 1e-3 * cap
    solver.Minimize(sum(W_LIKELY * s1[i] + W_POSSIBLE * s2[i] for i in range(n)) + eps * (z1 + z2))
    assert solver.Solve() == pywraplp.Solver.OPTIMAL
    alloc = np.array([v.solution_value() for v in x])
    leftover = cap - alloc.sum()
    if leftover > 1e-6:
        alloc += leftover * p90 / p90.sum()
    return alloc


vol = load_daily_volume()
daily_region = (vol.assign(region=vol.state.map(STATE_TO_REGION))
                   .groupby(["d", "region"]).y.sum().unstack("region")[regions])
point = to_region(pd.read_csv(RESULTS / "preds_lgbm.csv", parse_dates=["cutoff", "d"]), "forecast")
p90 = to_region(pd.read_csv(RESULTS / "preds_lgbm_p90.csv", parse_dates=["cutoff", "d"]), "forecast")

results, detail = [], []
for factor in CAPACITY_FACTORS:
    tot = {"history": 0.0, "lp": 0.0, "oracle": 0.0, "demand": 0.0}
    for c in CUTOFFS:
        trailing = daily_region.loc[c - pd.Timedelta(days=27):c]
        cap = factor * trailing.sum(axis=1).mean()
        share = (trailing.sum() / trailing.to_numpy().sum()).to_numpy()
        for d, pt in point.loc[c].iterrows():
            actual = daily_region.loc[d].to_numpy()
            x_hist = cap * share
            x_lp = solve_lp(pt.to_numpy(), p90.loc[c].loc[d].to_numpy(), cap)
            u_hist = np.maximum(actual - x_hist, 0).sum()
            u_lp = np.maximum(actual - x_lp, 0).sum()
            tot["history"] += u_hist
            tot["lp"] += u_lp
            tot["oracle"] += max(actual.sum() - cap, 0)       # best any split can do
            tot["demand"] += actual.sum()
            if factor == 1.0:
                detail.append({"cutoff": c.strftime("%Y-%m-%d"), "d": d.strftime("%Y-%m-%d"),
                               "capacity": cap, "demand": float(actual.sum()),
                               "unmet_history": float(u_hist), "unmet_lp": float(u_lp)})
    r = {"capacity_factor": factor,
         "unmet_history": tot["history"], "unmet_lp": tot["lp"], "unmet_oracle": tot["oracle"],
         "total_demand": tot["demand"],
         "unmet_history_pct": tot["history"] / tot["demand"], "unmet_lp_pct": tot["lp"] / tot["demand"],
         "unmet_oracle_pct": tot["oracle"] / tot["demand"],
         "lp_reduction_pct": 1 - tot["lp"] / tot["history"]}
    results.append(r)
    print(f"capacity x{factor}: unmet history={r['unmet_history_pct']:.2%} LP={r['unmet_lp_pct']:.2%} "
          f"oracle={r['unmet_oracle_pct']:.2%}  -> LP cuts unmet demand by {r['lp_reduction_pct']:.1%}")

save_json({"regions": REGIONS, "weights": {"likely": W_LIKELY, "possible": W_POSSIBLE},
           "results": results, "headline": next(r for r in results if r["capacity_factor"] == 1.0),
           "daily_capacity_1x": detail}, "capacity.json")
