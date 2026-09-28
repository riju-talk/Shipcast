"""Task A baseline: weekly seasonal-naive forecast on the rolling-origin backtest.

y_hat(c + h) = y(c + h - 7 * ceil(h / 7)) -> h=1..7 repeats last week, h=8..14 repeats it again.
Only reads data <= cutoff. Writes results/naive.json and results/preds_naive.csv.
"""

import numpy as np
import pandas as pd

from common import CUTOFFS, HORIZON, RESULTS, load_daily_volume, save_json, score_backtest

vol = load_daily_volume()
y = vol.pivot(index="d", columns="state", values="y")

rows = []
for c in CUTOFFS:
    hist = y.loc[:c]
    for h in range(1, HORIZON + 1):
        d = c + pd.Timedelta(days=h)
        src = d - pd.Timedelta(days=7 * int(np.ceil(h / 7)))
        assert src <= c
        for state in y.columns:
            rows.append((c, state, d, h, y.at[d, state], hist.at[src, state]))

preds = pd.DataFrame(rows, columns=["cutoff", "state", "d", "horizon", "actual", "forecast"])
preds.to_csv(RESULTS / "preds_naive.csv", index=False)

res = score_backtest(preds, "seasonal_naive")
for p in res["per_cutoff"]:
    print(f"  {p['cutoff']}: WAPE={p['wape']:.4f} bias={p['bias']:+.4f}")
print(f"seasonal-naive WAPE={res['wape']:.4f} bias={res['bias']:+.4f}")
save_json(res, "naive.json")
