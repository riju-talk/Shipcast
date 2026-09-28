"""README figures: forecast vs actuals, and late rate vs promise length."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import CUTOFFS, RESULTS, load_daily_volume

FIGS = RESULTS / "figs"
FIGS.mkdir(exist_ok=True)
INK, MUTED, GRID = "#1f2328", "#6b6f76", "#e6e6e3"
COLORS = {"seasonal-naive": "#eb6834", "LightGBM": "#2a78d6", "LSTM": "#1baf7a"}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID})

# ---- 1. forecast vs actuals (Brazil total and SP) ------------------------------------------
vol = load_daily_volume()
preds = {name: pd.read_csv(RESULTS / f"preds_{f}.csv", parse_dates=["cutoff", "d"])
         for name, f in [("seasonal-naive", "naive"), ("LightGBM", "lgbm"), ("LSTM", "lstm")]}
start = CUTOFFS[0] - pd.Timedelta(days=35)

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for ax, (title, states) in zip(axes, [("Brazil (all 27 states)", None), ("São Paulo (SP)", ["SP"])]):
    sel = vol if states is None else vol[vol.state.isin(states)]
    act = sel[sel.d >= start].groupby("d").y.sum()
    ax.plot(act.index, act.values, color=INK, lw=1.4, label="actual orders")
    for name, p in preds.items():
        ps = p if states is None else p[p.state.isin(states)]
        for i, (c, g) in enumerate(ps.groupby("cutoff")):
            s = g.groupby("d").forecast.sum()
            ax.plot(s.index, s.values, color=COLORS[name], lw=2, alpha=0.9,
                    ls="--" if name == "seasonal-naive" else "-", label=name if i == 0 else None)
    for c in CUTOFFS:
        ax.axvline(c, color=MUTED, lw=0.8, ls=":")
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel("orders / day")
axes[0].legend(ncol=4, frameon=False, loc="upper left")
axes[1].set_xlabel("dotted lines = backtest cutoffs; each forecast covers the next 14 days")
fig.tight_layout()
fig.savefig(FIGS / "forecast_vs_actual.png", dpi=150)
print("saved results/figs/forecast_vs_actual.png")

# ---- 2. late rate vs promise length ---------------------------------------------------------
pr = json.loads((RESULTS / "promise.json").read_text())
sw = pd.DataFrame(pr["sweep_test"])
ol = pr["olist"]["test"]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sw.avg_promise_days, sw.late_rate * 100, color="#2a78d6", lw=2, marker="o", ms=5,
        label="quantile LightGBM (one point per α)")
for a in (0.5, 0.9):
    r = sw[sw.alpha == a].iloc[0]
    ax.annotate(f"P{int(a * 100)}", (r.avg_promise_days, r.late_rate * 100), xytext=(8, 4),
                textcoords="offset points", color=INK, fontweight="bold")
ax.scatter([ol["avg_promise_days"]], [ol["late_rate"] * 100], marker="*", s=220, color="#eb6834",
           zorder=5, label="Olist's own promise")
ax.set_xlabel("average promised delivery time (days)")
ax.set_ylabel("late deliveries (%)")
ax.set_title("Test quarter Apr–Jun 2018: shorter promises cost lateness", loc="left", color=INK)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGS / "promise_late_rate.png", dpi=150)
print("saved results/figs/promise_late_rate.png")
