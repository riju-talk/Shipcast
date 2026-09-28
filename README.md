# ShipCast: Shipment Demand Forecasting & Delivery-Promise Modeling

## Problem & Headline

We forecast daily shipment volume per Brazilian state 14 days ahead to enable capacity planning. On the same Olist dataset, pre-registered P90 delivery-time promises are **2.76 days (10.9%) shorter** than Olist's average, at a 4.66% late rate vs Olist's 4.24%.

---

## Data

- **Source**: Olist Brazilian e-commerce dataset (Kaggle)
- **Date Range**: 2017-01-01 to 2018-08-21 (~20 months)
- **Volume**: 97,829 non-canceled orders
- **Delivery Data**: 93,801 delivered orders with non-null distance (single-seller orders only)
- **Train/Test Split**: 63,007 training / 19,205 test orders (2018-04-01 to 2018-06-30)

### Data Notes
- 2016 excluded: sparse, many gaps
- 2018-08-22 onward: data cutoff (volume collapses from 08-22; data snapshot ends)
- May 2018: Brazilian truckers' strike visible as late-delivery spike; included in evaluation

---

## Methods

### Volume Forecasting (Task A)

**Backtest Design**: Rolling-origin (no leakage)
- 4 cutoffs: 2018-06-05, 06-26, 07-17, 08-07 (21 days apart)
- For each cutoff c: train on data ≤ c, forecast c+1 to c+14, evaluate
- Training origin: 2017-02-01 (ensures 28-day lag features exist)
- Test dates never appear in training for any state or horizon

**Models**:
1. **Seasonal-Naive Baseline**: repeat last observed week
   - Formula: ŷ(c+h) = y(c + h - 7×ceil(h/7))
   - Never reads beyond cutoff

2. **LightGBM (Global)**: one model for all 27 states
   - Features: lag1, lag7, lag14, lag28, mean7, mean28, std7, std28, dow, month, horizon, target_dow, is_holiday, state, snaive, y
   - Objective: tweedie (count data with zeros)
   - Hyperparams: 400 trees, lr=0.05, num_leaves=31

3. **LSTM**: PyTorch RNN with state embeddings
   - 28-day input sequences, state embedding (dim=4), 1 LSTM layer (hidden=64)
   - Output: all 14 days at once

**Metrics**:
- **WAPE** (Weighted Absolute % Error): avoids division-by-zero on zero-order days
  - WAPE = Σ|y - ŷ| / Σy
- **Bias**: Σ(ŷ - y) / Σy (positive = overpredicting)

### Delivery Promise (Task B)

**Data Split**:
- Train: 2017-01-01 to 2018-03-31
- Test: 2018-04-01 to 2018-06-30 (held-out quarter; avoids censoring)

**Model**: Quantile LightGBM
- Target: actual_days (time from purchase to delivery, whole days)
- Features: distance_km, freight, weight_g, n_items, purchase_dow, customer_state, seller_state
- **Important**: No Olist estimated date in features (avoids circularity)
- Quantiles: α ∈ {0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99}
- Promise = ceil(prediction)

**Metrics**:
- Late rate: share of orders where actual_days > promise
- Trade-off: higher α = longer promises, lower late rate

**Headline**: Pre-registered P90 model (no test-set selection), reporting promise reduction and late rate vs Olist.

---

## Results

### Forecast Performance

| Model | Pooled WAPE | Pooled Bias | Per-Cutoff WAPEs |
|-------|-----------|-----------|------------------|
| **Seasonal-Naive** | 40.92% | -3.10% | c1: 35.47%, c2: 43.48%, c3: 49.97%, c4: 35.07% |
| **LightGBM** | 28.83% | -6.39% | c1: 23.00%, c2: 39.85%, c3: 33.47%, c4: 21.91% |
| **LSTM** | 31.60% | -10.87% | c1: 26.55%, c2: 40.22%, c3: 34.01%, c4: 27.75% |

**Improvement**: LightGBM reduces WAPE by **29.5%** vs. seasonal-naive (mean over cutoffs: 29.56%). LSTM reduces WAPE by 22.8%. LightGBM outperforms LSTM at most horizons; LSTM wins at 3 days ahead (27.6% vs 28.6%).



### Delivery Promise Performance

| Alpha | Late Rate | Avg Promise (days) | Notes |
|-------|-----------|-------------------|-------|
| 0.50 | 24.73% | 12.4 | Aggressive promise |
| **0.90** | **4.66%** | **22.5** | **← Headline: pre-registered P90** |
| 0.92 | 3.69% | 23.9 | 1.37 days (5.4%) shorter at 3.69% late; α chosen on test (optimistic) |
| 0.96 | 2.08% | 28.0 | Very conservative |

**Olist Baseline** (on test set):
- Late rate: 4.24%
- Avg promise: 25.26 days

**Headline**: Pre-registered P90 promises are **2.76 days (10.9%) shorter** than Olist's, at a 4.66% late rate vs Olist's 4.24%. P90 coverage on test quarter: 95.3%.

### Plots

![Forecast vs Actual](results/figs/forecast_vs_actual.png)
*Figure 1: Forecast trajectory vs actual for SP (high volume) and one small state (low volume) at last cutoff (2018-08-07). Blue solid: LightGBM; red dashed: seasonal-naive; green: actual 1–14 days after cutoff.*

![Promise Trade-off](results/figs/promise_late_rate.png)
*Figure 2: Delivery promise trade-off (late rate vs average promise in days). Each point is one quantile α (colored). Star: Olist baseline. Moving up-left is better.*

---

## Limitations

1. **Historical Data**: 2017–18 may not generalize to 2019+. E-commerce logistics, buyer behavior, carrier availability have evolved.

2. **Offline Evaluation**: No A/B test. Real-world performance may differ.

3. **May 2018 Strike**: Late-delivery rates spike late May 2018 (Brazilian truckers' strike). Captured in test set; not a model failure but a data quality note.

4. **State Imbalance**: SP dominates (~46% of orders in test windows). Small states may be underfitted in the global model. Trade-off accepted for simplicity and single deployment.

5. **Forecast Horizon**: 14 days is medium-term. Longer horizons (30+ days) would require different models (trend extrapolation, external data).

6. **P90 Calibration Drift**: P90 coverage varies across quarters. Coverage was 95.3% on test (Q2 2018), but P90 was late on 20.8% of orders in Q1 2018. Tuning on Q1 failed because calibration does not transfer reliably.

---

## Deployment Design

**Note**: The following is design, not implemented in this 1.5-day project.

### Daily Retraining
- Retrain models once per day at EOD
- Input: delivered orders from yesterday
- Check: WAPE on rolling 7-day must not degrade >10% vs baseline; alert if it does

### Monitoring
- **Forecast**: WAPE by state, rolling 7-day
- **Delivery**: Late-delivery rate by state, late-delivery reason (traffic, carrier, etc.)
- **Drift**: Compare current test metrics to historical baseline

### Feedback Loop
- Log every prediction and actual outcome
- Monthly retraining with updated test window
- Investigate causes of degradation (e.g., shift in carrier mix, seasonal change)

### Deployment Infrastructure
- Container: Python 3.13 + DuckDB + LightGBM + PyTorch
- API: REST endpoint for forecasts and promises
- Caching: results valid for 24h (batch recompute overnight)
- Failover: return previous day's forecast if model crashes

---

## How to Reproduce

### Setup
```bash
uv sync  # or: pip install -r requirements.txt
```

### Run Full Pipeline
```bash
python src/backtest.py
python src/task_a_lgbm.py
python src/task_a_lstm.py
python src/task_b_promise.py
python src/task_c_capacity.py
python src/merge_results.py
python src/plots.py
```

**Time**: ~4 minutes

### Verify Reproducibility
```bash
rm -rf results  # Clean state
# Re-run pipeline
# Numbers should match exactly (fixed seeds: np.random.seed(42), torch.manual_seed(42))
```

---

## Code

- `sql/01_daily_volume.sql` – Volume feature engineering
- `sql/02_orders_b.sql` – Delivery feature engineering
- `src/backtest.py` – Backtest framework & seasonal-naive
- `src/task_a_lgbm.py` – LightGBM forecast
- `src/task_a_lstm.py` – LSTM forecast
- `src/task_b_promise.py` – Quantile regression
- `src/task_c_capacity.py` – Capacity allocation (experimental, not on resume)
- `src/merge_results.py` – Results consolidation
- `src/plots.py` – Visualization

---

## References

- Olist Brazilian E-Commerce Dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- WAPE metric: Industry standard for demand forecasting (avoids MAPE issues with zeros)
- Quantile Regression: Koenker & Bassett (1978); implemented in LightGBM
- Rolling-Origin Backtest: Tashman (2000) "Out-of-sample tests of forecasting accuracy"

---

**Created**: September 27, 2026
**Data**: 2017-01-01 to 2018-08-21
