"""Task A: PyTorch LSTM that reads 28 days and predicts all 14 days at once.

Inputs per day: y scaled by the state's training-period mean, weekday one-hot (7),
holiday and Black Friday flags. A 4-dim state embedding is concatenated to the last
hidden state before the 14-output linear head. Scaling uses only data <= cutoff, and
training windows are restricted so that every target day is <= cutoff.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import (CUTOFFS, HORIZON, RESULTS, SEED, TRAIN_START, load_daily_volume,
                    save_json, score_backtest, seed_everything)

SEQ_LEN, HIDDEN, EMB, EPOCHS, BATCH, LR = 28, 64, 4, 30, 64, 1e-3

seed_everything()
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)


class LSTMForecaster(nn.Module):
    def __init__(self, n_states: int, n_inputs: int):
        super().__init__()
        self.emb = nn.Embedding(n_states, EMB)
        self.lstm = nn.LSTM(n_inputs, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN + EMB, HORIZON)

    def forward(self, x, s):
        _, (h, _) = self.lstm(x)
        return self.head(torch.cat([h[-1], self.emb(s)], dim=1))


vol = load_daily_volume()
states = sorted(vol.state.unique())
y = vol.pivot(index="d", columns="state", values="y")[states]
dates = y.index
cal = vol[vol.state == states[0]].set_index("d").loc[dates]
calendar = np.column_stack([np.eye(7)[cal.dow.to_numpy() - 1], cal.is_holiday, cal.is_black_friday])


def windows(scaled: np.ndarray, ends: np.ndarray, with_target: bool):
    """Stack input windows ending at index `ends` (inclusive) for every state."""
    X, S, Y = [], [], []
    for si in range(len(states)):
        for e in ends:
            seq = np.column_stack([scaled[e - SEQ_LEN + 1:e + 1, si], calendar[e - SEQ_LEN + 1:e + 1]])
            X.append(seq)
            S.append(si)
            if with_target:
                Y.append(scaled[e + 1:e + 1 + HORIZON, si])
    X = torch.tensor(np.array(X), dtype=torch.float32)
    S = torch.tensor(S, dtype=torch.long)
    return (X, S, torch.tensor(np.array(Y), dtype=torch.float32)) if with_target else (X, S)


rows = []
for c in CUTOFFS:
    ci = dates.get_loc(c)
    scale = y.loc[TRAIN_START:c].mean().clip(lower=0.1).to_numpy()   # training-only state means
    scaled = y.to_numpy() / scale
    first_end = max(dates.get_loc(TRAIN_START), SEQ_LEN - 1)
    ends = np.arange(first_end, ci - HORIZON + 1)                   # last target = end + 14 <= cutoff
    X, S, Y = windows(scaled, ends, with_target=True)

    torch.manual_seed(SEED)
    model = LSTMForecaster(len(states), X.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(SEED)
    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(X), generator=gen)
        total = 0.0
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(X[idx], S[idx]), Y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
    print(f"  cutoff {c.date()}: {len(X):,} windows, final train MSE={total / len(X):.4f}")

    model.eval()
    Xt, St = windows(scaled, np.array([ci]), with_target=False)
    with torch.no_grad():
        pred = model(Xt, St).numpy() * scale[:, None]
    for si, state in enumerate(states):
        for h in range(1, HORIZON + 1):
            d = dates[ci + h]
            rows.append((c, state, d, h, y.at[d, state], max(pred[si, h - 1], 0.0)))

preds = pd.DataFrame(rows, columns=["cutoff", "state", "d", "horizon", "actual", "forecast"])
preds.to_csv(RESULTS / "preds_lstm.csv", index=False)
res = score_backtest(preds, "lstm")
for p in res["per_cutoff"]:
    print(f"  {p['cutoff']}: WAPE={p['wape']:.4f} bias={p['bias']:+.4f}")
print(f"LSTM WAPE={res['wape']:.4f} bias={res['bias']:+.4f}")
save_json(res, "lstm.json")
