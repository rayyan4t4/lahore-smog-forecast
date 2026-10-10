import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

LOOKBACK, HORIZON = 72, 24
FEATS = ["pm25", "pm10", "temperature_2m", "relative_humidity_2m",
         "wind_speed_10m", "surface_pressure"]

df = pd.read_csv("lahore_hourly.csv", parse_dates=["time"])
df["hour_sin"] = np.sin(2 * np.pi * df.time.dt.hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * df.time.dt.hour / 24)
df["doy_sin"] = np.sin(2 * np.pi * df.time.dt.dayofyear / 365)
df["doy_cos"] = np.cos(2 * np.pi * df.time.dt.dayofyear / 365)
ALL = FEATS + ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]

X_all = df[ALL].to_numpy(np.float32)
y_all = df["pm25"].to_numpy(np.float32)

def windows(X, y):
    n = len(X) - LOOKBACK - HORIZON + 1
    Xw = np.stack([X[i:i + LOOKBACK] for i in range(n)])
    yw = np.stack([y[i + LOOKBACK:i + LOOKBACK + HORIZON] for i in range(n)])
    return Xw, yw

def split(mask):
    idx = np.where(mask)[0]
    s, e = idx.min(), idx.max()
    return windows(X_all[s:e + 1], y_all[s:e + 1])

tr = df.time < "2024-01-01"
va = (df.time >= "2024-01-01") & (df.time < "2025-01-01")
te = df.time >= "2025-01-01"
Xtr, ytr = split(tr); Xva, yva = split(va); Xte, yte = split(te)

mu, sd = Xtr.reshape(-1, len(ALL)).mean(0), Xtr.reshape(-1, len(ALL)).std(0) + 1e-8
ymu, ysd = ytr.mean(), ytr.std() + 1e-8
norm = lambda a: (a - mu) / sd
Xtr, Xva, Xte = norm(Xtr), norm(Xva), norm(Xte)
ytr_n, yva_n = (ytr - ymu) / ysd, (yva - ymu) / ysd

train_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr_n)),
                      batch_size=256, shuffle=True)

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(len(ALL), 64, 2, batch_first=True, dropout=0.2)
        self.head = nn.Linear(64, HORIZON)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])

torch.manual_seed(0)
model = LSTM()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
best, patience = float("inf"), 0
Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva_n)

for epoch in range(60):
    model.train()
    for xb, yb in train_dl:
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        vl = loss_fn(model(Xva_t), yva_t).item()
    if vl < best - 1e-4:
        best, patience = vl, 0
        torch.save(model.state_dict(), "model.pt")
    else:
        patience += 1
        if patience >= 8:
            break
    print(f"epoch {epoch+1}: val_mse={vl:.4f}", flush=True)

model.load_state_dict(torch.load("model.pt", weights_only=True))
model.eval()
with torch.no_grad():
    pred = (model(torch.from_numpy(Xte)).numpy() * ysd + ymu)

persist = np.repeat(yte[:, -1:], HORIZON, axis=1)
clim = np.tile(ytr.reshape(-1, HORIZON).mean(0), (len(yte), 1))

def scores(p, t):
    e = p - t
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))

m_mae, m_rmse = scores(pred, yte)
p_mae, p_rmse = scores(persist, yte)
c_mae, c_rmse = scores(clim, yte)
print(f"\nLSTM      MAE={m_mae:.2f} RMSE={m_rmse:.2f}")
print(f"persist   MAE={p_mae:.2f} RMSE={p_rmse:.2f}")
print(f"climatol. MAE={c_mae:.2f} RMSE={c_rmse:.2f}")

json.dump({"feat_mu": mu.tolist(), "feat_sd": sd.tolist(), "y_mu": float(ymu),
           "y_sd": float(ysd), "features": ALL, "lookback": LOOKBACK,
           "horizon": HORIZON},
          open("scaler.json", "w"))
json.dump({"test_mae": m_mae, "test_rmse": m_rmse,
           "persist_mae": p_mae, "persist_rmse": p_rmse,
           "clim_mae": c_mae, "clim_rmse": c_rmse,
           "n_test": len(yte)}, open("metrics.json", "w"), indent=1)
print("saved model.pt, scaler.json, metrics.json")
