import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
import torch.nn as nn

LAT, LON = 31.5497, 74.3436

BANDS = [
    (12.0, "Good", "#16a34a", "Air quality is fine — enjoy the outdoors."),
    (35.4, "Moderate", "#ca8a04", "Acceptable, but sensitive people should limit long outdoor exertion."),
    (55.4, "Unhealthy for Sensitive Groups", "#ea580c", "Children, elderly and asthma patients should stay indoors."),
    (150.4, "Unhealthy", "#dc2626", "Everyone should avoid outdoor exertion; wear an N95 mask outside."),
    (250.4, "Very Unhealthy", "#7e22ce", "Health alert — stay indoors, keep windows closed, run an air purifier."),
    (float("inf"), "Hazardous", "#7f1d1d", "Emergency — avoid all outdoor activity."),
]

def band(v):
    for lim, name, color, advice in BANDS:
        if v <= lim:
            return name, color, advice

class LSTM(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, 64, 2, batch_first=True)
        self.head = nn.Linear(64, 24)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])

@st.cache_resource
def load_model():
    sc = json.load(open("scaler.json"))
    m = LSTM(len(sc["features"]))
    m.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=True))
    m.eval()
    return m, sc

def fetch_recent():
    end = datetime.now().date()
    start = end - timedelta(days=4)
    aq = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality",
        params={"latitude": LAT, "longitude": LON, "hourly": "pm2_5,pm10",
                "start_date": str(start), "end_date": str(end), "timezone": "auto"},
        timeout=60).json()
    wx = requests.get("https://api.open-meteo.com/v1/forecast",
        params={"latitude": LAT, "longitude": LON,
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
                "start_date": str(start), "end_date": str(end), "timezone": "auto"},
        timeout=60).json()
    df = pd.DataFrame({"time": pd.to_datetime(aq["hourly"]["time"])})
    df["pm25"] = aq["hourly"]["pm2_5"]
    df["pm10"] = aq["hourly"]["pm10"]
    for k in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure"]:
        df[k] = wx["hourly"][k]
    return df.interpolate(limit_direction="both").dropna().reset_index(drop=True)

st.set_page_config(page_title="Lahore Smog Forecast", page_icon="🌫️")
st.title("🌫️ Lahore Smog Forecast")
st.write("LSTM neural network predicting the next 24 hours of PM2.5 in Lahore, Pakistan — "
         "trained on 4 years of hourly pollution and weather data.")

model, sc = load_model()
FEATS = sc["features"]
mu = np.array(sc["feat_mu"], np.float32)
sd = np.array(sc["feat_sd"], np.float32)

with st.spinner("Fetching latest Lahore air data and running the forecast..."):
    df = fetch_recent().tail(72).reset_index(drop=True)
    t = df["time"]
    d = df.copy()
    d["hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24)
    d["doy_sin"] = np.sin(2 * np.pi * t.dt.dayofyear / 365)
    d["doy_cos"] = np.cos(2 * np.pi * t.dt.dayofyear / 365)
    X = ((d[FEATS].to_numpy(np.float32) - mu) / sd)[None]
    with torch.no_grad():
        fc = (model(torch.from_numpy(X)).numpy()[0] * sc["y_sd"] + sc["y_mu"])
    last_t = t.iloc[-1]
    ftime = [last_t + timedelta(hours=i + 1) for i in range(24)]

obs = pd.DataFrame({"time": t, "PM2.5": df["pm25"], "type": "observed"})
fcd = pd.DataFrame({"time": ftime, "PM2.5": fc, "type": "forecast"})
st.line_chart(pd.concat([obs, fcd]).pivot(index="time", columns="type", values="PM2.5"))

now_v, peak_v = float(df["pm25"].iloc[-1]), float(fc.max())
c1, c2 = st.columns(2)
for col, v, label in [(c1, now_v, "Current PM2.5 (µg/m³)"), (c2, peak_v, "Peak forecast next 24h (µg/m³)")]:
    name, color, advice = band(v)
    col.metric(label, f"{v:.0f}")
    col.markdown(f"<b style='color:{color}'>{name}</b>", unsafe_allow_html=True)
_, _, advice = band(peak_v)
st.info(f"**Health advice:** {advice}")

with st.expander("How the model was evaluated"):
    m = json.load(open("metrics.json"))
    st.write(f"Test MAE **{m['test_mae']:.1f}** µg/m³, RMSE **{m['test_rmse']:.1f}** µg/m³ "
             f"on {m['n_test']:,} held-out hours (2025–2026).")
    st.write(f"Baselines — persistence: MAE {m['persist_mae']:.1f} | "
             f"hourly climatology: MAE {m['clim_mae']:.1f}.")
    st.write("2-layer LSTM (64 units), 72-hour lookback → 24-hour forecast, "
             "chronological train/val/test split, early stopping on validation loss.")
st.caption("Data: Open-Meteo (CAMS). For education — not a medical device.")
