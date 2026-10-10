import requests
import pandas as pd

LAT, LON = 31.5497, 74.3436
START, END = "2022-01-01", "2026-10-09"

aq = requests.get(
    "https://air-quality-api.open-meteo.com/v1/air-quality",
    params={"latitude": LAT, "longitude": LON, "hourly": "pm2_5,pm10",
            "start_date": START, "end_date": END, "timezone": "auto"},
    timeout=120,
).json()
wx = requests.get(
    "https://archive-api.open-meteo.com/v1/archive",
    params={"latitude": LAT, "longitude": LON,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
            "start_date": START, "end_date": END, "timezone": "auto"},
    timeout=120,
).json()

df = pd.DataFrame({"time": aq["hourly"]["time"]})
df["pm25"] = aq["hourly"]["pm2_5"]
df["pm10"] = aq["hourly"]["pm10"]
for k in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure"]:
    df[k] = wx["hourly"][k]
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)
df = df.interpolate(limit_direction="both").dropna().reset_index(drop=True)
df.to_csv("lahore_hourly.csv", index=False)
print(f"rows: {len(df)}, from {df.time.min()} to {df.time.max()}")
print(df.describe()[["pm25", "pm10"]].round(1).to_string())
