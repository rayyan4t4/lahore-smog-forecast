# Lahore Smog Forecast 🌫️

A 24-hour PM2.5 forecasting model for Lahore, Pakistan — built because the city's winter smog is a yearly public-health emergency and a reliable next-day forecast helps people plan.

## Model

- **Architecture:** 2-layer LSTM (64 hidden units) → 24-hour PM2.5 forecast from a 72-hour lookback
- **Features:** PM2.5, PM10, temperature, humidity, wind speed, pressure + cyclical time encodings
- **Data:** 41,832 hourly rows (2022–2026) from Open-Meteo (CAMS), Lahore coordinates
- **Split:** chronological — train 2022–23, validate 2024, test 2025–26 (no shuffling across time)
- **Training:** Adam, MSE loss, early stopping on validation loss

## Results (15,433 held-out test hours, 2025–2026)

| Model | MAE (µg/m³) | RMSE (µg/m³) |
|---|---|---|
| LSTM (this project) | **17.7** | **25.3** |
| Persistence baseline | 24.2 | 36.3 |
| Hourly climatology | 33.3 | 46.8 |

The LSTM beats persistence by 27% and climatology by 47% on MAE.

## Run it

```bash
pip install -r requirements.txt
python fetch_data.py   # pull fresh data
python train.py         # train + evaluate (saves model.pt)
streamlit run streamlit_app.py
```

The notebook `Lahore_Smog_Forecast.ipynb` walks through the full pipeline step by step (also runs on Google Colab's free tier).

## Live demo

[Streamlit app](https://lahore-smog-forecast.streamlit.app) — pulls the latest 72 hours from Open-Meteo and forecasts the next 24.

*For education — not a medical device.*
