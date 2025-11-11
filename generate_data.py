"""
Generate synthetic subscriber + weather data for multiple markets (DMAs).
Outputs CSV files: data/subscribers.csv, data/weather.csv
"""
import numpy as np
import pandas as pd
import os
from datetime import datetime

OUT = "data"
os.makedirs(OUT, exist_ok=True)

np.random.seed(42)

start = datetime(2024, 1, 1)
end = datetime(2024, 3, 31)
freq = "H"
dates = pd.date_range(start, end, freq=freq)

markets = ["DMA1", "DMA2", "DMA3"]

rows_sub = []
rows_weather = []

for m in markets:
    baseline = 1000 + 200 * (markets.index(m))  # market offset
    for ts in dates:
        hour = ts.hour
        dow = ts.weekday()
        diurnal = 1 + 0.2 * np.sin((hour / 24) * 2 * np.pi)
        weekly = 0.9 if dow >= 5 else 1.0
        drift = 1 + 0.0005 * (ts - start).days
        noise = np.random.normal(0, 20)
        net_adds = baseline * 0.001 * diurnal * weekly * drift + noise
        net_adds = max(int(net_adds + np.random.normal(0, 2)), -50)
        rows_sub.append({
            "market": m,
            "ts": ts,
            "activations": max(int(50 * diurnal + np.random.poisson(5)), 0),
            "cancellations": max(int(45 * (1/diurnal) + np.random.poisson(4)), 0),
            "net_adds": net_adds
        })
        base_temp = 60 + 10 * np.sin((ts.timetuple().tm_yday / 365) * 2 * np.pi)
        temp = base_temp + np.random.normal(0, 5)
        precip = 0
        if np.random.rand() < 0.02:
            precip = round(np.random.uniform(0.1, 2.0), 2)
        rows_weather.append({
            "market": m,
            "ts": ts,
            "temp": round(temp, 2),
            "precip": precip,
            "weather_flag": 1 if precip > 0 else 0
        })

pd.DataFrame(rows_sub).to_csv(os.path.join(OUT, "subscribers.csv"), index=False)
pd.DataFrame(rows_weather).to_csv(os.path.join(OUT, "weather.csv"), index=False)
print("Wrote data/subscribers.csv and data/weather.csv")
