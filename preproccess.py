# Merge subscriber and weather CSVs, engineer features, create windowed Parquet dataset.
import pandas as pd
import numpy as np
import os

IN_DIR = "data"
OUT = "data"
os.makedirs(OUT, exist_ok=True)

subs = pd.read_csv(os.path.join(IN_DIR, "subscribers.csv"), parse_dates=["ts"])
weather = pd.read_csv(os.path.join(IN_DIR, "weather.csv"), parse_dates=["ts"])

# demo holiday list — replace with your company calendar if available
holidays = pd.to_datetime(["2024-01-01", "2024-02-14", "2024-02-19", "2024-03-17"])
df = subs.merge(weather, on=["market", "ts"], how="left")

# simulate outage flag for demo (in real data use outage logs)
np.random.seed(1)
df["outage_flag"] = 0
df.loc[df.sample(frac=0.01).index, "outage_flag"] = 1

# basic time features
df["hour"] = df["ts"].dt.hour
df["dow"] = df["ts"].dt.weekday
df["dom"] = df["ts"].dt.day
df["month"] = df["ts"].dt.month
df["is_holiday"] = df["ts"].dt.normalize().isin(holidays).astype(int)

# per-market rolling / lag features and imputation
dfs = []
for m, g in df.groupby("market"):
    g = g.sort_values("ts").set_index("ts")
    g["lag_1"] = g["net_adds"].shift(1)
    g["lag_24"] = g["net_adds"].shift(24)
    g["roll_24_mean"] = g["net_adds"].rolling(window=24, min_periods=1).mean()
    g["temp_roll_6"] = g["temp"].rolling(6, min_periods=1).mean()
    # forward/backfill simple imputation (demo). Replace with domain imputation in production.
    g = g.fillna(method="ffill").fillna(method="bfill").fillna(0)
    g = g.reset_index()
    dfs.append(g)
df2 = pd.concat(dfs, ignore_index=True)

# windowing: past WINDOW hours -> predict sum of next PRED_HORIZON hours
WINDOW = 72
PRED_HORIZON = 24
examples = []
for m, g in df2.groupby("market"):
    g = g.sort_values("ts").reset_index(drop=True)
    for i in range(WINDOW, len(g) - PRED_HORIZON):
        past = g.loc[i - WINDOW:i - 1].copy()
        future = g.loc[i:i + PRED_HORIZON - 1]["net_adds"].sum()
        # keep a compact set of fields for the prompt / training
        window_dict = past[["ts", "net_adds", "temp", "precip", "hour", "is_holiday", "outage_flag"]].to_dict(orient="records")
        examples.append({
            "market": m,
            "ts": g.loc[i, "ts"],
            "window": window_dict,
            "target_sum_next_24": int(future)
        })

out_df = pd.DataFrame(examples)
out_df.to_parquet(os.path.join(OUT, "windows.parquet"), index=False)
print("Wrote data/windows.parquet with", len(out_df), "examples")
