"""
Train a simple baseline model (XGBoost) to predict sum next 24 using engineered features.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

df = pd.read_parquet("data/windows.parquet")

def featurize_window(window):
    arr = pd.DataFrame(window)
    return {
        "last_net": int(arr["net_adds"].iloc[-1]),
        "mean_net_24": float(arr["net_adds"].tail(24).mean()),
        "std_net_24": float(arr["net_adds"].tail(24).std()),
        "mean_temp": float(arr["temp"].mean()),
        "max_precip": float(arr["precip"].max()),
        "holiday_count": int(arr["is_holiday"].sum()),
        "outage_count": int(arr["outage_flag"].sum())
    }

feats = []
for _, r in df.iterrows():
    f = featurize_window(r["window"])
    f["market"] = r["market"]
    f["target"] = r["target_sum_next_24"]
    feats.append(f)

fdf = pd.DataFrame(feats)
fdf = pd.get_dummies(fdf, columns=["market"], drop_first=True)

X = fdf.drop(columns=["target"])
y = fdf["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {"objective": "reg:squarederror", "verbosity": 0}
bst = xgb.train(params, dtrain, num_boost_round=100)
preds = bst.predict(dtest)

mae = mean_absolute_error(y_test, preds)
mape = np.mean(np.abs((y_test - preds) / np.maximum(1, y_test))) * 100
print(f"Baseline MAE: {mae:.2f}, MAPE: {mape:.2f}%")
pd.DataFrame({"y_true": y_test, "y_pred": preds}).to_csv("data/baseline_preds.csv", index=False)
