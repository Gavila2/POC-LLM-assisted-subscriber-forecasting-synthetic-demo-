"""
Compute simple metrics (MAE and MAPE) for model predictions saved in CSVs.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

def eval_file(path):
    df = pd.read_csv(path)
    mae = mean_absolute_error(df["y_true"], df["y_pred"])
    mape = np.mean(np.abs((df["y_true"] - df["y_pred"]) / np.maximum(1, df["y_true"]))) * 100
    print(path, "MAE:", mae, "MAPE:", f"{mape:.2f}%")

eval_file("data/baseline_preds.csv")
# If you saved LLM predictions to data/llm_preds.csv, run eval_file("data/llm_preds.csv")
