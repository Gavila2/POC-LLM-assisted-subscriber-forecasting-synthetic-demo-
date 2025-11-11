"""
Convert windowed examples into LLM-friendly prompt/response pairs and save as JSONL.
"""
import json
import os
import pandas as pd

IN = "data/windows.parquet"
OUT = "data/prompts.jsonl"
df = pd.read_parquet(IN)

def make_prompt(row):
    header = f"Market: {row['market']}\nForecast time: {row['ts']}\n"
    header += "History (past 72 hourly rows):\n"
    header += "ts,net_adds,temp,precip,hour,is_holiday,outage\n"
    for r in row['window']:
        header += f"{r['ts']},{r['net_adds']},{r['temp']},{r['precip']},{r['hour']},{r['is_holiday']},{r['outage_flag']}\n"
    header += "\nTask: Predict the SUM of net_adds for the next 24 hours (integer)."
    return header

with open(OUT, "w") as fh:
    for _, row in df.sample(frac=0.02, random_state=42).iterrows():  # small sample for demo
        prompt = make_prompt(row)
        response = str(row["target_sum_next_24"])
        fh.write(json.dumps({"prompt": prompt, "response": response}) + "\n")
print("Wrote:", OUT)
