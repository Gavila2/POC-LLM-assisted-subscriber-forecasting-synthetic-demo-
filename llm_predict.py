"""
Example of formatting a prompt and calling an LLM (OpenAI) to get a numeric forecast.
Set OPENAI_API_KEY in environment to call API; otherwise it prints the prompt.
"""
import os
import json
import argparse
import openai

parser = argparse.ArgumentParser()
parser.add_argument("--example_idx", type=int, default=0)
args = parser.parse_args()

PROMPTS_FILE = "data/prompts.jsonl"
examples = [json.loads(line) for line in open(PROMPTS_FILE)]
ex = examples[args.example_idx]
prompt = ex["prompt"] + "\nAnswer (just the integer):"

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("OPENAI_API_KEY not set — printing prompt instead of calling API.\n")
    print(prompt[:2000])
    exit(0)

openai.api_key = api_key

resp = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=16,
    temperature=0.0
)
text = resp["choices"][0]["text"].strip()
print("Raw model output:", text)
try:
    pred = int(float(text.split()[0].replace(",", "")))
    print("Parsed prediction:", pred)
except Exception as e:
    print("Failed to parse model output into int:", e)
