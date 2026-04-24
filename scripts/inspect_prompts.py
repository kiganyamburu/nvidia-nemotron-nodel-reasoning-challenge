from pathlib import Path
import re
import pandas as pd

p = Path(".") / "train.csv"
df = pd.read_csv(p)
print("rows", len(df))
print("cols", df.columns.tolist())
for i in range(3):
    s = str(df.loc[i, "prompt"])
    print("\n--- prompt", i, "len", len(s))
    print("repr:", repr(s[:200]))
    print("preview:\n", s[:400])
    print("first120 ords:", [ord(c) for c in s[:120]])
    print("words sample:", re.findall(r"(?u)\b\w+\b", s)[:20])
