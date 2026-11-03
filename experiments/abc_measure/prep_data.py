"""GSM8K parquet (openai/gsm8k main) -> test.jsonl / train.jsonl with question, answer fields.

  python prep_data.py PARQUET_DIR OUT_DIR
"""
import json
import os
import sys

import pyarrow.parquet as pq

src, dst = sys.argv[1:3]
os.makedirs(dst, exist_ok=True)
for split in ("test", "train"):
    t = pq.read_table(f"{src}/{split}-00000-of-00001.parquet").to_pylist()
    with open(f"{dst}/{split}.jsonl", "w") as f:
        for r in t:
            f.write(json.dumps({"question": r["question"], "answer": r["answer"]}) + "\n")
    print(split, len(t))
