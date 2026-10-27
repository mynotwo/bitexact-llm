"""Shared pieces of the RL rollout/trainer mismatch experiment: GSM8K prompts, the rule-based grader,
the canonical log-prob of an integer logits row, and rollout file I/O.

Rollout file (JSON): {"engine": str, "meta": {...}, "seqs": {seed: {"p": prompt index, "tok": [...],
"lp": [...], "am": [...], "sha": [...]}}}. lp = sampler-side log p of each sampled token, am = the
sampler's argmax token, sha = first 16 hex digits of the logits row SHA-256 (BitExact only).
"""

import json
import re

import numpy as np

G = 8                      # samples per prompt (one GRPO group)
N_PROMPTS = 128
MAX_NEW = 256
TEMP = 1.0
SHOTS = (0, 1)             # GSM8K train examples used as the 2-shot prefix


def _clean(ans):
    return re.sub(r"<<[^>]*>>", "", ans).strip()


def _jsonl(path):
    return [json.loads(x) for x in open(path)]


def build_prompts(test_jsonl, train_jsonl, tokenizer, n=N_PROMPTS):
    """[(prompt ids with BOS, gold answer string)] for the first n test questions."""
    train = _jsonl(train_jsonl)
    head = "".join(f"Question: {train[i]['question']}\nAnswer: {_clean(train[i]['answer'])}\n\n" for i in SHOTS)
    out = []
    for ex in _jsonl(test_jsonl)[:n]:
        text = head + f"Question: {ex['question']}\nAnswer:"
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        out.append((ids, ex["answer"].split("####")[-1].strip()))
    return out


def seed_of(p, g):
    return 1 + G * p + g


def requests(prompts):
    """[(seed, prompt index, prompt ids)] in canonical order."""
    return [(seed_of(p, g), p, ids) for p, (ids, _) in enumerate(prompts) for g in range(G)]


_NUM = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")


def _num(s):
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def grade(text, gold):
    """Rule-based reward: (reward 0/1, flag in {'ok', 'wrong', 'no_answer'}). Only the text before the
    first follow-on 'Question:' counts (the 2-shot base model often keeps writing new questions)."""
    body = text.split("\nQuestion:")[0]
    m = _NUM.search(body)
    if not m:
        return 0, "no_answer"
    ok = _num(m.group(1)) is not None and _num(m.group(1)) == _num(gold)
    return (1, "ok") if ok else (0, "wrong")


def int_logp(row, tok, scale=2.0 ** -16):
    """Canonical log p of token `tok` from one int64 logits row (fixed point, unit 2^-16), T = 1:
    float64 log-softmax on the CPU with numpy. Same input bits -> same output bits on the same build."""
    x = np.asarray(row, dtype=np.float64) * scale
    m = x.max()
    return float(x[tok] - (m + np.log(np.exp(x - m).sum())))


def save(path, engine, meta, seqs):
    json.dump({"engine": engine, "meta": meta, "seqs": {str(k): v for k, v in seqs.items()}}, open(path, "w"))


def load(path):
    d = json.load(open(path))
    d["seqs"] = {int(k): v for k, v in d["seqs"].items()}
    return d


def first_div(a, b):
    """Index of the first differing token of two token lists (None if identical)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))
