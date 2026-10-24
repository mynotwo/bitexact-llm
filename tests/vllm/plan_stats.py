"""Summarize a run's .plan file (vLLM): how mixed the engine steps were.

  python tests/vllm/plan_stats.py RUN [RUN ...]
"""

import sys


def main():
    for p in sys.argv[1:]:
        rows = [list(map(int, ln.split())) for ln in open(p + ".plan")]
        mixed = sum(d > 0 and pr > 0 for _, d, pr, _ in rows)
        ctx = sum(c for *_, c in rows)
        print(f"{p}: forwards {len(rows)} mixed_prefill_decode {mixed} max_decode_batch "
              f"{max(d for _, d, _, _ in rows)} prefill_parts {sum(pr for _, _, pr, _ in rows)} "
              f"prefill_parts_after_pos0 {ctx} max_tokens {max(n for n, *_ in rows)}")


if __name__ == "__main__":
    main()
