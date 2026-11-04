# BitExact speed, final integration (branch exp/speed-final)

`exp/speed-final` = `exp/speed` (d10ac00) + merge `exp/speed-gemm` (2bbdba6, cuBLASLt int8 GEMM, default on CUDA) + merge `exp/speed-attn` (e758c1b, KV stored as 3 int8 limb planes). Both merges are clean merge commits, no conflicts.

## Cumulative speed

Host 138, one H100 80GB per run (each started at 0 MiB), vLLM 0.32.0, torch 2.13.0+cu130. Scorer: tok/s of teacher-forced log p (GSM8K = plugin rollouts at batch 8; 4k = 16 rollouts of ~4k tokens at batch 1 and 4). Decode: ms/step = wall of one `generate` / new tokens (prefill included), T = 1, vLLM CUDA graphs; B=1 and B=64 generate 512 tokens, B=512 generates 256 tokens. Final row: median of 3 repetitions (scorer: reps 2-3 of 3 warm, rep 1 includes compilation, median is rep 2/3).

| state | scorer GSM8K b8 | scorer 4k b1 | scorer 4k b4 | decode B=1 | decode B=64 | decode B=512 |
|---|---|---|---|---|---|---|
| original (eager scorer; plugin at e5eb24c) | 6,637 | 2,361 | 3,658 | 4.286 | 18.160 | n/m |
| `exp/speed` (compiled scorer, fused decode, items a + b) | 38,462 | 20,468 | 21,484 | 2.688 | 5.779 | 26.9 (1) |
| **exp/speed-final** | **42,442** | **22,071** | **23,022** | **2.448** | **5.349** | **23.856** |
| final vs original | 6.4x | 9.3x | 6.3x | 1.75x | 3.40x | n/m |
| final vs `exp/speed` | 1.10x | 1.08x | 1.07x | 1.10x | 1.08x | 1.13x |

n/m = not measured. (1) `exp/speed` B=512 from `experiments/speed_attn/RESULTS.md` (base of that run, same host). Original and `exp/speed` rows are copied from `experiments/speed/RESULTS.md`; the final row was measured in this round on 138 GPUs 2 (scorer) and 3 (decode), other jobs ran on other GPUs at the same time.

Contribution of each branch alone (their own RESULTS): GEMM (`lt` vs `intmm`) scorer 1.06-1.09x, decode B=1 1.03x, B=64 1.01x; limb KV decode B=64 5.853 -> 5.522 ms/step (-5.7%), B=512 26.9 -> 25.31 (-5.9%).

## Gates

| gate | result |
|---|---|
| full test suite, 138 (H100) | 90 passed, 5 skipped, 0 failed (`results/pytest_138.log`) |
| scorer gate, 138, 8 configurations (GSM8K 128 seq at b8, b64, b1, b8 shuffled: 28,083 rows; 4k 16 seq at b4, b1, b8, b4 shuffled: 65,536 rows): rows equal plugin hash, sequence hashes and float64 log p bits equal eager reference | 8 / 8 (`results/final.check.jsonl`) |
| scorer gate, 141 (H800, GPU 3, `/data41/audreymao/speed_final_20261104`), same 8 configurations | 8 / 8 (`results/scorer_gate_141.log`) |
| decode gate (`speed_decode/gate.py`, 32 GSM8K prompts x G8 = 256 sequences; T = 1: 53,387 tokens, greedy: 65,536 tokens; B=1 and B=64) vs base | 0 sequences differ in all 4 comparisons; B=1 == B=64 in both; GATE PASS (`results/decode_gate.txt`) |

## RL step time (branch exp/abc-rl-final = exp/abc-rl-speed + merge exp/speed-final)

Bit check (`bitcheck.py`, identical weights for 10 steps, C-old vs the final tree): rewards, tokens, sampler log p and scorer log p equal at all 10 steps, ALL EQUAL (`results/bitcheck_compare.txt`).

C-final run: seed 1, 30 steps, flags of `runs/C_new_v2_s1` (`--int-impl new --plugin-gpu-mem 0.20 --score-chunk 64`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--eval-every 0`), one H100 (138 GPU 4). Median of steps 5-29 (25 steps), seconds. Step 0 token hash and sampler/scorer log p hashes equal those of C-new v2.

| arm | s/step | sample | score | train (fwd + bwd) | quantize + optim |
|---|---|---|---|---|---|
| C-old | 30.9 | | | | |
| C-new v2 | 17.9 | 6.66 | 5.64 | | |
| **C-final** | **16.2** | **5.83** | **4.87** | **5.24** | **0.23** |
| D | 11.7 | | | | |

C-final is 1.91x faster than C-old, 1.11x faster than C-new v2 and 1.38x slower than D. C-old, C-new v2 and D values are from `experiments/abc_rl_speed/RESULTS.md`; the sample/score split of C-new v2 is the median of steps 5-29 of its `steps.jsonl`. Per-step records: `results/rl_C_final_s1_steps.jsonl`.
