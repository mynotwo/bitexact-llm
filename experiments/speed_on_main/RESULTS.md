# exp/speed-final ported onto main (branch exp/speed-on-main)

`exp/speed-on-main` = `origin/main` (6d42e87, perf2 + perf3) + the items of `exp/speed-final` that main does not already
have, one commit each. Every port passed the bit gates below; nothing that measured slower stayed in.

Result: the int8 scorer (teacher-forced log p) is 8.3x / 11.3x / 7.9x faster than main's default eager flow and
1.56x / 2.58x / 2.02x faster than main's flow with compiled segments (GSM8K b8 / 4k b1 / 4k b4). Plugin throughput
equals main's (main's README benchmark, same GPU, interleaved); at a cache-limited batch (512 sequences,
`gpu_memory_utilization` 0.15) a decode step is 1.12x faster because the KV cache holds 33% more tokens.
All outputs are bit-identical to main.

## Overlap map

(a) = main already has it (kept main's version), (b) = complementary (ported), (c) = both did it differently (kept the
faster one). Numbers: 138, H100, see "Setup".

| `exp/speed-final` item | call | evidence |
|---|---|---|
| packed varlen scorer (`int8/scorer.py`) | (b) | main has no scorer. Ported, adapted to main's limb-plane cache (`tri_paged.store`) and to `IntLlama`'s H / HKV / HD (8B geometry). Scorer 42,925 / 23,646 / 25,001 tok/s vs 41,137 / 12,529 / 24,770 for the padded flow with the same engine (GSM8K b8 / 4k b1 / 4k b4) |
| compiled segments by default | (b) | main's `IntLlama(compiled=False)` default; the plugin already passed `BITEXACT_COMPILE` explicitly. Default eager flow 6,168 tok/s, compiled 33,063 (GSM8K b8, padded flow) |
| fused causal attention in `IntLlama` | (b) | main's `IntLlama.attention` still materialises scores (`bmm` + softmax). Ported on main's limb-plane cache: padded flow 33,063 -> 41,137 (GSM8K b8), 13,741 -> 24,770 (4k b4) |
| overflow flag as a separate reduction | (a) | main cf55c9e proves the post-GEMM bound at load (`post_bound`, the flag disappears) and makes the remaining flags row-parallel (`segs._amax`). Same problem (one-CTA amax after a symbolic recompile), solved more fully |
| no-split attend path | (a) for decode, (b) for prefill / scorer tiles | main 5601138 `_attend_dec`: decode (M <= 4) with one split is a single kernel, no atomics, no zero fill. main's prefill path (M > 4) still zero-filled and used atomics at nsplit = 1. Ported to that path only: scorer 48,890 -> 49,980 (GSM8K b8), 25,686 -> 26,480 (4k b1), 26,947 -> 27,856 (4k b4) |
| static-shape compiled `lm_head` (`vllm_plugin/logits.py`) | (a) | main 4f2c7d4 / cf55c9e: `compute_logits` runs the LM head through `tri_sample.Head` (one Triton pass, no int64 logits) and the post-GEMM flag is gone, so the one-CTA amax it fixed does not exist on main |
| no host syncs in `compute_logits` / sampler | (a) | main 277e091: rows from `idx_mapping_np`, positions from `seq_lens_upper_bound_np`, same as ours (and multi-row speculative decoding handled) |
| `OverflowCheck` device assert vs main's late flag read | (c), main kept | main d87b9ce copies `m.err` to pinned memory and raises in `AsyncOutput.get_output`. Our `torch._assert_async` in its place, same GPU, interleaved, 3 rounds: decode ms/step 1.742 / 3.164 / 13.300 vs main 1.719 / 3.155 / 13.240 (B = 1 / 64 / 512); README bench at T = 0 (one round of 3 runs) 578 / 14,014 / 32,564 vs main 577 / 13,788 / 32,466 tok/s. Equal within noise, so main's version (a Python exception instead of a dead CUDA context) stays |
| single-group sampler without row gather | (a) | main's `integer_sample` takes the whole-batch path (`whole`) the same way |
| cuBLASLt int8 GEMM (`bmm.int_mm`, default on CUDA) vs main's skinny Triton GEMM | (c) + (b) | main a92973b: skinny Triton GEMM for decode rows (<= 256, table per shape) and `torch._int_mm` above. Kept main's for decode rows; cuBLASLt (sm90 wgmma) ported for rows > 256, where main had `torch._int_mm`. Scorer 42,925 -> 48,890 (GSM8K b8), 23,646 -> 25,686 (4k b1), 25,001 -> 26,947 (4k b4). Inside the vLLM plugin it measured slower (next section), so the plugin keeps `torch._int_mm` there |
| KV cache as 3 int8 limb planes | (a) for the layout, (b) for the allocation | main 9465603 stores the same planes (K limbs, then V limbs, 384 bytes) but still declares head size 2 HD, so each token-head row is 512 bytes with 128 unused. Ported the declaration (head size 3 HD / 2): 278,802 -> 371,741 cache tokens in the same 17.18 GiB (`gpu_memory_utilization` 0.3); cache-limited decode 16.943 -> 15.193 ms/step |

Also checked: with the same prompts and seeds, main's plugin and `exp/speed-final`'s plugin give identical tokens,
per-token log p and logits-row hashes (decode gate, 256 sequences x 4 configurations, 0 differ). The two lines compute
the same integers.

## Cumulative table

Each row adds one commit to the row above. Scorer: tok/s (prompt + response tokens) of the 138 scorer harness (GSM8K =
1,024 plugin rollouts at batch 8; 4k = 16 rollouts of about 4k tokens at batch 1 and 4), median of 3 timed passes after
a warm-up; rows "main" to "+ fused attention" are the padded flow (`score_padded`), later rows the packed scorer.
Decode: ms/step of one `generate` (prefill included), T = 1, vLLM CUDA graphs, B = 1 and 64 generate 512 tokens, B = 512
generates 256 tokens, median of 3 rounds. Bench: main's README benchmark (`tests/vllm/bench.py`: B requests of 128 in /
128 out, second of two identical generate calls timed), output tok/s, T = 0, median of 6 runs (2 interleaved rounds of
3; a third round ran for main only, the other trees' third round did not start because the GPU showed 1 MiB). At T = 1
final vs main: 530 / 12,958 / 31,623 vs 538 / 13,061 / 31,132.

| state | scorer GSM8K b8 | scorer 4k b1 | scorer 4k b4 | decode B=1 | decode B=64 | decode B=512 | bench B=1 | bench B=32 | bench B=128 |
|---|---|---|---|---|---|---|---|---|---|
| main, default `IntLlama` (eager) | 6,168 | 2,344 | 3,520 | 1.755 | 3.159 | 13.169 | 572 | 13,968 | 32,613 |
| + compiled by default | 33,063 | 10,235 | 13,741 | = main | = main | = main | = main | = main | = main |
| + fused causal attention | 41,137 | 12,529 | 24,770 | = main | = main | = main | = main | = main | = main |
| + packed scorer | 42,925 | 23,646 | 25,001 | = main | = main | = main | = main | = main | = main |
| + cuBLASLt GEMM (rows > 256, not in the plugin) | 48,890 | 25,686 | 26,947 | 1.736 | 3.145 | 13.255 | 568 | 13,869 | 32,622 |
| + no-split prefill attend | 49,980 | 26,480 | 27,856 | n/m | n/m | n/m | n/m | n/m | n/m |
| **+ 6 HD-byte KV rows (final)** | **51,460** | **26,586** | **27,759** | **1.755** | **3.144** | **13.160** | **571** | **13,965** | **32,596** |
| final vs main default | 8.3x | 11.3x | 7.9x | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |
| final vs main compiled flow | 1.56x | 2.58x | 2.02x | | | | | | |

"= main": the commit does not change any code the plugin runs (the plugin constructs `IntLlama` with explicit
flags and its own attention), so the plugin rows equal main's. n/m: not measured separately. The scorer row "main"
is main's padded flow at main's default (`compiled=False`); "+ compiled by default" is the same flow compiled.
Decode and bench rows of main, "+ cuBLASLt" and final were measured in interleaved rounds on one GPU (decode GPU 3,
bench GPU 2). Scorer rows of "+ packed scorer", "+ cuBLASLt", "+ no-split" and final were measured in one interleaved pass on GPU 2
together with main's compiled flow (33,031 / 10,286 / 13,724 there); a second pass on GPU 4 gave main 33,094 / 10,316 /
13,751 and the final cuBLASLt commit 48,539 / 25,453 / 26,930. The "+ compiled" and "+ fused attention" scorer rows
come from separate runs on other GPUs. The pass-1 trees `lt3`, `ns3`, `kv3` still had cuBLASLt in the plugin path; their
scorer code is the same as in the final commits.

Cache-limited decode (B = 512, 256 tokens, `gpu_memory_utilization` 0.15, 3 rounds interleaved, GPU 3): main 16.943
ms/step (86,012 cache tokens), final 15.193 (114,688), 1.12x. At 0.2 (main 150,275 tokens, `kv3` 199,321) the step
is not cache bound: main 13.28, `ns3` 12.33, `kv3` 12.37 ms/step (those two trees still had cuBLASLt in the plugin).

## What was dropped

- cuBLASLt inside the vLLM plugin. Same GPU, same tree (`kv3`), `BITEXACT_GEMM=lt` vs `intmm`, README bench at T = 0,
  3 rounds: B = 32 13,525 vs 14,171, B = 128 31,633 vs 33,031 tok/s (lt 4.6% / 4.2% slower). The decode harness
  went the other way: B = 512 12.377 (`lt3`) vs 13.240 (main) ms/step, same GPU, interleaved. Main's README metric
  decides, so `IntLlama.lt` is off in the plugin (`BITEXACT_PLUGIN_LT=1` turns it on); the plugin then measures equal to
  main on both (cumulative table, row "+ cuBLASLt"). Why cuBLASLt slows the README bench was not established.
- `OverflowCheck` (see the overlap map): equal speed, main's version kept.

## Gates

Per port, on 138 (H100), each GPU at 0 MiB at launch. Trees: `sc` = packed scorer commit, `lt4` = cuBLASLt commit,
`ns4` = no-split commit, `kv4` = final code commit.

| gate | sc | lt4 | ns4 | kv4 (final) |
|---|---|---|---|---|
| main's invariance suite (`tests/vllm/run_tests.sh`, 14 runs vs the integer reference) | 14/14, compare_exit 0, compare_long_exit 0 | same | same | same |
| suite hash files byte-equal to main's run | 14/14 | 14/14 | 14/14 | 14/14 |
| full test suite (`pytest tests`, H100, constants set) | 117 passed, 6 skipped | 124 passed, 6 skipped | 124 passed, 6 skipped | 124 passed, 6 skipped |
| scorer gate, 8 configurations (GSM8K 128 seq at b8 / b64 / b1 / b8 shuffled, 28,083 rows; 4k 16 seq at b4 / b1 / b8 / b4 shuffled, 65,536 rows): rows equal plugin hashes, sequence hashes and float64 log p bits equal the eager reference | 8/8 | 8/8 | 8/8 | 8/8 |
| decode gate (32 GSM8K prompts x G8 = 256 seqs; T = 1 53,387 tokens, greedy 65,536 tokens; B = 1 and B = 64) vs main | 0 differ, PASS | 0 differ, PASS | 0 differ, PASS | 0 differ, PASS |

main itself: 113 passed, 2 skipped; suite 14/14. The first two commits ("compiled", "fused attention") change no
plugin code; each passed the scorer gate 8/8 (padded flow), and every later tree, which contains them, passed all gates. The 6 skips are the exact-mode parametrisations of `test_invariance` and `test_scorer` (no
exact constants on 138) and `tests/vllm/test_overflow_check.py` (needs `BITEXACT_VLLM_MODEL`); run on its own with
the plugin model, that test passes on the final tree and on main (1 passed each).

H800 check (141, GPU 4, driver 535.183.06, `/data41/audreymao/speed_on_main_20261104`), final tree (39f517a, the
last code commit; the RESULTS commit changes no code), scorer gate with
the H100 references and plugin hashes: 8/8 (all 28,083 and 65,536 rows, 128 and 16 sequences equal).

## Setup

138: 8 x H100 80GB, vLLM 0.32.0, torch 2.13.0+cu130, Triton 3.7.1, venv `/home/audreymao/abc_measure/venv` (no
installed bitexact package: vLLM scripts run through `inproc.py`, which registers the plugin in-process). Llama-3.2-1B
int8 constants `~/abc_measure/const.pt`. Other jobs of ours ran on other GPUs at the same time; the host was otherwise
idle (load about 0.3 before the runs).

Runner: `run138.sh` (`sync`, `pytest`, `suite`, `scheck`, `stime`, `gate`, `dspeed`, `dcap`, `bench`, and `ab` for
same-GPU interleaved rounds), `run141.sh`, `summary.py`. Outputs on 138 in `~/som/out`.
