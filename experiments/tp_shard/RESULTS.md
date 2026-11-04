# Tensor-parallel sharding of the BitExact int8 engine

Question: `experiments/portability/RESULTS.md` argues that vLLM's batch-invariant bf16 mode loses bit
identity across tensor-parallel sizes, while an integer engine would not, because its reductions are exact
integer sums. That was marked "untested: BitExact has no sharded implementation". This experiment adds one
— a real Megatron-style split of every weight matrix, with `torch.distributed` NCCL collectives between
ranks — and measures whether the output stays bit-identical to the unsharded engine.

## Result

Every cell compares against the TP = 1 run of the same code path, on the same constants
(`e669be2a...`) and the same token histories. "identical" = the greedy/sampled token matches, the
SHA-256 of the **full 128,256-wide int64 logits row** (gathered from every rank) matches, and the
canonical `int_logp` of the chosen token matches, at every position.

| test | TP 2 vs TP 1 | TP 4 vs TP 1 |
|---|---|---|
| 4 fixed prompts, greedy, 32 new tokens (one prompt at a time) | 128/128 identical | 128/128 identical |
| 4 fixed prompts, greedy, 256 new tokens, one prompt at a time (KV cache, 256 sequential steps) | 1,024/1,024 identical | 1,024/1,024 identical |
| 4 fixed prompts, greedy, 256 new tokens, one right-padded batch | 1,024/1,024 identical | 1,024/1,024 identical |
| 4 fixed prompts, Gumbel-max T=1 (the plugin's sampler), 64 new tokens | 256/256 identical | 256/256 identical |
| GSM8K workload: 256 seeded rollouts, 53,387 teacher-forced positions, batch 8 | 53,387/53,387 identical | 53,387/53,387 identical |
| same workload at TP 2, batch 1 vs batch 8 (batching control, both vs TP 1 batch 8) | 53,387/53,387 identical | — |
| H100 (138) TP 1 and TP 2, same 53,387 positions, same constants | 53,387/53,387 identical to the H800 runs | — |
| **control: shipped `IntLlama`, unsharded** (TP 1 sharded path vs it) | 128/128 at 32 tokens; 1,024/1,024 at 256 tokens, sequential and batched | — |

`max |Δ log p| = 0` in every row; not a single logits row differed, so the claim that "any summation
order gives the same integer" is now measured, not inferred.

**Cost.** Decode throughput (H800, 64-token prefill, greedy, `steps` timed after warm-up; median of
3 runs on the same 5 devices, spread across runs as noted):

| GPU budget | B=1 ms/step | B=8 | B=32 | B=128 | B=256 | B=256 tok/s | run-to-run spread |
|---|---|---|---|---|---|---|---|
| unsharded `IntLlama` (1 GPU) | 101.1 | 101.0 | 100.8 | 102.9 | 105.1 | 2,435 | ≤12% |
| TP 1 (this code, 1 GPU) | 113.9 | 113.2 | 112.7 | 114.5 | 118.1 | 2,168 | ≤11% |
| TP 2 (2 GPUs) | 115.2 | 116.1 | 124.7 | 128.2 | 120.1 | 2,131 | ≤11% |
| TP 4 (4 GPUs) | 125.5 | 125.6 | 125.0 | 125.7 | 129.4 | 1,978 | ≤2% |

Interpretation: the step is kernel-launch and elementwise bound at this size, so splitting the GEMMs does
not shorten it. Spending 2 or 4 GPUs makes the step **slower**, not faster: TP 2 costs +10-25% and TP 4
+19-28% versus the unsharded engine, over B = 1…256. Only the TP 4 column is stable enough run to run
(≤2%) to read per-batch differences; TP 2's per-batch ordering sits inside its 11% spread, so read it as
"no better than unsharded", not as specific deltas. The TP 1 column is the same comparison minus all
communication, and it already costs +12% (extra indirection in `Segs` / `quant_rows_shared`), so any TP
gain would have to exceed that before it was visible here. Sharding at 1B buys **memory and exactness,
not speed**: 2 or 4 GPUs hold a model that already fits in one, and nothing gets faster.

## Facts

- **What is split.** `wqkv` by head (rank `r` owns query heads `[r·h, (r+1)·h)` and KV heads
  `[r·hkv, (r+1)·hkv)`), `wo` on the input side over those same head columns, `wgu` by FF channel with
  gate row `j` and up row `j` on the same rank (so `silu_mul` stays rank-local), `wd` on the input side
  over those FF channels, `lm` by vocabulary. H = 32, HKV = 8 (GQA 4:1), FF = 8192, V = 128,256, so
  TP 2 and TP 4 both divide all four exactly.
- **What is replicated.** The embedding, the RMSNorm gammas, the RoPE tables, the exp/SiLU/ln tables, and
  the pre-segment quantization of every norm input. Those are row-local pure integer functions, so
  replication is exact by construction; it is also the reason no communication is needed before a split
  linear layer.
- **What is communicated.** Four int64 all-reduces per layer under TP > 1: the `wo` accumulator over the
  input (head) split and the `wd` accumulator over the input (FF) split, plus one small all-reduce of the
  per-row `amax` vector before each of those two splits. Plus one `all_gather` of the vocabulary shard
  when the full logits row is needed. The all-reduce sums exact integers, so their order cannot matter.
- **The one non-obvious requirement.** `quant_rows` derives its per-row shift from `amax` over the
  **whole** row. A K-split therefore has to quantize with the globally reduced `amax`, or the ranks use
  different scales and the partial sums no longer reference the same fixed point. With N-splits this is
  free (the input row is replicated); with the two K-splits it costs one all-reduce of a `[M]` int64
  vector. `tpmodel.quant_rows_shared` / `pre_plain_shared` are the only arithmetic changes versus the
  unsharded `Segs`, and the TP 1 gate proves they are a no-op at world size 1.
- **Overlap of attention heads.** Each rank's `hkv` KV heads attend to their own `h/hkv` query-head group
  only, so the `B·HKV` batched attention degenerates into `B·hkv` independent problems per rank; the GQA
  grouping inside a rank is unchanged. This is the same decomposition Megatron uses for GQA.
- **TP 1 is a real gate, not a formality.** `run_sharded` at TP 1 (`shard(c, 0, 1)` is the identity
  split) reproduces the shipped `IntLlama` on 4 prompts × 32 greedy tokens and on both 256-token modes
  (sequential and one padded batch), including the logits-row hashes: 1,024/1,024 in each 256-token
  mode. So the identity results above are "sharded == unsharded", not merely "sharded == sharded".
- **Cross-GPU.** H100 (138) at TP 1 and TP 2 produces the same 53,387 logits-row hashes and log p as
  H800 (141), from the same constants file (`e669be2a...`), with a different CUDA build
  (2.13.0+cu130 vs 2.13.0+cu129).
- **Batch invariance survives sharding.** The TP 2 workload run at batch 1 and at batch 8 matches the same
  reference, so the per-row independence the engine claims is preserved by the split.
- **Throughput.** TP is a net loss at 1B and the loss grows with TP size: 2,435 tok/s unsharded on one
  GPU vs 2,131 (TP 2) and 1,978 (TP 4) at B = 256, and a wider gap at B = 1 (101 vs 115 vs 126 ms/step).
  TP 4's numbers are stable to 2%; TP 2's sit inside an 11% run-to-run spread, so its absolute position
  relative to TP 1 is not resolved — only "not better" is. The reason is that at 1B a GEMM is already
  launch-latency sized, so quartering it saves less than the 4 all-reduce round trips per layer cost.
  The TP 1 column isolates the code-path overhead: +12% over the shipped engine on the same GPU with the
  same weights and tokens.
- **One bug the gate caught.** `gather_logits` first used `broadcast(src=r)` over rank buffers; rank 0's
  own shard was overwritten by the broadcast from `r = 0` (a no-op send to itself is fine, but the loop
  read the buffer after it had been reused), so TP 2 logits were wrong while every layer's hidden state
  matched. The per-layer debug script (`dbg_tp.py`, hidden-state hashes after each attention and each MLP
  block) localised it to the gather, not the shard. `all_gather` fixes it. Worth recording because it is
  exactly the class of bug a hash-only end-to-end comparison hides.

## Inferences (not directly tested)

- This is what the portability writeup's inference claimed, now with an implementation behind it. The
  relevant consequence: a model trained through the integer engine at one TP size and served at another
  should keep identical log p, unlike vLLM's batch-invariant arm D, which moved by mean |Δ| 0.0241 across
  TP sizes. That serving comparison was not run: it needs the vLLM plugin to expose a sharded path, which
  it does not have (the plugin replicates the full model per rank).
- The throughput balance at 8B was not measured. At 1B the GEMMs are launch-latency sized and the split
  is clearly negative; 8B GEMMs are 8× wider, so arithmetic intensity per collective grows and the split
  could turn positive, but that is a prediction, not a measurement.
- Only TP ≤ 4 was run. TP 8 divides H = 32, HKV = 8, FF = 8192 and V = 128,256 exactly, so no new
  sharding case appears, but it was not executed.

## Not tested

- TP 8, and any TP size that does not divide H, HKV, FF and V (no padding / unbalanced split path
  exists; `shard()` asserts).
- Pipeline or sequence parallelism, FP8/bf16 weights, CPU ranks, or a mixed CPU/GPU process group.
- The vLLM plugin with a sharded model (the plugin still replicates; see portability RESULTS.md).
- Throughput at the 8B model size, and under real serving (continuous batching, prefix cache).

## Setup

- Host 141: 8× H800 80GB, driver 535, torch 2.13.0+cu129, Python 3.11.15, `/data41/audreymao/bitexact_port`
  (`venv`, `data/intllama_const.pt`, code synced to `tp_shard/{repo,code}`). GPUs used only when
  `memory.used == 0`.
- Host 138: 8× H100 80GB, driver 595, torch 2.13.0+cu130, `/home/audreymao/abc_measure/{venv,const.pt,
  deqmodel,gsm8k}` and `~/tp_shard/{repo,code,out}`.
- Constants: `intllama_const.pt`, internal SHA-256 `e669be2a7ad5b4575a521f9516d36fefed79157a25a4efe7f470375d24ec5bb5`,
  16 layers, Llama-3.2-1B geometry.
- Workload: `experiments/rl_mismatch` GSM8K protocol — 32 test questions, 2-shot prefix, G = 8, seed
  `1 + 8p + g`; the token histories are the 256 rollouts of `C_t1.json` (53,387 tokens, median length
  256), scored teacher-forced in right-padded batches of 8 (and of 1 for the batching control).
- Comparison: `compare_tp.py` (greedy/sampled replays) and `compare_gsm.py` (workload scores) over the
  shared seeds; both compare token, logits-row SHA-16 and `common.int_logp` per position.

## Files and commands

```
experiments/tp_shard/tpmodel.py     shard(constants, rank, world) + TPIntLlama + gather_logits
experiments/tp_shard/tp_run.py      greedy / Gumbel replay of fixed prompts, single or batched; --reference runs IntLlama
experiments/tp_shard/tp_gsm.py      teacher-forced scoring of a rollout file at TP N
experiments/tp_shard/compare_tp.py  replay comparison (token / sha / lp)
experiments/tp_shard/compare_gsm.py workload comparison over common seeds
experiments/tp_shard/bench_tp.py    decode throughput, TP N vs --ref (shipped engine)
experiments/tp_shard/dbg_tp.py      per-layer hidden-state hashes (the bisect tool that found the gather bug)
```

```
# 141. `env.sh` sets BITEXACT_INT8_CONST (not CONST) and ends with `cd $W/repo`,
# so use absolute paths and $C.
W=/data41/audreymao/bitexact_port; S=$W/tp_shard/code/experiments/tp_shard; . $W/env.sh; C=$BITEXACT_INT8_CONST
export PYTHONPATH=$W/tp_shard/repo/src:$S
CUDA_VISIBLE_DEVICES=a,b $PY -m torch.distributed.run --nproc_per_node=N --standalone \
  $S/tp_run.py --const $C --out $S/../out/seq --max-new 256 [--reference] [--sample] [--batch]
CUDA_VISIBLE_DEVICES=a,b $PY -m torch.distributed.run --nproc_per_node=N --standalone \
  $S/tp_gsm.py --const $C --tokenizer <deqmodel> --data <gsm8k> --rollout $W/data/C_t1.json --out X.json --bs 8
$PY $S/compare_tp.py out/seq/tp1.json out/seq/tp2.json out/seq/tp4.json
export PYTHONPATH=$PYTHONPATH:$W/tp_shard/code/experiments/rl_mismatch
$PY $S/compare_gsm.py out/gsm_tp1.json out/gsm_tp2.json out/gsm_tp4.json
$PY -m torch.distributed.run --nproc_per_node=N --standalone $S/bench_tp.py --const $C --bs 1 8 32 128 256 --steps 32 [--ref]
```

`run141.sh` and `bench.sh` wrap the two loops (idle-GPU selection, retries, summary). Raw outputs:
`$W/tp_shard/out/{seq,sample,gsm_tp*,h100_gsm_*}/` and `final_bench_tp*.json`.