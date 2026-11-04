# Portability: BitExact integer engine (C) vs vLLM batch-invariant mode (D)

Question: when the vLLM version, the tensor-parallel size, the batch size, the GPU type or the device (CPU) changes, do the per-token log p and the generated tokens stay bit-identical? Arm C = BitExact int8 vLLM plugin (sampler) + standalone int8 engine (scorer). Arm D = vLLM bf16, `VLLM_BATCH_INVARIANT=1`, `attention_backend=TRITON_ATTN`, sampler + the same engine scoring via `prompt_logprobs` (as in `experiments/abc_measure`).

## Result

Each cell compares against the reference run of the same arm (vLLM 0.32.0, TP 1, max_num_seqs 256, H100). "bitwise 0" = every compared log p bit-equal (C: also every int64 logits-row SHA-256) and every token sequence identical. "differs (mean|Δ|, % tokens differ)" is the scorer on identical input tokens, followed by sampler token agreement (identical sequences out of 256; seeded T = 1 and greedy).

| test | C (BitExact) | D (vLLM batch-invariant) |
|---|---|---|
| vLLM 0.31.0 (torch 2.13.0, Triton 3.7.1) | bitwise 0 (plugin T=1 and greedy, engine scorer) | bitwise 0 (sampler T=1 and greedy, scorer) |
| vLLM 0.33.0 (torch 2.14.0, Triton 3.8.0) | bitwise 0 (plugin T=1 and greedy, engine scorer) | differs (1.3e-4, 17.2%); tokens T=1 254/256, greedy 256/256 (greedy lp differ on 14.0% of tokens, max 4.8e-7) |
| vLLM 0.28.0 (torch 2.13.0, Triton 3.7.1) | plugin does not load (vLLM API missing); engine scorer under this venv's torch: bitwise 0 | differs (0.0244, 99.95%); tokens T=1 0/256, greedy 128/256 |
| vLLM 0.25.0 (torch 2.11.0, Triton 3.6.0) | plugin does not load (vLLM API missing); engine scorer under this venv's torch: bitwise 0 | differs (0.0243, 99.96%); tokens T=1 0/256, greedy 88/256 |
| TP 2 vs TP 1 | plugin: replication only (`tensor_parallel_size=2` runs the full model on both ranks), bitwise 0. Standalone engine with a real split: bitwise 0 on 53,387 tokens (`experiments/tp_shard`) | differs (0.0241, 99.97%); tokens T=1 25/256, greedy 80/256 |
| TP 4 vs TP 1 | not tested in the plugin (replication only, see TP 2); the standalone engine sharded for real at TP 2 and TP 4 is bitwise 0 vs TP 1 (`experiments/tp_shard`) | differs (0.0242, 99.96%); tokens T=1 32/256, greedy 112/256 |
| batch size within TP 1 (max_num_seqs 1 / 64 vs 256; scorer batch 8 vs 128 or engine batch 1 / 64 vs 8) | bitwise 0 (sampler and scorer) | bitwise 0 (sampler and scorer) |
| batch size within TP 2 and within TP 4 (max_num_seqs 1 / 64 vs 256) | not applicable in the plugin (replication). The sharded engine scored the workload at batch 1 and 8 with identical results (`experiments/tp_shard`) | bitwise 0 (both TP sizes) |
| GPU type: H800 sampler (abc_measure Q1) vs H100 sampler, 256 common seeds | bitwise 0 | bitwise 0 |
| GPU type: H800 plugin rollouts (1,024 seqs, 215,899 tokens) scored by H100 engine | bitwise 0 | (H800 sampler scored on H100: bitwise 0, measured in abc_measure) |
| CPU recomputation, 16 seqs | bitwise 0 (x86 CPU engine vs H100 engine and vs H100 plugin, 3,496 tokens, all logits-row hashes equal) | could not run batch-invariant mode on CPU; plain PyTorch CPU: bf16 differs (0.0253, 100%), fp32 differs (0.0229, 100%) |

**Does the integer engine keep an advantage?** Yes: C stayed bitwise identical in every test it could run (vLLM 0.31 / 0.33, sampler batch 1 to 256 and scorer batch 1 to 64, H800 vs H100, GPU vs CPU, torch 2.11 to 2.14). D held only within one vLLM build and one TP size. It broke across TP sizes, across vLLM 0.25 / 0.28 / 0.33, and on CPU.

## Facts

- **D is batch invariant inside a fixed configuration.** This holds in every vLLM version tested (0.25, 0.28, 0.31, 0.32) and every TP size tested (1, 2, 4): max_num_seqs 1 / 64 / 256 give identical tokens and log p, and the sampler and the `prompt_logprobs` scorer agree bit for bit (0.25.0 and 0.28.0: 256/256 sequences, 0 log p differences). So the cross-version and cross-TP differences are differences between configurations. Within each configuration the mode works.
- **D across TP size.** The difference has the same size as bf16 noise vs a different kernel stack (scorer mean |Δ| 0.024; abc_measure's vLLM bf16 vs HF bf16 was 0.025). In batch-invariant mode vLLM 0.32 uses a fixed-order all-reduce (`all_reduce_batch_invariant`: all-gather, then a fixed-order sum). That fixes the order for a given world size, but a different TP size splits each GEMM reduction into different partial sums.
- **D across vLLM versions.** 0.31.0 equals 0.32.0 bitwise. The two share torch 2.13.0 and Triton 3.7.1. 0.28.0 has the same torch and Triton as 0.32.0 and still differs at bf16-noise size (0.024), so the difference comes from vLLM's own kernels or graph. 0.33.0 (torch 2.14, Triton 3.8) differs by much less (scorer mean |Δ| 1.3e-4, 17% of tokens; greedy tokens all equal, greedy sampler log p max |Δ| 4.8e-7). With seeded T = 1, versions 0.25 and 0.28 diverge at the first generated token in most sequences (median first divergence = token 0). Greedy decoding with the same versions matches for a median of 31-33 tokens.
- **C across vLLM versions.** On 0.31.0 and 0.33.0 the plugin loads (in-process registration, no code change) and reproduces the reference tokens, log p and logits-row hashes. On 0.28.0 and 0.25.0 it fails at import: `ModuleNotFoundError: No module named 'vllm.v1.worker.gpu.sample.logits_processor'`. The plugin's integer sampler is written against the V2 model-runner interface of vLLM >= 0.31 (`pyproject.toml` pins `vllm==0.32.0`). The standalone engine needs only torch, and under the torch of every venv (2.11.0, 2.13.0, 2.14.0, all cu130) it gives the same 53,387 logits-row hashes.
- **C under `tensor_parallel_size=2`.** In-process registration does not reach TP worker processes (`Model architectures ['BitExactLlamaForCausalLM'] are not supported`). With the package installed in the experiment's own venv (entry points), TP 2 runs, both ranks log every row (106,946 rows = 2 x 53,473) with identical hashes, and the output equals TP 1. The model class has no tensor-parallel layers, so this is replication, not sharding.
- **C cross-GPU (baseline claim, now inside this branch).** Two checks. First, the H100 plugin re-sampling the first 256 Q1 requests reproduces the H800 plugin rollouts from abc_measure. Second, the H100 standalone engine scores all 1,024 H800 rollouts (215,899 tokens) with 0 hash differences. The int8 constants file rebuilt on 138 from the original checkpoint has the same internal SHA-256 (`e669be2a...`) as the one on 141.
- **D cross-GPU sampler (not measured before).** The H100 batch-invariant sampler reproduces the H800 one (abc_measure `q1_D`, vLLM 0.32.0+cu129 on driver 535) for all 256 common sequences, bit for bit. The H100 run used vLLM 0.32.0 with torch 2.13.0+cu130 on driver 595, so this pair also changes the CUDA build (12.9 to 13.0).
- **C with real sharding (new, measured in `experiments/tp_shard`).** An integer engine split by tensor parallelism stays bit-invariant. `tpmodel.shard` splits `wqkv` by head (`wo` on the input side over those columns), `wgu` by FF channel with gate row `j` on the same rank as up row `j` (`wd` on the input side), and `lm` by vocabulary; the two K-splits all-reduce their int64 accumulators, and both quantize with a globally all-reduced row `amax` (the only correctness requirement beyond the split itself). Against TP 1 of the same code path — itself identical to the shipped `IntLlama` — TP 2 and TP 4 agree on every position: 53,387/53,387 logits-row hashes and `int_logp` on the GSM8K workload, 1,024/1,024 on greedy 256-token replay, 256/256 on Gumbel-max T=1, at batch 1 and at batch 8, on H800 and on H100. Cost, not benefit: 2,131 vs 2,435 tok/s at B = 256 (TP 2 vs unsharded on one GPU; TP 4 is 1,978), and roughly +25% wall time per decode step, because at 1B the step is launch-bound and the four all-reduces per layer cost more than the shrunken GEMMs save.
- **D on CPU, attempts:**
  1. vLLM 0.32.0 (CUDA wheel) with no GPU visible: engine init fails, `AssertionError: DP adjusted local rank 0 is out of bounds for 0 devices`.
  2. vLLM's `init_batch_invariance()` called in a CPU-only process before an HF forward: aborts with `Cuda error ... 'no CUDA-capable device is detected'`. In 0.32.0 the mode registers its aten overrides only for CUDA and XPU dispatch keys.
  3. The only CPU vLLM wheel on the internal mirror is `vllm_cpu` 0.9.2. It is far older than the versions above and was not tried.
  4. Plain PyTorch CPU (HF transformers 5.19, eager attention, batch 1, float32 log-softmax): bf16 mean |Δ| 0.0253 (max 0.28), fp32 0.0229 (max 0.18); every token differs from the GPU batch-invariant log p.

## Inferences (not directly tested)

- D's guarantee covers one software build and one parallel layout. A trainer that recomputes log p needs the same vLLM build, the same TP size and a GPU. A model trained at TP 1 and served at TP 4, or rescored after a vLLM upgrade, gets bf16-sized mismatch back (0.024 mean |Δ|, the size of the A arm in abc_measure).
- The C failures on vLLM 0.25 / 0.28 are an API-compatibility cost, not a numerical one. The engine that produces the numbers runs unchanged on every torch tested. Porting the sampler hook to an older vLLM was not tried.

## Not tested

- vLLM version changes on H800 (141). The vLLM 0.31 / 0.33 wheels on the mirror pull cu130 torch, which does not start on 141's driver 535 (`torch.cuda.is_available() == False`). All version tests ran on H100.
- A CUDA-build change with everything else fixed (same vLLM, same GPU, cu129 vs cu130). It is only mixed into the H800 vs H100 pair.
- C through the vLLM plugin at TP 4 (the plugin replicates; the sharded standalone engine was tested at TP 4).
- D on a CPU build of vLLM (only 0.9.2 available).
- Batch size 1 vs 64 for the plugin under TP 2.

## Setup

- Host 138 (8x H100 80GB HBM3, driver 595.71.05, Xeon Platinum 8457C), GPUs used only when they showed 0 MiB. Reference venv: the existing `~/abc_measure/venv`, read only (vLLM 0.32.0, torch 2.13.0+cu130, Triton 3.7.1, transformers 5.19.0). New venvs `~/portability/venv_{0.25.0,0.28.0,0.31.0,0.33.0}` came from the internal mirror (`install.sh`, `pip install vllm==VER numpy==2.3.5`). Transformers in them: 5.22.1 / 5.22.1 / 5.17.0 / 5.20.2. bitexact was added (`pip install --no-deps -e`) only to `venv_0.31.0`, for the TP 2 run. Everything else loaded bitexact through `PYTHONPATH` and registered the plugin in-process.
- Weights: the dequantized int8 Llama-3.2-1B checkpoint (fp32) for D, int8 constants for C, both from abc_measure.
- Workload: the first 32 GSM8K test questions, 2-shot prefix, G = 8, request seed 1 + 8p + g (the first 256 requests of abc_measure Q1), at most 256 new tokens, stop at EOS. Seeded T = 1 (no top-k/top-p) and greedy. D sampler log p = vLLM raw log p. D scorer = `prompt_logprobs` on prompt + reference tokens, max_num_seqs 128, shuffled. C log p = float64 log-softmax of the int64 logits row. CPU subset: seeds 1, 9, ..., 121 (`seeds16.txt`).
- Comparison (`compare.py`): over the seeds both files contain. For sampler files, positions up to the first differing token (exclusive) are compared. For scorer files, all positions are compared. "% tokens differ" = log p not bit-equal / positions compared.
- Pilot (8 prompts x 2): about 1 min per vLLM run (engine start-up dominates) and 3 min per plugin run (torch.compile). The whole matrix took about 35 min of wall time on 5-6 GPUs.

## Commands

```
# 138, from ~/portability (env138.sh sets PY, X, O, DEQ, INT, CONST, DATA)
bash code/experiments/portability/run138.sh ref 1          # references, batch sizes, C cross-GPU
bash code/experiments/portability/run138.sh tp 3,4,5,7     # D at TP 2 / 4 (+ batch sizes), C TP 2 in-process attempt
bash code/experiments/portability/run138.sh ver 0          # vLLM 0.31.0, 0.33.0
VERS="0.28.0 0.25.0" bash code/experiments/portability/run138.sh ver 0
bash code/experiments/portability/run138.sh verx 0         # within-version controls for 0.25 / 0.28
CUDA_VISIBLE_DEVICES= bash code/experiments/portability/run138.sh cpu ""
CUDA_VISIBLE_DEVICES=6,7 venv_0.31.0/bin/python $X/sample.py --engine bitexact --model $INT --data $DATA \
    --n-prompts 32 --g 8 --tp 2 --out $O/C_t1_tp2_inst.json
$PY $X/analyze.py $O results
```

Example single runs: D sampler `VLLM_BATCH_INVARIANT=1 $PY sample.py --engine bf16 --attn TRITON_ATTN --model $DEQ --data $DATA --n-prompts 32 --g 8 [--temp 0] [--tp N] [--max-seqs M] --out D.json`. D scorer `VLLM_BATCH_INVARIANT=1 $PY score_d.py --attn TRITON_ATTN --rollout D.json --model $DEQ --data $DATA --max-seqs 128 [--tp N] --out Ds.json`. C engine `$PY ../rl_mismatch/bitexact_engine.py score --const $CONST --rollout C.json --tokenizer $DEQ --data $DATA --device cuda|cpu --bs 8 --out Cs.json`.

Files: `results/cmp_<name>.json` (one per comparison, with versions in `x_meta`), `results/summary.json`, `results/analyze.txt`. Raw rollouts and scores are on 138 in `~/portability/out` (per-run logs `*.json.log`).
