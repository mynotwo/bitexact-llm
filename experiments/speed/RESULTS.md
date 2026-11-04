# BitExact speed-up, round 2 (branch exp/speed)

`exp/speed` = `exp/speed-scorer` + merge commit of `exp/speed-decode` + two more bit-exact items. One H100 80GB per run (host 138, each started on a GPU showing 0 MiB; other jobs on other GPUs), torch 2.13.0+cu130, Triton 3.7.1, vLLM 0.32.0. Median of 3 timed passes after one warm-up.

## Cumulative table

Scorer: tok/s of teacher-forced log p over prompt + response (GSM8K = 1,024 plugin rollouts at batch 8; 4k = 16 rollouts of ~4k tokens at batch 1 and 4). Decode: ms/step = wall of a 512-token generate (prefill included) / 512, T = 1, vLLM CUDA graphs, B requests of 178-260-token GSM8K prompts. Prefill = cold 64-request (or 1-request) prefill plus one generated token.

| state | scorer GSM8K b8 | 4k b1 | 4k b4 | decode B=1 ms/step | decode B=64 ms/step | B=64 512-token generate | B=64 prefill |
|---|---|---|---|---|---|---|---|
| original (eager scorer; plugin at e5eb24c) | 6,637 | 2,361 | 3,658 | 4.286 | 18.160 | 9.30 s | n/m |
| compiled scorer default (speed-scorer a) | 27,239 | 9,391 | 12,922 | | | | |
| exp/speed-scorer + exp/speed-decode merged (start of this round) | 36,016 | 19,963 | 20,880 | 2.683 | 7.019 | 3.59 s | 2.17 s |
| a. overflow flag as a separate reduction | 36,610 | 20,028 | 21,121 | 2.683 | 6.139 | 3.14 s | 0.59 s |
| b. no-split path in `tri_paged.attend` | 38,462 | 20,468 | 21,484 | 2.688 | 5.779 | 2.96 s | 0.57 s |
| total vs original | 5.8x | 8.7x | 5.9x | 1.59x | 3.14x | | |
| items a+b vs merged start | 1.07x | 1.03x | 1.03x | 1.00x | 1.21x | 1.21x | 3.8x |

n/m = not measured. Original rows are copied from `experiments/speed_scorer/RESULTS.md` and `experiments/speed_decode/RESULTS.md`; "original" scorer is `compiled=False, fused=False` as abc_measure ran it. The merged, a and b rows were measured in this round. The merged and a rows ran at the same time on two GPUs of 138, b ran alone afterwards with the same scripts (`experiments/speed/`). Decode B=1 differs by up to 0.8% between GPUs of 138 (the same tree gives 2.683 and 2.704 on two different GPUs), so B=1 is unchanged by a and b within that spread. Scorer differences of 1-2% between runs are noise of the same size.

## Item a: prefill single-CTA kernel

`Segs.post` / `post_add` computed the overflow flag as `t.abs().amax() >= 2^38` over the whole `[M, N]` output inside one `torch.compile(dynamic=None)` function. After the first recompile with symbolic sizes Inductor emits a grid (1,1,1) kernel for that reduction (found in `exp/speed-decode`), so each GEMM epilogue of a large prefill chunk runs on one CTA. Change: `post` returns the per-row maximum of `|t|` (a reduction along each row; the kernel grids were not inspected, the prefill timing below is the evidence), and `IntLlama._linear` tests `tmax.amax() >= 2^38` outside the fused function. The values `y` are untouched; the flag is the same maximum compared with the same bound, so the flag fires for the same inputs. The plugin's `lm_logits` keeps the whole-tensor form (`post(..., whole=True)`), where its own `dynamic=False` compile already splits the reduction.

Prefill of 64 requests (plus one generated token): 2.17 s -> 0.59 s. The 512-token B=64 generate: 3.59 s -> 3.14 s (7.02 -> 6.14 ms/step). Prefill of one request: 0.11 s -> 0.04 s. The scorer is unaffected (its compiled segments use `dynamic=False` and already split the reduction): 36,016 -> 36,610, 19,963 -> 20,028, 20,880 -> 21,121 (within run-to-run noise, not slower).

## Item b: no-split path in `tri_paged.attend`

With `nsplit == 1` each `(query tile, request, head)` has exactly one program, so `_max_kernel` stores its row maxima and `_acc_kernel` stores `num` and `den` instead of `atomic_max` / `atomic_add`; `num` and `den` are `torch.empty`, not `zeros`, and `mx` is a per-stream reused scratch (a fresh allocation while a CUDA graph is being captured, so the graph owns its memory). A row with no visible key (`qpos = -1`) gets `mx = NEG`, `num = 0`, `den = 0`, as in the split path. The integer softmax is unchanged: two sweeps, same table, same limb arithmetic. `nsplit > 1` runs the old code (the kernels are specialised on a `SPLIT` constexpr). Which calls take the new path: tiles >= 264, i.e. decode at B=64 (8 KV heads x 64 requests), long prefill chunks and the scorer's chunks. B=1 decode has 8 tiles and keeps the split path.

Decode B=64: 6.14 -> 5.78 ms/step (-0.36 ms). Scorer: 36,610 -> 38,462 (+5%), 20,028 -> 20,468 (+2%), 21,121 -> 21,484 (+2%). New unit test `test_attend_nosplit_reuse_and_empty_rows` fills the scratch with garbage, changes sizes between calls, includes `qpos = -1` rows and compares `nsplit=1` with 3 and 16.

## Gates (each item)

| gate | a | b |
|---|---|---|
| scorer, 8 configurations (GSM8K 128 seq at b8, b64, b1, b8 shuffled: 28,083 rows; 4k 16 seq at b4, b1, b8, b4 shuffled: 65,536 rows): rows equal to plugin hash, sequence hashes and float64 log p bits equal to the eager reference | 8 / 8 | 8 / 8 |
| decode, seeded T = 1 (53,387 tokens) and greedy (65,536 tokens), 32 GSM8K prompts x G8 = 256 sequences, B=1 and B=64: tokens, argmax, log p, logits-row hash equal to the pre-change plugin (`base`) | 6 / 6 comparisons, 0 sequences differ | 6 / 6, 0 differ |
| decode B=1 equals B=64 | PASS | PASS |
| CPU reference path, 16 GSM8K sequences (3,501 rows) equal plugin hashes | 3,501 / 3,501 | 3,501 / 3,501 |
| speed not slower (table above) | yes | yes |

"Pre-change plugin" = the rollouts of `exp/speed-decode`'s `base` run (the plugin at e5eb24c), recorded before any of its changes. Decode gates ran with `BITEXACT_VERIFY_HOST_ROWS=1`.

Merge: `exp/speed` full test suite on 138 (H100): 69 passed, 8 skipped (the exact-engine parametrisation, as before). `ruff check` clean on `src`, `tests`, `experiments/speed*`.

## Cross-GPU, H800 of host 141

Final tree (commit of the push) on GPU 3 of host 141 (H800, 0 MiB before the run; torch 2.13.0+cu129; env `/data41/audreymao/bitexact_port/venv`). Hashes and float64 log p bits compared with the eager reference recorded on the H100 of 138 and with the plugin hashes:

| config on H800 | rows equal plugin hash | sequences equal ref hash / log p bits |
|---|---|---|
| GSM8K 128 seq, batch 8 | 28,083 / 28,083 | 128 / 128 and 128 / 128 |
| GSM8K 128 seq, batch 64, shuffled | 28,083 / 28,083 | 128 / 128 and 128 / 128 |
| 4k 16 seq, batch 4 | 65,536 / 65,536 | 16 / 16 and 16 / 16 |
| 4k 16 seq, batch 1 | 65,536 / 65,536 | 16 / 16 and 16 / 16 |

Both new code paths ran there (item a in every GEMM epilogue; item b in the scorer's attention).

## Not done

- Single-pass online softmax in `tri_paged` (would change the integer definition; excluded by the brief).
- CUDA graphs for the scorer (see `experiments/speed_scorer/RESULTS.md`, d).
- No fresh vLLM decode run on H800 (141 has no vLLM environment for 0.32; unchanged from `exp/speed-decode`). The H800 check is for the scorer only.

## Files

`experiments/speed/`: `env_scorer.sh`, `env_decode.sh`, `run_scorer.sh` (time / check), `run_decode.sh` (gate / speed), `sync.sh` (copy a tree or a commit to 138). `experiments/speed_decode/speed.py` gained `--prefill`. Code: `src/bitexact/int8/segs.py`, `int8/model.py`, `vllm_plugin/tri_paged.py`, `vllm_plugin/logits.py`, `tests/test_vllm_kernels.py`.
