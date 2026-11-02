# Llama-3.1-8B, exact mode, bit identity across hardware

Workload `workload_T0.7_8x128.json` (SHA-256 `2384dafe…f889`): A11's T = 0.7 samples sid 0–7, prompt plus
the first 128 generated tokens, 595 positions (sid 0, 6 and 3 are shorter than 128). Teacher forcing on the
recorded tokens, so a subset run (`--texts K --positions N`) computes exactly a prefix of the full run's rows.
Each position stores the SHA-256 of the int64 logits row, the greedy token and the T = 0.7 integer
Gumbel-max token.

Model: `NousResearch/Meta-Llama-3.1-8B` (same bf16 weights as `meta-llama/Llama-3.1-8B`), 4 shards:

```
f8b9704ab09cdeb097aa4a0a24bca96f906eec36bad63ab495bc21475058601b  model-00001-of-00004.safetensors
c28b25e7541751056ee126627e007f8d4288319733285e9f7b17b9ff6eb313f0  model-00002-of-00004.safetensors
d8e9504dd4e4a146d484c52a97584ec14dac92237c46b064934af67a85e7d383  model-00003-of-00004.safetensors
e4486f35c040f683f7d790354f66c169c109eb9fa0954a4a35d7c458a108405d  model-00004-of-00004.safetensors
```

Constants: `python examples/build_constants.py --model <8B dir> --mode exact --out const_8b.pt`
(CPU, about 12 min and 51 GB peak RSS on a Xeon 8457C, 24.7 GB file); internal SHA-256 must be
`2f02ac8e43f08ac337b397d18fc740eb48dca27158c62ef742fbd4a9e729bfa9` (`run_ident.py` checks the prefix).

## Run and compare

```
PYTHONPATH=src python experiments/xhw_8b/run_ident.py --const const_8b.pt --device cuda:0 --mode bs1 --out X_bs1.json
#   --mode bs1 | bs8 | incr      [--compiled] [--gemm cublas]
#   CPU: --device cpu --threads N; --cpu-lazy keeps RAM at ~30 GB (else ~97 GB); --texts K --positions N for a subset
python experiments/xhw_8b/compare.py REF.json X_*.json --out compare_X.json   # exit 0 only if all identical
```

`compare.py` refuses runs with a different constants or workload SHA-256 and reports, per run, positions in
common, logits rows that differ, greedy and sampled tokens that differ, and positions missing from the reference.

## Measured (H800 reference on 141)

| Run | Positions | Time | Peak |
|---|---|---|---|
| H800 bs1 / bs8 / incr / incr+compile / bs8 cuBLAS | 595 each | 18.5 / 5.2 / 41.1 / 48.2 / 4.1 s | 24–28 GB GPU |
| x86 CPU (Xeon 8457C), bs1, 32 threads | 595 | 88 s | 97 GB RSS |
| x86 CPU, bs8, 32 threads | 595 | 257 s | 97 GB RSS |
| x86 CPU, incr, 32 threads, `--positions 16` | 113 | 50 s (about 3 s per step) | 97 GB RSS |
| x86 CPU, bs1, 14 threads, `--cpu-lazy`, 3 texts | 176 | 116 s | 30 GB RSS |
| x86 CPU, bs1, 14 threads, `--cpu-lazy`, 1 text × 4 positions (3 exist) | 3 | 27 s | 30 GB RSS |

All CPU runs used 141's CPU (the H800 host). All of these have 0 differing logits rows and tokens against `h800_bs1.json`. A bs1 forward costs about
the same for 3 or 128 positions of a text, so size CPU subsets by texts, not positions.

Result files: `results/h800_141/` (reference: `h800_bs1.json`), `results/cpu_141/`, summary `results/compare_141.json`.

## Merlin (L20 and other GPUs)

`merlin/make_entry.sh OUT.sh TAG` embeds this checkout's HEAD (`src`, `examples`, `experiments/xhw_8b`)
into `merlin/entry_8b.sh.in`. The job downloads the 4 shards (hf-mirror, then huggingface.co, then
through the proxy), checks their SHA-256, builds the constants in the job, checks `2f02ac8e…`, runs
bs1 / bs8 / incr / incr+compile and prints each JSON as gzip+base64. `merlin/fork_8b.py NAME OUT.sh`
forks the L20 job `ef1b5b5295886b0e` (L20 hl spot by default) with that script; `merlin/extract.py LOG DIR`
writes the JSONs back out.
