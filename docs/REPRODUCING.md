# Reproduce a bounded claim

BitExact has two distinct entry points. A CPU arithmetic check needs no GPU,
model weights, Hugging Face account or network after installation. A model
identity check needs constants built from your own authorized Llama-3.2-1B
checkpoint. Passing the first does not establish the second.

## 1. Start without model weights

From a checkout, with Python 3.11 or 3.12:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m bitexact.repro self-check --out arithmetic.json
```

The last line is the one-command entry point once installed. It exercises the
production integer shifts, floor division, square root and exact bf16 linear
path, including nonzero sparse residuals. Expected values come from Python
integers, `math.isqrt` and `Fraction`, rather than from a second copy of the
production algorithm. Eight fixtures must pass. The output names its scope:

```text
PASS: 8 arithmetic fixtures; 5053b757aad61165cb7d22d271f6b87acbc0edbd29ec06f352f1f4e022b1b214
This is not a Llama model or cross-device inference verification.
```

The JSON includes the fixture transcript and its SHA-256, OS/architecture,
Python/torch versions and source commit when run from a checkout. It excludes
hostnames, account names and absolute machine paths. A filename must be new;
existing reports are never overwritten.

The [CPU workflow](../.github/workflows/cpu.yml) runs the test suite, builds a
wheel and uploads its own arithmetic JSON and JUnit results. Its green status
covers CPU checks only. CUDA and model cases skip without their prerequisites.

## 2. Record actual model identity

Obtain Llama-3.2-1B through your own authorized model access, retaining its model
license. The checkpoint directory must contain `config.json` and one
`model.safetensors`. We do not distribute Meta weights or derived constants.
Only load a `.pt` constants file you built yourself: the existing constants
format uses Python pickle. Public reproduction submissions should be JSON.

Build one mode's constants:

```bash
python examples/build_constants.py --model PATH/Llama-3.2-1B --mode exact --out exact_const.pt
```

Exact constants occupy about 4.8 GB; running the CPU engine requires additional
memory for weights, intermediates and caches. Installation success does not
establish that a machine has enough RAM. Int8 constants use `--mode int8`.

Record and check incremental generation against a right-padded full-forward
batch on the same device:

```bash
python -m bitexact.repro record --const exact_const.pt --device cpu --max-new 16 --batch-size 2 --out cpu.json
python -m bitexact.repro record --const exact_const.pt --device cuda:0 --max-new 16 --batch-size 2 --out gpu.json
python -m bitexact.repro compare cpu.json gpu.json
```

With another device's v1 reference JSON available, record and compare in one command:

```bash
python -m bitexact.repro record --const exact_const.pt --device cuda:0 --max-new 16 --out mine.json --reference cpu.json
```

This uses four fixed public token-id prompts and temperature 7/10. Each prompt
has its own seed; the integer sampler's global seed base is also recorded. Every
generated position records the SHA-256 of the complete int64 logits row and its
sampled token. The recorder recomputes the generated sequence in full-forward
batches before accepting a report and calls the engine's bound check.

Compare only reports with the same verified constants digest, exact/int8 mode,
layer count, token-id prompts, sampling settings and coverage. Threads, devices,
software versions, compiled/eager mode and batch size may differ and are recorded
as runtime provenance. Use `--threads`, `--batch-size`, `--compiled` and exact
mode's `--gemm triton|cublas` to identify the execution variant you checked.
A reduced-layer constants file is reported with that layer count; it is not a
full-model result. The standalone context limits remain 4,096 positions for
exact and 2,048 for int8. MPS and unlisted model architectures are unsupported.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Local checks passed; with reference/compare, all compared rows also match |
| 1 | Compatible reports disagree in at least one logits hash or sampled token |
| 2 | Invalid/incompatible report, failed local check, invalid configuration or I/O/runtime error |

The comparator rejects empty, missing and extra rows, malformed hashes, failed
checks, duplicate JSON keys, incompatible experiments and a single report. It
does not accept legacy unversioned reports: rerun the recorder. The original
`examples/check_bit_identity.py` delegates to the new CLI; its current default
is 16 generated positions per prompt, recorded explicitly in the contract.

## 3. Share an external reproduction

Open [a device reproduction report](https://github.com/mynotwo/bitexact-llm/issues/new?template=reproduction.yml).
Include the source revision, device/runtime, commands, exit codes, generated JSON
and reference source. A mismatch is useful evidence too. Review the JSON before
posting; never upload private checkpoints, constants, credentials or request data.

A submitted report is a participant's claim until its provenance is reviewed.
Two copied JSON files are not independent reproductions. Equal hashes compare
these recorded outputs; they do not authenticate the reporter, prove equality to
fp32, cover all prompts or establish a throughput advantage. We have not run a
new Llama/CUDA reproduction as part of this CPU entry-point release.

## Arithmetic guard scope

Exact o_proj/down_proj linear outputs now flag shift loss and values outside
`(-2^62, 2^62)` before residual addition; final logits use the same checked
epilogue. Nonzero values needing an unsupported positive shift above 30 are
rejected. Regression tests cover boundary values, cancellation, wraparound and
2,000 cases checked with Python big integers/Fraction.

These are additional failure checks, not a proof of all intermediate bounds.
They retain the existing GEMM, sparse accumulation and constants input contracts.
Historical whole-model identity and throughput tables in the README predate the
new guards. Re-establishing them on the changed source requires actual model and
hardware runs. The vLLM plugin is an int8 path with its own existing suite;
standalone reports do not certify that plugin or `vllm serve`.
