"""Strict, portable reproduction reports. Run ``python -m bitexact.repro --help``."""

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

SCHEMA = "bitexact.identity.v1"
CHECKS = {"constants_digest", "integer_bounds", "incremental_vs_batched_full"}
HEX = re.compile(r"[0-9a-f]{64}\Z")
PROMPTS = [
    [128000, 791, 23070, 19044, 6267, 304, 813, 1515, 430, 279, 13766, 1047],
    [128000, 755, 11117, 91907, 1577, 43298, 997, 262, 4304, 52586, 50917, 28090, 14781, 198],
    [128000, 644, 220, 9674, 17, 279, 33715, 8813, 279, 18671, 13962, 11, 1405, 279, 6098, 15916],
    [128000, 29880, 369, 264, 4062, 42120, 19724, 25, 1176, 11, 8798, 1403, 56488, 315, 33213, 5707, 323],
]


def _positive(value):
    if type(value) is not int or value < 1:
        raise ValueError("expected a positive integer")
    return value


def validate(report):
    """Reject missing coverage or incompatible contracts before comparing any hashes."""
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        raise ValueError("expected a v1 model identity report; re-record legacy or self-check reports")
    c = report.get("contract", {})
    fields = {"constants_sha256", "mode", "n_layers", "prompts", "max_new", "temperature",
              "seed_base", "sampler_seed_base"}
    if not isinstance(c, dict) or set(c) != fields:
        raise ValueError("missing or unexpected experiment contract fields")
    if not isinstance(c["constants_sha256"], str) or not HEX.fullmatch(c["constants_sha256"]):
        raise ValueError("invalid constants digest")
    if c["mode"] not in ("exact", "int8") or not 1 <= _positive(c["n_layers"]) <= 16:
        raise ValueError("unsupported engine or layer count")
    n = _positive(c["max_new"])
    prompts = c["prompts"]
    if not isinstance(prompts, list) or not prompts or len(prompts) > 16:
        raise ValueError("expected 1 to 16 nonempty token-id prompts")
    limit = 4096 if c["mode"] == "exact" else 2048
    for prompt in prompts:
        if not isinstance(prompt, list) or not prompt or len(prompt) + n > limit:
            raise ValueError("empty prompt or context exceeds the standalone engine limit")
        if any(type(t) is not int or not 0 <= t < 128256 for t in prompt):
            raise ValueError("invalid Llama-3.2 token id")
    temp = c["temperature"]
    if not isinstance(temp, list) or len(temp) != 2 or any(type(t) is not int for t in temp):
        raise ValueError("temperature must be [integer numerator, positive denominator]")
    if temp[0] < 0 or temp[1] <= 0 or type(c["seed_base"]) is not int or c["seed_base"] < 0:
        raise ValueError("invalid sampling contract")
    if type(c["sampler_seed_base"]) is not int or c["sampler_seed_base"] < 0:
        raise ValueError("invalid integer sampler seed base")
    checks = report.get("checks")
    if not isinstance(checks, dict) or set(checks) != CHECKS or any(v is not True for v in checks.values()):
        raise ValueError("all required recording checks must pass")
    if not isinstance(report.get("env"), dict) or not report["env"]:
        raise ValueError("missing runtime provenance")
    rows = report.get("rows")
    expected = {f"{i}:{t}" for i in range(len(prompts)) for t in range(n)}
    if not isinstance(rows, dict) or set(rows) != expected:
        raise ValueError("recorded row set must match the entire experiment, with no missing or extra rows")
    for row in rows.values():
        if not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str) or not HEX.fullmatch(row[0]):
            raise ValueError("invalid logits hash row")
        if type(row[1]) is not int or not 0 <= row[1] < 128256:
            raise ValueError("invalid sampled token")
    return report


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_report(path):
    with open(path, encoding="utf-8") as f:
        return validate(json.load(f, object_pairs_hook=_unique_keys))


def compare_reports(reports):
    if len(reports) < 2:
        raise ValueError("provide at least two reports")
    for r in reports:
        validate(r)
    ref = reports[0]
    results = []
    for r in reports[1:]:
        if r["contract"] != ref["contract"]:
            raise ValueError("experiment contracts differ (constants, mode, prompts, sampling or coverage)")
        keys = ref["rows"]
        results.append({"positions": len(keys),
                        "logits_differ": sum(ref["rows"][k][0] != r["rows"][k][0] for k in keys),
                        "tokens_differ": sum(ref["rows"][k][1] != r["rows"][k][1] for k in keys)})
    return results


def compare(paths):
    if len({str(Path(p).resolve()) for p in paths}) != len(paths):
        raise ValueError("use distinct report files; comparing a file with itself is not a reproduction")
    results = compare_reports([read_report(p) for p in paths])
    for path, result in zip(paths[1:], results):
        print(json.dumps({"report": str(path), **result}))
    return int(any(r["logits_differ"] or r["tokens_differ"] for r in results))


def environment():
    import torch

    from bitexact import __version__

    env = {"python": platform.python_version(), "torch": str(torch.__version__),
           "system": platform.system(), "machine": platform.machine(), "package": __version__}
    root = Path(__file__).resolve().parents[2]
    if not (root / "src/bitexact/repro.py").is_file():
        env["git_commit"] = None
        return env
    try:
        env["git_commit"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=3).decode().strip()
        env["tracked_dirty"] = subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet", "HEAD"], timeout=3).returncode != 0
    except (OSError, subprocess.SubprocessError):
        env["git_commit"] = None
    return env


def write_report(path, report):
    # Never overwrite an existing reference or another participant's run.
    with open(path, "x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")


def self_check():
    """Small nonzero fixtures through production arithmetic, checked against Python integer/Fraction oracles."""
    import math
    from fractions import Fraction

    import torch

    from bitexact.exact.consts import FR, decompose
    from bitexact.exact.model import Lin
    from bitexact.exact.segs import lin_y
    from bitexact.intops import fdiv, isqrt, rsh

    def equal(name, got, expected):
        if got != expected:
            raise ValueError(f"arithmetic self-check failed: {name}")
        return got

    vectors = {}
    values = [-(1 << 50) + 3, -17, -1, 0, 1, 17, (1 << 50) - 3]
    for shift in [-3, 0, 1, 7, 31]:
        want = [(Fraction(v) / Fraction(2) ** shift + Fraction(1, 2)).__floor__() for v in values]
        vectors[f"round_shift_{shift}"] = equal("round_shift", rsh(torch.tensor(values), shift).tolist(), want)
    denom = [3, 7, 11, 13, 17, 19, 23]
    vectors["floor_division"] = equal("floor_division", fdiv(torch.tensor(values), torch.tensor(denom)).tolist(),
                                      [a // b for a, b in zip(values, denom)])
    squares = [0, 1, 2, 3, 4, 17, (1 << 60) - 1]
    vectors["integer_sqrt"] = equal("integer_sqrt", isqrt(torch.tensor(squares)).tolist(),
                                    [math.isqrt(x) for x in squares])
    w = torch.tensor([[1.0, -0.5, 2 ** -30, -2 ** -28, 0.125, 0.0, 0.75, -0.25],
                      [-0.25, 0.5, -0.125, 1.0, -2 ** -29, 2 ** -31, -0.75, 0.0]], dtype=torch.bfloat16)
    c = {f"fixture.{k}": v for k, v in decompose(w).items()}
    q = torch.tensor([[13, -7, 3, 11, 0, -5, 17, -19], [-2, 5, 7, -11, 3, 0, 1, 4]], dtype=torch.int32)
    shifts = [-10, -6]
    y, bad = lin_y(*Lin(c, "fixture", "cpu", "cpu", False)(q), torch.tensor(shifts), checked=True)
    if bool(bad):
        raise ValueError("arithmetic fixture exceeded its linear bound")
    expected = [[(sum(Fraction(float(weight)) * int(x) for weight, x in zip(row, qs))
                  * Fraction(2) ** (shift + FR) + Fraction(1, 2)).__floor__() for row in w]
                for qs, shift in zip(q, shifts)]
    vectors["bf16_linear_with_sparse_residual"] = equal("bf16_linear", y.tolist(), expected)
    encoded = json.dumps(vectors, sort_keys=True, separators=(",", ":")).encode()
    return {"schema": "bitexact.arithmetic.v1", "scope": "CPU arithmetic fixtures, not model inference",
            "checks_passed": len(vectors), "vectors_sha256": hashlib.sha256(encoded).hexdigest(),
            "vectors": vectors, "env": environment()}


def _batch_verify(model, prompts, tokens, rows, batch_size, tn=7, td=10):
    import torch

    from bitexact.generate import choose, logits_sha
    from bitexact.llama import PAD

    for start in range(0, len(prompts), batch_size):
        indices = list(range(start, min(start + batch_size, len(prompts))))
        sequences = [prompts[i] + tokens[i][:-1] for i in indices]
        length = max(map(len, sequences))
        ids = torch.full((len(indices), length), PAD, dtype=torch.long, device=model.dev)
        for r, seq in enumerate(sequences):
            ids[r, :len(seq)] = torch.tensor(seq, device=model.dev)
        pos = torch.arange(length, device=model.dev).expand(len(indices), length).contiguous()
        hidden = model.forward(ids, pos, model.new_cache(len(indices), length))
        for r, i in enumerate(indices):
            p = len(prompts[i]) - 1
            logits = model.logits(hidden[r, p:p + len(tokens[i])])
            sampled = choose(model, logits, [(i, t) for t in range(len(tokens[i]))], tn, td).tolist()
            for t, token in enumerate(sampled):
                if rows[f"{i}:{t}"] != [logits_sha(logits[t]), token]:
                    raise ValueError(f"incremental/batched-full identity failed at prompt {i}, position {t}")
        model.check()


def record(args):
    import torch

    from bitexact import generate, load
    from bitexact.exact.consts import sha256 as exact_digest
    from bitexact.int8.consts import sha256 as int8_digest
    from bitexact.sampling import SEED_BASE

    torch.set_num_threads(args.threads)
    # Load only a constants file you built yourself; torch pickle is not an untrusted-report format.
    c = torch.load(args.const, map_location="cpu", weights_only=False)
    mode = "exact" if "lm.limbs" in c else "int8"
    digest = (exact_digest if mode == "exact" else int8_digest)(c)
    if c.get("sha256") != digest:
        raise ValueError("constants digest does not match its stored identity")
    report = {"schema": SCHEMA, "contract": {"constants_sha256": digest, "mode": mode,
              "n_layers": c["n_layers"], "prompts": PROMPTS, "max_new": args.max_new,
              "temperature": [7, 10], "seed_base": 0, "sampler_seed_base": SEED_BASE},
              "checks": dict.fromkeys(CHECKS, True),
              "env": environment(), "rows": {}}
    # Validate the configuration before loading a large engine or running inference.
    contract_only = {**report, "rows": {f"{i}:{t}": ["0" * 64, 0]
                                      for i in range(len(PROMPTS)) for t in range(args.max_new)}}
    validate(contract_only)
    kw = {"compiled": args.compiled, **({"gemm": args.gemm} if mode == "exact" else {})}
    model = load(c, args.device, **kw)
    report["env"].update({"device": str(model.dev), "threads": args.threads, "compiled": args.compiled,
                           "batch_size": args.batch_size})
    if model.dev.type == "cuda":
        report["env"].update({"gpu": torch.cuda.get_device_name(model.dev), "cuda": torch.version.cuda})
    elif model.dev.type != "cpu":
        raise ValueError("only CPU and CUDA paths are supported")
    if mode == "exact":
        report["env"]["gemm"] = model.lm.gemm
    tokens = []
    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            seq, hashes = generate(model, prompt, args.max_new, 7, 10, seed=i)
            tokens.append(seq)
            report["rows"].update({f"{i}:{t}": [h, tok] for t, (h, tok) in enumerate(zip(hashes, seq))})
            print(f"recorded prompt {i + 1}/{len(PROMPTS)}", flush=True)
        _batch_verify(model, PROMPTS, tokens, report["rows"], args.batch_size)
        model.check()
    validate(report)
    status = 0
    if args.reference:
        results = compare_reports([read_report(args.reference), report])
        print(json.dumps(results[0]))
        status = int(any(r["logits_differ"] or r["tokens_differ"] for r in results))
    write_report(args.out, report)
    print(f"wrote {args.out}: {len(report['rows'])} positions; same-device schedule checks passed")
    if not args.reference:
        print("Cross-device identity is not established by this single report. Compare another device's report.")
    return status


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("self-check", help="CPU arithmetic checks; no GPU, weights or network")
    smoke.add_argument("--out", required=True)
    r = sub.add_parser("record", help="record a local constants file and verify incremental/batched identity")
    r.add_argument("--const", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--reference", help="optional v1 report from another device; failure exits nonzero")
    r.add_argument("--device", default="cpu")
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--max-new", type=int, default=16)
    r.add_argument("--batch-size", type=int, default=2)
    r.add_argument("--gemm", choices=["triton", "cublas"], default="triton")
    r.add_argument("--compiled", action="store_true")
    cmp = sub.add_parser("compare", help="compare complete reports with identical experiment contracts")
    cmp.add_argument("paths", nargs="+")
    args = ap.parse_args(argv)
    try:
        if args.command == "self-check":
            result = self_check()
            write_report(args.out, result)
            print(f"PASS: {result['checks_passed']} arithmetic fixtures; {result['vectors_sha256']}")
            print("This is not a Llama model or cross-device inference verification.")
            return 0
        if args.command == "compare":
            return compare(args.paths)
        if not 1 <= args.max_new <= 4096 or not 1 <= args.batch_size <= len(PROMPTS) or args.threads < 1:
            raise ValueError("max-new, batch-size or threads outside the supported range")
        if Path(args.out).exists():
            raise ValueError("output already exists; choose a new filename")
        return record(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, OverflowError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
