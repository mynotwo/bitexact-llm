"""The v1 model identity report: what a valid report holds, and how reports are read, written and compared.

Plain JSON rules only (no torch, no engine). ``bitexact.repro`` produces the reports and is the CLI.
"""

import json
import re

SCHEMA = "bitexact.identity.v1"
CHECKS = {"constants_digest", "integer_bounds", "incremental_vs_batched_full"}
HEX = re.compile(r"[0-9a-f]{64}\Z")


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


def write_report(path, report):
    # Never overwrite an existing reference or another participant's run.
    with open(path, "x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")


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
