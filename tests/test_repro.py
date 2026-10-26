"""A hash match is meaningful only after identity, sampling and complete coverage agree."""

import copy
import json

import pytest

from bitexact.repro import CHECKS, SCHEMA, compare, compare_reports, read_report, validate


def report():
    return {
        "schema": SCHEMA,
        "contract": {
            "constants_sha256": "a" * 64,
            "mode": "exact",
            "n_layers": 16,
            "prompts": [[1, 2]],
            "max_new": 2,
            "temperature": [7, 10],
            "seed_base": 0,
            "sampler_seed_base": 20261022,
        },
        "env": {"device": "cpu"},
        "checks": dict.fromkeys(CHECKS, True),
        "rows": {"0:0": ["b" * 64, 2], "0:1": ["c" * 64, 3]},
    }


def test_matching_reports_may_use_different_environments():
    a, b = report(), report()
    b["env"] = {"device": "cuda:0", "torch": "other"}
    assert compare_reports([a, b]) == [{"positions": 2, "logits_differ": 0, "tokens_differ": 0}]


@pytest.mark.parametrize(
    "key,value",
    [
        ("constants_sha256", "d" * 64),
        ("mode", "int8"),
        ("n_layers", 2),
        ("prompts", [[3, 4]]),
        ("temperature", [0, 1]),
        ("seed_base", 1),
        ("sampler_seed_base", 3),
    ],
)
def test_incompatible_experiments_fail_before_comparison(key, value):
    a, b = report(), report()
    b["contract"][key] = value
    with pytest.raises(ValueError, match="contracts differ"):
        compare_reports([a, b])


@pytest.mark.parametrize("mutation", ["missing", "extra", "empty", "bad_hash", "bad_token", "failed_check"])
def test_bad_coverage_and_failed_checks_never_pass(mutation):
    b = report()
    if mutation == "missing":
        del b["rows"]["0:1"]
    if mutation == "extra":
        b["rows"]["0:2"] = ["c" * 64, 3]
    if mutation == "empty":
        b["rows"] = {}
    if mutation == "bad_hash":
        b["rows"]["0:0"][0] = "not a hash"
    if mutation == "bad_token":
        b["rows"]["0:0"][1] = True
    if mutation == "failed_check":
        b["checks"]["integer_bounds"] = False
    with pytest.raises(ValueError):
        compare_reports([report(), b])


def test_mismatch_is_reported_not_relabelled_invalid():
    b = copy.deepcopy(report())
    b["rows"]["0:1"] = ["d" * 64, 9]
    assert compare_reports([report(), b]) == [{"positions": 2, "logits_differ": 1, "tokens_differ": 1}]


def test_single_report_and_duplicate_json_keys_rejected(tmp_path):
    with pytest.raises(ValueError):
        compare_reports([report()])
    path = tmp_path / "bad.json"
    path.write_text('{"schema":"x","schema":"y"}')
    with pytest.raises(ValueError, match="duplicate"):
        read_report(path)
    with pytest.raises(ValueError, match="distinct"):
        compare([path, path])


def test_legacy_report_rejected():
    with pytest.raises(ValueError, match="re-record"):
        validate({"env": {}, "rows": {}})


def test_compare_exit_status(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(report()))
    b.write_text(json.dumps(report()))
    assert compare([a, b]) == 0
    r = report()
    r["rows"]["0:0"][1] = 4
    b.write_text(json.dumps(r))
    assert compare([a, b]) == 1


def test_integer_check_flags_are_not_booleans():
    b = report()
    b["checks"]["integer_bounds"] = 1
    with pytest.raises(ValueError, match="checks"):
        validate(b)


def test_self_check_uses_production_arithmetic_without_model_weights():
    from bitexact.repro import self_check

    r = self_check()
    assert r["schema"] == "bitexact.arithmetic.v1"
    assert r["checks_passed"] == 8
    assert r["vectors"]["bf16_linear_with_sparse_residual"] != [[0, 0], [0, 0]]
    # A separately checked fixture transcript, not a model-inference reference.
    assert r["vectors_sha256"] == "5053b757aad61165cb7d22d271f6b87acbc0edbd29ec06f352f1f4e022b1b214"
    assert "hostname" not in r["env"]
    with pytest.raises(ValueError, match="self-check"):
        validate(r)


def test_self_check_cli_refuses_overwrite(tmp_path):
    from bitexact.repro import main

    out = tmp_path / "run.json"
    assert main(["self-check", "--out", str(out)]) == 0
    original = out.read_bytes()
    assert main(["self-check", "--out", str(out)]) == 2
    assert out.read_bytes() == original


def test_batch_verifier_checks_the_full_recomputed_sequence():
    import torch

    from bitexact import logits_sha
    from bitexact.repro import _batch_verify

    class Engine:
        dev = torch.device("cpu")

        def new_cache(self, batch, length):
            return None

        def forward(self, ids, pos, cache):
            return (ids.cumsum(1) % 7).unsqueeze(-1)

        def logits(self, hidden):
            return -((torch.arange(7) - hidden) ** 2)

        def check(self):
            pass

    m = Engine()
    rows = {
        f"{i}:{t}": [logits_sha(m.logits(torch.tensor([[token]]))[0]), token]
        for i in range(2)
        for t, token in enumerate([3, 6])
    }
    _batch_verify(m, [[1, 2], [3]], [[3, 6], [3, 6]], rows, 2, tn=0, td=1)
    rows["1:1"][0] = "0" * 64
    with pytest.raises(ValueError, match="prompt 1, position 1"):
        _batch_verify(m, [[1, 2], [3]], [[3, 6], [3, 6]], rows, 2, tn=0, td=1)


@pytest.fixture
def recorder_fixture(monkeypatch):
    """Exercise CLI orchestration with an explicit test engine, never as model evidence."""
    from types import SimpleNamespace

    import torch

    import bitexact
    import bitexact.exact.consts
    from bitexact.llama import V, ln_table

    class Engine:
        dev = torch.device("cpu")
        FR = 0
        ln = ln_table()
        lm = SimpleNamespace(gemm="cpu")
        fail = False

        def new_cache(self, batch, length):
            return torch.zeros(batch, dtype=torch.long)

        def forward(self, ids, pos, cache):
            sums = ids.cumsum(1) + cache[:, None]
            cache.copy_(sums[:, -1])
            return (sums % 7).unsqueeze(-1)

        def logits(self, hidden):
            return -((torch.arange(V) - hidden) ** 2)

        def check(self):
            if self.fail:
                raise OverflowError("injected engine-bound failure")

    engine = Engine()
    constants = {"lm.limbs": None, "sha256": "a" * 64, "n_layers": 16}
    monkeypatch.setattr(torch, "load", lambda *a, **kw: constants)
    monkeypatch.setattr(bitexact.exact.consts, "sha256", lambda c: "a" * 64)
    monkeypatch.setattr(bitexact, "load", lambda *a, **kw: engine)
    return engine, constants


def test_record_cli_writes_only_after_batched_and_bound_checks(tmp_path, recorder_fixture):
    from bitexact.repro import main

    out = tmp_path / "model.json"
    assert main(["record", "--const", "fixture.pt", "--out", str(out), "--max-new", "2"]) == 0
    r = read_report(out)
    assert len(r["rows"]) == 8
    assert r["env"]["gemm"] == "cpu"
    assert r["contract"]["sampler_seed_base"] == 20261022
    # The --reference path compares the same complete contract in one command.
    second = tmp_path / "second.json"
    assert (
        main(["record", "--const", "fixture.pt", "--out", str(second), "--max-new", "2", "--reference", str(out)]) == 0
    )


def test_record_digest_and_bound_failures_produce_no_report(tmp_path, recorder_fixture):
    from bitexact.repro import main

    engine, constants = recorder_fixture
    out = tmp_path / "bad.json"
    constants["sha256"] = "b" * 64
    assert main(["record", "--const", "fixture.pt", "--out", str(out), "--max-new", "1"]) == 2
    assert not out.exists()
    constants["sha256"] = "a" * 64
    engine.fail = True
    assert main(["record", "--const", "fixture.pt", "--out", str(out), "--max-new", "1"]) == 2
    assert not out.exists()
