"""All portability comparisons: python analyze.py OUT_DIR RESULTS_DIR. Writes RESULTS_DIR/cmp_<name>.json per pair and
RESULTS_DIR/summary.json; prints one line per pair. Missing files are reported as missing (not tested / failed)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import compare  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS16 = set(int(s) for s in open(os.path.join(HERE, "seeds16.txt")).read().split())
VERS = ["0.25.0", "0.28.0", "0.31.0", "0.33.0"]

PAIRS = [  # (name, test, arm, ref, x, kind, subset)
    ("D_sampler_to_scorer", "baseline", "D", "D_t1", "Ds_t1", "score", None),
    ("C_sampler_to_scorer", "baseline", "C", "C_t1", "Cs_t1", "score", None),
]
for v in VERS:
    PAIRS += [(f"D_ver{v}_t1", "version", "D", "D_t1", f"D_t1_v{v}", "sample", None),
              (f"D_ver{v}_t0", "version", "D", "D_t0", f"D_t0_v{v}", "sample", None),
              (f"D_ver{v}_score", "version", "D", "Ds_t1", f"Ds_t1_v{v}", "score", None),
              (f"C_ver{v}_t1", "version", "C", "C_t1", f"C_t1_v{v}", "sample", None),
              (f"C_ver{v}_t0", "version", "C", "C_t0", f"C_t0_v{v}", "sample", None),
              (f"C_ver{v}_score", "version", "C", "Cs_t1", f"Cs_t1_v{v}", "score", None)]
for v in ["0.25.0", "0.28.0"]:
    PAIRS += [(f"D_ver{v}_own_ms64_vs_ms256", "version_control", "D", f"D_t1_v{v}", f"D_t1_v{v}_ms64", "sample", None),
              (f"D_ver{v}_own_sampler_to_scorer", "version_control", "D", f"D_t1_v{v}", f"Ds_own_v{v}", "score", None)]
for tp in (2, 4):
    PAIRS += [(f"D_tp{tp}_t1", "tp", "D", "D_t1", f"D_t1_tp{tp}", "sample", None),
              (f"D_tp{tp}_t0", "tp", "D", "D_t0", f"D_t0_tp{tp}", "sample", None),
              (f"D_tp{tp}_score", "tp", "D", "Ds_t1", f"Ds_t1_tp{tp}", "score", None),
              (f"D_tp{tp}_ms64_vs_ms256", "batch", "D", f"D_t1_tp{tp}", f"D_t1_tp{tp}_ms64", "sample", None),
              (f"D_tp{tp}_ms1_vs_ms256", "batch", "D", f"D_t1_tp{tp}", f"D_t1_tp{tp}_ms1", "sample", None)]
PAIRS += [
    ("D_tp1_ms64_vs_ms256", "batch", "D", "D_t1", "D_t1_ms64", "sample", None),
    ("D_tp1_ms1_vs_ms256", "batch", "D", "D_t1", "D_t1_ms1", "sample", None),
    ("D_score_ms8_vs_ms128", "batch", "D", "Ds_t1", "Ds_t1_ms8", "score", None),
    ("C_tp2_t1", "tp", "C", "C_t1", "C_t1_tp2_inst", "sample", None),  # vLLM 0.31 venv with bitexact installed (entry points)
    ("C_ms64_vs_ms256", "batch", "C", "C_t1", "C_t1_ms64", "sample", None),
    ("C_ms1_vs_ms256", "batch", "C", "C_t1", "C_t1_ms1", "sample", None),
    ("C_score_bs1_vs_bs8", "batch", "C", "Cs_t1", "Cs_t1_bs1", "score", None),
    ("C_score_bs64_vs_bs8", "batch", "C", "Cs_t1", "Cs_t1_bs64", "score", None),
    ("D_xgpu_sampler_H800_vs_H100", "xgpu", "D", "h800_q1_D", "D_t1", "sample", None),
    ("C_xgpu_sampler_H800_vs_H100", "xgpu", "C", "h800_q1_B", "C_t1", "sample", None),
    ("C_xgpu_H800plugin_to_H100engine", "xgpu", "C", "h800_q1_B", "Cs_h800B", "score", None),
    ("C_cpu_engine_vs_gpu_engine", "cpu", "C", "Cs_t1", "Cs_t1_cpu", "score", SEEDS16),
    ("C_cpu_engine_vs_gpu_plugin", "cpu", "C", "C_t1", "Cs_t1_cpu", "score", SEEDS16),
    ("D_cpu_hf_bf16_vs_gpu", "cpu", "D", "Ds_t1", "Ds_t1_cpu_hf_bfloat16", "score", SEEDS16),
    ("D_cpu_hf_fp32_vs_gpu", "cpu", "D", "Ds_t1", "Ds_t1_cpu_hf_float32", "score", SEEDS16),
    ("D_cpu_vllm_vs_gpu", "cpu", "D", "Ds_t1", "Ds_t1_cpu_vllm", "score", SEEDS16),
]


def main():
    out_dir, res = sys.argv[1], sys.argv[2]
    os.makedirs(res, exist_ok=True)
    summ = []
    for name, test, arm, ref, x, kind, sub in PAIRS:
        fr, fx = f"{out_dir}/{ref}.json", f"{out_dir}/{x}.json"
        row = {"name": name, "test": test, "arm": arm, "ref": ref, "x": x, "kind": kind}
        if not (os.path.exists(fr) and os.path.exists(fx)):
            row["status"] = "failed" if os.path.exists(fx + ".failed") or os.path.exists(fr + ".failed") else "missing"
            log = fx + ".log"
            if os.path.exists(log):
                errs = [ln.strip() for ln in open(log, errors="replace") if "Error" in ln or "error:" in ln]
                row["error"] = errs[-1][:400] if errs else None
        else:
            r = compare(fr, fx, sub)
            if kind == "score":   # same input tokens; token comparison not meaningful for score files
                r.pop("median_first_div", None)
            row.update(r, status="ok", bitwise=r["lp_differ"] == 0 and r["sha_differ"] == 0
                       and (kind == "score" or r["seqs_identical"] == r["seqs"]))
            mx = json.load(open(fx)).get("meta", {})
            row["x_meta"] = {k: mx.get(k) for k in ("vllm", "torch", "cuda", "gpu", "device_name", "tp", "max_seqs", "bs",
                                                    "temp", "dtype", "note") if k in mx}
            json.dump(row, open(f"{res}/cmp_{name}.json", "w"), indent=1)
        summ.append(row)
        if row["status"] == "ok":
            print(f"{name:40s} bitwise={row['bitwise']!s:5s} seqs {row['seqs_identical']}/{row['seqs']} "
                  f"tok {row['tokens_compared']} lp_diff {100 * row['lp_differ_frac']:.2f}% mean|d| {row['mean_abs_d']:.3g} "
                  f"max|d| {row['max_abs_d']:.3g} sha_diff {row['sha_differ']}/{row['sha_compared']}")
        else:
            print(f"{name:40s} {row['status']} {row.get('error') or ''}"[:300])
    json.dump(summ, open(f"{res}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
