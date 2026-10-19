"""The Python autotuner against `phobos-kbench`'s, in one session on one
card: `gemm_fp32` at 4096^3 and `flash_attention_fp32` at 4096 queries and
keys, D 64. Each tunes the same kernel source over the same space; the
winners are compared within the spread of the Python run, and
`gemm_fp32`'s fraction of cuBLAS is compared with kbench's.

    python python/tests/gpu/autotune_vs_kbench.py [--out FILE.json] [--flush-l2] [--keep DIR]

kbench runs first, with `--retune` so it tunes rather than reads its
winner cache. The Python tuner then runs without its L2 flush, as kbench
times, unless `--flush-l2`. The fraction of cuBLAS is cuBLAS's time over
the winner's, both by median and by minimum, cuBLAS timed by torch with
TF32 off, between the same events the tuner uses. Needs a GPU, torch,
numpy and cargo; checks the card for other processes first.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python" / "examples"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from phobos import autotune, driver  # noqa: E402

SIZE = 4096


def contention():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    # A desktop's graphics clients are listed with no memory of their own.
    return "\n".join(line for line in out.splitlines() if line.strip() and "[N/A]" not in line)


def kbench(bench, keep):
    """kbench's log and JSON for `bench`, kept in `keep`: a run already kept
    there is read back rather than run again."""
    log_path, json_path = keep / f"kbench_{bench}.log", keep / f"kbench_{bench}.json"
    if not log_path.exists():
        args = ["cargo", "run", "-q", "--release", "-p", "phobos-kbench", "--", "--bench", bench]
        if bench == "gemm_fp32":
            args += ["--retune", "--json", str(json_path)]
        proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"kbench {bench} failed:\n{(proc.stdout + proc.stderr)[-4000:]}")
        log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    data = json.loads(json_path.read_text()) if json_path.exists() else None
    return log_path.read_text(encoding="utf-8"), data


def settings_of(text):
    return {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", text)}


def kbench_gemm_winner(data):
    row = next(r for r in data["results"] if r.get("benchmark") == "gemm_fp32")
    return {k: int(v) for k, v in row["winner"]["config"].items()}, row["phobos_over_cublas_median"]


def kbench_flash_winner(log):
    m = re.search(r"autotune winner: (.*)", log)
    return settings_of(m.group(1)) if m else None


def timed(fn, flush, rounds=20):
    start, end = driver.Event(), driver.Event()
    samples = []
    for i in range(rounds + 5):
        if flush is not None:
            driver.memset(flush.ptr, 0, flush.nbytes)
        start.record()
        fn()
        end.record()
        end.synchronize()
        if i >= 5:
            samples.append(end.elapsed_ms(start))
    samples.sort()
    return samples[0], samples[len(samples) // 2]


def compare_winners(name, result, other):
    by = {tuple(sorted(v["settings"].items())): v for v in result.variants}
    ours = by[tuple(sorted(result.winner.items()))]
    theirs = by.get(tuple(sorted((other or {}).items())))
    line = {"kernel": name, "python_winner": result.winner, "kbench_winner": other,
            "python_winner_min_ms": ours["min_ms"], "python_winner_spread_ms": ours["spread_ms"]}
    if theirs is None:
        line["verdict"] = "kbench's winner is not among the Python run's verified variants"
    else:
        gap = theirs["min_ms"] - ours["min_ms"]
        line.update({"kbench_winner_min_ms_in_python": theirs["min_ms"], "kbench_winner_spread_ms": theirs["spread_ms"], "gap_ms": gap})
        if result.winner == other:
            line["verdict"] = "same winner"
        elif gap <= max(ours["spread_ms"], theirs["spread_ms"]):
            line["verdict"] = "different winners, within the spread"
        else:
            line["verdict"] = "different winners, outside the spread"
    return line


def gemm_tolerance(k):
    # kbench's bound (gemm/verify.rs) for f32 FMA accumulation, over the
    # condition sum_k |a| |b|: (2K) min (16 sqrt K) u, plus 2u of output.
    u = 2.0 ** -24
    return min(2 * k, 16 * k ** 0.5) * u + 2 * u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--flush-l2", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", choices=("gemm", "flash"))
    ap.add_argument("--keep", default=str(Path(tempfile.gettempdir()) / "phobos-kbench-vs-python"),
                    help="where kbench's runs are kept, and read back on a rerun")
    args = ap.parse_args()
    busy = contention()
    if busy and not args.force:
        sys.exit(f"other processes on the card, which would skew every figure:\n{busy}\n(--force to run anyway)")
    torch.backends.cuda.matmul.allow_tf32 = False
    card = driver.device_name()
    report = {"card": card, "chip_capability": driver.compute_capability(), "l2_flush": args.flush_l2, "rows": []}
    print(f"card: {card}")

    keep = Path(args.keep)
    keep.mkdir(parents=True, exist_ok=True)
    _, data = kbench("gemm_fp32", keep)
    kb_gemm, kb_ratio = kbench_gemm_winner(data)
    print(f"kbench gemm_fp32 winner {kb_gemm}, {100 * kb_ratio:.1f}% of cuBLAS (medians)")
    log_flash, _ = kbench("flash_fp32", keep)
    kb_flash = kbench_flash_winner(log_flash)
    print(f"kbench flash_fp32 winner {kb_flash}")

    if args.only == "flash":
        return flash(args, report, kb_flash)
    from gemm_fp32 import gemm

    gemm.key = ("M", "N", "K")
    torch.manual_seed(0)
    a = torch.rand(SIZE, SIZE, device="cuda") * 2 - 1
    b = torch.rand(SIZE, SIZE, device="cuda") * 2 - 1
    c = torch.zeros(SIZE, SIZE, device="cuda")
    cond = (np.abs(a.cpu().numpy()).astype(np.float64) @ np.abs(b.cpu().numpy()).astype(np.float64)).max()

    def gemm_reference(A, B, C, alpha, beta):
        return {"C": alpha * (A.astype(np.float64) @ B.astype(np.float64)) + beta * C}

    grid = lambda meta: (meta["M"] // meta["TILE_M"], meta["N"] // meta["TILE_N"])  # noqa: E731
    result = gemm.tune(grid, a, b, c, 1.0, 0.0, reference=gemm_reference, rtol=0.0, atol=gemm_tolerance(SIZE) * cond,
                       flush_l2=args.flush_l2)
    winner = next(v for v in result.variants if v["settings"] == result.winner)
    flush = autotune.DeviceArray([2 * driver.attribute(driver.DEVICE_L2_CACHE_SIZE)], "u8") if args.flush_l2 else None
    blas_min, blas_median = timed(lambda: torch.matmul(a, b, out=c), flush)
    row = compare_winners("gemm_fp32", result, kb_gemm)
    row.update({"cublas_min_ms": blas_min, "cublas_median_ms": blas_median,
                "python_fraction_of_cublas_median": blas_median / winner["median_ms"],
                "python_fraction_of_cublas_min": blas_min / winner["min_ms"], "kbench_fraction_of_cublas_median": kb_ratio,
                "verified": sum(v["verified"] for v in result.variants), "variants": len(result.variants)})
    report["rows"].append(row)
    if args.only != "gemm":
        flash(args, report, kb_flash)
    finish(args, report)


def finish(args, report):
    for row in report["rows"]:
        print(json.dumps(row, indent=2))
    if busy := contention():
        print(f"warning: other processes on the card at the end:\n{busy}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


def flash(args, report, kb_flash):
    from flash_attention_fp32 import flash_attention

    flash_attention.key = ("Nq", "Nk")
    q = torch.rand(SIZE, 64, device="cuda") * 2 - 1
    k = torch.rand(SIZE, 64, device="cuda") * 2 - 1
    v = torch.rand(SIZE, 64, device="cuda") * 2 - 1
    o = torch.zeros(SIZE, 64, device="cuda")
    scale = 1.0 / 8.0

    def flash_reference(Q, K, V, O, s):
        scores = (Q.astype(np.float64) @ K.astype(np.float64).T) * s
        scores -= scores.max(axis=1, keepdims=True)
        p = np.exp(scores)
        return {"O": (p @ V.astype(np.float64)) / p.sum(axis=1, keepdims=True)}

    # kbench's flash check (flash.rs): 2e-2 relative, with a floor of 0.1.
    # kbench compiles a `@tensorcore` kernel at 64-bit indices on every chip
    # (autotune::wide_index), which takes the mma.sync path; the runtime
    # widens only from sm_80. The comparison compiles as kbench does.
    result = flash_attention.tune(lambda meta: (meta["Nq"] // meta["BR"],), q, k, v, o, scale, reference=flash_reference,
                                  rtol=2e-2, atol=2e-3, flush_l2=args.flush_l2, index_bits=64)
    row = compare_winners("flash_attention_fp32", result, kb_flash)
    row.update({"verified": sum(v["verified"] for v in result.variants), "variants": len(result.variants), "index_bits": 64})
    report["rows"].append(row)
    if args.only == "flash":
        finish(args, report)


if __name__ == "__main__":
    main()
