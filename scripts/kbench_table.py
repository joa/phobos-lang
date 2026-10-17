"""Summarize phobos-kbench --json GEMM results as a Markdown table.

    python scripts/kbench_table.py RESULTS.json...

One row per benchmark and problem: the numerical contract, kernel-only
median throughput of Phobos and of the cuBLAS call with the same contract,
the ratio of medians, the run-to-run spread, the winning configuration and
its resources, and whether every correctness check passed.
"""

import json
import sys


def spread(t):
    """p10..p90 as a percentage of the median."""
    return 100.0 * (t["p90_ms"] - t["p10_ms"]) / t["median_ms"]


def main(paths):
    print("| benchmark | M x N x K | contract | baseline | phobos TFLOP/s | cuBLAS TFLOP/s | phobos/cuBLAS | spread p10-p90 (phobos, cuBLAS) | winner | regs / smem / blocks per SM | checks |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for path in paths:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        for r in doc["results"]:
            s = r["shape"]
            shape = f'{s["m"]}x{s["n"]}x{s["k"]}'
            if "error" in r:
                print(f'| {r["benchmark"]} | {shape} | | | | | | | error: {r["error"]} | | |')
                continue
            c = r["contract"]
            contract = f'{c["a_b"]} in, {c["c"]} out, {c["mul"]} mul, {c["acc"]} acc'
            t = r["timing_kernel_only"]
            blas_key = "cublas_with_convert" if r["baseline_converts_ab"] else "cublas"
            phobos, blas = t["phobos"], t[blas_key]
            baseline = r["baseline"] + (" + f32->f16 convert" if r["baseline_converts_ab"] else "")
            w = r["winner"]
            cfg = " ".join(f"{k}={v}" for k, v in w["config"].items())
            res = w["resources"]
            ratio = blas["median_ms"] / phobos["median_ms"]
            print(
                f'| {r["benchmark"]} | {shape} | {contract} | {baseline} '
                f'| {phobos["tflops_median"]:.2f} | {blas["tflops_median"]:.2f} | {100 * ratio:.1f}% '
                f'| {spread(phobos):.1f}%, {spread(blas):.1f}% | {cfg} '
                f'| {res["registers"]} / {res["static_smem_bytes"]} B / {res["blocks_per_sm"]} '
                f'| {"pass" if r["correctness"]["pass"] else "FAIL"} |'
            )


if __name__ == "__main__":
    main(sys.argv[1:])
