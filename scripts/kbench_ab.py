"""Compare arms of a phobos-kbench A/B: ARM-CONFIG-rN.json files in one dir.

    python scripts/kbench_ab.py DIR [BASE_ARM]

For each config and problem, every arm's Phobos kernel-only medians across
its runs, the median of those, and the ratio to the base arm. cuBLAS ran
interleaved in every process, so the Phobos/cuBLAS ratio is also given; it
cancels clock drift between processes.
"""

import collections
import glob
import json
import os
import statistics
import sys


def main(d, base):
    runs = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(d, "*.json"))):
        name = os.path.basename(path)[:-5]
        if name.startswith("cache") or "-r" not in name:
            continue
        stem = name.rsplit("-r", 1)[0]
        arm, cfg = stem.split("-M", 1)
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        for r in doc["results"]:
            if "error" in r:
                continue
            s = r["shape"]
            t = r["timing_kernel_only"]
            runs[("M" + cfg, f'{s["m"]}x{s["n"]}x{s["k"]}', arm)].append(
                (t["phobos"]["median_ms"], t["cublas"]["median_ms"] / t["phobos"]["median_ms"],
                 r["winner"]["resources"], r["correctness"]["pass"]))
    keys = sorted({k[:2] for k in runs})
    arms = sorted({k[2] for k in runs}, key=lambda a: (a != base, a))
    print("| config | M x N x K | arm | phobos median ms per run | median | vs base | phobos/cuBLAS | regs / blocks per SM | checks |")
    print("|---|---|---|---|---|---|---|---|---|")
    for cfg, shape in keys:
        if (cfg, shape, base) not in runs:
            continue
        ref = statistics.median(x[0] for x in runs[(cfg, shape, base)])
        for arm in arms:
            xs = runs.get((cfg, shape, arm))
            if not xs:
                continue
            med = statistics.median(x[0] for x in xs)
            ratio = statistics.median(x[1] for x in xs)
            res = xs[0][2]
            ok = all(x[3] for x in xs)
            per_run = ", ".join(f"{x[0]:.4f}" for x in xs)
            print(f"| {cfg} | {shape} | {arm} | {per_run} | {med:.4f} | {ref / med:.3f}x | {100 * ratio:.1f}% "
                  f"| {res['registers']} / {res['blocks_per_sm']} | {'pass' if ok else 'FAIL'} |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "base")
