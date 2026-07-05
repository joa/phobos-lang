#!/usr/bin/env python3
"""Plot phobos-bench results as a higher-is-better bar chart.

Reads the CSV written by `phobos-bench --csv` and draws achieved GFLOP/s per
benchmark (grouped by implementation) against the GPU's theoretical peak, which
is shown as a translucent backdrop bar with the percent-of-peak annotated on
each bar. Rows without a peak (if any) are skipped.

Usage:
    python phobos-bench/plot.py phobos-bench.csv
    python phobos-bench/plot.py phobos-bench.csv -o bench.png --title "RTX 4090"

Requires matplotlib:  pip install matplotlib
"""

import argparse
import csv
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt


def read_rows(path):
    """Parsed rows that carry a numeric peak (defensive: every row should)."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not r.get("peak_gflops"):
                continue
            rows.append(
                {
                    "benchmark": r["benchmark"],
                    "impl": r["impl"],
                    "gflops": float(r["gflops"]),
                    "peak": float(r["peak_gflops"]),
                    "pct": float(r["pct_of_peak"]),
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="results CSV from `phobos-bench --csv`")
    ap.add_argument("-o", "--out", help="output image (default: show interactively)")
    ap.add_argument("--title", default="phobos-bench: achieved vs theoretical peak")
    args = ap.parse_args()

    rows = read_rows(args.csv)
    if not rows:
        sys.exit(f"no FLOP-bound rows in {args.csv} (nothing with a peak to plot)")

    # Stable benchmark order (CSV order) and the set of implementations.
    benches = list(OrderedDict.fromkeys(r["benchmark"] for r in rows))
    impls = list(OrderedDict.fromkeys(r["impl"] for r in rows))
    by_key = {(r["benchmark"], r["impl"]): r for r in rows}

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(benches)), 5))
    group_w = 0.8
    bar_w = group_w / len(impls)
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # Fixed colors per implementation; cuBLAS gets the NVIDIA green.
    impl_colors = {"cuBLAS": "#76b900"}

    def color_for(impl, idx):
        return impl_colors.get(impl, palette[idx % len(palette)])

    for gi, bench in enumerate(benches):
        # Theoretical peak for this benchmark (same precision across its rows).
        peak = max(r["peak"] for r in rows if r["benchmark"] == bench)
        ax.bar(gi, peak, width=group_w, color="0.85", zorder=0,
               label="theoretical peak" if gi == 0 else None)

        for ii, impl in enumerate(impls):
            r = by_key.get((bench, impl))
            if r is None:
                continue
            x = gi - group_w / 2 + bar_w * (ii + 0.5)
            ax.bar(x, r["gflops"], width=bar_w * 0.92, color=color_for(impl, ii),
                   zorder=2, label=impl if gi == 0 else None)
            ax.text(x, r["gflops"], f"{r['pct']:.0f}%", ha="center", va="bottom",
                    fontsize=8, zorder=3)

    ax.set_xticks(range(len(benches)))
    ax.set_xticklabels(benches)
    ax.set_ylabel("GFLOP/s")
    ax.set_title(args.title)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.margins(y=0.12)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
