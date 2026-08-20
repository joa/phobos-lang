#!/usr/bin/env python3
"""Plot what phobos-bench measured: GFLOP/s against the card's roofline peak.

Reads the CSV that `cargo run -r -p phobos-bench -- --csv PATH` writes
(columns: benchmark, impl, precision, gflops, peak_gflops, pct_of_peak) and
draws one horizontal bar per (benchmark, impl), grouped by benchmark, labelled
with achieved GFLOP/s and percent of peak. Kernels only phobos has a shim for
(no cuBLAS entry, e.g. flash attention) draw a single bar rather than a pair.

Unlike scripts/plot.py this has nothing to average: phobos-bench times its own
best autotuned configuration once per kernel, not several interleaved rounds,
so there is no round, no repetition and no error bar here.

Usage:
    cargo run -r -p phobos-bench -- --csv results/results.csv
    python scripts/plot_bench.py results/results.csv -o results/bench.svg
    python scripts/plot_bench.py results/results.csv --dark -o results/bench-dark.svg

The format follows each -o extension, so SVG, PDF and PNG all come off the
same call. Requires matplotlib:  pip install matplotlib
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# Fixed, so the element ids in an SVG are stable across runs.
matplotlib.rcParams["svg.hashsalt"] = "phobos-bench"

# Same palette as scripts/plot.py, so the two charts read as one family.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834"],
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926"],
    },
}

# Fixed per implementation regardless of theme, carried over from the plotter
# this replaces (phobos-bench/plot.py, deleted in 523d4df): cuBLAS gets
# NVIDIA's own brand green rather than a palette color, since it is a vendor
# library and not one of phobos's own series.
IMPL_COLORS = {"cuBLAS": "#76b900"}


def load(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        sys.exit(f"no rows in {path}")
    wanted = {"benchmark", "impl", "precision", "gflops", "peak_gflops", "pct_of_peak"}
    missing = wanted - set(rows[0])
    if missing:
        sys.exit(
            f"{path} is not what phobos-bench --csv writes:"
            f" missing {', '.join(sorted(missing))}"
        )
    out = []
    for r in rows:
        out.append(
            {
                "benchmark": r["benchmark"],
                "impl": r["impl"],
                "precision": r["precision"],
                "gflops": float(r["gflops"]),
                "peak_gflops": float(r["peak_gflops"]),
                "pct_of_peak": float(r["pct_of_peak"]),
            }
        )
    return out


def fmt_gflops(v):
    if v >= 1000:
        return f"{v / 1000:.1f} TFLOP/s"
    return f"{v:.0f} GFLOP/s"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("results", help="the CSV phobos-bench --csv wrote")
    ap.add_argument(
        "-o",
        "--out",
        action="append",
        default=[],
        metavar="FILE",
        help="where to write it, the format taken from the extension;"
        " repeatable, shown interactively if absent",
    )
    ap.add_argument(
        "--title", default="phobos vs cuBLAS, achieved GFLOP/s (higher is better)"
    )
    ap.add_argument("--dark", action="store_true", help="draw for a dark surface instead")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    rows = load(args.results)
    theme = THEMES["dark" if args.dark else "light"]
    impls = sorted({r["impl"] for r in rows}, key=lambda i: i != "phobos")
    # IMPL_COLORS first (cuBLAS's fixed green), theme series for the rest, in
    # order, so phobos gets the first series color whichever theme is active.
    series = iter(theme["series"])
    colors = {i: IMPL_COLORS.get(i) or next(series) for i in impls}

    benchmarks = list(dict.fromkeys(r["benchmark"] for r in rows))
    by_bench = {b: [r for r in rows if r["benchmark"] == b] for b in benchmarks}

    bar_h = 0.32
    fig_h = 0.85 * sum(len(v) for v in by_bench.values()) + 0.45 * len(benchmarks) + 1.4
    fig, ax = plt.subplots(figsize=(8.4, fig_h))
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])

    y = 0.0
    ticks, tick_labels = [], []
    scale_max = max(r["gflops"] for r in rows) * 1.28
    for bench in benchmarks:
        group = by_bench[bench]
        group_top = y
        for row in sorted(group, key=lambda r: impls.index(r["impl"])):
            ax.barh(
                y,
                row["gflops"],
                height=bar_h,
                color=colors[row["impl"]],
                zorder=2,
                label=row["impl"] if bench == benchmarks[0] else None,
            )
            note = f"{fmt_gflops(row['gflops'])}  ({row['pct_of_peak']:.0f}% of {fmt_gflops(row['peak_gflops'])})"
            ax.text(
                row["gflops"] + scale_max * 0.015,
                y,
                note,
                va="center",
                ha="left",
                fontsize=8,
                color=theme["secondary"],
            )
            y -= bar_h * 1.15
        ticks.append((group_top + y + bar_h * 1.15) / 2)
        precision = group[0]["precision"]
        tick_labels.append(f"{bench}\n{precision}")
        y -= bar_h * 0.9

    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=9, color=theme["primary"])
    ax.set_ylim(y + bar_h * 0.9, bar_h * 1.5)
    ax.set_xlim(0, scale_max)
    ax.set_xlabel("GFLOP/s", fontsize=8, color=theme["muted"])
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=theme["grid"], linewidth=0.8)
    ax.yaxis.grid(False)
    ax.tick_params(colors=theme["muted"], labelsize=8, length=0)
    for name, spine in ax.spines.items():
        spine.set_visible(name == "left")
        spine.set_color(theme["axis"])
        spine.set_linewidth(0.8)

    ax.set_title(args.title, fontsize=13, color=theme["primary"], pad=14, loc="left")

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles,
        labels,
        loc="lower right",
        frameon=False,
        fontsize=9,
    )
    for text in legend.get_texts():
        text.set_color(theme["secondary"])

    fig.tight_layout()

    if not args.out:
        plt.show()
        return
    for path in args.out:
        fig.savefig(
            path,
            dpi=args.dpi,
            facecolor=theme["surface"],
            metadata={"Date": None} if Path(path).suffix in (".svg", ".pdf") else None,
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
