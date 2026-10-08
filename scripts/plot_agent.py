#!/usr/bin/env python3
"""Plot what scripts/agent_bench.py measured: an agent's work, per task and engine.

Reads the results.json agent_bench.py writes and draws three panels side by
side, wall time, prompt rate and decode rate, with a group of bars per task of
bench/ and one bar per engine. Each bar is the mean over repetitions, and the
whisker is the standard error over them. Wall time is lower-is-better, the two
rates higher-is-better, and each panel has its own scale.

Rates are token-weighted per repetition, summed tokens over summed seconds, the
same figures the harness prints. The prompt rate counts only the tokens an
engine had to process, not the ones its kept session reused.

Usage:
    python scripts/agent_bench.py -r 3 --replay RUN/*.requests.jsonl --out OUT
    python scripts/plot_agent.py OUT/results.json -o results/agent.svg
    python scripts/plot_agent.py results/agent.json --dark -o agent-dark.svg

The format follows each -o extension. Requires matplotlib.
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from plot import THEMES, engine_key, fit_labels, fmt_rate, plural, widen_for_chrome

# (key, title, unit, higher is better)
PANELS = [
    ("wall", "wall time", "seconds", False),
    ("pp", "prompt", "tokens/s", True),
    ("tg", "decode", "tokens/s", True),
]


def task_name(child):
    """The task of bench/ a result belongs to. A replay names its tasks after
    the trace, as in phobos-rep1-fib."""
    return re.sub(r"^.*-rep\d+-", "", child)


def figures(task):
    """Wall seconds and the token-weighted prompt and decode rates of one task."""
    reqs = task["requests"]
    fresh = sum(r["prompt"] - r["reused"] for r in reqs)
    p_secs = sum(r["prompt_secs"] for r in reqs)
    gen = sum(r["generated"] for r in reqs)
    d_secs = sum(r["decode_secs"] for r in reqs)
    return {
        "wall": task["wall_secs"],
        "pp": fresh / p_secs if p_secs else 0.0,
        "tg": gen / d_secs if d_secs else 0.0,
    }


def load(path):
    try:
        run = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as err:
        sys.exit(f"cannot read {path}: {err.strerror}")
    tasks = run.get("tasks")
    if not tasks or "wall_secs" not in tasks[0]:
        sys.exit(f"{path} is not what scripts/agent_bench.py writes")
    return run.get("meta", {}), tasks


def summarize(tasks):
    """(engine, task, key) -> (mean, standard error, repetitions)."""
    per_rep = {}
    for t in tasks:
        for key, value in figures(t).items():
            per_rep.setdefault((t["engine"], task_name(t["child"]), key), []).append(value)
    out = {}
    for key, values in per_rep.items():
        err = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
        out[key] = (statistics.fmean(values), err, len(values))
    return out


def draw_panel(ax, panel, names, engines, stats, theme, colors, base):
    key, title, unit, higher = panel
    bar_h = min(0.6 / len(engines), 0.3)
    group_h = bar_h * len(engines)
    scale_max = max((v[0] for k, v in stats.items() if k[2] == key), default=1.0)
    pad = scale_max * 0.02
    labels = []
    for gi, name in enumerate(names):
        for ei, engine in enumerate(engines):
            got = stats.get((engine, name, key))
            if not got:
                continue
            mean, err, _ = got
            y = gi - group_h / 2 + bar_h * (ei + 0.5)
            ax.barh(
                y, mean, height=bar_h * 0.86, color=colors[engine], zorder=2,
                label=engine if gi == 0 else None,
            )
            if err > 0:
                ax.errorbar(
                    mean, y, xerr=err, fmt="none", ecolor=theme["secondary"],
                    elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=3,
                )
            note = f"{mean:.1f} s" if key == "wall" else fmt_rate(mean)
            if engine == base:
                rivals = [
                    stats[(e, name, key)][0]
                    for e in engines
                    if e != base and (e, name, key) in stats
                ]
                if len(rivals) == 1 and rivals[0] and mean:
                    ratio = mean / rivals[0] if higher else rivals[0] / mean
                    note += f"   {ratio:.2f}x"
            labels.append(
                ax.text(
                    mean + err + pad, y, note, va="center", ha="left", fontsize=8,
                    color=theme["secondary"],
                )
            )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9, color=theme["primary"])
    ax.set_ylim(-0.55, len(names) - 0.45)
    ax.invert_yaxis()
    ax.set_xlim(0, scale_max * 1.06)
    better = "higher" if higher else "lower"
    ax.set_xlabel(f"{unit}, {better} is better", fontsize=8, color=theme["muted"])
    ax.set_title(title, fontsize=10, color=theme["primary"], pad=8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=theme["grid"], linewidth=0.8)
    ax.yaxis.grid(False)
    ax.tick_params(colors=theme["muted"], labelsize=8, length=0)
    for side, spine in ax.spines.items():
        spine.set_visible(side == "left")
        spine.set_color(theme["axis"])
        spine.set_linewidth(0.8)
    return labels


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("results", help="the results.json scripts/agent_bench.py wrote")
    ap.add_argument(
        "-o", "--out", action="append", default=[], metavar="FILE",
        help="where to write it, the format taken from the extension;"
        " repeatable, shown interactively if absent",
    )
    ap.add_argument("--title", default="phobos vs llama.cpp, a coding agent's work replayed")
    ap.add_argument(
        "--context", default="",
        help="leads the subtitle, e.g. the model, the card and the engines' settings",
    )
    ap.add_argument("--dark", action="store_true", help="draw for a dark surface instead")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    _, tasks = load(args.results)
    stats = summarize(tasks)
    names = list(dict.fromkeys(task_name(t["child"]) for t in tasks))
    engines = sorted({t["engine"] for t in tasks}, key=engine_key)
    base = engines[0]

    theme = THEMES["dark" if args.dark else "light"]
    if len(engines) > len(theme["series"]):
        sys.exit(f"{len(engines)} engines but only {len(theme['series'])} validated colors")
    colors = dict(zip(engines, theme["series"]))

    panel_h = 0.26 * len(names) * len(engines) + 0.34 * len(names) + 0.95
    fig, axes = plt.subplots(
        1, len(PANELS), figsize=(1.2 + 4.3 * len(PANELS), panel_h + 1.55), squeeze=False
    )
    fig.patch.set_facecolor(theme["surface"])
    panels = []
    for column, panel in enumerate(PANELS):
        ax = axes[0][column]
        ax.set_facecolor(theme["surface"])
        labels = draw_panel(ax, panel, names, engines, stats, theme, colors, base)
        if column:
            ax.set_yticklabels([])
        panels.append((ax, labels))

    reps = sorted({v[2] for v in stats.values()})
    counted = f"{plural(reps[0], 'rep')} per task" if len(reps) == 1 else "reps vary by task"
    bits = [b for b in (args.context, counted, "mean +/- standard error") if b]

    height_inches = fig.get_size_inches()[1]
    chrome = [
        fig.text(0.012, 1 - 0.32 / height_inches, args.title, fontsize=13,
                 color=theme["primary"], ha="left", va="center"),
        fig.text(0.012, 1 - 0.58 / height_inches, ", ".join(bits), fontsize=8,
                 color=theme["muted"], ha="left", va="center"),
    ]
    footer = 0.0
    if len(engines) > 1:
        footer = 0.42
        handles, labels = panels[0][0].get_legend_handles_labels()
        legend = fig.legend(
            handles, labels, loc="center left", bbox_to_anchor=(0.012, 0.18 / height_inches),
            ncols=len(engines), frameon=False, fontsize=9,
        )
        for text in legend.get_texts():
            text.set_color(theme["secondary"])
        note = fig.text(
            0.988, 0.18 / height_inches, f"Nx is how much faster {base} is",
            fontsize=8, color=theme["muted"], ha="right", va="center",
        )
        chrome.append((legend, note))

    widen_for_chrome(fig, chrome)
    fig.tight_layout(rect=(0, footer / height_inches, 1, 1 - 0.78 / height_inches))
    for ax, labels in panels:
        fit_labels(fig, ax, labels)

    if not args.out:
        plt.show()
        return
    for path in args.out:
        fig.savefig(
            path, dpi=args.dpi, facecolor=theme["surface"],
            metadata={"Date": None} if Path(path).suffix in (".svg", ".pdf") else None,
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
