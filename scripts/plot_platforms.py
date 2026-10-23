#!/usr/bin/env python3
"""Plot Phobos over llama.cpp on every tested card: one line per card.

Reads each results/platforms/*/card.json and the bench.py run it names, keeps
the uncontended rounds, and draws each test's ratio of the two engines' means,
prompt sizes then decode sizes. The website's platforms page draws the same
chart from the same files.

A ratio compares two engines on one card in one session, so it is the one
quantity that shares an axis across cards and tests. Rates on different
rentals are not comparable, and prompt and decode rates differ a hundredfold.

Usage:
    python scripts/plot_platforms.py -o results/platforms.svg
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from plot import THEMES, keep_uncontended, load, summarize, test_key, test_kind

matplotlib.rcParams["svg.hashsalt"] = "phobos-platforms"

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen3.5-4B-Q4_K_M"

# The website's card slots, validated as an adjacent categorical set.
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}


def chip_number(chip):
    return int(chip.removeprefix("sm_"))


def cards():
    """Every card with a bench.py run, oldest chip first, with its color slot and ratios.

    A slot counts every card, measured or not, as the website's does."""
    every = [json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "results/platforms").glob("*/card.json")]
    every.sort(key=lambda card: (chip_number(card["chip"]), card["card"]))
    out = []
    for slot, card in enumerate(every):
        if "bench" not in card:
            continue
        samples, meta = load(ROOT / card["bench"])
        samples, _, _ = keep_uncontended(samples, meta, False, card["bench"])
        stats = summarize([s for s in samples if s["model"] == MODEL])
        ratios = {}
        for (engine, _, test), (mean, _, _) in stats.items():
            if not engine.startswith("phobos"):
                continue
            other = [v for (e, _, t), v in stats.items() if t == test and not e.startswith("phobos")]
            if other and other[0][0] > 0:
                ratios[test] = mean / other[0][0]
        if ratios:
            out.append((slot, card["card"], ratios, meta))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", action="append", required=True, help="output file; repeatable")
    ap.add_argument("--dark", action="store_true", help="the dark theme")
    args = ap.parse_args()

    mode = "dark" if args.dark else "light"
    theme = THEMES[mode]
    found = cards()
    if not found:
        sys.exit("no card has a bench.py run")
    tests = sorted({test for _, _, ratios, _ in found for test in ratios}, key=test_key)
    prompts = [t for t in tests if test_kind(t) == "pp"]
    # One empty slot between the prompt and the decode sizes.
    position = {t: i + (1 if test_kind(t) != "pp" and prompts else 0) for i, t in enumerate(tests)}

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])

    values = [1.0]
    ends = []
    for slot, name, ratios, _ in found:
        color = SERIES[mode][slot]
        for kind in ("pp", "tg"):
            run = [t for t in tests if test_kind(t) == kind and t in ratios]
            if not run:
                continue
            xs = [position[t] for t in run]
            ys = [ratios[t] for t in run]
            values += ys
            ax.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round", zorder=3,
                    label=name if kind == "pp" or not any(test_kind(t) == "pp" for t in ratios) else None)
            ax.scatter(xs, ys, s=36, color=color, edgecolors=theme["surface"], linewidths=1.5, zorder=4)
        last = max(ratios, key=lambda t: position[t])
        ends.append([ratios[last], name, position[last]])

    # End labels, pushed apart so no two overlap: a gap of 4.5% of the axis.
    low, high = min(values), max(values)
    gap = (high - low) * 1.16 * 0.045
    ends.sort()
    for index in range(1, len(ends)):
        ends[index][0] = max(ends[index][0], ends[index - 1][0] + gap)
    for y, name, x in ends:
        ax.text(x + 0.25, y, name, va="center", fontsize=8, color=theme["primary"])

    ax.axhline(1.0, color=theme["secondary"], linewidth=1, linestyle=(0, (4, 4)), zorder=2)
    ax.set_xticks([position[t] for t in tests])
    ax.set_xticklabels(tests, fontsize=8, color=theme["secondary"])
    ax.set_xlim(-0.6, max(position.values()) + 1.8)
    pad = (high - low) * 0.08
    ax.set_ylim(low - pad, high + pad)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}x"))
    ax.tick_params(axis="y", labelsize=8, colors=theme["secondary"])
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8, zorder=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])
    ax.set_ylabel("Phobos over llama.cpp", fontsize=9, color=theme["secondary"])
    if prompts:
        ax.text((position[prompts[0]] + position[prompts[-1]]) / 2, -0.12, "prompt processing",
                transform=ax.get_xaxis_transform(), ha="center", fontsize=9, color=theme["primary"])
    decodes = [t for t in tests if test_kind(t) != "pp"]
    if decodes:
        ax.text((position[decodes[0]] + position[decodes[-1]]) / 2, -0.12, "token generation",
                transform=ax.get_xaxis_transform(), ha="center", fontsize=9, color=theme["primary"])

    fig.suptitle(f"{MODEL}, Phobos's rate over llama.cpp's on each card, higher is better",
                 x=0.06, ha="left", fontsize=11, color=theme["primary"])
    commits = sorted({meta.get("commit", "?") for *_, meta in found})
    ax.set_title(f"one bench.py run per card, uncontended rounds, Phobos {', '.join(commits)}; "
                 "rates on different cards do not compare",
                 loc="left", fontsize=8, color=theme["muted"])
    legend = ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=len(found), frameon=False,
                       fontsize=8, handlelength=1.5)
    for text in legend.get_texts():
        text.set_color(theme["primary"])
    fig.tight_layout()
    for out in args.out:
        fig.savefig(out, facecolor=theme["surface"])
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
