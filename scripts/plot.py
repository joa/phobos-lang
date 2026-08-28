#!/usr/bin/env python3
"""Plot what scripts/bench.py measured: tokens per second, higher is better.

Reads the CSVs that `bench.py --csv` writes, or the JSONs that `bench.py --json`
writes, and draws one panel per test with a bar per engine and model. The bars
carry the same figure the table prints, a mean over rounds of each round's mean,
with the standard error over rounds as the whisker.

One panel per test rather than one axis for everything: a prompt pass and a
decode step are both tokens per second and are an order of magnitude apart, so
sharing an axis would flatten the decode rows into the baseline. The panels are
laid out a row per kind, prompt above decode, and a row shares one scale, so
pp128 against pp512 is a fair comparison of bar lengths while pp against tg is
not. Every bar is labelled with its own number regardless.

Several files draw as several blocks, one under the other, each with its own
rows and its own scale. A file is one invocation of bench.py, which is one visit
to the card with the engines interleaved against each other, so bars are a fair
comparison within a block and not across two. The scale is per block for the
same reason it is per row: a 27B that only just fits generates two orders of
magnitude slower than a model that does, and on the small models' scale it would
draw as a sliver.

Given the JSON it plots only the rounds bench.py found the card to itself
between, which is the same number its "t/s uncontended" column reports, and says
so under the title. The CSV carries no such record, so everything in it is
plotted. --all-rounds turns the filter off.

Usage:
    python scripts/bench.py --json bench.json
    python scripts/plot.py bench.json -o bench.png -o bench.svg
    python scripts/plot.py bench.json bench-qwen38.json -o inference.svg
    python scripts/plot.py bench.csv --dark -o bench-dark.svg

The format follows each -o extension, so SVG, PDF and PNG all come off the same
call. Requires matplotlib:  pip install matplotlib
"""

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# Fixed, so the element ids in an SVG are stable across runs.
matplotlib.rcParams["svg.hashsalt"] = "phobos-bench"

# The chart's colors, one theme per mode, both validated as a categorical set
# against their own surface. Series are assigned in a fixed order, so an engine
# keeps its color whatever else is in the file.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}


# What comes off a model name to leave the label. Only these: a version is a
# dot too, and Path.stem would take Qwen3.8-27B-UD-IQ1_M down to Qwen3.
MODEL_SUFFIXES = (".gguf", ".onnx")


# --------------------------------------------------------------------------- #
# reading


def model_name(value):
    """The model's name without its directory or its format's extension.

    bench.py writes the name already, but a CSV assembled by hand can carry the
    path the run named."""
    name = Path(value).name
    for suffix in MODEL_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def load(path):
    """The samples and whatever the file knows beyond them.

    Both of bench.py's outputs carry the same per-repetition rows; the JSON adds
    the card, the commit and which rounds were uncontended."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as err:
        sys.exit(f"cannot read {path}: {err.strerror}")
    if text.lstrip().startswith("{"):
        run = json.loads(text)
        rows = run.get("samples", [])
        meta = dict(run.get("meta", {}))
        meta["clean"] = set(run.get("summary", {}).get("uncontended_rounds", []))
        meta["median_clock"] = run.get("summary", {}).get("median_clock")
    else:
        rows = list(csv.DictReader(text.splitlines()))
        meta = {"clean": None}
    if not rows:
        sys.exit(f"no samples in {path}")

    # phobos-bench writes a CSV too, of GFLOP/s against a theoretical peak,
    # which is a different measurement and not what this draws.
    wanted = {"round", "engine", "backend", "model", "test", "rate"}
    missing = wanted - set(rows[0])
    if missing:
        sys.exit(
            f"{path} is not what scripts/bench.py writes:"
            f" missing {', '.join(sorted(missing))}"
        )

    samples = []
    for r in rows:
        samples.append(
            {
                "round": int(r["round"]),
                "engine": r["engine"],
                "backend": r["backend"],
                "model": model_name(r["model"]),
                "test": r["test"],
                "rate": float(r["rate"]),
            }
        )
    if not samples:
        sys.exit(f"no samples in {path}")
    return samples, meta


def keep_uncontended(samples, meta, all_rounds, name):
    """The rounds bench.py had the card to itself around, and how many went.

    A JSON records which rounds those were; a CSV does not, and neither does a
    run where the card was busy throughout, so both plot whole."""
    rounds = {s["round"] for s in samples}
    clean = meta.get("clean")
    if not clean or all_rounds:
        return samples, len(rounds), 0
    kept = rounds & set(clean)
    if not kept:
        print(
            f"every round in {name} was contended; plotting them all", file=sys.stderr
        )
        return samples, len(rounds), 0
    return [s for s in samples if s["round"] in kept], len(kept), len(rounds - kept)


def test_kind(test):
    """The letters a test name starts with: pp, tg, or whatever else appears."""
    found = re.match(r"([a-z]+)", test)
    return found.group(1) if found else test


def test_key(test):
    """pp before tg, then by token count, so the panels read prompt to decode."""
    found = re.match(r"([a-z]+)(\d+)$", test)
    if not found:
        return (True, 0, test)
    kind, count = found.groups()
    return (kind != "pp", int(count), test)


def kinds_of(samples):
    """The tests one file carries, a list per kind, prompt before decode."""
    grid = {}
    for test in sorted({s["test"] for s in samples}, key=test_key):
        grid.setdefault(test_kind(test), []).append(test)
    return grid


def engine_key(engine):
    """phobos first, since it is what the ratios are taken against."""
    return (not engine.startswith("phobos"), engine)


def summarize(samples):
    """Round means, then the mean and standard error over rounds.

    The same two steps bench.py's table takes. Averaging every repetition
    together instead would treat a round as several independent samples, and a
    round is one visit to the card at one set of clocks."""
    per_round = {}
    for s in samples:
        key = (s["engine"], s["model"], s["test"])
        per_round.setdefault(key, {}).setdefault(s["round"], []).append(s["rate"])

    out = {}
    for key, rounds in per_round.items():
        means = [statistics.fmean(v) for v in rounds.values()]
        err = statistics.stdev(means) / len(means) ** 0.5 if len(means) > 1 else 0.0
        out[key] = (statistics.fmean(means), err, len(means))
    return out


# --------------------------------------------------------------------------- #
# drawing


def fmt_rate(rate):
    """A rate at three significant-ish figures. The label sits inside the axis,
    so every character it does not need is axis the bars get to keep."""
    if rate >= 1000:
        return f"{rate:,.0f}"
    return f"{rate:.1f}"


def panel_inches(models, engines):
    """How tall a panel of `models` groups of `engines` bars each wants to be."""
    return 0.26 * models * engines + 0.34 * models + 0.95


def draw_panel(ax, test, models, engines, stats, theme, colors, base, scale_max):
    """One test: a bar per engine within a group per model, laid out along y so
    the model names read horizontally.

    scale_max is the widest bar anywhere in this panel's row, since the row
    shares one axis."""
    # Capped, so a file with one engine draws a bar of the same weight as one
    # with three rather than a block filling its group.
    bar_h = min(0.6 / len(engines), 0.3)
    group_h = bar_h * len(engines)
    pad = scale_max * 0.02
    labels = []  # the value labels, measured once drawn to size the axis

    ticks = []
    for gi, model in enumerate(models):
        ticks.append(gi)
        for ei, engine in enumerate(engines):
            got = stats.get((engine, model, test))
            if not got:
                continue
            mean, err, _ = got
            # The y axis is inverted below, so subtracting here puts the first
            # engine at the top of its group and keeps the bars in the order the
            # legend lists them.
            y = gi - group_h / 2 + bar_h * (ei + 0.5)
            ax.barh(
                y,
                mean,
                height=bar_h * 0.86,  # the gap between adjacent bars
                color=colors[engine],
                zorder=2,
                label=engine if gi == 0 else None,
            )
            if err > 0:
                ax.errorbar(
                    mean,
                    y,
                    xerr=err,
                    fmt="none",
                    ecolor=theme["secondary"],
                    elinewidth=1.0,
                    capsize=2.5,
                    capthick=1.0,
                    zorder=3,
                )
            # The value on the bar, in ink rather than the series color: the two
            # panels have different scales, so an unlabelled bar is only
            # comparable within its own panel.
            note = fmt_rate(mean)
            other = stats.get((base, model, test)) if engine != base else None
            if other and other[0]:
                note += f"   {mean / other[0]:.2f}x"
            labels.append(
                ax.text(
                    mean + err + pad,
                    y,
                    note,
                    va="center",
                    ha="left",
                    fontsize=8,
                    color=theme["secondary"],
                )
            )

    ax.set_yticks(ticks)
    ax.set_yticklabels(models, fontsize=9, color=theme["primary"])
    ax.set_ylim(-0.55, len(models) - 0.45)
    ax.invert_yaxis()
    ax.set_xlim(0, scale_max * 1.06)
    ax.set_xlabel("tokens/s", fontsize=8, color=theme["muted"])
    named = {"pp": "prompt", "tg": "decode"}.get(test_kind(test))
    ax.set_title(
        f"{test}  ({named})" if named else test,
        fontsize=10,
        color=theme["primary"],
        pad=8,
    )

    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=theme["grid"], linewidth=0.8)
    ax.yaxis.grid(False)
    ax.tick_params(colors=theme["muted"], labelsize=8, length=0)
    for name, spine in ax.spines.items():
        spine.set_visible(name == "left")
        spine.set_color(theme["axis"])
        spine.set_linewidth(0.8)
    return labels


def fit_labels(fig, ax, labels):
    """Widen the axis until every value label fits inside it.

    The labels sit in data coordinates but are as wide as their glyphs, so the
    room one needs depends on the scale it is measured under. Widening the axis
    moves the label left as fast as it frees room to the right, hence solving for
    the limit rather than padding by a guess: for a label starting at x with a
    width that is some fraction f of the axis, x + f * limit <= limit."""
    fig.canvas.draw()
    axis_pixels = ax.get_window_extent().width
    limit = ax.get_xlim()[1]
    for label in labels:
        share = label.get_window_extent().width / axis_pixels
        if share < 0.9:
            limit = max(limit, label.get_position()[0] / (1 - share) * 1.01)
    ax.set_xlim(0, limit)


def widen_for_chrome(fig, chrome):
    """Grow the figure until the title, the subtitle and the footer line fit.

    The panels set the width, and a tall narrow file (one test per kind, say)
    can end up narrower than the sentence describing the card it ran on, which
    would then run off the edge: none of the chrome wraps."""
    fig.canvas.draw()
    width_inches, height_inches = fig.get_size_inches()
    lines = [row if isinstance(row, tuple) else (row,) for row in chrome]
    needed = max(
        sum(part.get_window_extent().width for part in line) / fig.dpi for line in lines
    )
    if needed + 0.4 > width_inches:
        fig.set_size_inches(needed + 0.4, height_inches)


def plural(count, noun):
    return f"{count} {noun}{'s' if count != 1 else ''}"


def agreed(sources, field, fmt=str):
    """One rendering of a field per distinct value, in the order the files came.

    Two files are two sessions, so the card and the commit are theirs to agree
    on rather than the plot's to assume."""
    seen = [fmt(s["meta"][field]) for s in sources if s["meta"].get(field)]
    return list(dict.fromkeys(seen))


def subtitle(sources):
    """What the numbers rest on, in one line: the card, the commit, how many
    rounds each file carries, and whether any of them were thrown out."""
    bits = []
    for field, fmt, shape in (
        ("card", str, "{}"),
        ("median_clock", lambda v: f"{v:.0f}", "{} MHz median under load"),
        ("commit", str, "phobos {}"),
    ):
        values = agreed(sources, field, fmt)
        if values:
            bits.append(shape.format("/".join(values)))

    if len(sources) == 1:
        one = sources[0]
        counted = f"{plural(one['rounds'], 'round')} x {plural(one['reps'], 'rep')}"
        bits.append(f"{counted}, mean +/- standard error")
        if one["dropped"]:
            bits.append(f"{plural(one['dropped'], 'contended round')} dropped")
        return ", ".join(bits)

    for source in sources:
        counted = f"{plural(source['rounds'], 'round')} x {plural(source['reps'], 'rep')}"
        if source["dropped"]:
            counted += f", {source['dropped']} contended dropped"
        bits.append(f"{source['name']} {counted}")
    bits.append("mean +/- standard error, one scale per file")
    return ", ".join(bits)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "results",
        nargs="+",
        help="the CSVs or JSONs scripts/bench.py wrote, a block of rows each",
    )
    ap.add_argument(
        "-o",
        "--out",
        action="append",
        default=[],
        metavar="FILE",
        help="where to write it, the format taken from the extension;"
        " repeatable, shown interactively if absent",
    )
    ap.add_argument("--title", default="phobos vs llama.cpp, tokens/s (higher is better)")
    ap.add_argument(
        "--dark", action="store_true", help="draw for a dark surface instead"
    )
    ap.add_argument(
        "--all-rounds",
        action="store_true",
        help="plot every round, including any the card was not ours around",
    )
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    sources = []
    for path in args.results:
        samples, meta = load(path)
        samples, rounds, dropped = keep_uncontended(
            samples, meta, args.all_rounds, path
        )
        cells = len({(s["round"], s["engine"], s["model"], s["test"]) for s in samples})
        sources.append(
            {
                "name": Path(path).name,
                "meta": meta,
                "samples": samples,
                "stats": summarize(samples),
                "models": sorted({s["model"] for s in samples}),
                "kinds": kinds_of(samples),
                "rounds": rounds,
                "dropped": dropped,
                "reps": len(samples) // max(cells, 1),
            }
        )

    drawn = [s for source in sources for s in source["samples"]]
    engines = sorted({s["engine"] for s in drawn}, key=engine_key)
    base = engines[0]

    theme = THEMES["dark" if args.dark else "light"]
    if len(engines) > len(theme["series"]):
        sys.exit(
            f"{len(engines)} engines but only {len(theme['series'])} validated colors;"
            " plot a subset of the file"
        )
    colors = dict(zip(engines, theme["series"]))

    # A row per kind of test within a file, so prompt sits above decode and the
    # panels of a row can share one scale. A file's rows stay together, and a
    # row is only ever as tall as that file's models need.
    plan = [
        (source, kind, tests)
        for source in sources
        for kind, tests in source["kinds"].items()
    ]
    columns = max(len(tests) for _, _, tests in plan)
    heights = [panel_inches(len(s["models"]), len(engines)) for s, _, _ in plan]
    fig, axes = plt.subplots(
        len(plan),
        columns,
        figsize=(1.2 + 4.3 * columns, sum(heights) + 1.55),
        squeeze=False,
        sharex="row",
        gridspec_kw={"height_ratios": heights},
    )
    fig.patch.set_facecolor(theme["surface"])

    panels = []
    for row, (source, kind, tests) in enumerate(plan):
        # One scale for the row, taken from its widest bar, so a longer prompt
        # reading faster than a shorter one is visible as a longer bar.
        stats = source["stats"]
        scale_max = max(
            (stats[k][0] for k in stats if test_kind(k[2]) == kind), default=1.0
        )
        for column in range(columns):
            ax = axes[row][column]
            if column >= len(tests):
                ax.set_visible(False)
                continue
            ax.set_facecolor(theme["surface"])
            labels = draw_panel(
                ax, tests[column], source["models"], engines, stats, theme, colors,
                base, scale_max,
            )
            if column:
                # The models are the same in every panel of a row, so naming
                # them once at the left is enough.
                ax.set_yticklabels([])
            panels.append((ax, labels))

    # The chrome is placed in inches off the edges, so a file with more models
    # grows the panels and leaves the title and the legend where they were.
    height_inches = fig.get_size_inches()[1]
    chrome = [
        fig.text(
            0.012,
            1 - 0.32 / height_inches,
            args.title,
            fontsize=13,
            color=theme["primary"],
            ha="left",
            va="center",
        )
    ]
    chrome.append(
        fig.text(
            0.012,
            1 - 0.58 / height_inches,
            subtitle(sources),
            fontsize=8,
            color=theme["muted"],
            ha="left",
            va="center",
        )
    )
    # Identity is never color alone: the legend names every engine, and the
    # backend rides along with it, since a CPU-served row is not a comparison.
    footer = 0.0
    if len(engines) > 1:
        footer = 0.42
        backends = {s["engine"]: s["backend"] for s in drawn}
        # Gathered over every panel rather than the first: an engine missing
        # from the first file still belongs in the legend.
        handles, labels = [], []
        for ax, _ in panels:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        labels = [
            l if backends.get(l, "").lower() in l.lower() else f"{l} ({backends.get(l, '?')})"
            for l in labels
        ]
        legend = fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.012, 0.18 / height_inches),
            ncols=len(engines),
            frameon=False,
            fontsize=9,
        )
        for text in legend.get_texts():
            text.set_color(theme["secondary"])
        note = fig.text(
            0.988,
            0.18 / height_inches,
            f"Nx is that engine's rate over {base}'s",
            fontsize=8,
            color=theme["muted"],
            ha="right",
            va="center",
        )
        # The legend sits left and the note right on one line, so together they
        # set a width the same way a single run of text does.
        chrome.append((legend, note))

    widen_for_chrome(fig, chrome)
    fig.tight_layout(
        rect=(0, footer / height_inches, 1, 1 - 0.78 / height_inches), h_pad=1.8
    )
    # After the layout, so the panels are the width they will be printed at.
    for ax, labels in panels:
        fit_labels(fig, ax, labels)

    if not args.out:
        plt.show()
        return
    for path in args.out:
        # No creation date in the vector formats, so re-running on the same
        # samples produces the same file rather than a diff.
        fig.savefig(
            path,
            dpi=args.dpi,
            facecolor=theme["surface"],
            metadata={"Date": None} if Path(path).suffix in (".svg", ".pdf") else None,
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
