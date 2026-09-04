#!/usr/bin/env python3
"""Phobos against llama.cpp, on the same models, in one session, on a warm card.

Runs the GGUF path's `bench` example and `llama-bench` over the same models with
the same pp/tg sizes and reports one table, plus the phobos-to-llama.cpp ratio
per row.

What this does that a pair of hand-run benchmarks does not:

 1. Refuses to start on a contended card.
 2. Warms the card before the first timed run and confirms the SM clock has
    reached a plateau.
 3. Interleaves. Every round runs each engine once per model, and the order
    reverses on odd rounds, so the first slot of a round (worth about a percent)
    is shared out instead of always landing on the same engine.
 4. Checks the card again between rounds, when none of our work is on it, and
    reports the summary twice if anything was found.
 5. Gets llama.cpp onto the GPU, and proves it did. A llama.cpp release ships
    ggml-cuda.dll without the CUDA runtime it links against, and lacking it the
    build serves the CPU while still printing a benchmark.
 6. Records what each engine ran with: the build, the runtime, and any PHOBOS_
    override in the environment.

Not controlled for: llama.cpp runs its own defaults, which are its best
configuration rather than a match for phobos's internals. Its KV cache is f16 and
flash attention is on where the card takes it. That is left alone deliberately,
since pinning it down to whatever phobos does would measure a configuration
nobody runs, but it is a difference between the two columns and not only between
the two engines.

Usage:
    python scripts/bench.py
    python scripts/bench.py -p 128 512 -n 32 128 512 -r 3 -R 6 --csv bench.csv
    python scripts/bench.py --llama-bench C:/path/to/llama-bench.exe --force
    python scripts/bench.py --cuda-lib C:/path/to/cuda/bin

Needs nvidia-smi and cargo on PATH, and an NVIDIA card. No third-party imports.
"""

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODELS = [
    "models/Qwen3.5-0.8B-Q8_0.gguf",
    "models/minicpm5-1b-Q8_0.gguf",
]


# --------------------------------------------------------------------------- #
# the card


def smi(query, mode="gpu"):
    """One nvidia-smi query, first GPU, as a list of stripped strings."""
    flag = f"--query-{mode}={query}"
    out = subprocess.run(
        ["nvidia-smi", flag, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lines = [l for l in out.splitlines() if l.strip()]
    if mode == "gpu":
        return [f.strip() for f in lines[0].split(",")]
    return [[f.strip() for f in l.split(",")] for l in lines]


def card_now():
    """Clock, utilization, power and temperature as numbers, for one sample."""
    sm, util, power, temp = smi("clocks.sm,utilization.gpu,power.draw,temperature.gpu")

    def num(v):
        try:
            return float(v)
        except ValueError:
            return float("nan")

    return {"clock": num(sm), "util": num(util), "power": num(power), "temp": num(temp)}


def compute_apps():
    """The pids holding the GPU. Counted, never named: which processes those
    are is not the benchmark's business, and a workload heavy enough to move a
    measurement shows up in what the card draws."""
    try:
        rows = smi("pid", mode="compute-apps")
    except subprocess.CalledProcessError:
        return []
    return [r[0] for r in rows if r and r[0]]


class Monitor:
    """Polls the card while something else runs, so the clocks recorded for a
    run are the clocks it ran at. Sampling before or after says nothing: the
    card leaves its plateau within a moment of the last launch."""

    def __init__(self, interval_secs):
        self.interval_secs = interval_secs
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.samples.append(card_now())
            except Exception:
                pass
            self._stop.wait(self.interval_secs)

    def __enter__(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=5)
        return False

    def clocks(self):
        return [s["clock"] for s in self.samples if s["clock"] == s["clock"]]


# --------------------------------------------------------------------------- #
# engines


def sizes(counts):
    """A token-count list as both engines take it on one flag."""
    return ",".join(str(c) for c in counts)


@dataclass
class Run:
    """What one invocation of one engine on one model produced."""

    backend: str
    rates: dict  # test name -> per-repetition tokens/s
    label: str = ""
    clocks: list = field(default_factory=list)


class Phobos:
    name = "phobos"

    def __init__(self, exe):
        self.exe = str(exe)
        self.env = None
        self.lib = ""

    def command(self, model, prompt_tokens, gen_tokens, reps):
        return [
            self.exe,
            "-m",
            str(model),
            "-p",
            sizes(prompt_tokens),
            "-n",
            sizes(gen_tokens),
            "-r",
            str(reps),
        ]

    def parse(self, stdout, stderr):
        # The per-repetition seconds off the progress lines rather than the mean
        # off the summary table, so the samples line up with llama-bench's
        # samples_ts and the spread can be computed the same way for both.
        rates = {}
        for test, count, secs in re.findall(
            r"^\s*((?:pp|tg)(\d+)) rep \d+/\d+: ([\d.]+) s", stderr, re.M
        ):
            rates.setdefault(test, []).append(int(count) / float(secs))
        found = re.search(r"backend ([^,]+),", stderr)
        served = found.group(1) if found else "?"
        label = ""
        arch = re.search(r"\(([^)]*?), \d+ vocab\)", stderr)
        if arch:
            label = arch.group(1)
        return Run(
            backend="CUDA" if "GPU" in served else "CPU", rates=rates, label=label
        )


class LlamaCpp:
    name = "llama.cpp"

    def __init__(self, exe, extra=()):
        self.exe = str(exe)
        # Its own PATH, not the session's: a llama.cpp release ships
        # ggml-cuda.dll with no CUDA runtime beside it, and a directory that
        # supplies one has to reach this process without also reaching the one
        # phobos runs in, which finds its own driver and needs no help.
        self.env = None
        self.lib = ""
        # Whatever the caller wants llama.cpp configured with, appended last so
        # it overrides the defaults above. The probe sees these too, so the
        # backend reported is the one the timed runs will use.
        self.extra = list(extra)

    def command(self, model, prompt_tokens, gen_tokens, reps):
        return [
            self.exe,
            "-m",
            str(model),
            "-p",
            sizes(prompt_tokens),
            "-n",
            sizes(gen_tokens),
            "-r",
            str(reps),
            "-ngl",
            "99",
            "-o",
            "json",
            *self.extra,
        ]

    def parse(self, stdout, stderr):
        start, end = stdout.find("["), stdout.rfind("]")
        if start < 0 or end < 0:
            raise ValueError("llama-bench printed no JSON")
        entries = json.loads(stdout[start : end + 1])
        rates, backend, label = {}, "?", ""
        for e in entries:
            test = f"tg{e['n_gen']}" if e["n_gen"] else f"pp{e['n_prompt']}"
            rates[test] = list(e["samples_ts"])
            backend = e.get("backends", "?")
            label = e.get("model_type", "")
        return Run(backend=backend, rates=rates, label=label)


def invoke(engine, model, args, monitor_interval):
    """One engine, one model, both tests, with the card watched throughout."""
    cmd = engine.command(model, args.prompt_tokens, args.gen_tokens, args.reps)
    with Monitor(monitor_interval) as mon:
        done = subprocess.run(
            cmd, capture_output=True, text=True, cwd=ROOT, env=engine.env
        )
    if done.returncode != 0:
        tail = (done.stderr or done.stdout).strip().splitlines()[-3:]
        raise RuntimeError(f"{engine.name} exited {done.returncode}: {' / '.join(tail)}")
    run = engine.parse(done.stdout, done.stderr)
    run.clocks = mon.clocks()
    return run


# --------------------------------------------------------------------------- #
# setup


def find_llama_benches(explicit):
    """Given paths first, then anything next to the phobos checkout, then PATH."""
    found = [Path(p) for p in explicit]
    if not found:
        for sibling in sorted(ROOT.parent.glob("llama*")):
            exe = sibling / ("llama-bench.exe" if os.name == "nt" else "llama-bench")
            if exe.is_file():
                found.append(exe)
        on_path = shutil.which("llama-bench")
        if on_path and Path(on_path) not in found:
            found.append(Path(on_path))
    return found


def cuda_lib_dirs(explicit):
    """Directories worth putting on a llama-bench's PATH.

    A llama.cpp release carries ggml-cuda.dll but not the cudart and cublas it
    links against, and without them the backend never loads: the build serves the
    CPU and says so only in one line of its own log, which reads as a working
    benchmark and quietly turns a GPU comparison into a CPU one. Ollama ships a
    matching runtime per major version, so try each of those."""
    if explicit:
        return [Path(p) for p in explicit]
    root = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "lib" / "ollama"
    return [d for d in sorted(root.glob("cuda_v*"), reverse=True) if d.is_dir()]


def with_lib(directory):
    """The environment plus one directory at the front of PATH."""
    if directory is None:
        return None, ""
    env = dict(os.environ)
    env["PATH"] = f"{directory}{os.pathsep}{env.get('PATH', '')}"
    return env, str(directory)


def best_backend(engine, model, libs):
    """The engine set up the best way it can be, and which backend that reaches.

    Tried bare first, so a build that finds its own runtime is left alone, then
    once per candidate directory. CUDA short-circuits; anything else is kept only
    as a fallback, because a CPU row that could have been a GPU row is the whole
    failure this is here to avoid."""
    fallback = None
    for directory in [None, *libs]:
        engine.env, engine.lib = with_lib(directory)
        backend, why = probe(engine, model)
        if backend == "CUDA":
            return backend, None
        if backend and fallback is None:
            fallback = (backend, engine.env, engine.lib)
    if fallback:
        backend, engine.env, engine.lib = fallback
        return backend, None
    engine.env, engine.lib = None, ""
    return None, why


def build_phobos(args):
    """Compile the bench example before anything is timed, so no repetition
    pays for a cargo build or a first-use kernel compile."""
    exe = ROOT / "target" / "release" / "examples" / (
        "bench.exe" if os.name == "nt" else "bench"
    )
    if args.no_build:
        if not exe.is_file():
            sys.exit(f"--no-build but {exe} is missing")
        return exe
    cmd = [
        "cargo",
        "build",
        "--release",
        "-p",
        "phobos-gguf",
        "--features",
        "cuda",
        "--example",
        "bench",
    ]
    print("building phobos bench...", flush=True)
    done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(done.stderr[-2000:])
    return exe


def probe(engine, model):
    """A tiny run to find out whether the engine works at all and which backend
    actually served it. A llama.cpp build can carry ggml-cuda.dll, load it, and
    still fail at the first launch, or quietly fall back to the CPU; either one
    read as a GPU comparison would be wrong."""
    cmd = engine.command(model, [16], [4], 1)
    done = subprocess.run(
        cmd, capture_output=True, text=True, cwd=ROOT, timeout=900, env=engine.env
    )
    if done.returncode != 0:
        why = (done.stderr or done.stdout).strip().splitlines()
        return None, (why[-1] if why else f"exit {done.returncode}")
    try:
        run = engine.parse(done.stdout, done.stderr)
    except Exception as err:
        return None, str(err)
    return run.backend, None


def warm_card(engine, args):
    """Sustained work until the SM clock stops climbing. Reports the ramp, since
    a plateau that never arrives means the numbers below are measuring the card
    warming up rather than the engines.

    Prompt passes only, no decode: a decode step is launch-bound and leaves the
    card idle enough that it never asks for its boost clock, so warming with one
    would report a plateau the prompt rows then climb straight past."""
    before = card_now()
    print(f"warming the card ({before['clock']:.0f} MHz now)...", flush=True)
    cmd = [
        str(engine.exe),
        "-m",
        str(args.models[0]),
        "-p",
        "512",
        "-n",
        "0",
        "-r",
        "400",
    ]
    # The engine's own environment, not the session's: a llama-bench warming
    # the card without its CUDA runtime on PATH warms the CPU instead.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=ROOT,
        env=engine.env,
    )
    deadline = time.time() + args.warm_secs
    seen, peak, plateau = [], 0.0, False

    try:
        while time.time() < deadline and proc.poll() is None:
            time.sleep(1.0)
            now = card_now()["clock"]
            if now != now:
                continue
            peak = max(peak, now)
            seen.append(now)
            recent = seen[-4:]
            plateau = (
                len(seen) >= 10
                and (max(recent) - min(recent)) / max(recent) < 0.02
                and now > 0.5 * peak
            )
            if plateau:
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    print(
        f"  clock {before['clock']:.0f} -> {seen[-1] if seen else float('nan'):.0f} MHz"
        f" (peak {peak:.0f}) over {len(seen)} s,"
        f" {'plateau' if plateau else 'STILL DRIFTING'}",
        flush=True,
    )
    return plateau


def gap_scan(settle_secs=1.5, samples=4):
    """What the card is doing with none of our work on it, between two rounds.

    This is the signal a round is judged on.

    The middle of several samples rather than the worst, because the desktop
    catches up on its own drawing the moment a GPU-heavy process exits and one
    spike there is not a contended session."""
    time.sleep(settle_secs)
    seen = []
    for _ in range(samples):
        seen.append(card_now())
        time.sleep(0.4)
    return {
        "util": statistics.median(s["util"] for s in seen),
        "power": statistics.median(s["power"] for s in seen),
    }


def busy(state, args):
    """Why the card is not ours, or None.

    Judged on what the card is doing, never on which processes hold it: a
    workload large enough to bend a measurement draws power and occupies SMs,
    and the thresholds sit above an ordinary busy desktop."""
    if state["power"] > args.max_idle_power:
        return f"{state['power']:.0f} W drawn with none of our work running"
    if state["util"] > args.max_idle_util:
        return f"{state['util']:.0f}% utilization by others"
    return None


def preflight(args):
    """The card, and whether it is ours to measure on."""
    name, driver, clock_max = smi("name,driver_version,clocks.max.sm")
    now = card_now()
    apps = compute_apps()
    # The same median-of-samples the rounds are judged on, so the gate to start
    # and the gate on each round cannot disagree about the same card.
    state = gap_scan(settle_secs=0.0)
    print(f"card {name}, driver {driver}, max SM clock {clock_max} MHz")
    print(
        f"idle: {now['clock']:.0f} MHz, {state['util']:.0f}% util,"
        f" {state['power']:.0f} W, {now['temp']:.0f} C,"
        f" {len(apps)} compute apps"
    )
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    why = busy(state, args)
    if why:
        print(f"\nthe card is not free: {why}")
        if not args.force:
            sys.exit(
                "\nrefusing to benchmark. A contended card has read half to a\n"
                "tenth of the real figure while the clocks looked fine, and the result\n"
                "is self-consistent, so it cannot be corrected afterwards. Close the\n"
                "other process, or pass --force to measure anyway."
            )
        print("  --force given, measuring anyway; the numbers are not comparable\n")
    return {
        "card": name,
        "driver": driver,
        "clock_max": clock_max,
        "commit": commit or "?",
    }


# --------------------------------------------------------------------------- #
# statistics and reporting


def mean_stderr(values):
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, statistics.stdev(values) / len(values) ** 0.5


def table(rows, headers):
    """A markdown table, columns padded to the widest cell."""
    widths = [
        max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
        for i, h in enumerate(headers)
    ]
    out = ["| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"]
    out.append("| " + " | ".join("-" * w for w in widths) + " |")
    for r in rows:
        out.append(
            "| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |"
        )
    return "\n".join(out)


def summarize(samples, rounds_kept, engines, models, tests):
    """Round means, then the mean and standard error over rounds."""
    out = {}
    for engine in engines:
        for model in models:
            for test in tests:
                per_round = []
                for rnd in rounds_kept:
                    hits = [
                        s["rate"]
                        for s in samples
                        if s["round"] == rnd
                        and s["engine"] == engine
                        and s["model"] == model
                        and s["test"] == test
                    ]
                    if hits:
                        per_round.append(statistics.fmean(hits))
                if per_round:
                    out[(engine, model, test)] = (*mean_stderr(per_round), len(per_round))
    return out


def drift(samples, backends, rounds):
    """Per round, how far its GPU-backed rows sat from their own medians.

    A diagnostic, not a filter. Selecting rounds by the rates being summarized
    would pull every mean toward its median and shrink the error bars to match,
    so what a round is judged on is the contention scan beside it, which knows
    nothing about the rates. This is here to be looked at: an ambient slowdown
    moves every row of a round together, which is visible in one column."""
    by_row = {}
    for s in samples:
        if backends.get(s["engine"]) == "CPU":
            continue
        by_row.setdefault((s["engine"], s["model"], s["test"]), {}).setdefault(
            s["round"], []
        ).append(s["rate"])
    if len(rounds) < 3:
        return {}
    out = {}
    for rnd in rounds:
        moves = []
        for per_round in by_row.values():
            middle = statistics.median(
                statistics.fmean(v) for v in per_round.values()
            )
            here = per_round.get(rnd)
            if here and middle:
                moves.append(statistics.fmean(here) / middle - 1.0)
        if moves:
            out[rnd] = statistics.fmean(moves)
    return out


def report(samples, cell_clocks, health, args, meta, backends, labels, paths, order):
    engines = [e for e in order if any(s["engine"] == e for s in samples)]
    models = list(dict.fromkeys(s["model"] for s in samples))
    tests = list(dict.fromkeys(s["test"] for s in samples))
    all_rounds = sorted({s["round"] for s in samples})

    watched = [
        c
        for (_, engine, _), c in cell_clocks.items()
        if backends.get(engine) != "CPU" and c == c
    ]
    median = statistics.median(watched) if watched else 0.0
    # Judged on what the card was doing between rounds, which is independent of
    # the rates being summarized.
    dirty = [rnd for rnd in all_rounds if health.get(rnd) and busy(health[rnd], args)]
    clean = [rnd for rnd in all_rounds if rnd not in dirty]
    moved = drift(samples, backends, all_rounds)

    wide = summarize(samples, all_rounds, engines, models, tests)
    tight = summarize(samples, clean, engines, models, tests)

    print(f"\n{'=' * 78}\n{meta['card']}, driver {meta['driver']}, phobos {meta['commit']}")
    print(
        f"{args.rounds} rounds x {args.reps} repetitions, {', '.join(tests)}"
    )
    if watched:
        print(
            f"SM clock under GPU work: median {median:.0f} MHz,"
            f" {min(watched):.0f} to {max(watched):.0f}"
        )
    print(
        f"rounds with the card to ourselves between them:"
        f" {len(clean)} of {len(all_rounds)}"
    )
    for engine in engines:
        print(f"{engine}: {backends[engine]}, {paths[engine]}")
    for model in models:
        told = labels.get((engines[0], model), "?")
        print(f"{model}: {told}")
    overrides = {k: v for k, v in os.environ.items() if k.startswith("PHOBOS_")}
    print(f"phobos env: {overrides or 'nothing set, so every default is in force'}")

    rows = []
    for model in models:
        for test in tests:
            for engine in engines:
                got = wide.get((engine, model, test))
                if not got:
                    continue
                mean, err, count = got
                fine = tight.get((engine, model, test))
                row = [
                    model,
                    f"{engine} ({backends[engine]})",
                    test,
                    f"{mean:.2f}",
                    f"+/- {err:.2f}",
                    f"{count}",
                ]
                if dirty:
                    row.append(f"{fine[0]:.2f}" if fine else "-")
                rows.append(row)
    headers = ["model", "engine", "test", "t/s", "stderr", "rounds"]
    if dirty:
        headers.append("t/s uncontended")
    print("\n" + table(rows, headers))

    if moved:

        def round_clock(rnd):
            here = [
                c
                for (r, engine, _), c in cell_clocks.items()
                if r == rnd and backends.get(engine) != "CPU" and c == c
            ]
            return statistics.fmean(here) if here else float("nan")

        spread = [
            [
                str(rnd),
                f"{100 * moved[rnd]:+.2f}%",
                f"{round_clock(rnd):.0f}",
                f"{health[rnd]['util']:.0f}%" if health.get(rnd) else "-",
                "contended" if rnd in dirty else "",
            ]
            for rnd in sorted(moved)
        ]
        print(
            "\nper round, against each row's own median (a diagnostic, not a filter):\n"
            + table(spread, ["round", "GPU rows", "SM MHz", "util between", "note"])
        )

    # The ratio, and a marker when the two engines were not served by the same
    # backend, which makes the row a curiosity rather than a comparison.
    if len(engines) >= 2:
        base, *rest = engines
        rows = []
        for model in models:
            for test in tests:
                mine = wide.get((base, model, test))
                for engine in rest:
                    theirs = wide.get((engine, model, test))
                    if not mine or not theirs:
                        continue
                    mixed = backends[base] != backends[engine]
                    rows.append(
                        [
                            model,
                            test,
                            f"{mine[0]:.2f}",
                            f"{theirs[0]:.2f}",
                            f"{mine[0] / theirs[0]:.2f}x",
                            f"{engine} on {backends[engine]}" if mixed else "",
                        ]
                    )
        print(
            f"\n{base} against {', '.join(rest)}:\n"
            + table(rows, ["model", "test", base, "other", "ratio", "not comparable"])
        )
        crossed = [e for e in rest if backends[e] != backends[base]]
        if crossed:
            print(
                f"\nThe ratio above crosses backends: {', '.join(crossed)} ran on"
                f" {backends[crossed[0]]}\nwhile {base} ran on {backends[base]},"
                " so what it compares is the two engines on the\nhardware each"
                " reached, not the two engines. Read it as a floor, not a speedup."
            )

    if dirty:
        print(
            f"\nsomething else held the card around round(s) {dirty}."
            " Where the two t/s columns\ndisagree the uncontended one is the number,"
            f" and it rests on {len(clean)} round(s)."
        )
    if not clean:
        print(
            "\nEvery round was contended, so nothing here is a comparison. Find what"
            " else is\nusing the card and run it again."
        )
    return {
        "median_clock": median,
        "uncontended_rounds": clean,
        "rounds": all_rounds,
        "drift": moved,
    }


def write_csv(path, samples):
    with open(path, "w", newline="", encoding="utf-8") as f:
        out = csv.DictWriter(
            f, ["round", "slot", "engine", "backend", "model", "test", "rep", "rate", "clock"]
        )
        out.writeheader()
        out.writerows(samples)
    print(f"\nsamples written to {path}")


# --------------------------------------------------------------------------- #


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-m", "--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument(
        "-p",
        "--prompt-tokens",
        type=int,
        nargs="+",
        default=[128, 512],
        help="prompt sizes, one pp row each",
    )
    ap.add_argument(
        "-n",
        "--gen-tokens",
        type=int,
        nargs="+",
        default=[32, 128, 512],
        help="decode lengths, one tg row each",
    )
    ap.add_argument("-r", "--reps", type=int, default=3, help="repetitions per run")
    ap.add_argument("-R", "--rounds", type=int, default=5, help="interleaved rounds")
    ap.add_argument(
        "--llama-bench",
        action="append",
        default=[],
        metavar="EXE",
        help="llama-bench to use; repeatable, discovered next to the checkout if absent",
    )
    ap.add_argument(
        "--cuda-lib",
        action="append",
        default=[],
        metavar="DIR",
        help="directory holding the CUDA runtime a llama-bench needs on PATH;"
        " repeatable, Ollama's bundled ones are tried if absent",
    )
    ap.add_argument(
        "--llama-args",
        default="",
        metavar="FLAGS",
        help="extra flags for llama-bench, as one string:"
        ' --llama-args "-ctk f32 -ctv f32" puts its KV cache in f32, which is'
        " what phobos carries. Recorded in the report, since it changes what the"
        " other column means",
    )
    ap.add_argument("--no-llama", action="store_true", help="phobos only")
    ap.add_argument(
        "--no-phobos",
        action="store_true",
        help="llama.cpp only, for a model phobos cannot load yet: it measures"
        " the target rather than a comparison",
    )
    ap.add_argument("--no-build", action="store_true", help="use the existing bench.exe")
    ap.add_argument(
        "--warm-secs", type=float, default=45.0, help="cap on the warmup (default 45)"
    )
    ap.add_argument(
        "--max-idle-util",
        type=float,
        default=45.0,
        help="percent utilization by others that condemns a round (default 45)",
    )
    ap.add_argument(
        "--max-idle-power",
        type=float,
        default=120.0,
        help="watts drawn by others that condemns a round (default 120)",
    )
    ap.add_argument(
        "--poll", type=float, default=0.5, help="clock sampling interval, seconds"
    )
    ap.add_argument("--force", action="store_true", help="measure on a busy card anyway")
    ap.add_argument("--csv", help="write every repetition here")
    ap.add_argument("--json", help="write the run, including the summary, here")
    args = ap.parse_args()
    args.models = [
        str(p if Path(p).is_absolute() else ROOT / p) for p in args.models
    ]
    for model in args.models:
        if not Path(model).is_file():
            sys.exit(f"no such model: {model}")
    return args


def main():
    args = parse_args()
    if args.no_phobos and args.no_llama:
        sys.exit("--no-phobos and --no-llama together leave no engine to run")
    meta = preflight(args)

    engines = []
    if not args.no_phobos:
        engines.append(Phobos(build_phobos(args)))
    if not args.no_llama:
        for exe in find_llama_benches(args.llama_bench):
            engines.append(LlamaCpp(exe, shlex.split(args.llama_args)))

    # Probe every engine before the session starts. An engine that cannot run is
    # dropped here with its reason, rather than aborting a round halfway.
    print("\nprobing engines:")
    libs = cuda_lib_dirs(args.cuda_lib)
    backends, labels, working, seen = {}, {}, [], set()
    for engine in engines:
        tag = f"{engine.name} [{Path(engine.exe).parent.name}]"
        wanted = libs if isinstance(engine, LlamaCpp) else []
        backend, why = best_backend(engine, args.models[0], wanted)
        if backend is None:
            print(f"  {tag}: unusable, skipped ({why})")
            continue
        print(
            f"  {tag}: backend {backend}"
            + (f", runtime from {engine.lib}" if engine.lib else "")
        )
        # One build per (engine, backend) pair: two llama.cpp builds that both
        # fall back to the CPU would measure the same thing twice under
        # different names, and one of them would win the ratio row by accident.
        if (engine.name, backend) in seen:
            print("    same backend as a build already kept, skipped")
            continue
        seen.add((engine.name, backend))
        if engine.name != "phobos":
            engine.name = f"{engine.name} {backend}"
        backends[engine.name] = backend
        working.append(engine)
    if not working:
        sys.exit("no engine ran; nothing to measure")
    if not args.no_phobos and not any(e.name == "phobos" for e in working):
        sys.exit("phobos itself did not run; nothing to compare")
    if len(working) == 1:
        print(
            f"\nonly {working[0].name} is usable, so this run measures it alone"
            " rather than comparing anything."
        )
        if not args.no_phobos:
            print(
                "Pass --llama-bench to point at a build, and --cuda-lib at the"
                " CUDA runtime it needs."
            )

    # Warmed by whichever engine is present: the command is the pp-only shape
    # both binaries take, so it does not matter which one holds the card.
    if not warm_card(working[0], args) and not args.force:
        print("  (proceeding anyway; the clean-round filter below will show it)")

    # phobos's own warmup covers a batch of at least 128 rows, so a shallower
    # prompt row is a shape it has not compiled and the first repetition pays
    # for the compile. llama.cpp warms at the shape it is about to time.
    shallow = [] if args.no_phobos else [n for n in args.prompt_tokens if n < 128]
    if shallow:
        print(
            f"\n{', '.join(f'pp{n}' for n in shallow)} is below the 128 rows phobos warms"
            " with, so that\nprompt row includes a kernel compile and understates it."
            " Use -p 128 or more to compare."
        )

    samples, cell_clocks, health = [], {}, {}
    cells = [(e, m) for m in args.models for e in working]
    for rnd in range(1, args.rounds + 1):
        # Reversed on odd rounds: the first slot of a round is systematically
        # the faster one here, worth about a percent, which is larger than the
        # differences this is meant to resolve.
        turn = cells if rnd % 2 == 0 else list(reversed(cells))
        print(f"\nround {rnd}/{args.rounds}")
        for slot, (engine, model) in enumerate(turn):
            try:
                run = invoke(engine, model, args, args.poll)
            except Exception as err:
                print(f"  {engine.name} {Path(model).stem}: FAILED, {err}")
                continue
            clock = statistics.fmean(run.clocks) if run.clocks else float("nan")
            # Recorded by name, not by path: the results are published.
            name = Path(model).stem
            cell_clocks[(rnd, engine.name, name)] = clock
            labels[(engine.name, name)] = run.label
            shown = []
            for test, rates in sorted(run.rates.items()):
                for rep, rate in enumerate(rates):
                    samples.append(
                        {
                            "round": rnd,
                            "slot": slot,
                            "engine": engine.name,
                            "backend": run.backend,
                            "model": name,
                            "test": test,
                            "rep": rep,
                            "rate": rate,
                            "clock": clock,
                        }
                    )
                shown.append(f"{test} {statistics.fmean(rates):.2f} t/s")
            print(f"  {engine.name:<16} {name:<22} {'  '.join(shown)}   [{clock:.0f} MHz]")
        health[rnd] = gap_scan()
        why = busy(health[rnd], args)
        if why:
            print(f"  between rounds the card was not ours: {why}")

    if not samples:
        sys.exit("no measurements survived")
    summary = report(
        samples,
        cell_clocks,
        health,
        args,
        meta,
        backends,
        labels,
        {
            e.name: e.exe
            + (f" (runtime {Path(e.lib).name})" if e.lib else "")
            + (f" {' '.join(e.extra)}" if getattr(e, "extra", None) else "")
            for e in working
        },
        [e.name for e in working],
    )
    if args.csv:
        write_csv(args.csv, samples)
    if args.json:
        # Same reason as the samples: names, not the paths they were read from.
        recorded = dict(
            vars(args),
            models=[Path(m).stem for m in args.models],
            llama_bench=[Path(p).name for p in args.llama_bench],
            cuda_lib=[Path(p).name for p in args.cuda_lib],
        )
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta": meta,
                    "args": {k: str(v) for k, v in recorded.items()},
                    "backends": backends,
                    "labels": {f"{e}/{m}": v for (e, m), v in labels.items()},
                    "summary": summary,
                    "health": {str(k): v for k, v in health.items()},
                    "samples": samples,
                },
                f,
                indent=2,
            )
        print(f"run written to {args.json}")


if __name__ == "__main__":
    main()
