#!/usr/bin/env python3
"""An agent's real work, served by phobos and by llama.cpp, timed end to end.

For each engine, one server at a time on the port the agent is configured for:

 1. Start the server and wait until it can answer. Phobos listens before its
    weights are loaded, so for phobos this waits for its `warmed up` line.
    One small untimed request follows, so neither engine is measured cold.
 2. Copy every child of bench/ into a fresh scratch folder, leaving out its
    `check.*` file so the agent never sees how it is judged.
 3. For each child, run `pi -p @<child>/prompt.md` in the child's folder.
    The task finishes when pi exits. It counts as completed when the agent's
    last message contains DONE, and as correct when the child's check passes.
    The agent's folder stays under the results as `<engine>-rep<N>/<child>`,
    which is where to look at what pelican drew.
 4. Read the server's log for the run: per-request prompt and decode rates,
    how much of each prompt was cached, and for phobos the expert cache's
    hit rates.

Both servers get the same settings where their defaults would differ.
pi sends no sampling fields, so both get Qwen's recommended sampler for the
mode. Neither thinks by default; with `--thinking`, pi asks for reasoning and
llama.cpp is told to reason. The context is pi's, 16k with one slot.

The expert split is left to each engine. llama.cpp fits `-ncmoe` to the card
itself (`--fit`, on by default), and phobos sizes its expert cache from free
memory. Neither is tuned by hand.

Rates are token-weighted, summed tokens over summed seconds. Wall time only
means something beside the token counts, since the agent can take a different
path on each engine.

A proxy records every request the agent sends, with the length of its answer,
into `<engine>-rep<N>-<child>.requests.jsonl`. `--replay` sends such a file to
each engine instead of running the agent. The prompts are the same and each
answer is capped at the recorded length, so two engines or two builds do the
same work. The answers' text still differs from the recording, so an engine's
kept session is not the one the recording had.

`--squeeze` takes 2 GiB of device memory away partway through each task, the
way another program opening on the desktop would, and holds it until the task
ends, or until the request `--squeeze-until` names, as if the program closed
again. scripts/vram_hog.py holds it from a process of its own and keeps
rewriting it, since the driver pages an idle allocation out first. It starts
before the task's Nth chat request (`--squeeze-at`, the 4th by default), so
both engines take the hit at the same point in the same work, and both load
against the same unsqueezed card. The report then splits each task's rates
into the requests before, during and after the squeeze, and for phobos lists
each time its expert cache shrank or grew.

Usage:
    python scripts/agent_bench.py
    python scripts/agent_bench.py --engines phobos -r 3
    python scripts/agent_bench.py --replay RUN/phobos-rep1-fib.requests.jsonl -r 3
    python scripts/agent_bench.py --engines phobos --phobos-env PHOBOS_MOE_HOST_DECODE=0
    python scripts/agent_bench.py --replay RUN/phobos-rep1-fib.requests.jsonl -r 3 --squeeze

Needs pi, nvidia-smi and cargo on PATH. No third-party imports.
"""

import argparse
import http.client
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import card_now, compute_apps, cuda_lib_dirs, smi, with_lib  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = "models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
# pi is configured for 8080; the engine serves behind the recording proxy.
PORT = 8080
ENGINE_PORT = 8081
CONTEXT = 16384
MAX_TOKENS = 8192

# Qwen's recommendations for the non-thinking mode and for thinking.
SAMPLER = {"temp": 0.7, "top_k": 20, "top_p": 0.8, "min_p": 0.0, "presence_penalty": 1.5}
THINKING_SAMPLER = {"temp": 0.6, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "presence_penalty": 0.0}
SEED = 0


# --------------------------------------------------------------------------- #
# what a server logs about one request


@dataclass
class Request:
    prompt: int  # positions evaluated plus positions reused
    reused: int
    generated: int
    prompt_secs: float
    decode_secs: float


# phobos: "3120 prompt (2980 reused) + 61 generated, 212.3 pp, 17.1 tg tok/s"
PHOBOS_LINE = re.compile(
    r"(\d+) prompt \((\d+) reused\) \+ (\d+) generated, ([\d.]+) pp, ([\d.]+) tg tok/s"
)
PHOBOS_EXPERTS = re.compile(
    r"experts resident: decode (?:([\d.]+)% of (\d+)|none), prompt (?:([\d.]+)% of (\d+)|none); ([\d.]+) GB copied"
)
# llama.cpp: "prompt eval time =    1232.22 ms /    39 tokens" per request,
# and the same without "prompt " for the decode.
LLAMA_PROMPT = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
LLAMA_DECODE = re.compile(r"(?<!prompt )\beval time =\s*([\d.]+) ms /\s*(\d+) tokens")
PHOBOS_RESIZE = re.compile(r"expert cache: (\d+) MiB of the card left free; (?:shrinking|growing) from (\d+) to (\d+) slots")


def phobos_requests(text):
    out = []
    for m in PHOBOS_LINE.finditer(text):
        prompt, reused, generated = int(m[1]), int(m[2]), int(m[3])
        pp, tg = float(m[4]), float(m[5])
        fresh = prompt - reused
        out.append(
            Request(
                prompt=prompt,
                reused=reused,
                generated=generated,
                prompt_secs=fresh / pp if pp > 0 else 0.0,
                decode_secs=generated / tg if tg > 0 else 0.0,
            )
        )
    return out


def llama_requests(text):
    """One request per `eval time` pair.

    llama.cpp logs only the positions it evaluated. The full prompt length
    comes from the agent's side."""
    out = []
    for (p_ms, p_n), (d_ms, d_n) in zip(LLAMA_PROMPT.findall(text), LLAMA_DECODE.findall(text)):
        out.append(
            Request(
                prompt=int(p_n),
                reused=0,
                generated=int(d_n),
                prompt_secs=float(p_ms) / 1e3,
                decode_secs=float(d_ms) / 1e3,
            )
        )
    return out


def phobos_resizes(text):
    """Each time the expert cache shrank or grew, as (free MiB, slots a block
    before, after)."""
    return [tuple(map(int, m)) for m in PHOBOS_RESIZE.findall(text)]


def phobos_experts(text):
    """Decode and prompt hit rates over a stretch of log, lookup-weighted."""
    d_hit = d_n = p_hit = p_n = 0.0
    copied = 0.0
    for m in PHOBOS_EXPERTS.finditer(text):
        if m[1]:
            d_hit += float(m[1]) / 100 * int(m[2])
            d_n += int(m[2])
        if m[3]:
            p_hit += float(m[3]) / 100 * int(m[4])
            p_n += int(m[4])
        copied += float(m[5])
    if not d_n and not p_n:
        return None
    return {
        "decode_hit": d_hit / d_n if d_n else None,
        "prompt_hit": p_hit / p_n if p_n else None,
        "copied_gb": copied,
    }


# --------------------------------------------------------------------------- #
# engines


class Engine:
    name = ""
    ready_line = None

    def __init__(self, args):
        self.args = args
        self.env = dict(os.environ)
        self.proc = None
        self.log_path = None
        self.log = None

    def command(self):
        raise NotImplementedError

    def requests(self, text):
        raise NotImplementedError

    def start(self, log_path):
        self.log_path = log_path
        self.log = open(log_path, "wb")
        cmd = self.command()
        print(f"  $ {' '.join(map(str, cmd))}")
        self.proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=ROOT,
            env=self.env,
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + self.args.load_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                sys.exit(f"{self.name} exited while loading, see {log_path}")
            if self.ready():
                return
            time.sleep(1)
        self.stop()
        sys.exit(f"{self.name} not ready after {self.args.load_timeout} s, see {log_path}")

    def ready(self):
        if not port_open(ENGINE_PORT):
            return False
        return self.ready_line is None or self.ready_line in self.text()

    def text(self, start=0):
        with open(self.log_path, "rb") as f:
            f.seek(start)
            return f.read().decode("utf-8", "replace")

    def offset(self):
        self.log.flush()
        return os.path.getsize(self.log_path)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.log:
            self.log.close()
        for _ in range(30):
            if not port_open(ENGINE_PORT):
                return
            time.sleep(1)

    def describe(self, text):
        """What the engine chose for itself, off its load log."""
        return {}


class Phobos(Engine):
    name = "phobos"
    ready_line = "warmed up in"

    def __init__(self, args):
        super().__init__(args)
        self.env.update(kv.split("=", 1) for kv in args.phobos_env)

    def command(self):
        exe = ROOT / "target" / "release" / ("phobos-cli.exe" if os.name == "nt" else "phobos-cli")
        s = self.args.sampler
        return [
            exe,
            "--gguf", self.args.model,
            "--listen", f"127.0.0.1:{ENGINE_PORT}",
            "--no-tui",
            "-n", MAX_TOKENS,
            "--temp", s["temp"],
            "--top-k", s["top_k"],
            "--top-p", s["top_p"],
            "--min-p", s["min_p"],
            "--presence-penalty", s["presence_penalty"],
            "--seed", SEED,
            *self.args.phobos_arg,
        ]

    def requests(self, text):
        return phobos_requests(text)

    def describe(self, text):
        """The expert cache's layout, the last if it has shrunk already."""
        found = re.findall(r"expert cache: \d+ slots[^\n]*", text)
        return {"expert_cache": found[-1].strip()} if found else {}


class Llama(Engine):
    name = "llama.cpp"

    def __init__(self, args):
        super().__init__(args)
        exe = Path(args.llama_server)
        if not exe.is_file():
            sys.exit(f"no llama-server at {exe}")
        self.exe = exe
        # A release ships ggml-cuda.dll without its CUDA runtime and silently
        # serves on the CPU; see bench.py. Keeps the first environment under
        # which the server lists a CUDA device.
        for lib in [None, *cuda_lib_dirs(args.cuda_lib)]:
            env, _ = with_lib(lib)
            env = env or dict(os.environ)
            listed = subprocess.run([exe, "--list-devices"], env=env, capture_output=True, text=True).stdout
            if "CUDA0" in listed:
                self.env = env
                break
        else:
            sys.exit(f"{exe} lists no CUDA device with any runtime tried; pass --cuda-lib")
        given = args.llama_arg
        if "-ncmoe" in given or "--n-cpu-moe" in given:
            at = given.index("-ncmoe" if "-ncmoe" in given else "--n-cpu-moe")
            self.fit = f"experts of {given[at + 1]} blocks on the CPU, as given"
        else:
            self.fit = self.fitted()

    def fitted(self):
        """The expert split `--fit` will choose, which the server does not log.

        `llama-fit-params` runs the same fit and prints one `-ot` override per
        block whose experts stay on the CPU. Must run before the server starts,
        or it fits against what the server leaves free."""
        fit = self.exe.with_name(self.exe.name.replace("llama-server", "llama-fit-params"))
        if not fit.is_file():
            return None
        printed = subprocess.run(
            [fit, "-m", self.args.model, "-c", str(CONTEXT), "-np", "1"],
            cwd=ROOT, env=self.env, capture_output=True, text=True,
        ).stdout
        whole = set(re.findall(r"blk\\\.(\d+)\\\.ffn_\(up\|down", printed))
        part = set(re.findall(r"blk\\\.(\d+)\\\.ffn_down\.\*=CPU", printed)) - whole
        more = f", part of {len(part)} more" if part else ""
        return f"experts of {len(whole)} blocks on the CPU{more}"

    def command(self):
        s = self.args.sampler
        return [
            self.exe,
            "-m", self.args.model,
            "--port", ENGINE_PORT,
            "--host", "127.0.0.1",
            "-c", CONTEXT,
            "-np", 1,
            "--reasoning", "on" if self.args.thinking else "off",
            "--temp", s["temp"],
            "--top-k", s["top_k"],
            "--top-p", s["top_p"],
            "--min-p", s["min_p"],
            "--presence-penalty", s["presence_penalty"],
            "-s", SEED,
            *self.args.llama_arg,
        ]

    def ready(self):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{ENGINE_PORT}/health", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def requests(self, text):
        return llama_requests(text)

    def describe(self, text):
        return {"fit": self.fit} if self.fit else {}


def port_open(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def warm_request():
    """One short untimed chat, so no timed request is the first."""
    body = json.dumps(
        {
            "model": "x",
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 8,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{ENGINE_PORT}/v1/chat/completions",
        data=body,
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        r.read()


# --------------------------------------------------------------------------- #
# another program taking device memory mid-task


class Squeeze:
    """Starts scripts/vram_hog.py before a task's `at`th chat request and
    stops it before the `until`th, or when the task ends if `until` is 0.
    Off when `mib` is 0."""

    def __init__(self, mib, at, until, touch_ms):
        self.mib, self.at, self.until, self.touch_ms = mib, at, until, touch_ms
        self.lock = threading.Lock()
        self.seen = 0
        self.hog = None
        self.started = self.ended = 0
        self.note = ""

    def before_request(self):
        """Counts one chat request, and starts or ends the squeeze before it."""
        if not self.mib:
            return
        with self.lock:
            self.seen += 1
            if self.seen == self.until and self.hog:
                self.stop()
                self.ended = self.seen
                print(f"  squeeze ended before request {self.seen}: card {card_line()}")
            if self.seen != self.at:
                return
            before = card_line()
            self.hog = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts" / "vram_hog.py"), str(self.mib), "--touch-ms", str(self.touch_ms)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            held = self.hog.stdout.readline().strip()
            if not held.startswith("holding"):
                sys.exit(f"vram_hog.py did not take its memory: {held or 'it exited'}")
            self.started = self.seen
            self.note = f"{held}; card {before}, then {card_line()}"
            print(f"  squeeze before request {self.at}: {self.note}")

    def stop(self):
        self.hog.stdin.close()
        self.hog.wait(timeout=30)
        self.hog = None

    def release(self):
        """Ends the squeeze, if any, and resets the count for the next task.
        Returns the requests it started and ended before, 0 for neither."""
        if self.hog:
            self.stop()
        span = (self.started, self.ended)
        self.seen = self.started = self.ended = 0
        return span


# --------------------------------------------------------------------------- #
# recording what the agent asks, and asking it again


class Recorder:
    """A proxy from the agent's port to the engine's.

    While `path` is set, appends each chat request and its answer length to
    that file, so the work can be replayed on any engine."""

    def __init__(self, squeeze):
        self.path = None
        self.squeeze = squeeze
        self.server = ThreadingHTTPServer(("127.0.0.1", PORT), self.handler())
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def handler(recorder):
        class Forward(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                self.forward(None)

            def do_POST(self):
                self.forward(self.rfile.read(int(self.headers.get("content-length", 0))))

            def forward(self, body):
                if body and self.path.endswith("/chat/completions"):
                    recorder.squeeze.before_request()
                conn = http.client.HTTPConnection("127.0.0.1", ENGINE_PORT, timeout=3600)
                keep = ("content-type", "authorization", "accept")
                conn.request(self.command, self.path, body, {k: v for k, v in self.headers.items() if k.lower() in keep})
                resp = conn.getresponse()
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                        self.send_header(k, v)
                # Close ends the answer, so a stream passes through as it
                # arrives.
                self.send_header("connection", "close")
                self.end_headers()
                answer = bytearray()
                while chunk := resp.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    answer += chunk
                conn.close()
                if body and recorder.path and self.path.endswith("/chat/completions"):
                    recorder.write(body, bytes(answer))

        return Forward

    def write(self, body, answer):
        usage = answer_usage(answer)
        record = {"body": json.loads(body), "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def answer_usage(answer):
    """The usage block of a chat answer, streamed or not."""
    text = answer.decode("utf-8", "replace")
    usage = {}
    for line in text.splitlines():
        if line.startswith("data: {"):
            try:
                usage = json.loads(line[6:]).get("usage") or usage
            except json.JSONDecodeError:
                pass
    if not usage:
        try:
            usage = json.loads(text).get("usage") or {}
        except json.JSONDecodeError:
            pass
    return usage


def replay_trace(trace, squeeze):
    """Sends a recorded agent's requests again, in order, each capped at the
    recorded answer length.

    Returns the wall time and the prompt lengths the engine reported."""
    records = [json.loads(line) for line in Path(trace).read_text("utf-8").splitlines() if line.strip()]
    started = time.perf_counter()
    prompts = []
    for record in records:
        squeeze.before_request()
        body = dict(record["body"], stream=True, stream_options={"include_usage": True})
        body.pop("max_tokens", None)
        if record.get("completion_tokens"):
            body["max_completion_tokens"] = record["completion_tokens"]
        req = urllib.request.Request(
            f"http://127.0.0.1:{ENGINE_PORT}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3600) as r:
            prompts.append(answer_usage(r.read()).get("prompt_tokens") or 0)
    return time.perf_counter() - started, prompts


# --------------------------------------------------------------------------- #
# the agent


@dataclass
class Task:
    engine: str
    child: str
    rep: int
    wall_secs: float = 0.0
    exited: str = ""
    said_done: bool = False
    check: str = "none"
    turns: int = 0
    requests: list = field(default_factory=list)
    experts: dict = None
    squeezed_at: int = 0  # the request the squeeze started before, 0 without one
    squeeze_ended: int = 0  # the request it ended before, 0 if it held to the end
    squeeze: str = ""
    resizes: list = field(default_factory=list)

    def totals(self, part=slice(None)):
        rs = [Request(**r) if isinstance(r, dict) else r for r in self.requests][part]
        prompt = sum(r.prompt for r in rs)
        reused = sum(r.reused for r in rs)
        generated = sum(r.generated for r in rs)
        p_secs = sum(r.prompt_secs for r in rs)
        d_secs = sum(r.decode_secs for r in rs)
        return {
            "requests": len(rs),
            "prompt": prompt,
            "reused": reused,
            "generated": generated,
            "prompt_secs": p_secs,
            "decode_secs": d_secs,
            "pp": (prompt - reused) / p_secs if p_secs else 0.0,
            "tg": generated / d_secs if d_secs else 0.0,
        }


def copy_children(scratch, children):
    """Copies the children of bench/ into scratch, without the `check.*`
    files the agent must not see."""
    for child in children:
        shutil.copytree(
            ROOT / "bench" / child,
            scratch / child,
            ignore=shutil.ignore_patterns("check.*"),
        )


def run_check(child, workdir):
    for check in sorted((ROOT / "bench" / child).glob("check.*")):
        runner = {".js": ["node"], ".py": [sys.executable]}.get(check.suffix)
        if runner is None:
            continue
        done = subprocess.run(
            [*runner, str(check)], cwd=workdir, capture_output=True, text=True, timeout=60
        )
        return "pass" if done.returncode == 0 else "fail"
    return "none"


def run_agent(pi, workdir, out_path, timeout_secs, thinking):
    """Runs pi in print mode until it exits.

    Its stdin is closed, since print mode waits on an open one before it
    sends anything."""
    prompt = workdir / "prompt.md"
    cmd = [pi, "-p", "--offline", "--mode", "json", "--no-session", f"@{prompt}"]
    if thinking:
        # pi then asks for reasoning_effort, which phobos reads as thinking.
        cmd[1:1] = ["--model", "reasoning=true", "--thinking", "medium"]
    started = time.perf_counter()
    with open(out_path, "wb") as out:
        proc = subprocess.Popen(cmd, cwd=workdir, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=timeout_secs)
            exited = "ok" if code == 0 else f"exit {code}"
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            exited = "timeout"
    wall = time.perf_counter() - started

    last_text, turns, prompts = "", 0, []
    for line in Path(out_path).read_text("utf-8", "replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "message_end" and event["message"].get("role") == "assistant":
            turns += 1
            usage = event["message"].get("usage", {})
            prompts.append(usage.get("input", 0) + usage.get("cacheRead", 0))
            text = "".join(c.get("text", "") for c in event["message"].get("content", []) if c.get("type") == "text")
            if text.strip():
                last_text = text
    return wall, exited, "DONE" in last_text, turns, prompts


def kill_tree(proc):
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.kill()
    proc.wait()


# --------------------------------------------------------------------------- #
# the card


def used_gib():
    return float(smi("memory.used")[0]) / 1024


def card_line():
    used, total = smi("memory.used,memory.total")
    now = card_now()
    return (
        f"{float(used) / 1024:.2f} of {float(total) / 1024:.1f} GiB used,"
        f" {now['util']:.0f}% util, {now['power']:.0f} W, {len(compute_apps())} compute apps"
    )


def preflight(args):
    for port in (PORT, ENGINE_PORT):
        if port_open(port):
            sys.exit(f"something already listens on {port}; the proxy and the engine need {PORT} and {ENGINE_PORT}")
    print(f"card: {card_line()}")
    now = card_now()
    if (now["util"] > args.max_idle_util or now["power"] > args.max_idle_power) and not args.force:
        sys.exit("the card is busy; close what uses it, or pass --force")


def build_phobos():
    print("building phobos-cli")
    subprocess.run(
        ["cargo", "build", "--release", "-p", "phobos-cli", "--features", "cuda"],
        cwd=ROOT,
        check=True,
    )


# --------------------------------------------------------------------------- #
# reporting


def fmt_table(rows, headers):
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
    line = lambda r: "  ".join(str(c).rjust(w) for c, w in zip(r, widths))  # noqa: E731
    return "\n".join([line(headers), line(["-" * w for w in widths]), *map(line, rows)])


def report(tasks, engines_meta):
    headers = ["engine", "child", "rep", "wall s", "done", "check", "turns", "reqs", "prompt", "reused", "gen", "pp t/s", "tg t/s", "hit dec", "GB bus"]
    rows = []
    for t in tasks:
        s = t.totals()
        e = t.experts or {}
        rows.append(
            [
                t.engine, t.child, t.rep, f"{t.wall_secs:.1f}",
                "yes" if t.said_done else t.exited if t.exited != "ok" else "no",
                t.check, t.turns, s["requests"], s["prompt"], s["reused"], s["generated"],
                f"{s['pp']:.1f}", f"{s['tg']:.1f}",
                f"{100 * e['decode_hit']:.1f}%" if e.get("decode_hit") is not None else "",
                f"{e['copied_gb']:.0f}" if e else "",
            ]
        )
    print("\n" + fmt_table(rows, headers))

    print()
    summary = []
    for name in dict.fromkeys(t.engine for t in tasks):
        mine = [t for t in tasks if t.engine == name]
        s = [t.totals() for t in mine]
        fresh = sum(x["prompt"] - x["reused"] for x in s)
        p_secs = sum(x["prompt_secs"] for x in s)
        gen = sum(x["generated"] for x in s)
        d_secs = sum(x["decode_secs"] for x in s)
        walls = [t.wall_secs for t in mine]
        summary.append(
            [
                name, len(mine), f"{statistics.median(walls):.1f}", f"{sum(walls):.1f}",
                sum(t.said_done for t in mine), sum(t.check == "pass" for t in mine),
                fresh, f"{fresh / p_secs if p_secs else 0:.1f}", f"{p_secs:.1f}",
                gen, f"{gen / d_secs if d_secs else 0:.1f}", f"{d_secs:.1f}",
            ]
        )
    print(fmt_table(summary, ["engine", "tasks", "wall med", "wall sum", "DONE", "pass", "pp tok", "pp t/s", "pp s", "tg tok", "tg t/s", "tg s"]))
    for name, meta in engines_meta.items():
        for k, v in meta.items():
            print(f"{name} {k}: {v}")

    squeezed = [t for t in tasks if t.squeezed_at]
    if squeezed:
        print("\nbefore, during and after the squeeze, token-weighted:")
        rows = []
        for t in squeezed:
            start, end = t.squeezed_at - 1, (t.squeeze_ended - 1 if t.squeeze_ended else None)
            pre, during = t.totals(slice(None, start)), t.totals(slice(start, end))
            post = t.totals(slice(end, None)) if end is not None else None
            rate = lambda part, key: f"{part[key]:.1f}" if part else ""  # noqa: E731
            slots = ", ".join(f"{a}->{b}" for _, a, b in t.resizes)
            rows.append(
                [
                    t.engine, t.child, t.rep, f"{t.squeezed_at}-{t.squeeze_ended or 'end'}",
                    rate(pre, "pp"), rate(during, "pp"), rate(post, "pp"),
                    rate(pre, "tg"), rate(during, "tg"), rate(post, "tg"),
                    f"{during['tg'] / pre['tg']:.2f}x" if pre["tg"] else "",
                    len(t.resizes), slots,
                ]
            )
        print(fmt_table(rows, ["engine", "child", "rep", "squeezed", "pp before", "pp during", "pp after", "tg before", "tg during", "tg after", "tg kept", "resizes", "slots a block"]))


# --------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--engines", default="phobos,llama", help="comma-separated, in order: phobos, llama")
    p.add_argument("-m", "--model", default=MODEL)
    p.add_argument("-r", "--reps", type=int, default=1)
    p.add_argument("--children", nargs="*", help="which children of bench/ (default: all)")
    p.add_argument("--replay", nargs="+", metavar="TRACE", help="replay recorded *.requests.jsonl traces instead of running the agent")
    p.add_argument("--timeout", type=float, default=1800, help="seconds one agent task may take")
    p.add_argument("--load-timeout", type=float, default=1800, help="seconds a server may take to load")
    p.add_argument("--llama-server", default=str(ROOT.parent / "llama.cpp" / ("llama-server.exe" if os.name == "nt" else "llama-server")))
    p.add_argument("--cuda-lib", nargs="*", default=[])
    p.add_argument("--phobos-arg", action="append", default=[], help="extra phobos-cli argument")
    p.add_argument("--llama-arg", action="append", default=[], help="extra llama-server argument")
    p.add_argument("--phobos-env", action="append", default=[], help="NAME=VALUE for the phobos server")
    p.add_argument("--greedy", action="store_true", help="temperature 0 on both instead of Qwen's sampler")
    p.add_argument("--thinking", action="store_true", help="both engines think: pi asks for reasoning, llama.cpp --reasoning on")
    p.add_argument("--no-build", action="store_true")
    p.add_argument("--out", help="folder for logs and results (default: a new temp folder)")
    p.add_argument(
        "--squeeze", type=int, nargs="?", const=2048, default=0, metavar="MIB",
        help="take MIB of device memory, 2048 if not given, partway through each task",
    )
    p.add_argument("--squeeze-at", type=int, default=4, metavar="N", help="squeeze before the Nth chat request of a task")
    p.add_argument("--squeeze-until", type=int, default=0, metavar="N", help="end the squeeze before the Nth chat request (default: at the task's end)")
    p.add_argument("--squeeze-touch-ms", type=int, default=250, metavar="MS", help="how often the squeeze rewrites its memory")
    p.add_argument("--max-idle-util", type=float, default=30)
    p.add_argument("--max-idle-power", type=float, default=80)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    args.sampler = THINKING_SAMPLER if args.thinking else SAMPLER
    if args.greedy:
        args.sampler = dict(args.sampler, temp=0.0)
    return args


def run_engine(engine, kind, rep, children, pi, out, meta, recorder, squeeze):
    """One round on one engine: start it, run every child or replay every
    trace, then stop it."""
    print(f"\nround {rep}, {engine.name}: loading, card {card_line()}")
    loaded, idle_gib = time.perf_counter(), used_gib()
    engine.start(out / f"{kind}-rep{rep}-server.log")
    tasks = []
    try:
        warm_request()
        print(f"{engine.name}: ready in {time.perf_counter() - loaded:.0f} s, card {card_line()}")
        # An engine that fell back to the CPU takes no card memory.
        if used_gib() - idle_gib < 1.0:
            sys.exit(f"{engine.name} is up but took no card memory; it is not on the GPU")
        meta.setdefault(engine.name, engine.describe(engine.text()))
        scratch = out / f"{kind}-rep{rep}"
        if not engine.args.replay:
            copy_children(scratch, children)
        for child in children:
            workdir = scratch / child
            before = engine.offset()
            if engine.args.replay:
                wall, prompts = replay_trace(child, squeeze)
                exited, done, turns = "ok", True, len(prompts)
                child = Path(child).name.removesuffix(".requests.jsonl")
            else:
                recorder.path = out / f"{kind}-rep{rep}-{child}.requests.jsonl"
                wall, exited, done, turns, prompts = run_agent(pi, workdir, out / f"{kind}-rep{rep}-{child}.jsonl", engine.args.timeout, engine.args.thinking)
                recorder.path = None
            squeezed_at, squeeze_ended = squeeze.release()
            time.sleep(0.5)  # the last request's summary line
            text = engine.text(before)
            requests = engine.requests(text)
            if len(requests) == len(prompts):
                # Full prompt length from the agent, evaluated share from
                # the engine's log.
                for r, total in zip(requests, prompts):
                    fresh = r.prompt - r.reused
                    r.prompt, r.reused = max(total, fresh), max(total - fresh, 0)
            else:
                print(f"  {len(requests)} requests logged against {len(prompts)} agent turns; prompt totals are the engine's")
            task = Task(
                engine=engine.name,
                child=child,
                rep=rep,
                wall_secs=wall,
                exited=exited,
                said_done=done,
                check="none" if engine.args.replay else run_check(child, workdir),
                turns=turns,
                requests=[asdict(r) for r in requests],
                experts=phobos_experts(text) if kind == "phobos" else None,
                squeezed_at=squeezed_at,
                squeeze_ended=squeeze_ended,
                squeeze=squeeze.note if squeezed_at else "",
                resizes=phobos_resizes(text) if kind == "phobos" else [],
            )
            tasks.append(task)
            s = task.totals()
            print(
                f"  {child}: {wall:.1f} s, {exited}, DONE {done}, check {task.check},"
                f" {s['requests']} requests, pp {s['pp']:.1f} tg {s['tg']:.1f} t/s"
            )
    finally:
        squeeze.release()
        engine.stop()
    return tasks


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    kinds = {"phobos": Phobos, "llama": Llama, "llama.cpp": Llama}
    order = [e.strip() for e in args.engines.split(",") if e.strip()]
    if unknown := [e for e in order if e not in kinds]:
        sys.exit(f"unknown engine {unknown}; pick from phobos, llama")
    if args.replay:
        children, pi = [str(Path(t).resolve()) for t in args.replay], None
    else:
        children = args.children or sorted(d.name for d in (ROOT / "bench").iterdir() if d.is_dir())
        pi = shutil.which("pi")
        if not pi:
            sys.exit("pi is not on PATH")

    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="phobos-agent-bench-"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"results in {out}")
    preflight(args)
    if "phobos" in order and not args.no_build:
        build_phobos()

    if args.squeeze_until and args.squeeze_until <= args.squeeze_at:
        sys.exit("--squeeze-until has to come after --squeeze-at")
    squeeze = Squeeze(args.squeeze, args.squeeze_at, args.squeeze_until, args.squeeze_touch_ms)
    recorder = None if args.replay else Recorder(squeeze)
    tasks, meta = [], {}
    for rep in range(1, args.reps + 1):
        # Each engine starts fresh every round. The order reverses on even
        # rounds so neither always goes first.
        for kind in order if rep % 2 else order[::-1]:
            tasks += run_engine(kinds[kind](args), kind, rep, children, pi, out, meta, recorder, squeeze)
            (out / "results.json").write_text(json.dumps({"meta": meta, "tasks": [asdict(t) for t in tasks]}, indent=1))

    report(tasks, meta)
    print(f"\nlogs and results in {out}")


if __name__ == "__main__":
    main()
