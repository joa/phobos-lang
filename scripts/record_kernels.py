"""Records the kernels every GGUF model under models/ compiles, for phobos-cache warm.

    python scripts/record_kernels.py [--manifest DIR] [--models DIR] [--epoch FINGERPRINT] [NAME...]

Runs each GGUF file through phobos-bench with PHOBOS_KERNEL_MANIFEST set,
so the manifest holds every compile request a run makes. A kernel's source
depends on the model's shapes and, for some, on the prompt's row count and the
cache depth, so the bench shape covers a short prompt, a full 512-row batch,
a ragged remainder past it, and decode both fresh and at depth. A request is recorded on a cache hit too: pass the
fingerprint a warm cache was built under as --epoch and the runs only load.

NAME filters the models by substring; --models points at a directory other
than models/, which a release's worktree does not carry. The binary is built
under CARGO_TARGET_DIR when that is set. The manifest is chip-independent;
`phobos-cache warm --manifest DIR` then compiles it for every supported chip.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ".exe" if os.name == "nt" else ""
BENCH = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")) / "release" / f"phobos-bench{EXE}"

BENCH_SHAPE = ["-p", "7,100,512,600", "-n", "16", "-d", "0,2048", "-r", "1", "--no-warmup"]


def runs(models, names):
    jobs = [(p.name, [str(BENCH), "-m", str(p), *BENCH_SHAPE]) for p in sorted(models.glob("*.gguf"))]
    return [(name, cmd) for name, cmd in jobs if not names or any(n in name for n in names)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=ROOT / "target" / "kernel-manifest")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--epoch", help="PHOBOS_KERNEL_CACHE_EPOCH for the runs, to hit an existing cache")
    ap.add_argument("--dry-run", action="store_true", help="print the runs without building or running")
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()

    jobs = runs(args.models, args.names)
    if args.dry_run:
        for name, cmd in jobs:
            print(f"{name}: {' '.join(cmd)}")
        return 0

    build = ["cargo", "build", "--release", "--features", "cuda", "-p", "phobos-bench"]
    subprocess.run(build, cwd=ROOT, check=True)

    env = dict(os.environ, PHOBOS_KERNEL_MANIFEST=str(args.manifest.resolve()))
    if args.epoch:
        env["PHOBOS_KERNEL_CACHE_EPOCH"] = args.epoch
    args.manifest.mkdir(parents=True, exist_ok=True)

    failed = []
    for name, cmd in jobs:
        print(f"\n=== {name}", flush=True)
        if subprocess.run(cmd, cwd=ROOT, env=env).returncode != 0:
            failed.append(name)

    count = sum(1 for p in args.manifest.iterdir() if not p.name.startswith("."))
    print(f"\n{count} requests in {args.manifest}; {len(jobs) - len(failed)} of {len(jobs)} runs succeeded")
    for name in failed:
        print(f"FAILED {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
