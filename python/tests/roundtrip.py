"""The front end's completeness check: every source round-trips. Each `.ph`
is printed in Python by `ph2py`, the front end writes the Phobos that Python
spells, and both are emitted under the five targets the emit sweep diffs;
the modules must be identical, and a source that fails to emit must fail on
both sides.

    python python/tests/roundtrip.py [--work DIR] [DIR...]

With no DIR, the corpus is every `.ph` git tracks and the fused chains
against their stage library. Needs no GPU; builds `ph2py` and the
`snapshot` example in release.
"""

import argparse
import filecmp
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from phobos import frontend  # noqa: E402


def tree_sources():
    out = subprocess.run(["git", "ls-files", "*.ph"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    sources = [(f"tree_{p.replace('/', '_')[:-3]}", (ROOT / p).read_text(encoding="utf-8").replace("\r", "")) for p in out.split()]
    fuse = ROOT / "phobos-gguf/src/backend/fuse"
    lib = (fuse / "lib.ph").read_text(encoding="utf-8").replace("\r", "")
    for chain in sorted((fuse / "chains").glob("*.ph")):
        sources.append((f"chain_{chain.stem}", lib + "\n" + chain.read_text(encoding="utf-8").replace("\r", "")))
    return sources


def dir_sources(i, d):
    return [(f"d{i}_{p.stem}", p.read_text(encoding="utf-8").replace("\r", "")) for p in sorted(Path(d).rglob("*.ph"))]


def cargo_example(name, *args):
    return subprocess.run(["cargo", "run", "-q", "--release", "-p", "phobos-lang", "--example", name, "--", *args],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8")


def cargo_bin(name, *args):
    return subprocess.run(["cargo", "run", "-q", "--release", "-p", "phobos-compile", "--bin", name, "--", *args],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--work")
    args = ap.parse_args()
    work = Path(args.work or tempfile.mkdtemp(prefix="phobos-roundtrip-"))
    if work.exists():
        shutil.rmtree(work)
    original, python, spelled, emitted_a, emitted_b = (work / d for d in ("ph", "py", "spelled", "emit_ph", "emit_spelled"))
    for d in (original, python, spelled):
        d.mkdir(parents=True)

    sources = tree_sources() if not args.dirs else [s for i, d in enumerate(args.dirs) for s in dir_sources(i, d)]
    for stem, text in sources:
        (original / f"{stem}.ph").write_text(text, encoding="utf-8", newline="\n")

    printed = cargo_bin("ph2py", "--out", str(python), str(original))
    # One line a source it cannot spell, its first error, starting with the path.
    unspelled = [line for line in printed.stdout.splitlines() if line.startswith(str(original))]
    if printed.returncode != 0:
        print(printed.stderr)
        sys.exit(2)

    untranslated = []
    for py in sorted(python.glob("*.py")):
        text = py.read_text(encoding="utf-8")
        try:
            program = frontend.translate(str(py), text)
        except frontend.FrontendError as e:
            untranslated.append(f"{py.stem}: {e}")
            continue
        (spelled / f"{py.stem}.ph").write_text(program.text, encoding="utf-8", newline="\n")

    for src, out in ((original, emitted_a), (spelled, emitted_b)):
        run = cargo_example("snapshot", str(out), str(src))
        if run.returncode != 0:
            print(run.stderr)
            sys.exit(2)

    stems = sorted(p.stem for p in spelled.glob("*.ph"))
    identical, differ, fail_one_side = 0, [], []
    for stem in stems:
        for a in sorted(emitted_a.glob(f"{stem}.*.mlir")) + sorted(emitted_a.glob(f"{stem}.*.err")):
            target = a.name[len(stem) + 1:].rsplit(".", 1)[0]
            mlir_b, err_b = emitted_b / f"{stem}.{target}.mlir", emitted_b / f"{stem}.{target}.err"
            if a.suffix == ".mlir":
                if not mlir_b.exists():
                    fail_one_side.append(f"{stem} {target}: emits from the .ph, fails from the .py: {err_b.read_text(encoding='utf-8').strip()[:300]}")
                elif filecmp.cmp(a, mlir_b, shallow=False):
                    identical += 1
                else:
                    differ.append(f"{stem} {target}")
            elif mlir_b.exists():
                fail_one_side.append(f"{stem} {target}: fails from the .ph, emits from the .py")
            else:
                identical += 1

    print(f"{len(sources)} sources; ph2py could not spell {len(unspelled)}, the front end could not read {len(untranslated)}")
    for line in unspelled + untranslated:
        print("  " + line)
    print(f"{identical} source-target pairs agree, {len(differ)} emit different modules, {len(fail_one_side)} fail on one side only")
    for line in differ + fail_one_side:
        print("  " + line)
    print(f"work: {work}")
    sys.exit(0 if not (unspelled or untranslated or differ or fail_one_side) else 1)


if __name__ == "__main__":
    main()
