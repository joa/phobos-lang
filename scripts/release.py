"""Builds a Phobos release for a tag, x64, Windows and Linux.

    python scripts/release.py TAG [--models DIR] [--skip NAME]... [--jobs N] [--seed-manifest DIR] [--local] [--publish]
    python scripts/release.py toolchain [--llvm DIR] [--llvm-version V] [--ref BRANCH] [--pack-only]

For TAG, in order:

 1. Checks the tag out into a clean worktree under target/dist/TAG/src, since
    the compiler fingerprint hashes the files on disk and one uncommitted edit
    would key the cache to a build nobody ships. Pushes the tag if origin
    lacks it, which starts .github/workflows/release.yml.
 2. Builds phobos-cache and phobos-bench there and reads their fingerprint.
 3. Warms the previous release's kernel manifest under that fingerprint, so
    the recording runs after it hit instead of compiling on the GPU box one
    model at a time. The slow kernels land here, on every core.
 4. Records this tag's manifest by running every GGUF model under --models
    through phobos-bench, then warms whatever it added and prunes whatever it
    no longer asks for, so the cache ships exactly this build's kernels.
 5. Waits for CI, downloads both builds, and refuses to go on unless every
    binary carries the fingerprint the cache was warmed under.
 6. Packages each platform with the cache, a README and the license, and
    uploads a draft release (published with --publish) holding the packages
    and the manifest the next release seeds from.

--local skips CI and the upload: it builds the Windows binaries in the
worktree and packages them, which exercises everything but the other OS.

`toolchain` packs a local Windows LLVM/MLIR install into the
toolchain-llvm-VERSION release, creating it, and dispatches the workflow that
builds the Linux one. Needs gh, logged in, and the MLIR toolchain the rest of
the tree builds with.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "target"
EXE = ".exe" if os.name == "nt" else ""
BINARIES = ["phobos-cli", "phobos-bench", "phobos-cache", "phobos-compile"]
CRT_DLLS = ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"]
MANIFEST_ASSET = "kernel-manifest.tar.gz"
LLVM_VERSION = "22.1.7"

PACKAGE_README = """\
Phobos {tag}, {platform}

  phobos-cli      runs a GGUF model: a prompt, a REPL, or an OpenAI-compatible
                  server with --listen. --help lists the flags.
  phobos-bench    a GGUF model's prompt and generation throughput, as
                  llama-bench reports it.
  phobos-compile  compiles a Phobos kernel source to PTX.
  phobos-cache    lists and clears the kernel cache.

Needs an NVIDIA card of compute capability 7.5 or newer (Turing onwards) and
a driver from the CUDA 13 line (R580 or newer): the kernels are PTX ISA 9.0,
which older drivers refuse to load.

kernel-cache/ holds every kernel the supported models compile, for sm_75,
sm_80, sm_86, sm_89, sm_90 and sm_120, so a first launch loads instead of
compiling for minutes. Keep it beside the binaries. Kernels it lacks are
compiled on first use and kept in ~/.phobos/kernel-cache, which
`phobos-cache clear` empties; `phobos-cache list` shows what is there.

Compiler fingerprint {fingerprint}.
"""


# The shell's PHOBOS_* never reach a step: a leftover cache epoch would key
# the shipped cache and every fingerprint.txt alike, pass the check, and miss
# on every user's machine.
BASE_ENV = {k: v for k, v in os.environ.items() if not k.startswith("PHOBOS_")} | {"CARGO_TARGET_DIR": str(TARGET)}


def run(cmd, cwd=ROOT, env=BASE_ENV, check=True, capture=False):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    done = subprocess.run(
        [str(c) for c in cmd], cwd=cwd, env=env, check=False, text=True, capture_output=capture
    )
    if check and done.returncode != 0:
        if capture:
            sys.stderr.write(done.stdout + done.stderr)
        sys.exit(f"failed ({done.returncode}): {cmd[0]}")
    return done


def output(cmd, cwd=ROOT):
    return run(cmd, cwd=cwd, capture=True).stdout.strip()


def fingerprint_of(version_line):
    """The fingerprint in `phobos-X 0.1.0 (compiler FINGERPRINT)`."""
    found = re.search(r"\(compiler ([0-9a-f]+)\)", version_line)
    if not found:
        sys.exit(f"no compiler fingerprint in {version_line!r}")
    return found.group(1)


def worktree(tag, dist):
    """A clean checkout of the tag, reused when it is already there."""
    src = dist / "src"
    commit = output(["git", "rev-parse", f"{tag}^{{commit}}"])
    if src.is_dir():
        if output(["git", "rev-parse", "HEAD"], cwd=src) == commit and not output(
            ["git", "status", "--porcelain"], cwd=src
        ):
            return src, commit
        run(["git", "worktree", "remove", "--force", src])
    run(["git", "worktree", "add", "--detach", src, tag])
    return src, commit


def build(src, binaries):
    cmd = ["cargo", "build", "--release", "--locked"]
    for bin in binaries:
        cmd += ["-p", bin]
    features = [f"{b}/cuda" for b in binaries if b in ("phobos-cli", "phobos-bench")]
    if features:
        cmd += ["--features", ",".join(features)]
    run(cmd, cwd=src)


def exe(name):
    return TARGET / "release" / f"{name}{EXE}"


def seed_manifest(tag, dist, explicit, online):
    """--seed-manifest, or the manifest the previous release shipped, or the
    one a local session last recorded; None for the very first release."""
    if explicit:
        return explicit
    releases = []
    if online:
        releases = json.loads(output(["gh", "release", "list", "--json", "tagName,isDraft", "--limit", "50"]))
    for release in releases:
        name = release["tagName"]
        if name == tag or name.startswith("toolchain-") or release["isDraft"]:
            continue
        into = dist / "seed"
        if run(["gh", "release", "download", name, "-p", MANIFEST_ASSET, "-D", into, "--clobber"], check=False).returncode == 0:
            with tarfile.open(into / MANIFEST_ASSET) as archive:
                archive.extractall(into, filter="data")
            print(f"seeding from {name}'s manifest")
            return into / "kernel-manifest"
    local = TARGET / "kernel-manifest"
    return local if local.is_dir() else None


def warm_cache(src, dist, models, skip, jobs, seed):
    cache, manifest = dist / "kernel-cache", dist / "kernel-manifest"
    cache_tool = exe("phobos-cache")
    if seed:
        run([cache_tool, "warm", "--manifest", seed, "--dir", cache, "--jobs", jobs])
    shutil.rmtree(manifest, ignore_errors=True)
    env = BASE_ENV | {"PHOBOS_KERNEL_CACHE_DIR": str(cache)}
    recorded = run(
        [sys.executable, src / "scripts" / "record_kernels.py", "--manifest", manifest, "--models", models,
         *[arg for name in skip for arg in ("--skip", name)]],
        cwd=src, env=env, check=False,
    )
    if recorded.returncode != 0:
        print("some models did not record; their kernels compile on first use", flush=True)
    if not any(manifest.iterdir()):
        sys.exit(f"nothing recorded into {manifest}")
    run([cache_tool, "warm", "--manifest", manifest, "--dir", cache, "--jobs", jobs])
    run([cache_tool, "prune", "--manifest", manifest, "--dir", cache])
    return cache, manifest


def ensure_pushed(tag):
    if not output(["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}"]):
        run(["git", "push", "origin", f"refs/tags/{tag}"])


def ci_builds(commit, dist):
    """Waits for release.yml on `commit` and downloads its two artifacts."""
    for _ in range(60):
        runs = json.loads(output([
            "gh", "run", "list", "--workflow", "release.yml", "--commit", commit,
            "--json", "databaseId,status,conclusion", "--limit", "5",
        ]))
        if runs:
            break
        time.sleep(10)
    else:
        sys.exit(f"no release.yml run for {commit}")
    run_id = str(runs[0]["databaseId"])
    run(["gh", "run", "watch", run_id, "--exit-status", "--interval", "30"])
    into = dist / "ci"
    shutil.rmtree(into, ignore_errors=True)
    run(["gh", "run", "download", run_id, "-D", into])
    return {p.name.removeprefix("phobos-"): p for p in into.iterdir() if p.is_dir()}


def local_build(src, dist):
    """The Windows binaries from the worktree, staged as CI stages them."""
    build(src, BINARIES)
    out = dist / "local" / "windows-x64"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for bin in BINARIES:
        shutil.copy2(exe(bin), out)
    vswhere = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    vs = Path(output([vswhere, "-latest", "-property", "installationPath"]))
    crt = sorted((vs / "VC" / "Redist" / "MSVC").glob("*/x64/Microsoft.VC14*.CRT"))[-1]
    for dll in CRT_DLLS:
        shutil.copy2(crt / dll, out)
    (out / "fingerprint.txt").write_text(output([out / "phobos-cache.exe", "--version"]) + "\n")
    return {"windows-x64": out}


def package(tag, platform, built, cache, fingerprint, dist):
    """`phobos-TAG-PLATFORM.zip` or `.tar.gz` holding one top-level directory."""
    name = f"phobos-{tag}-{platform}"
    stage = dist / "pkg" / name
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for item in built.iterdir():
        if item.name != "fingerprint.txt":
            shutil.copy2(item, stage)
    shutil.copytree(cache, stage / "kernel-cache")
    shutil.copy2(ROOT / "LICENSE", stage)
    readme = PACKAGE_README.format(tag=tag, platform=platform, fingerprint=fingerprint)
    (stage / "README.txt").write_text(readme, newline="\r\n" if platform.startswith("windows") else "\n")

    if platform.startswith("windows"):
        archive = dist / f"{name}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for path in sorted(stage.rglob("*")):
                z.write(path, Path(name) / path.relative_to(stage))
    else:
        archive = dist / f"{name}.tar.gz"

        # An artifact download drops the executable bit, so it is set here.
        def executable(info):
            if Path(info.name).name in BINARIES:
                info.mode = 0o755
            return info

        with tarfile.open(archive, "w:gz") as t:
            t.add(stage, arcname=name, filter=executable)
    print(f"packaged {archive} ({archive.stat().st_size >> 20} MiB)")
    return archive


def release(args):
    tag = args.tag
    dist = TARGET / "dist" / tag
    dist.mkdir(parents=True, exist_ok=True)
    if not args.local:
        run(["gh", "auth", "status"], capture=True)
    src, commit = worktree(tag, dist)
    if not args.local:
        ensure_pushed(tag)

    build(src, ["phobos-cache", "phobos-bench"])
    fingerprint = fingerprint_of(output([exe("phobos-cache"), "--version"]))
    print(f"{tag} at {commit[:12]}, compiler {fingerprint}")

    seed = seed_manifest(tag, dist, args.seed_manifest, online=not args.local)
    cache, manifest = warm_cache(src, dist, args.models.resolve(), args.skip, args.jobs, seed)

    builds = local_build(src, dist) if args.local else ci_builds(commit, dist)
    for platform, built in builds.items():
        theirs = fingerprint_of((built / "fingerprint.txt").read_text())
        if theirs != fingerprint:
            sys.exit(f"{platform} was built under compiler {theirs}, the cache under {fingerprint}")

    assets = [package(tag, p, b, cache, fingerprint, dist) for p, b in sorted(builds.items())]
    manifest_archive = dist / MANIFEST_ASSET
    with tarfile.open(manifest_archive, "w:gz") as t:
        t.add(manifest, arcname="kernel-manifest")
    assets.append(manifest_archive)

    if args.local:
        print("local build: nothing uploaded")
        for asset in assets:
            print(f"  {asset}")
        return
    exists = run(["gh", "release", "view", tag], check=False, capture=True).returncode == 0
    if exists:
        run(["gh", "release", "upload", tag, *assets, "--clobber"])
    else:
        flags = [] if args.publish else ["--draft"]
        run(["gh", "release", "create", tag, *assets, *flags, "--title", tag, "--verify-tag", "--generate-notes"])


def toolchain(args):
    """Packs the Windows LLVM/MLIR install CI links against and starts the
    Linux build: `lib/` without clang's libraries, `include/`, and the four
    tools the build scripts call."""
    llvm = args.llvm.resolve()
    version = args.llvm_version
    tag = f"toolchain-llvm-{version}"
    archive = TARGET / "dist" / f"llvm-{version}-windows-x64.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    tools = ["llvm-config.exe", "libclang.dll", "mlir-tblgen.exe", "llvm-tblgen.exe"]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for top in ("include", "lib"):
            for path in sorted((llvm / top).rglob("*")):
                if path.is_file() and not (top == "lib" and path.parent == llvm / "lib" and path.name.startswith("clang")):
                    z.write(path, Path("llvm-install") / path.relative_to(llvm))
        for tool in tools:
            z.write(llvm / "bin" / tool, Path("llvm-install") / "bin" / tool)
    print(f"packed {archive} ({archive.stat().st_size >> 20} MiB)")
    if args.pack_only:
        return

    if run(["gh", "release", "view", tag], check=False, capture=True).returncode != 0:
        notes = f"LLVM and MLIR {version}, static, for the release workflow to link against. Not a Phobos release."
        run(["gh", "release", "create", tag, "--prerelease", "--title", tag, "--notes", notes])
    run(["gh", "release", "upload", tag, archive, "--clobber"])
    run(["gh", "workflow", "run", "toolchain.yml", "--ref", args.ref, "-f", f"llvm_version={version}"])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "toolchain":
        ap = argparse.ArgumentParser(prog="release.py toolchain")
        ap.add_argument("--llvm", type=Path, default=Path(os.environ.get("LLVM_SYS_221_PREFIX", "")))
        ap.add_argument("--llvm-version", default=LLVM_VERSION)
        ap.add_argument("--ref", default="main", help="the branch holding toolchain.yml")
        ap.add_argument("--pack-only", action="store_true", help="pack the archive, upload nothing")
        toolchain(ap.parse_args(sys.argv[2:]))
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag")
    ap.add_argument("--models", type=Path, default=ROOT / "models", help="the GGUF files to record")
    ap.add_argument("--skip", action="append", default=[], help="leave out models whose name holds this")
    ap.add_argument("--jobs", default=str(max(1, (os.cpu_count() or 2) // 2)), help="concurrent kernel compiles")
    ap.add_argument("--seed-manifest", type=Path, help="warm this manifest first instead of the last release's")
    ap.add_argument("--local", action="store_true", help="build Windows here, skip CI and the upload")
    ap.add_argument("--publish", action="store_true", help="publish the release instead of leaving a draft")
    release(ap.parse_args())


if __name__ == "__main__":
    main()
