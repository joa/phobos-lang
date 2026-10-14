# Race check

`phobos_lang::race` checks a kernel's shared memory for races on the host,
with no GPU. It runs the MLIR the compiler emitted, every thread of one CTA
side by side, so it sees the barriers exactly as the membar pass kept and
elided them.

## Running it

```
cargo run --release -p phobos-lang --example race -- [-v] PATH...
```

`PATH` is a `.ph` source, which it compiles, or emitted `.mlir` as the `emit`
example prints it, or a directory of either. `PHOBOS_CHIP` and
`PHOBOS_INDEX_BITS` pick the target, as for `emit`. It exits 1 when a CTA
races. `-v` also lists the CTAs that stopped early and the findings that
are only possible or within a warp.

`cargo test -p phobos-lang` runs it over every `.ph` in the tree on sm_75 at
32 and 64 bits and sm_80 at 64 bits. The model kernels are generated in Rust;
dump them with `PHOBOS_DUMP_DIR` (see `ENV.md`) and run the example over the
dump. A change to barrier placement does both.

From Rust: `race::check_source`, `race::check_text` or `race::check_module`,
each returning one `Report` per kernel and CTA.

## What it reports

| Finding | Meaning |
|---|---|
| race | Two threads of different warps touch one shared byte between barriers, at least one writing. Fails the test. |
| within a warp | Two lanes of one warp do the same with no warp-wide op between them (a shuffle, `mma.sync`, `ldmatrix`, a wmma op). Two lanes storing through one instruction are one store. |
| possible | One side's address depended on loaded data, so the access covered its whole view. |
| divergent barrier | A barrier some threads of the CTA did not reach. Fails the test. |
| out of bounds | A shared access outside its buffer. Fails the test. |
| stopped | A trap, or control the run could not follow. Fails the test. |

## How a run is modelled

- It computes integers and memref layouts, never data. A load from a tensor
  or a tile yields nothing.
- An address built from loaded data, such as a gather from a staged table,
  covers everything its view can reach. A conflict with it is possible, not
  certain.
- `cp.async` bytes stay written, across barriers, until the issuing thread
  waits for their group.
- `ldmatrix`: the first `8 * numTiles` lanes each name one 16-byte row. A wmma
  load or store moves the whole fragment as one access of the warp.
- Dynamic tensor extents are 256, rounded up to what each `@aligned` check
  at entry divides by, so the run does not trap. Integer scalars are 4.
- Two CTAs are run, `[0, 0, 0]` and `[1, 1, 0]`, in a `[2, 2, 1]` grid.

## Known issues

- **Reports name MLIR ops, not source lines.** The emitted MLIR carries no
  locations. Source spans through the parser, the IR and MLIR are the next
  piece of work, then a `phobos-compile --check-races` flag.
- **Long loops are sampled.** A loop with uniform bounds and more than four
  turns runs its first two and last two. A hazard between turns shows between
  neighbouring ones, but one that only appears from some later turn on, say
  keyed on `kt % 4`, is missed. What the loop carries across the skipped
  turns becomes unknown.
- **Waits on data run once.** A `while` whose condition comes from loaded
  data (a flag, a grid barrier's spin) is taken to end after one pass, so a
  race only a later spin reaches is not seen.
- **Few launch shapes.** Two CTAs and one extent per dimension. A race that
  only a ragged last CTA, or another extent, reaches is not seen.
- **Within-warp findings do not fail the test.** On Volta and later the
  lanes of a warp can diverge, so a hazard between them with no warp-wide op
  between is real; it is reported, but not enforced yet.
- **Possible findings do not fail the test.** They rest on data the run
  does not have.
- **Shared-memory atomics record no access.** Two atomics never race, but an
  atomic against a plain access to the same byte is not checked.
- **Global memory is not checked**, nor races between CTAs.
- **Only the tree's `.ph` files are in `cargo test`.** The generated model
  kernels need a dump and a manual run.
- **Three codegen-test kernels race by construction**: every warp writes the
  same element of a tile. They are compile tests, and the race is correct to
  report.
- **One `nvgpu` op does not parse back.** `device_async_create_group` with no
  tokens prints in a form MLIR does not parse; `check_text` rewrites it to the
  generic form first.
