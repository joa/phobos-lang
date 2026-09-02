// IQ1_S matvec: decodes straight from the raw block bytes (see q2k.rs) plus
// a grid lookup via IQ1S_GRID. Byte reads go through (b + 256) % 256 first
// to avoid truncating-division mismatches on signed bytes.
//
// Per-lane quantities are substituted as text at each use rather than bound
// to a name, since a name bound to a computed tile is released back to the
// shared-memory pool on first use. `qb`, `d`, `grid`, `iota` are the
// exception: they're views/staged buffers the pool doesn't own.

use super::qgemm::IQ1_GRID4_LEN;
use super::quant::qdot_i8_cta;

use std::fmt::Write as _;

/// Output columns per CTA.
pub(crate) const IQ1S_TN: usize = 8;

/// Output tile for the dp4a variant, which gives a warp several columns and
/// needs a wider tile than `IQ1S_TN` to keep the CTA busy.
pub(crate) const IQ1S_I8_TN: usize = 64;

/// The tile for an `n` that 64 does not divide. A warp then takes two
/// columns rather than eight, which is slower but still well ahead of the
/// float path it would otherwise fall back to.
pub(crate) const IQ1S_I8_NARROW_TN: usize = 16;

const LANES: usize = 32;
const LANE: usize = 8;
const BLOCK_BYTES: usize = 48;
const QS_OFF: usize = 0;
const QH_OFF: usize = 32;
const DELTA: f32 = 0.125;

/// [`crate::quant::iq1s_flat_grid`]'s length: 2048 grid entries, eight `i32`
/// lanes apiece.
pub(crate) const IQ1S_GRID_LEN: usize = 2048 * 8;

/// Byte/halfword offsets for lane `is` (ib = is/4, l = is%4, matching
/// quant/iq1_s.rs::dequantize).
fn run_geometry(is: usize) -> (usize, usize, usize, usize) {
    let ib = is / 4;
    let l = is % 4;
    let qs_off = QS_OFF + 4 * ib + l;
    let qh_lo_off = QH_OFF + 2 * ib;
    let qh_hi_off = qh_lo_off + 1;
    let shift_div = 8usize.pow(l as u32);
    (qs_off, qh_lo_off, qh_hi_off, shift_div)
}

/// Emits `let decoded{is} = ...` plus its output offset within the
/// 256-wide k-block.
fn decoded_lane(is: usize) -> (usize, String) {
    let (qs_off, qh_lo_off, qh_hi_off, shift_div) = run_geometry(is);
    let out_off = is * LANE;
    let qsb = format!("((i32(qb[:, {qs_off} :+ 1]) + 256) % 256)");
    let qh = format!(
        "(((i32(qb[:, {qh_lo_off} :+ 1]) + 256) % 256) \
          + ((i32(qb[:, {qh_hi_off} :+ 1]) + 256) % 256) * 256)"
    );
    let dl = format!("(f32(d) * f32((({qh} / 4096) % 8) * 2 + 1))");
    let delta = format!(
        "({DELTA} - {twice_delta} * f32(({qh} / 32768) % 2))",
        twice_delta = 2.0 * DELTA
    );
    let base_idx = format!("({qsb} + (({qh} / {shift_div}) % 8) * 256)");
    let idx8 = format!("({base_idx} * {LANE} + iota)");
    (
        out_off,
        format!("    let decoded{is} = {dl} * (f32(gather(grid, {idx8})) + {delta})\n"),
    )
}

pub(crate) fn iq1s_matvec_src(tn: usize) -> String {
    let mut body = String::new();
    for is in 0..LANES {
        let (out_off, decode) = decoded_lane(is);
        let a_is = format!("A[pm :+ 1, kb * 256 + {out_off} :+ {LANE}]");
        let _ = writeln!(body, "{decode}    acc = acc + dot_t({a_is}, decoded{is})");
    }

    format!(
        "@launch(256)
@autotune(TN in [{tn}])
kernel iq1s_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                   D: tensor<f16>[N, NB], GRID: tensor<i32>[1, {IQ1S_GRID_LEN}],
                   IOTA: tensor<i32>[1, {LANE}], C: tensor<f32>[M, N]) {{
  let pn = program_id(0)
  let pm = program_id(1)
  let grid = GRID[0 :+ 1, :]
  let iota = IOTA[0 :+ 1, :]
  var acc: tile<f32>[1, TN] = 0.0
  for kb in range(0, NB, 1) {{
    var qb = QB[pn * TN :+ TN, kb * {BLOCK_BYTES} :+ {BLOCK_BYTES}]
    let d = D[pn * TN :+ TN, kb :+ 1]
{body}  }}
  C[pm :+ 1, pn * TN :+ TN] = acc
}}
"
    )
}

/// Same as [`iq1s_matvec_src`] for `m == 1`, fused into one `iq1s_qdot_t`
/// call. `@aligned(N = TN)` is required: `iq1s_qdot_t` assumes in-bounds
/// slices, only true when N is a multiple of tn
/// ([`crate::backend::device::DeviceBackend::project_raw`] guards this;
/// [`iq1s_matvec_src`] is the masked fallback).
pub(crate) fn iq1s_qdot_matvec_src(tn: usize) -> String {
    format!(
        "@launch(256)
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel iq1s_qdot_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                        D: tensor<f16>[N, NB], GRID: tensor<i8>[1, {IQ1S_GRID_LEN}],
                        C: tensor<f32>[M, N]) {{
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = iq1s_qdot_t(A[0 :+ 1, :], QB[pn * TN :+ TN, :],
                                         D[pn * TN :+ TN, :], GRID[0 :+ 1, :])
}}
"
    )
}

/// [`iq1s_qdot_matvec_src`] against an activation already quantized to int8
/// by the `quantize` kernel, contracted in `dp4a`. Same result, a third fewer
/// instructions a weight, and a quarter of the activation traffic.
pub(crate) fn iq1s_qdot_i8_matvec_src(tn: usize) -> String {
    let cta = qdot_i8_cta(tn);
    // Four CTAs of 256 resident: 64 registers a thread.
    let min_blocks = 1024 / cta;
    format!(
        "@launch({cta}, {min_blocks})
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel iq1s_qdot_i8_matvec(AQ: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                           QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                           GRID: tensor<i8>[1, {IQ1_GRID4_LEN}],
                           C: tensor<f32>[M, N]) {{
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = iq1s_qdot_i8_t(AQ[0 :+ 1, :], AS[0 :+ 1, :],
                                            QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :],
                                            GRID[0 :+ 1, :])
}}
"
    )
}

/// Same per-(tile, k-block) decode as [`iq1s_matvec_src`] but stored into a
/// `[K, N]` scratch instead of reduced, so a matmul over multiple rows only
/// decodes once. `transpose` matches `Backend::matmul`'s `[K, N]` layout.
pub(crate) fn iq1s_dequant_src(tn: usize) -> String {
    let mut body = String::new();
    for is in 0..LANES {
        let (out_off, decode) = decoded_lane(is);
        let _ = writeln!(
            body,
            "{decode}    SCRATCH[kb * 256 + {out_off} :+ {LANE}, pn * TN :+ TN] = transpose(decoded{is})"
        );
    }

    format!(
        "@launch(256)
@autotune(TN in [{tn}])
kernel iq1s_dequant(QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                    GRID: tensor<i32>[1, {IQ1S_GRID_LEN}], IOTA: tensor<i32>[1, {LANE}],
                    SCRATCH: tensor<f32>[K, N]) {{
  let pn = program_id(0)
  let grid = GRID[0 :+ 1, :]
  let iota = IOTA[0 :+ 1, :]
  for kb in range(0, NB, 1) {{
    var qb = QB[pn * TN :+ TN, kb * {BLOCK_BYTES} :+ {BLOCK_BYTES}]
    let d = D[pn * TN :+ TN, kb :+ 1]
{body}  }}
}}
"
    )
}

/// [`iq1s_dequant_src`]'s decode as a single `iq1s_qdecode_t` call: nothing
/// staged in shared memory, no barrier, and a thread's eight decoded weights
/// written straight to the rows they belong in. `@aligned(N = TN)` is required
/// for the reason [`iq1s_qdot_matvec_src`] needs it, and
/// [`crate::backend::device::DeviceBackend::project_raw_dense`] keeps
/// [`iq1s_dequant_src`] for a strip that is not a whole number of `TN`.
pub(crate) fn iq1s_qdecode_src(tn: usize) -> String {
    format!(
        "@launch(256)
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel iq1s_qdecode(QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                    GRID: tensor<i8>[1, {IQ1S_GRID_LEN}], SCRATCH: tensor<f32>[K, N]) {{
  let pn = program_id(0)
  SCRATCH[:, pn * TN :+ TN] = iq1s_qdecode_t(QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :],
                                             GRID[0 :+ 1, :])
}}
"
    )
}

/// Rows and columns of the fused prompt projection's output tile.
///
/// The same 128 x 64 the Q8_0 projection settled on, and for the same reason:
/// operand loads and scale arithmetic are per output element however the tiles
/// are arranged, so what pays for them is the tensor-core tiles in a warp's
/// patch, and the output tile bounds the patch.
/// The fused projection's tile, and the CTA that carries it.
///
/// `PHOBOS_QMMA_TILE=TMxTNxCTA` overrides all three together, which is how the
/// shape gets swept without a rebuild. The tile has to divide the batch, so
/// `TM` above 128 takes the expansion path at `pp128`, and the staged form
/// needs exactly one warp patch a warp.
pub(crate) fn qmma_tile() -> (usize, usize, usize) {
    let Ok(spec) = std::env::var("PHOBOS_QMMA_TILE") else {
        return (IQ1S_QMMA_TM, IQ1S_QMMA_TN, IQ1S_QMMA_CTA);
    };
    let mut parts = spec.split('x').map(str::parse::<usize>);
    match (parts.next(), parts.next(), parts.next()) {
        (Some(Ok(tm)), Some(Ok(tn)), Some(Ok(cta))) => (tm, tn, cta),
        _ => (IQ1S_QMMA_TM, IQ1S_QMMA_TN, IQ1S_QMMA_CTA),
    }
}

pub(crate) const IQ1S_QMMA_TM: usize = 128;
pub(crate) const IQ1S_QMMA_TN: usize = 64;

/// Threads the fused projection's CTA carries.
///
/// Narrower than `q8_qmma`'s because `qmma_patch` will not grow a patch past
/// the point where it leaves warps of the CTA idle, and the patch is what pays
/// for the decode: at `rm` row-tiles a weight fragment is decoded once and fed
/// to `rm` tensor instructions. Four warps hold it to `rm = 4`, two let it
/// reach 8. Emitted, per k step:
///
/// | CTA | registers | mma | global loads |
/// | ---: | ---: | ---: | ---: |
/// | 32 | 251 | 128 | 104 |
/// | **64** | **251** | **128** | **104** |
/// | 128 | 168 | 64 | 92 |
/// | 256 | 101 | 32 | 52 |
///
/// 1.23 tensor instructions a load against 0.70, and no spill either way.
pub(crate) const IQ1S_QMMA_CTA: usize = 64;

/// Twice [`IQ1S_GRID_LEN`]: the signed table carries both foldings of every
/// entry so the group's sign bit is an index bit rather than arithmetic. See
/// `crate::quant::iq1s_signed_grid`.
pub(crate) const IQ1S_SIGNED_GRID_LEN: usize = 2 * IQ1S_GRID_LEN;

/// IQ1_S's prompt projection, decode and contraction in one kernel.
///
/// What this replaces is `iq1s_qdecode` writing an expanded weight and
/// `matmul_tc` reading it back: 23.8 GB of traffic against the 2.3 GB the
/// weights themselves are, plus the scratch to hold it, on a card where the
/// weights are 6.37 GiB of 8. `@aligned` is required, as it is for
/// `iq1s_qdot_matvec`: `iq1s_qmma_t` assumes in-bounds slices, and `K = 256`
/// because a lane indexes the block bytes itself.
pub(crate) fn iq1s_qmma_src(block: usize, tm: usize, tn: usize) -> String {
    // The staged form decodes each column once for the whole CTA instead of
    // once per warp that needs it: pp128 117.7 against 112.4, reproducible, and
    // the same answer to four digits in `backend_check`. Chosen here rather
    // than inside codegen so the two produce different source text, which is
    // what keeps them apart in the kernel cache; a flag read during codegen
    // would collide with the entry the other one wrote.
    //
    // It needs one patch a warp, which `IQ1S_QMMA_TM`, `IQ1S_QMMA_TN` and
    // `IQ1S_QMMA_CTA` give it; the intrinsic refuses rather than guesses if a
    // future tile does not. `PHOBOS_QMMA_STAGE=0` goes back to the register
    // form.
    let intrinsic = match std::env::var("PHOBOS_QMMA_STAGE").as_deref() {
        Ok("0") => "iq1s_qmma_t",
        _ => "iq1s_qmma_staged_t",
    };
    format!(
        "@launch({block})
@autotune(TM in [{tm}], TN in [{tn}])
@aligned(M = TM, N = TN, K = 256)
kernel iq1s_qmma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                 QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                 GRID: tensor<i8>[1, {IQ1S_SIGNED_GRID_LEN}],
                 C: tensor<f32>[M, N]) {{
  let pm = program_id(0)
  let pn = program_id(1)
  C[pm * TM :+ TM, pn * TN :+ TN] = {intrinsic}(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :],
                                                GRID[0 :+ 1, :])
}}
"
    )
}

