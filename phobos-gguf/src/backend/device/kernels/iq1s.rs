// IQ1_S matvec. Decodes straight from the raw block bytes (see q2k.rs) plus
// a grid lookup. Fields are read with `>>` and `&`; a whole byte used as an
// index is `i32(b) & 255`.
//
// Per-lane quantities are substituted as text at each use rather than bound
// to a name, because a name bound to a computed tile is released back to
// the shared-memory pool on first use. `qb`, `d`, `grid` and `iota` are
// exempt, since the pool does not own them.


use std::fmt::Write as _;

/// Output columns per CTA.
pub(crate) const IQ1S_TN: usize = 8;

const LANES: usize = 32;
const LANE: usize = 8;
const BLOCK_BYTES: usize = 48;
const QS_OFF: usize = 0;
const QH_OFF: usize = 32;
const DELTA: f32 = 0.125;

/// [`crate::quant::iq1s_flat_grid`]'s length: 2048 grid entries of eight
/// lanes each.
pub(crate) const IQ1S_GRID_LEN: usize = 2048 * 8;

/// Byte offsets and index shift for lane `is`, with `ib = is / 4` and
/// `l = is % 4` as in `quant/iq1_s.rs::dequantize`.
fn run_geometry(is: usize) -> (usize, usize, usize, usize) {
    let ib = is / 4;
    let l = is % 4;
    let qs_off = QS_OFF + 4 * ib + l;
    let qh_lo_off = QH_OFF + 2 * ib;
    let qh_hi_off = qh_lo_off + 1;
    (qs_off, qh_lo_off, qh_hi_off, 3 * l)
}

/// Emits `let decoded{is} = ...` and returns it with its offset in the
/// 256-wide k-block.
fn decoded_lane(is: usize) -> (usize, String) {
    let (qs_off, qh_lo_off, qh_hi_off, shift) = run_geometry(is);
    let out_off = is * LANE;
    let qsb = format!("(i32(qb[:, {qs_off} :+ 1]) & 255)");
    // Bits 0 to 15 of the word are exact; the high byte's sign extension
    // lands above them.
    let qh = format!(
        "((i32(qb[:, {qh_lo_off} :+ 1]) & 255) + (i32(qb[:, {qh_hi_off} :+ 1]) << 8))"
    );
    let dl = format!("(f32(d) * f32((({qh} >> 12) & 7) * 2 + 1))");
    let delta = format!(
        "({DELTA} - {twice_delta} * f32(({qh} >> 15) & 1))",
        twice_delta = 2.0 * DELTA
    );
    let base_idx = format!("({qsb} + (({qh} >> {shift}) & 7) * 256)");
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
        let _ = writeln!(body, "{decode}    acc += dot_t({a_is}, decoded{is})");
    }

    format!(
        "@launch(256)
@autotune(TN in [{tn}])
kernel iq1s_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB],
                   D: tensor<f16>[N, NB], GRID: tensor<i32>[1, {IQ1S_GRID_LEN}],
                   IOTA: tensor<i32>[1, {LANE}], C: tensor<f32>[M, N]) {{
  let pn = ctx.block.x
  let pm = ctx.block.y
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

/// [`iq1s_matvec_src`] for `m == 1`, written with lanes; see `iq_f32.rs`.
/// [`crate::backend::device::DeviceBackend::project_raw`] ensures `N` is a
/// multiple of `tn`, and falls back to [`iq1s_matvec_src`] otherwise.
pub(crate) fn iq1s_qdot_matvec_src(tn: usize) -> String {
    super::qdot_matvec_src(super::IqFmt::Iq1s, tn)
}

/// [`iq1s_matvec_src`]'s decode, stored into a `[K, N]` scratch instead of
/// reduced, so a matmul over many rows decodes only once. The `transpose`
/// matches `Backend::matmul`'s `[K, N]` layout.
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
  let pn = ctx.block.x
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

/// [`iq1s_dequant_src`]'s decode, written with lanes; see `iq_f32.rs`.
/// [`crate::backend::device::DeviceBackend::project_raw_dense`] uses
/// [`iq1s_dequant_src`] for a strip that is not a whole number of `TN`.
pub(crate) fn iq1s_qdecode_src(tn: usize) -> String {
    super::qdecode_src(super::IqFmt::Iq1s, tn)
}

/// The fused projection's tile and CTA width, as `(TM, TN, CTA)`.
///
/// `PHOBOS_QMMA_TILE=TMxTNxCTA` overrides all three, to sweep the shape
/// without a rebuild. The tile must divide the batch, and the staged form
/// needs exactly one patch per warp.
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

/// Rows and columns of the fused prompt projection's output tile, the same
/// as the Q8_0 projection's. The output tile bounds a warp's patch of
/// tensor-core tiles, which is what amortizes operand loads and scales.
pub(crate) const IQ1S_QMMA_TM: usize = 128;
pub(crate) const IQ1S_QMMA_TN: usize = 64;

/// Threads the fused projection's CTA carries.
///
/// Narrower than `q8_qmma`'s, so each warp gets a larger patch. With `rm`
/// row-tiles in a patch, one decoded weight fragment feeds `rm` tensor
/// instructions. `qmma_patch` never grows a patch so far that warps sit
/// idle, so two warps allow `rm = 8` where four allow only 4.
pub(crate) const IQ1S_QMMA_CTA: usize = 64;

/// Twice [`IQ1S_GRID_LEN`]. The signed table holds both sign foldings of
/// every entry, so the group's sign bit is an index bit rather than
/// arithmetic. See `crate::quant::iq1s_signed_grid`.
pub(crate) const IQ1S_SIGNED_GRID_LEN: usize = 2 * IQ1S_GRID_LEN;

/// IQ1_S's prompt projection, decode and contraction in one kernel. It
/// avoids writing an expanded weight to scratch for `matmul_tc` to read.
///
/// `@aligned` is required, as for `iq1s_qdot_matvec`, because
/// `iq1s_qmma_t` assumes in-bounds slices. `K = 256` because a lane
/// indexes the block bytes itself.
pub(crate) fn iq1s_qmma_src(block: usize, tm: usize, tn: usize) -> String {
    // The staged form decodes each column once per CTA instead of once per
    // warp. The choice is made here so the two forms differ in source text,
    // which the kernel cache keys on.
    //
    // The staged form needs exactly one patch per warp, and the intrinsic
    // rejects a tile that does not give it. `PHOBOS_QMMA_STAGE=0` selects
    // the register form.
    let intrinsic = match phobos_base::env::flag_on("PHOBOS_QMMA_STAGE") {
        true => "iq1s_qmma_staged_t",
        false => "iq1s_qmma_t",
    };
    format!(
        "@launch({block})
@autotune(TM in [{tm}], TN in [{tn}])
@aligned(M = TM, N = TN, K = 256)
kernel iq1s_qmma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                 QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                 GRID: tensor<i8>[1, {IQ1S_SIGNED_GRID_LEN}],
                 C: tensor<f32>[M, N]) {{
  let pm = ctx.block.x
  let pn = ctx.block.y
  C[pm * TM :+ TM, pn * TN :+ TN] = {intrinsic}(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :],
                                                GRID[0 :+ 1, :])
}}
"
    )
}

