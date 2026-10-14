// The staged prompt projections written in `.ph`. Each half of a 256-element
// block is staged: the activation tile copied, the weight's 64 columns
// decoded into an int8 tile with their per-32 scales beside it, then
// contracted 32 elements at a time by `acc += f32(dot_t(a, w)) * ws * da`,
// an accumulator the integer tensor cores keep in registers across `k`.
// The activation tile and its scales are the loop's pipelined stages:
// copied asynchronously on sm_80 and on, carried in registers across the
// contraction before. The words a thread decodes are loaded a stage ahead
// the same way, by hand. Both int8 tiles have 128-byte rows, which the
// compiler swizzles so a fragment's eight rows fall in different banks.

use std::fmt::Write as _;

use super::iq_lanes::{LOW_BITS, SPREAD};
use super::{QGEMM_CTA, QGEMM_TM, QGEMM_TN, qgemm_tables};
use crate::quant::Quant;

// A stage is the 64 columns' four 32-element groups, one a thread.
const _: () = assert!(QGEMM_TN * 4 == QGEMM_CTA);

/// A format's decode: the words it reads from the block, as `(name, expr)`
/// with `BO` for the block's byte offset in the thread's row and `GG` for
/// its group, and the statements that write the group from them.
struct Decode {
    fetch: Vec<(&'static str, String)>,
    body: String,
}

/// A format's `.ph` staged projection, for the formats written so far, whole
/// or as one slice of a split `k`. None on a chip it has not been measured
/// on, which keeps the format's `_qgemm_t` intrinsic.
pub(crate) fn qgemm_ph_src(quant: Quant, split: Option<Split>) -> Option<String> {
    if !measured_on(phobos_kernels::compile::chip()) {
        return None;
    }
    match quant {
        Quant::IQ2_XXS => Some(staged(quant, "iq2xxs", 64, &iq2xxs_decode(), split)),
        _ => None,
    }
}

/// Whether the `.ph` projections are measured on `chip`: sm_75 only so far,
/// where the stages are carried in registers. From sm_80 on they copy with
/// `cp.async`, and sm_120 contracts with a half-used `m16n8k16` where the
/// intrinsics take `m16n8k32`.
fn measured_on(chip: &str) -> bool {
    chip == "sm_75"
}

/// A split projection's slice: `gn` column tiles, and a slice's `k`, its
/// activation scales, its payload bytes in a row and its blocks.
#[derive(Clone, Copy)]
pub(crate) struct Split {
    pub(crate) gn: usize,
    pub(crate) sk: usize,
    pub(crate) skb: usize,
    pub(crate) srb: usize,
    pub(crate) sb: usize,
}

/// IQ2_XXS: a 32-element group is eight bytes, four grid indices then a word
/// of four 7-bit sign indices and a 4-bit scale `s` on top, weighting the
/// group by `d (2s + 1) / 8`. The eighth sign of an octet is the parity of
/// its seven.
fn iq2xxs_decode() -> Decode {
    Decode {
        fetch: vec![
            ("lo", "word(QB[jr, BO + 8 * GG :+ 8], 0)".into()),
            ("hi", "word(QB[jr, BO + 8 * GG :+ 8], 1)".into()),
        ],
        body: format!(
            "    ws[gl, jj] = d * 0.125 * f32(2 * ((hi >> 28) & 15) + 1)
    for<unroll 4> l in range(0, 4, 1) {{
      let gi = ((lo >> (8 * l)) & 255) * 8
      let sv = (hi >> (7 * l)) & 127
      let sb = sv | ((popcount(sv) & 1) << 7)
{octet}    }}
",
            octet = signed_octet("w[jj, 32 * gl + 8 * l :+ 8]")
        ),
    }
}

/// Stores the octet at grid entry `gi` signed by the eight bits `sb` into
/// `dst`, as the decode matvecs sign it.
fn signed_octet(dst: &str) -> String {
    format!(
        "      let m0 = (((sb & 15) * {SPREAD}) & {LOW_BITS}) * 255
      let m1 = ((((sb >> 4) & 15) * {SPREAD}) & {LOW_BITS}) * 255
      let s0 = (word(grid[0, gi :+ 8], 0) ^ m0) + (m0 & {LOW_BITS})
      let s1 = (word(grid[0, gi :+ 8], 1) ^ m1) + (m1 & {LOW_BITS})
      store({dst}, s0, s1)
"
    )
}

/// The kernel around a format's `decode`. Four threads share a column `jj`,
/// so a warp reads whole runs of the grouped weight's row `jr` and stores
/// whole rows of `w`. Each stage a thread decodes its group `gl` of the
/// column's half block, with the block's scale `d`, into `w` and `ws`. The
/// words a stage reads and its scale are loaded during the stage before.
/// `bytes` is a block on the device, its scale held apart in `D`. The
/// parameters match the intrinsic's, so the launch does not change; the
/// first table, the grid, is staged into `grid`. With `split`, program
/// `(pm, ps * gn + pn)` contracts slice `ps` into column tile `ps * gn + pn`
/// of the partial `P`; a slice of the grouped weight starts at eight times
/// its block offset in the group's first row.
fn staged(quant: Quant, name: &str, bytes: usize, decode: &Decode, split: Option<Split>) -> String {
    let tables = qgemm_tables(quant).unwrap_or(&[]);
    let params: String = tables
        .iter()
        .enumerate()
        .map(|(i, len)| format!("T{i}: tensor<i8>[1, {len}], "))
        .collect();
    let grid = tables.first().map_or(String::new(), |len| format!("  var grid = T0[0 :+ 1, 0 :+ {len}]\n"));
    // Each operand's offset to the program's slice, and the slice's `k`.
    let (kernel, aligned, ids, out, k) = match split {
        None => (
            format!("{name}_qgemm"),
            "K = 256, KB = 8".to_string(),
            "  let pn = ctx.block.y\n".to_string(),
            "C: tensor<f32>[M, N]) {\n".to_string(),
            "K".to_string(),
        ),
        Some(Split { gn, sk, skb, srb, sb }) => (
            format!("{name}_qgemm_split"),
            format!("K = {sk}, KB = {skb}, RB = {srb}, NB = {sb}, NP = TN"),
            format!("  let pid = ctx.block.y\n  let ps = pid / {gn}\n  let pn = pid - ps * {gn}\n"),
            "P: tensor<f32>[M, NP]) {\n".to_string(),
            sk.to_string(),
        ),
    };
    let off = |by: usize| format!("ps * {by} + ");
    let (a_off, as_off, qb_off, d_off) = match split {
        None => Default::default(),
        Some(s) => (off(s.sk), off(s.skb), off(8 * s.srb), off(8 * s.sb)),
    };
    let store = match split {
        None => "C[pm * TM :+ TM, pn * TN :+ TN] = acc",
        Some(_) => "P[pm * TM :+ TM, pid * TN :+ TN] = acc",
    };
    // Where stage `kh` reads, written out where it is read so the offset's
    // divisor stays provable: its block's byte offset and its group.
    let at = |expr: &str, kh: &str| {
        expr.replace("BO", &format!("{qb_off}(({kh}) / 2 * 8 + c8) * {bytes}"))
            .replace("GG", &format!("(4 * (({kh}) % 2) + gl)"))
    };
    let (mut first, mut next, mut take) = (String::new(), String::new(), String::new());
    for (v, expr) in &decode.fetch {
        let _ = writeln!(first, "  var {v}n = {}", at(expr, "0"));
        let _ = writeln!(next, "      {v}n = {}", at(expr, "kh + 1"));
        let _ = writeln!(take, "    let {v} = {v}n");
    }
    let mut contract = String::new();
    for s in 0..4 {
        let _ = write!(
            contract,
            "    let a{s} = at[0 :+ TM, {k} :+ 32]
    let da{s} = das[0 :+ TM, {s} :+ 1]
    let w{s} = w[0 :+ TN, {k} :+ 32]
    let ws{s} = ws[{s} :+ 1, 0 :+ TN]
    acc += f32(dot_t(a{s}, w{s})) * ws{s} * da{s}
",
            k = 32 * s
        );
    }
    format!(
        "@launch({QGEMM_CTA}, 2)
@pipeline
@swizzle
@autotune(TM in [{QGEMM_TM}], TN in [{QGEMM_TN}])
@aligned(M = TM, N = TN, {aligned})
kernel {kernel}(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                  QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                  {params}{out}  let pm = ctx.block.x
{ids}{grid}  var w: tile<i8>[TN, 128]
  var ws: tile<f32>[4, TN]
  var acc: tile<f32>[TM, TN] = 0.0
  let jj = ctx.thread / 4
  let gl = ctx.thread % 4
  let jr = (pn * TN + jj) / 8 * 8
  let c8 = jj % 8
{first}  var dn = D[jr, {d_off}c8]
  for kh in range(0, {k} / 128, 1) {{
    var at = A[pm * TM :+ TM, {a_off}kh * 128 :+ 128]
    var das = AS[pm * TM :+ TM, {as_off}kh * 4 :+ 4]
{take}    let d = f32(dn)
    if kh + 1 < {k} / 128 {{
{next}      dn = D[jr, {d_off}(kh + 1) / 2 * 8 + c8]
    }}
{body}    barrier()
{contract}  }}
  {store}
}}
",
        body = decode.body,
    )
}
