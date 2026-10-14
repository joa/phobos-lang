// The staged prompt projections written in `.ph`. Each half of a 256-element
// block is staged: the activation tile copied, the weight's 64 columns
// decoded into an int8 tile with their per-32 scales beside it, then
// contracted 32 elements at a time by `acc += f32(dot_t(a, w)) * ws * da`,
// an accumulator the integer tensor cores keep in registers across `k`.
// The activation tile and its scales are staged each turn. A format with a
// light decode pipelines them, carried in registers across the contraction;
// a heavier one, which would spill holding them, has its stage's loads issue
// at the copy and its stores wait for the barrier, past the decode. The
// words a thread decodes are loaded a stage ahead by hand. Both int8 tiles
// have 128-byte rows, which the compiler swizzles so a fragment's eight rows
// fall in different banks.

use std::fmt::Write as _;

use super::iq_lanes::{LOW_BITS, SPREAD};
use super::{QGEMM_CTA, QGEMM_TM, QGEMM_TN, qgemm_tables};
use crate::quant::Quant;

// A stage is the 64 columns' four 32-element groups, one a thread.
const _: () = assert!(QGEMM_TN * 4 == QGEMM_CTA);

/// A format's decode: the words it reads from the block, as `(name, expr)`
/// with `BO` for the block's byte offset in the thread's row and `GG` for
/// its group, and the statements that write the group from them. The words
/// in `fetch` are loaded a stage ahead, those in `now` in their own stage.
/// `pipelined` carries the next activation stage across the contraction.
/// With `halves`, each 16 elements of a group has its own scale, row
/// `2 gl + h` of `ws`.
struct Decode {
    fetch: Vec<(&'static str, String)>,
    now: Vec<(&'static str, String)>,
    body: String,
    pipelined: bool,
    halves: bool,
}

/// A format's `.ph` staged projection, for the formats written so far, whole
/// or as one slice of a split `k`. None on a chip it has not been measured
/// on, which keeps the format's `_qgemm_t` intrinsic.
pub(crate) fn qgemm_ph_src(quant: Quant, split: Option<Split>) -> Option<String> {
    if !measured_on(phobos_kernels::compile::chip()) {
        return None;
    }
    match quant {
        Quant::IQ1_S => Some(staged(quant, "iq1s", 48, &iq1s_decode(), split)),
        Quant::IQ2_XXS => Some(staged(quant, "iq2xxs", 64, &iq2xxs_decode(), split)),
        Quant::IQ2_XS => Some(staged(quant, "iq2xs", 72, &iq2xs_decode(), split)),
        Quant::IQ2_S => Some(staged(quant, "iq2s", 80, &iq2s_decode(), split)),
        Quant::IQ3_XXS => Some(staged(quant, "iq3xxs", 96, &iq3xxs_decode(), split)),
        Quant::IQ3_S => Some(staged(quant, "iq3s", 108, &iq3s_decode(), split)),
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
        now: Vec::new(),
        pipelined: true,
        halves: false,
        body: format!(
            "    ws[gl, jj] = d * 0.125 * f32(2 * ((hi >> 28) & 15) + 1)
    for<unroll 4> l in range(0, 4, 1) {{
      let gi = ((lo >> (8 * l)) & 255) * 8
      let sv = (hi >> (7 * l)) & 127
      let sb = sv | ((popcount(sv) & 1) << 7)
{octet}    }}
",
            octet = signed_octet(
                "w[jj, 32 * gl + 8 * l :+ 8]",
                "word(grid[0, gi :+ 8], 0)",
                "word(grid[0, gi :+ 8], 1)",
                ""
            )
        ),
    }
}

/// IQ3_XXS: a 32-element group is eight grid indices, each a four-byte
/// entry, and a word at byte `64 + 4g` of four 7-bit sign indices and a
/// 4-bit scale `s` on top, weighting the group by `d (2s + 1) / 4`.
fn iq3xxs_decode() -> Decode {
    let mut body = "    ws[gl, jj] = d * 0.25 * f32(2 * ((aux >> 28) & 15) + 1)\n".to_string();
    for l in 0..4 {
        // Octet `l` is grid indices `2l` and `2l + 1`, in `lo` for the first
        // two octets and `hi` for the last two.
        let (q, at) = (if l < 2 { "lo" } else { "hi" }, 16 * (l % 2));
        let _ = write!(
            body,
            "    let g{l}a = (({q} >> {at}) & 255) * 4
    let g{l}b = (({q} >> {b}) & 255) * 4
    let sv{l} = (aux >> {sh}) & 127
    let sb{l} = sv{l} | ((popcount(sv{l}) & 1) << 7)
{octet}",
            b = at + 8,
            sh = 7 * l,
            octet = signed_octet(
                &format!("w[jj, 32 * gl + {o} :+ 8]", o = 8 * l),
                &format!("word(grid[0, g{l}a :+ 4], 0)"),
                &format!("word(grid[0, g{l}b :+ 4], 0)"),
                &l.to_string(),
            )
            .replace("      ", "    ")
        );
    }
    Decode {
        fetch: vec![
            ("lo", "word(QB[jr, BO + 8 * GG :+ 8], 0)".into()),
            ("hi", "word(QB[jr, BO + 8 * GG :+ 8], 1)".into()),
        ],
        now: vec![("aux", "word(QB[jr, BO + 64 + 4 * GG :+ 4], 0)".into())],
        body,
        pipelined: false,
        halves: false,
    }
}

/// IQ1's byte tables for `byteperm`: byte `c` of a two-bit code `c` (0, +1
/// or, as 3, -1) is `8c + 1`, or `8c - 1` under a negative delta.
const IQ1_PLUS: u32 = 0xF900_0901;
const IQ1_MINUS: u32 = 0xF700_07FF;

/// IQ1_S: a 32-element group is four low bytes of 11-bit grid indices at
/// `4g` and a halfword at `32 + 2g` holding their high bits three at a time,
/// a 3-bit scale `s` and the delta's sign on top, weighting the group by
/// `d (2s + 1) / 8`. A grid entry is eight two-bit codes, spread into the
/// nibbles `byteperm` picks each element's `8c +- 1` byte with.
fn iq1s_decode() -> Decode {
    let mut body = format!(
        "    ws[gl, jj] = d * 0.125 * f32(2 * ((qh >> 12) & 7) + 1)
    let lut = where(((qh >> 15) & 1) == 1, {IQ1_MINUS}, {IQ1_PLUS})
"
    );
    for l in 0..4 {
        let _ = write!(
            body,
            "    let gi{l} = (((qs >> {b}) & 255) | (((qh >> {h}) & 7) << 8)) * 2
    let e{l} = (elem(grid[0, gi{l} :+ 2], 0) & 255) | ((elem(grid[0, gi{l} :+ 2], 1) & 255) << 8)
    let x{l}a = (e{l} | (e{l} << 8)) & {S8}
    let x{l}b = (x{l}a | (x{l}a << 4)) & {S4}
    let x{l} = (x{l}b | (x{l}b << 2)) & {S2}
    store(w[jj, 32 * gl + {o} :+ 8], byteperm(lut, lut, x{l}), byteperm(lut, lut, x{l} >> 16))
",
            b = 8 * l,
            h = 3 * l,
            o = 8 * l,
            S8 = 0x00FF_00FFu32,
            S4 = 0x0F0F_0F0Fu32,
            S2 = 0x3333_3333u32,
        );
    }
    Decode {
        fetch: vec![
            ("qs", "word(QB[jr, BO + 4 * GG :+ 4], 0)".into()),
            ("ql", "elem(QB[jr, BO + 32 + 2 * GG :+ 2], 0)".into()),
            ("qhh", "elem(QB[jr, BO + 32 + 2 * GG :+ 2], 1)".into()),
        ],
        now: Vec::new(),
        body: format!("    let qh = (ql & 255) | ((qhh & 255) << 8)\n{body}"),
        pipelined: false,
        halves: false,
    }
}

/// IQ3_S: a 32-element group is eight low bytes of 9-bit grid indices at
/// `8g`, each a four-byte entry, their high bits in byte `64 + g`, four sign
/// bytes at `72 + 4g`, and nibble `g % 2` of byte `104 + g / 2` a scale `s`
/// weighting the group by `d (2s + 1)`. A block is only four-aligned, so the
/// index bytes come as two words.
fn iq3s_decode() -> Decode {
    let mut body = "    ws[gl, jj] = d * f32(2 * ((sc >> (4 * (gl % 2))) & 15) + 1)\n".to_string();
    for l in 0..4 {
        let q = if l < 2 { "lo" } else { "hi" };
        let at = 16 * (l % 2);
        let _ = write!(
            body,
            "    let g{l}a = ((({q} >> {at}) & 255) | (((qh >> {h}) & 1) << 8)) * 4
    let g{l}b = ((({q} >> {b}) & 255) | (((qh >> {h1}) & 1) << 8)) * 4
    let sb{l} = (sg >> {sh}) & 255
{octet}",
            b = at + 8,
            h = 2 * l,
            h1 = 2 * l + 1,
            sh = 8 * l,
            octet = signed_octet(
                &format!("w[jj, 32 * gl + {o} :+ 8]", o = 8 * l),
                &format!("word(grid[0, g{l}a :+ 4], 0)"),
                &format!("word(grid[0, g{l}b :+ 4], 0)"),
                &l.to_string(),
            )
            .replace("      ", "    ")
        );
    }
    Decode {
        fetch: vec![
            ("lo", "word(QB[jr, BO + 8 * GG :+ 4], 0)".into()),
            ("hi", "word(QB[jr, BO + 8 * GG + 4 :+ 4], 0)".into()),
            ("qh", "elem(QB[jr, BO + 64 + GG :+ 1], 0)".into()),
            ("sg", "word(QB[jr, BO + 72 + 4 * GG :+ 4], 0)".into()),
            ("sc", "elem(QB[jr, BO + 104 + GG / 2 :+ 1], 0)".into()),
        ],
        now: Vec::new(),
        body,
        pipelined: false,
        halves: false,
    }
}

/// IQ2_S: a 32-element group is four low bytes of 10-bit grid indices at
/// `4g`, four sign bytes at `32 + 4g`, the indices' high bits two at a time
/// in byte `64 + g`, and a scale byte at `72 + g` whose nibble `h` weights
/// half `h` of the group by `d (2s + 1) / 8`.
fn iq2s_decode() -> Decode {
    let mut body = String::new();
    for h in 0..2 {
        let _ = writeln!(body, "    ws[2 * gl + {h}, jj] = d * 0.125 * f32(2 * ((sc >> {}) & 15) + 1)", 4 * h);
    }
    for l in 0..4 {
        let _ = write!(
            body,
            "    let gi{l} = (((qs >> {b}) & 255) | (((qh >> {q}) & 3) << 8)) * 8
    let sb{l} = (sg >> {b}) & 255
{octet}",
            b = 8 * l,
            q = 2 * l,
            octet = signed_octet(
                &format!("w[jj, 32 * gl + {o} :+ 8]", o = 8 * l),
                &format!("word(grid[0, gi{l} :+ 8], 0)"),
                &format!("word(grid[0, gi{l} :+ 8], 1)"),
                &l.to_string(),
            )
            .replace("      ", "    ")
        );
    }
    Decode {
        fetch: vec![
            ("qs", "word(QB[jr, BO + 4 * GG :+ 4], 0)".into()),
            ("sg", "word(QB[jr, BO + 32 + 4 * GG :+ 4], 0)".into()),
            ("qh", "elem(QB[jr, BO + 64 + GG :+ 1], 0)".into()),
            ("sc", "elem(QB[jr, BO + 72 + GG :+ 1], 0)".into()),
        ],
        now: Vec::new(),
        body,
        pipelined: false,
        halves: true,
    }
}

/// IQ2_XS: a 32-element group is four halfwords, each a 9-bit grid index
/// under 7 sign bits, and a scale byte at `64 + g` whose nibble `h` weights
/// half `h` of the group by `d (2s + 1) / 8`.
fn iq2xs_decode() -> Decode {
    let mut body = String::new();
    for h in 0..2 {
        let _ = writeln!(body, "    ws[2 * gl + {h}, jj] = d * 0.125 * f32(2 * ((sc >> {}) & 15) + 1)", 4 * h);
    }
    for l in 0..4 {
        let (q, at) = (if l < 2 { "lo" } else { "hi" }, 16 * (l % 2));
        let _ = write!(
            body,
            "    let q{l} = ({q} >> {at}) & 65535
    let gi{l} = (q{l} & 511) * 8
    let sv{l} = q{l} >> 9
    let sb{l} = sv{l} | ((popcount(sv{l}) & 1) << 7)
{octet}",
            octet = signed_octet(
                &format!("w[jj, 32 * gl + {o} :+ 8]", o = 8 * l),
                &format!("word(grid[0, gi{l} :+ 8], 0)"),
                &format!("word(grid[0, gi{l} :+ 8], 1)"),
                &l.to_string(),
            )
            .replace("      ", "    ")
        );
    }
    Decode {
        fetch: vec![
            ("lo", "word(QB[jr, BO + 8 * GG :+ 8], 0)".into()),
            ("hi", "word(QB[jr, BO + 8 * GG :+ 8], 1)".into()),
            ("sc", "elem(QB[jr, BO + 64 + GG :+ 1], 0)".into()),
        ],
        now: Vec::new(),
        body,
        pipelined: false,
        halves: true,
    }
}

/// Stores the octet whose magnitudes are the words `lo` and `hi`, signed by
/// the eight bits `sb{t}`, into `dst`, as the decode matvecs sign it. `t`
/// tags its names, for octets written out side by side.
fn signed_octet(dst: &str, lo: &str, hi: &str, t: &str) -> String {
    format!(
        "      let m{t}lo = (((sb{t} & 15) * {SPREAD}) & {LOW_BITS}) * 255
      let m{t}hi = ((((sb{t} >> 4) & 15) * {SPREAD}) & {LOW_BITS}) * 255
      let q{t}lo = ({lo} ^ m{t}lo) + (m{t}lo & {LOW_BITS})
      let q{t}hi = ({hi} ^ m{t}hi) + (m{t}hi & {LOW_BITS})
      store({dst}, q{t}lo, q{t}hi)
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
    for (v, expr) in &decode.now {
        let _ = writeln!(take, "    let {v} = {}", at(expr, "kh"));
    }
    // One update per scale: per group, or per half group.
    let per_group = if decode.halves { 2 } else { 1 };
    let width = 32 / per_group;
    let mut contract = String::new();
    for s in 0..4 * per_group {
        let _ = write!(
            contract,
            "    let a{s} = at[0 :+ TM, {k} :+ {width}]
    let da{s} = das[0 :+ TM, {g} :+ 1]
    let w{s} = w[0 :+ TN, {k} :+ {width}]
    let ws{s} = ws[{s} :+ 1, 0 :+ TN]
    acc += f32(dot_t(a{s}, w{s})) * ws{s} * da{s}
",
            k = width * s,
            g = s / per_group
        );
    }
    format!(
        "@launch({QGEMM_CTA}, 2)
{pipeline}@swizzle
@autotune(TM in [{QGEMM_TM}], TN in [{QGEMM_TN}])
@aligned(M = TM, N = TN, {aligned})
kernel {kernel}(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                  QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                  {params}{out}  let pm = ctx.block.x
{ids}{grid}  var w: tile<i8>[TN, 128]
  var ws: tile<f32>[{scales}, TN]
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
        scales = 4 * per_group,
        pipeline = if decode.pipelined { "@pipeline\n" } else { "" },
    )
}
