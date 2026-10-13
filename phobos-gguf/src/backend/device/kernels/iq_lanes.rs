// The IQ formats' and PTQ1_0's decode matvecs written with lanes:
// `<fmt>_qdot_i8_t`'s mapping in `.ph`, four lanes a column, each a quarter
// of every block, the weight grouped by eight columns as the device holds it.
// A format's grid is staged into a shared tile once per CTA, which the lanes
// then read.
//
// A format names what it reads of a block, its fetch list, which the kernel
// loads at the top of each turn. Carrying it two blocks ahead in registers,
// as the intrinsics' pipeline does, measured 13 to 24% slower.
//
// A grid entry's eight magnitudes come from the format's packed grid. Its
// eight sign bits become two words of 0/-1 byte masks, `nibble * 0x204081`
// laying a nibble's bits seven apart and `0x01010101` keeping one a byte,
// and apply on the packed word as `(g ^ m) + (m & 0x01010101)`; a grid holds
// no zero, so no byte carries. The 7-bit sign indices of IQ2_XXS, IQ2_XS
// and IQ3_XXS leave the eighth sign to parity. Every octet is then two
// `dp4a` against the lane's activations.

use std::fmt::Write as _;

use super::{IQ1_GRID4_LEN, IQ2S_GRID_LEN, IQ2XS_GRID_LEN, IQ2XXS_GRID_LEN, IQ3S_GRID_LEN, IQ3XXS_GRID_LEN};

/// `0x0101_0101`: one bit of every byte of a word.
const LOW_BITS: u32 = 0x0101_0101;
/// `0x0020_4081`: a nibble's four bits laid seven apart, one to a byte.
const SPREAD: u32 = 0x0020_4081;

/// A format's matvec. `bytes` is the device block size; `grid` the length
/// of the `GRID` table, staged into the shared tile `grid`; `fetch` the
/// values a block's decode reads, as `(name, expr)` with `BLK` standing for
/// the block's byte offset in the lane's group row and `BB` for its index;
/// `body` the decode, which reads the fetched names and `a`, the lane's first
/// activation, and adds into `acc`.
fn lanes_matvec(name: &str, bytes: usize, grid: Option<usize>, fetch: &[(&str, String)], body: &str) -> String {
    // The table's static length proves the staging copy in bounds, which is
    // what lets it move sixteen bytes at a time.
    let (param, stage) = match grid {
        Some(len) => (
            format!("GRID: tensor<i8>[1, {len}], "),
            format!("  var grid = GRID[0 :+ 1, 0 :+ {len}]\n"),
        ),
        None => (String::new(), String::new()),
    };
    let gw = bytes * 8;
    let at = |bb: &str, expr: &str| {
        expr.replace("BLK", &format!("({bb}) * {gw} + c8 * {bytes}"))
            .replace("BB", &format!("({bb})"))
    };
    let mut take = String::new();
    for (v, expr) in fetch {
        let _ = writeln!(take, "    let {v} = {}", at("b", expr));
    }
    format!(
        "fn {name}_block(b) {{
{body}}}

@launch(256)
@aligned(K = 256, GW = {gw}, DW = 8)
kernel {name}_qdot_lanes(AQ: tensor<i8>[1, K], AS: tensor<f32>[1, KB],
                         QB: tensor<i8>[G, GW], D: tensor<f16>[G, DW], {param}C: tensor<f32>[1, N]) {{
{stage}  let col = ctx.block.x * 64 + ctx.warp * 8 + ctx.lane / 4
  let q = ctx.lane % 4
  let g = col / 8
  let c8 = col % 8
  let nb = K / 256
  var acc: f32 = 0.0
  for b in range(0, nb, 1) {{
{take}    let a = b * 256 + 64 * q
    {name}_block(b)
  }}
  let total = sum<over ctx.lane, width 4>(acc)
  if q == 0 && col < N {{
    C[0, col] = total
  }}
}}
"
    )
}

/// The block scale from the grouped plane.
fn d_plane() -> (&'static str, String) {
    ("d", "f32(D[g, BB * 8 + c8])".to_string())
}

/// An octet into `dot`: its eight magnitudes from the grid at byte offset
/// `entry` (a `.ph` index expression), as one entry of eight or, with
/// `pair`, two of four at `entry` and `pair`; signed by the eight bits
/// `signs`, or by seven and their parity when `parity`; against activations
/// `a + at .. a + at + 8`. `tag` names its locals.
#[allow(clippy::too_many_arguments)]
fn signed_octet(
    out: &mut String,
    tag: &str,
    dot: &str,
    at: usize,
    entry: &str,
    pair: Option<&str>,
    signs: &str,
    parity: bool,
) {
    let act = format!("AQ[0, a + {base} :+ 16]", base = at / 16 * 16);
    let t = at % 16 / 4;
    let (g0, g1) = match pair {
        None => (
            format!("word(grid[0, gi{tag} :+ 8], 0)"),
            format!("word(grid[0, gi{tag} :+ 8], 1)"),
        ),
        Some(second) => {
            let _ = writeln!(out, "  let gj{tag} = {second}");
            (
                format!("word(grid[0, gi{tag} :+ 4], 0)"),
                format!("word(grid[0, gj{tag} :+ 4], 0)"),
            )
        }
    };
    let bits = if parity {
        format!("sv{tag} | ((popcount(sv{tag}) & 1) << 7)")
    } else {
        format!("sv{tag}")
    };
    let _ = writeln!(
        out,
        "  let gi{tag} = {entry}
  let sv{tag} = {signs}
  let sb{tag} = {bits}
  let m{tag}0 = (((sb{tag} & 15) * {SPREAD}) & {LOW_BITS}) * 255
  let m{tag}1 = ((((sb{tag} >> 4) & 15) * {SPREAD}) & {LOW_BITS}) * 255
  {dot} = dp4a(({g0} ^ m{tag}0) + (m{tag}0 & {LOW_BITS}), word({act}, {t}), {dot})
  {dot} = dp4a(({g1} ^ m{tag}1) + (m{tag}1 & {LOW_BITS}), word({act}, {t1}), {dot})",
        t1 = t + 1
    );
}

/// `word t` of the lane's run of `len` bytes at `BLK + off`.
fn word(off: &str, len: usize, t: usize) -> String {
    format!("word(QB[g, BLK + {off} :+ {len}], {t})")
}

/// Byte `BLK + off`, unsigned.
fn byte(off: &str) -> String {
    format!("QB[g, BLK + {off}] & 255")
}

/// IQ2_XXS: 64 bytes a block, a lane's sixteen at `16q` two groups of eight.
/// A group's first word holds four grid indices, its second four 7-bit sign
/// indices and the 4-bit scale `n` on top, the group scaled by
/// `d (2n + 1) / 8`.
pub(crate) fn iq2xxs_qdot_lanes_src() -> String {
    let fetch = [
        ("lo0", word("16 * q", 16, 0)),
        ("hi0", word("16 * q", 16, 1)),
        ("lo1", word("16 * q", 16, 2)),
        ("hi1", word("16 * q", 16, 3)),
        d_plane(),
    ];
    let mut body = String::new();
    for s in 0..2 {
        let _ = writeln!(body, "  var dot{s} = 0");
        for l in 0..4 {
            let entry = format!("((lo{s} >> {sh}) & 255) * 8", sh = 8 * l);
            let signs = format!("(hi{s} >> {sh}) & 127", sh = 7 * l);
            signed_octet(&mut body, &format!("{s}{l}"), &format!("dot{s}"), 32 * s + 8 * l, &entry, None, &signs, true);
        }
        let _ = writeln!(
            body,
            "  acc += d * 0.125 * f32(2 * ((hi{s} >> 28) & 15) + 1) * AS[0, b * 8 + 2 * q + {s}] * f32(dot{s})"
        );
    }
    lanes_matvec("iq2xxs", 64, Some(IQ2XXS_GRID_LEN), &fetch, &body)
}

/// IQ2_XS: 72 bytes a block, only eight-aligned: a lane's sixteen at `16q`
/// are two groups of four 16-bit codes, each a 9-bit grid index and a 7-bit
/// sign index, then a scale byte per group at `64 + 2q`, a nibble `n` per
/// sixteen elements scaled by `d (2n + 1) / 8`.
pub(crate) fn iq2xs_qdot_lanes_src() -> String {
    let fetch = [
        ("w0", word("16 * q", 8, 0)),
        ("w1", word("16 * q", 8, 1)),
        ("w2", word("16 * q + 8", 8, 0)),
        ("w3", word("16 * q + 8", 8, 1)),
        ("sc0", byte("64 + 2 * q")),
        ("sc1", byte("65 + 2 * q")),
        d_plane(),
    ];
    let mut body = String::new();
    for s in 0..2 {
        for h in 0..2 {
            let _ = writeln!(body, "  var dot{s}{h} = 0");
            for l in 2 * h..2 * h + 2 {
                let _ = writeln!(
                    body,
                    "  let u{s}{l} = (w{w} >> {sh}) & 65535",
                    w = 2 * s + l / 2,
                    sh = 16 * (l % 2)
                );
                let entry = format!("(u{s}{l} & 511) * 8");
                let signs = format!("(u{s}{l} >> 9) & 127");
                signed_octet(&mut body, &format!("{s}{l}"), &format!("dot{s}{h}"), 32 * s + 8 * l, &entry, None, &signs, true);
            }
            let _ = writeln!(
                body,
                "  acc += d * 0.125 * f32(2 * ((sc{s} >> {sh}) & 15) + 1) * AS[0, b * 8 + 2 * q + {s}] * f32(dot{s}{h})",
                sh = 4 * h
            );
        }
    }
    lanes_matvec("iq2xs", 72, Some(IQ2XS_GRID_LEN), &fetch, &body)
}

/// IQ2_S: 80 bytes a block: 32 grid-index bytes (a lane's eight at `8q`),
/// 32 sign bytes (at `32 + 8q`), a `qh` byte per group (at `64 + 2q`) whose
/// bits `2l, 2l + 1` are octet `l`'s index bits 8 and 9, and a scale byte per
/// group (at `72 + 2q`), a nibble per sixteen elements, as IQ2_XS.
pub(crate) fn iq2s_qdot_lanes_src() -> String {
    let fetch = [
        ("qs0", word("8 * q", 8, 0)),
        ("qs1", word("8 * q", 8, 1)),
        ("sg0", word("32 + 8 * q", 8, 0)),
        ("sg1", word("32 + 8 * q", 8, 1)),
        ("qh0", byte("64 + 2 * q")),
        ("qh1", byte("65 + 2 * q")),
        ("sc0", byte("72 + 2 * q")),
        ("sc1", byte("73 + 2 * q")),
        d_plane(),
    ];
    let mut body = String::new();
    for s in 0..2 {
        for h in 0..2 {
            let _ = writeln!(body, "  var dot{s}{h} = 0");
            for l in 2 * h..2 * h + 2 {
                let entry = format!(
                    "(((qs{s} >> {b}) & 255) | (((qh{s} >> {q2}) & 3) << 8)) * 8",
                    b = 8 * l,
                    q2 = 2 * l
                );
                let signs = format!("(sg{s} >> {b}) & 255", b = 8 * l);
                signed_octet(&mut body, &format!("{s}{l}"), &format!("dot{s}{h}"), 32 * s + 8 * l, &entry, None, &signs, false);
            }
            let _ = writeln!(
                body,
                "  acc += d * 0.125 * f32(2 * ((sc{s} >> {sh}) & 15) + 1) * AS[0, b * 8 + 2 * q + {s}] * f32(dot{s}{h})",
                sh = 4 * h
            );
        }
    }
    lanes_matvec("iq2s", 80, Some(IQ2S_GRID_LEN), &fetch, &body)
}

/// IQ3_XXS: 96 bytes a block: 64 grid-index bytes (a lane's sixteen at
/// `16q`, two four-byte entries an octet), then a word per group (at
/// `64 + 8q`) of four 7-bit sign indices and the scale `n` on top, the group
/// scaled by `d (2n + 1) / 4`.
pub(crate) fn iq3xxs_qdot_lanes_src() -> String {
    let fetch = [
        ("q0", word("16 * q", 16, 0)),
        ("q1", word("16 * q", 16, 1)),
        ("q2", word("16 * q", 16, 2)),
        ("q3", word("16 * q", 16, 3)),
        ("aux0", word("64 + 8 * q", 8, 0)),
        ("aux1", word("64 + 8 * q", 8, 1)),
        d_plane(),
    ];
    let mut body = String::new();
    for s in 0..2 {
        let _ = writeln!(body, "  var dot{s} = 0");
        for l in 0..4 {
            let w = format!("q{}", 2 * s + l / 2);
            let first = format!("(({w} >> {b}) & 255) * 4", b = 16 * (l % 2));
            let second = format!("(({w} >> {b}) & 255) * 4", b = 16 * (l % 2) + 8);
            let signs = format!("(aux{s} >> {sh}) & 127", sh = 7 * l);
            signed_octet(&mut body, &format!("{s}{l}"), &format!("dot{s}"), 32 * s + 8 * l, &first, Some(&second), &signs, true);
        }
        let _ = writeln!(
            body,
            "  acc += d * 0.25 * f32(2 * ((aux{s} >> 28) & 15) + 1) * AS[0, b * 8 + 2 * q + {s}] * f32(dot{s})"
        );
    }
    lanes_matvec("iq3xxs", 96, Some(IQ3XXS_GRID_LEN), &fetch, &body)
}

/// IQ3_S: 108 bytes a block, only four-aligned: 64 grid-index bytes (a
/// lane's sixteen at `16q`), a `qh` byte per group (at `64 + 2q`) whose bit
/// `i` is index byte `i`'s ninth bit, 32 sign bytes (at `72 + 8q`), and a
/// scale byte per lane (at `104 + q`), a nibble `n` per group scaled by
/// `d (2n + 1)`.
pub(crate) fn iq3s_qdot_lanes_src() -> String {
    let fetch = [
        ("q0", word("16 * q", 4, 0)),
        ("q1", word("16 * q + 4", 4, 0)),
        ("q2", word("16 * q + 8", 4, 0)),
        ("q3", word("16 * q + 12", 4, 0)),
        ("qh0", byte("64 + 2 * q")),
        ("qh1", byte("65 + 2 * q")),
        ("sg0", word("72 + 8 * q", 4, 0)),
        ("sg1", word("76 + 8 * q", 4, 0)),
        ("sc", byte("104 + q")),
        d_plane(),
    ];
    let mut body = String::new();
    for s in 0..2 {
        let _ = writeln!(body, "  var dot{s} = 0");
        for l in 0..4 {
            let w = format!("q{}", 2 * s + l / 2);
            let index = |i: usize| {
                format!(
                    "((({w} >> {b}) & 255) | (((qh{s} >> {h}) & 1) << 8)) * 4",
                    b = 8 * (2 * (l % 2) + i),
                    h = 2 * l + i
                )
            };
            let signs = format!("(sg{s} >> {b}) & 255", b = 8 * l);
            signed_octet(&mut body, &format!("{s}{l}"), &format!("dot{s}"), 32 * s + 8 * l, &index(0), Some(&index(1)), &signs, false);
        }
        let _ = writeln!(
            body,
            "  acc += d * f32(2 * ((sc >> {sh}) & 15) + 1) * AS[0, b * 8 + 2 * q + {s}] * f32(dot{s})",
            sh = 4 * s
        );
    }
    lanes_matvec("iq3s", 108, Some(IQ3S_GRID_LEN), &fetch, &body)
}

/// `IQ1`'s byte tables: byte `t` is `8 (t - 1) + 1` or, for a negative
/// delta, `8 (t - 1) - 1`, as `0x0009_01F9` and `0x0007_FFF7`.
const IQ1_PLUS: u32 = 0x0009_01F9;
const IQ1_MINUS: u32 = 0x0007_FFF7;

/// An IQ1 octet into `dot`: grid entry `idx` of the nibble grid, whose
/// eight nibbles `g + 1` pick each element's `8g +- 1` byte from the delta's
/// table with `byteperm`; against activations `a + at .. a + at + 8`.
fn ternary_octet(out: &mut String, tag: &str, dot: &str, at: usize, idx: &str, neg: &str) {
    let act = format!("AQ[0, a + {base} :+ 16]", base = at / 16 * 16);
    let t = at % 16 / 4;
    let _ = writeln!(
        out,
        "  let e{tag} = word(grid[0, ({idx}) * 4 :+ 4], 0)
  let lut{tag} = where(({neg}) == 1, {IQ1_MINUS}, {IQ1_PLUS})
  {dot} = dp4a(byteperm(lut{tag}, lut{tag}, e{tag}), word({act}, {t}), {dot})
  {dot} = dp4a(byteperm(lut{tag}, lut{tag}, e{tag} >> 16), word({act}, {t1}), {dot})",
        t1 = t + 1
    );
}

/// IQ1_S: 48 bytes a block: 32 grid-index bytes (a lane's eight at `8q`) and
/// a 16-bit `qh` per group (at `32 + 4q`): octet `l`'s index bits 8 to 10 at
/// `3l`, the scale `n` at 12 and the delta's sign at 15. A group is
/// `d (2n + 1) / 8` times its `8g +- 1` bytes.
pub(crate) fn iq1s_qdot_lanes_src() -> String {
    let fetch = [
        ("qs0", word("8 * q", 8, 0)),
        ("qs1", word("8 * q", 8, 1)),
        ("qhw", word("32 + 4 * q", 4, 0)),
        d_plane(),
    ];
    let mut body = String::new();
    for s in 0..2 {
        let _ = writeln!(
            body,
            "  let qh{s} = (qhw >> {sh}) & 65535
  var dot{s} = 0",
            sh = 16 * s
        );
        for l in 0..4 {
            let idx = format!("((qs{s} >> {b}) & 255) | (((qh{s} >> {h}) & 7) << 8)", b = 8 * l, h = 3 * l);
            ternary_octet(&mut body, &format!("{s}{l}"), &format!("dot{s}"), 32 * s + 8 * l, &idx, &format!("(qh{s} >> 15) & 1"));
        }
        let _ = writeln!(
            body,
            "  acc += d * 0.125 * f32(2 * ((qh{s} >> 12) & 7) + 1) * AS[0, b * 8 + 2 * q + {s}] * f32(dot{s})"
        );
    }
    lanes_matvec("iq1s", 48, Some(IQ1_GRID4_LEN), &fetch, &body)
}

/// IQ1_M: 56 bytes a block: 32 grid-index bytes (a lane's eight at `8q`),
/// two `qh` bytes per group (at `32 + 4q + 2s`) holding octets `2i` and
/// `2i + 1` in their low and high nibbles (index bits 8 to 10, then the
/// delta's sign), and a 16-bit scale word per lane (at `48 + 2q`) with a
/// 3-bit `n` per sixteen elements at `3 (2s + h)`.
pub(crate) fn iq1m_qdot_lanes_src() -> String {
    let fetch = [
        ("qs0", word("8 * q", 8, 0)),
        ("qs1", word("8 * q", 8, 1)),
        ("qhw", word("32 + 4 * q", 4, 0)),
        ("scw", word("48 + 4 * (q / 2)", 4, 0)),
        d_plane(),
    ];
    let mut body = String::from("  let sc = (scw >> (16 * (q % 2))) & 65535\n");
    for s in 0..2 {
        let _ = writeln!(body, "  let qh{s} = (qhw >> {sh}) & 65535", sh = 16 * s);
        for h in 0..2 {
            let _ = writeln!(body, "  var dot{s}{h} = 0");
            for l in 2 * h..2 * h + 2 {
                let nib = format!("(qh{s} >> {n})", n = 4 * l);
                let idx = format!("((qs{s} >> {b}) & 255) | (({nib} & 7) << 8)", b = 8 * l);
                ternary_octet(&mut body, &format!("{s}{l}"), &format!("dot{s}{h}"), 32 * s + 8 * l, &idx, &format!("({nib} >> 3) & 1"));
            }
            let _ = writeln!(
                body,
                "  acc += d * 0.125 * f32(2 * ((sc >> {sh}) & 7) + 1) * AS[0, b * 8 + 2 * q + {s}] * f32(dot{s}{h})",
                sh = 3 * (2 * s + h)
            );
        }
    }
    lanes_matvec("iq1m", 56, Some(IQ1_GRID4_LEN), &fetch, &body)
}

/// One step of PTQ1_0's trit decode on two lane pairs `lo{t}` and `hi{t}`,
/// each 16-bit lane some byte times a power of three mod 256: times three,
/// the high bytes are the next trit of each, and the low bytes go on.
fn trit_step(out: &mut String, t: &str, plane: &str) {
    let _ = writeln!(
        out,
        "  let lo3{plane} = lo{t} * 3
  let hi3{plane} = hi{t} * 3
  let p{plane} = byteperm(lo3{plane}, hi3{plane}, 30001)
  lo{t} = lo3{plane} & 16711935
  hi{t} = hi3{plane} & 16711935"
    );
}

/// PTQ1_0: 56 bytes a block. Quarter `q` owns the three words at `12q`, the
/// tail byte at `48 + q` and the `f16` scale at `52 + 2 (q / 2)`. Trit `n` of
/// byte `j` of word `w` is weight `4 (5w + n) + j` of the quarter, so one
/// trit plane of a word is a `dp4a` operand; the tail's four trits are
/// weights 60..64. Trits are 0..2 for weights `d (t - 1)`, the 1 coming off
/// as the run's activation sum.
pub(crate) fn ptq1_qdot_lanes_src() -> String {
    let fetch = [
        ("w0", word("12 * q", 4, 0)),
        ("w1", word("12 * q + 4", 4, 0)),
        ("w2", word("12 * q + 8", 4, 0)),
        ("tb", byte("48 + q")),
        ("sw", word("52", 4, 0)),
    ];
    let mut body = String::from("  let d = f32(f16bits((sw >> (16 * (q / 2))) & 65535))\n");
    for w in 0..3 {
        let _ = writeln!(
            body,
            "  var lo{w} = byteperm(w{w}, 0, 16704)
  var hi{w} = byteperm(w{w}, 0, 17218)"
        );
        for n in 0..5 {
            trit_step(&mut body, &w.to_string(), &format!("{}", 5 * w + n));
        }
    }
    let _ = writeln!(
        body,
        "  var lo3 = (tb & 255) | (((tb * 3) & 255) << 16)
  var hi3 = ((tb * 9) & 255) | (((tb * 27) & 255) << 16)"
    );
    trit_step(&mut body, "3", "15");
    for s in 0..2 {
        let _ = writeln!(body, "  var dot{s} = 0");
        for p in 8 * s..8 * s + 8 {
            let _ = writeln!(
                body,
                "  dot{s} = dp4a(p{p}, word(AQ[0, a + {at} :+ 16], {t}), dot{s})",
                at = 16 * (p / 4),
                t = p % 4
            );
        }
        let _ = writeln!(
            body,
            "  let run{s} = sum(i32(AQ[0, a + {at} :+ 16])) + sum(i32(AQ[0, a + {at1} :+ 16]))
  acc += d * AS[0, b * 8 + 2 * q + {s}] * f32(dot{s} - run{s})",
            at = 32 * s,
            at1 = 32 * s + 16
        );
    }
    lanes_matvec("ptq1", 56, None, &fetch, &body)
}
