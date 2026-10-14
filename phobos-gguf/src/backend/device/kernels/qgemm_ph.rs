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

// A stage is the 64 columns' four 32-element groups, one a thread, and a
// minimum's group sums are a whole number a thread.
const _: () = assert!(QGEMM_TN * 4 == QGEMM_CTA && (QGEMM_TM * 4).is_multiple_of(QGEMM_CTA));

/// A format's decode: the words it reads from the block, as `(name, expr)`
/// with `BO` for the block's byte offset in the thread's row and `GG` for
/// its group, and the statements that write the group from them. The words
/// in `fetch` are loaded a stage ahead, those in `now` in their own stage.
/// `pipelined` carries the next activation stage across the contraction.
/// With `halves`, each 16 elements of a group has its own scale, row
/// `2 gl + h` of `ws`. With `mins`, the decode also writes the group's
/// minimum into `mw`, which the contraction subtracts against the
/// activation's group sums.
struct Decode {
    fetch: Vec<(&'static str, String)>,
    now: Vec<(&'static str, String)>,
    body: String,
    pipelined: bool,
    halves: bool,
    mins: bool,
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
        Quant::IQ1_M => Some(staged(quant, "iq1m", 56, &iq1m_decode(), split)),
        Quant::IQ2_XXS => Some(staged(quant, "iq2xxs", 64, &iq2xxs_decode(), split)),
        Quant::IQ2_XS => Some(staged(quant, "iq2xs", 72, &iq2xs_decode(), split)),
        Quant::IQ2_S => Some(staged(quant, "iq2s", 80, &iq2s_decode(), split)),
        Quant::IQ3_XXS => Some(staged(quant, "iq3xxs", 96, &iq3xxs_decode(), split)),
        Quant::IQ3_S => Some(staged(quant, "iq3s", 108, &iq3s_decode(), split)),
        Quant::Q4_K => Some(staged(quant, "q4k", 144, &kquant_decode(false), split)),
        Quant::Q5_K => Some(staged(quant, "q5k", 176, &kquant_decode(true), split)),
        Quant::Q6_K => Some(staged(quant, "q6k", 208, &q6k_decode(), split)),
        Quant::PTQ1_0 => Some(staged(quant, "ptq1", 56, &ptq1_decode(), split)),
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
        mins: false,
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
        mins: false,
    }
}

/// Q4_K, or with `high` Q5_K: `d` and `dmin` lead the block as halves, then
/// twelve bytes pack each group's 6-bit scale `s` and minimum `m`, and group
/// `g` is nibble `g % 2` of 32 bytes at `16 + 32 (g / 2)`, or for Q5_K at
/// `48 + 32 (g / 2)` under bit `g` of the 32 bytes at 16. An element is
/// `d s q - dmin m`. Groups 0-3 keep `s` and `m` in bytes `g` and `g + 4`;
/// groups 4-7 take their low bits from byte `g + 4` and their top two from
/// those bytes' spare ones, `gl` and `gl + 4`, so a thread reads bytes `gl`,
/// `gl + 4` and `gl + 8` either way.
fn kquant_decode(high: bool) -> Decode {
    let qs = if high { 48 } else { 16 };
    let mut body = String::from(
        "    let dk = f32(f16bits((d0 & 255) | ((d1 & 255) << 8)))
    let mk = f32(f16bits((d2 & 255) | ((d3 & 255) << 8)))
    let pa = pa0 & 255
    let pb = pb0 & 255
    let pc = pc0 & 255
    let upper = kh % 2
    let sk = where(upper == 1, (pc & 15) | ((pa >> 6) << 4), pa & 63)
    let mn = where(upper == 1, (pc >> 4) | ((pb >> 6) << 4), pb & 63)
    ws[gl, jj] = dk * f32(sk)
    mw[gl, jj] = mk * f32(mn)
"
    );
    for o in 0..4 {
        for t in 0..2 {
            let hi = if high {
                format!(" | (((h{o}{t} >> (4 * upper + gl)) & {LOW1}) << 4)", LOW1 = 0x0101_0101u32)
            } else {
                String::new()
            };
            let _ = writeln!(
                body,
                "    let v{o}{t} = ((l{o}{t} >> (4 * (gl % 2))) & {LOW4}){hi}",
                LOW4 = 0x0F0F_0F0Fu32
            );
        }
        let _ = writeln!(body, "    store(w[jj, 32 * gl + {b} :+ 8], v{o}0, v{o}1)", b = 8 * o);
    }
    let names = ["00", "01", "10", "11", "20", "21", "30", "31"];
    let lows = ["l00", "l01", "l10", "l11", "l20", "l21", "l30", "l31"];
    let highs = ["h00", "h01", "h10", "h11", "h20", "h21", "h30", "h31"];
    let mut now = vec![
        ("d0", "elem(QB[jr, BO :+ 4], 0)".to_string()),
        ("d1", "elem(QB[jr, BO :+ 4], 1)".to_string()),
        ("d2", "elem(QB[jr, BO :+ 4], 2)".to_string()),
        ("d3", "elem(QB[jr, BO :+ 4], 3)".to_string()),
        ("pa0", "elem(QB[jr, BO + 4 + GG % 4 :+ 1], 0)".to_string()),
        ("pb0", "elem(QB[jr, BO + 8 + GG % 4 :+ 1], 0)".to_string()),
        ("pc0", "elem(QB[jr, BO + 12 + GG % 4 :+ 1], 0)".to_string()),
    ];
    for (i, _) in names.iter().enumerate() {
        let (o, t) = (i / 2, i % 2);
        now.push((lows[i], format!("word(QB[jr, BO + {qs} + 32 * (GG / 2) + {b} :+ 8], {t})", b = 8 * o)));
        if high {
            now.push((highs[i], format!("word(QB[jr, BO + 16 + {b} :+ 8], {t})", b = 8 * o)));
        }
    }
    Decode {
        fetch: Vec::new(),
        now,
        body,
        pipelined: false,
        halves: false,
        mins: true,
    }
}

/// PTQ1_0: quarter `q = g / 2` of a device block is three words at `12q`,
/// five base-3 trits a byte, plus the tail byte `48 + q`, and the halfword at
/// `52 + 2 (q / 2)` is its scale. Trit plane `n` of a word is four weights,
/// taken a plane a step: a byte in a 16-bit lane times 3 carries its next
/// trit into the top byte. A quarter's first half is word 0's five planes
/// and word 1's first three; its second half word 1's last two, word 2's
/// five and the tail. A trit `t` is the weight `t - 1`.
fn ptq1_decode() -> Decode {
    let mut body = String::from(
        "    let qq = 2 * (kh % 2) + gl / 2
    let lead = gl % 2 == 0
    ws[gl, jj] = f32(f16bits((sc >> (16 * (qq / 2))) & 65535))
    let outer = where(lead, w0, w2)
",
    );
    // A word's two lanes of 16-bit bytes, then five planes.
    for (name, src) in [("f", "outer"), ("m", "w1")] {
        let _ = writeln!(
            body,
            "    let {name}lo0 = byteperm({src}, 0, {LO})
    let {name}hi0 = byteperm({src}, 0, {HI})",
            LO = 0x4140,
            HI = 0x4342
        );
        for n in 0..5 {
            let _ = writeln!(
                body,
                "    let {name}a{n} = {name}lo{n} * 3
    let {name}b{n} = {name}hi{n} * 3
    let {name}{n} = byteperm({name}a{n}, {name}b{n}, {TOPS})
    let {name}lo{m} = {name}a{n} & {MASK}
    let {name}hi{m} = {name}b{n} & {MASK}",
                m = n + 1,
                TOPS = 0x7531,
                MASK = 0x00FF_00FFu32
            );
        }
    }
    let _ = writeln!(
        body,
        "    let tb = (tl >> (8 * qq)) & 255
    let ta = (tb * {P0}) & {MASK}
    let tc = (tb * {P1}) & {MASK}
    let tail = byteperm(ta * 3, tc * 3, {TOPS})",
        P0 = 0x0003_0001u32,
        P1 = 0x001B_0009u32,
        MASK = 0x00FF_00FFu32,
        TOPS = 0x7531
    );
    let first = ["f0", "f1", "f2", "f3", "f4", "m0", "m1", "m2"];
    let second = ["m3", "m4", "f0", "f1", "f2", "f3", "f4", "tail"];
    for (i, (a, b)) in first.iter().zip(second).enumerate() {
        let _ = writeln!(
            body,
            "    let t{i} = (where(lead, {a}, {b}) + {BIAS}) ^ {FLIP}",
            BIAS = 0x7F7F_7F7Fu32,
            FLIP = 0x8080_8080u32
        );
    }
    for l in 0..4 {
        let _ = writeln!(body, "    store(w[jj, 32 * gl + {o} :+ 8], t{a}, t{b})", o = 8 * l, a = 2 * l, b = 2 * l + 1);
    }
    Decode {
        fetch: vec![
            ("w0", "word(QB[jr, BO + 12 * (GG / 2) :+ 12], 0)".into()),
            ("w1", "word(QB[jr, BO + 12 * (GG / 2) :+ 12], 1)".into()),
            ("w2", "word(QB[jr, BO + 12 * (GG / 2) :+ 12], 2)".into()),
            ("tl", "word(QB[jr, BO + 48 :+ 8], 0)".into()),
            ("sc", "word(QB[jr, BO + 48 :+ 8], 1)".into()),
        ],
        now: Vec::new(),
        body,
        pipelined: false,
        halves: false,
        mins: false,
    }
}

/// Q6_K: group `g` of a block is stream `j = g % 4` of half `n = g / 4`:
/// its 6-bit values are nibble `j / 2` of the bytes at `64n + 32 (j % 2)`
/// under bits `2j` of the bytes at `128 + 32n`, less 32, and each 16 of them
/// is weighted by `d` times a signed scale byte at `192 + 8n + 2j`.
fn q6k_decode() -> Decode {
    let mut body = String::new();
    for h in 0..2 {
        let _ = writeln!(body, "    ws[2 * gl + {h}, jj] = d * f32(s{h})");
    }
    for o in 0..4 {
        for t in 0..2 {
            let _ = writeln!(
                body,
                "    let q{o}{t} = ((l{o}{t} >> (4 * (gl / 2))) & {LOW4}) | (((h{o}{t} >> (2 * gl)) & {LOW2}) << 4)
    let v{o}{t} = ((q{o}{t} | {HIGH}) - {SUB}) ^ {HIGH}",
                LOW4 = 0x0F0F_0F0Fu32,
                LOW2 = 0x0303_0303u32,
                HIGH = 0x8080_8080u32,
                SUB = 0x2020_2020u32,
            );
        }
        let _ = writeln!(body, "    store(w[jj, 32 * gl + {b} :+ 8], v{o}0, v{o}1)", b = 8 * o);
    }
    let mut now = Vec::new();
    for o in 0..4 {
        for t in 0..2 {
            now.push((
                ["l00", "l01", "l10", "l11", "l20", "l21", "l30", "l31"][2 * o + t],
                format!("word(QB[jr, BO + 64 * (GG / 4) + 32 * (GG % 2) + {b} :+ 8], {t})", b = 8 * o),
            ));
            now.push((
                ["h00", "h01", "h10", "h11", "h20", "h21", "h30", "h31"][2 * o + t],
                format!("word(QB[jr, BO + 128 + 32 * (GG / 4) + {b} :+ 8], {t})", b = 8 * o),
            ));
        }
    }
    now.push(("s0", "elem(QB[jr, BO + 192 + 8 * (GG / 4) + 2 * (GG % 4) :+ 2], 0)".into()));
    now.push(("s1", "elem(QB[jr, BO + 192 + 8 * (GG / 4) + 2 * (GG % 4) :+ 2], 1)".into()));
    Decode {
        fetch: Vec::new(),
        now,
        body,
        pipelined: false,
        halves: true,
        mins: false,
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
        mins: false,
    }
}

/// IQ1_M: as IQ1_S, but the halfword at `32 + 2g` gives each octet a nibble,
/// three index bits under its own delta sign, and the halfword at
/// `48 + 2 (g / 2)` holds two 3-bit scales a group, from bit `6 (g % 2)`,
/// weighting half `h` by `d (2s + 1) / 8`.
fn iq1m_decode() -> Decode {
    let mut body = "    let qh = (qa & 255) | ((qb & 255) << 8)
    let sc = (sa & 255) | ((sb & 255) << 8)
"
    .to_string();
    for h in 0..2 {
        let _ = writeln!(
            body,
            "    ws[2 * gl + {h}, jj] = d * 0.125 * f32(2 * ((sc >> (6 * (gl % 2) + {sh})) & 7) + 1)",
            sh = 3 * h
        );
    }
    for l in 0..4 {
        let _ = write!(
            body,
            "    let lut{l} = where(((qh >> {n}) & 1) == 1, {IQ1_MINUS}, {IQ1_PLUS})
    let gi{l} = (((qs >> {b}) & 255) | (((qh >> {m}) & 7) << 8)) * 2
    let e{l} = (elem(grid[0, gi{l} :+ 2], 0) & 255) | ((elem(grid[0, gi{l} :+ 2], 1) & 255) << 8)
    let x{l}a = (e{l} | (e{l} << 8)) & {S8}
    let x{l}b = (x{l}a | (x{l}a << 4)) & {S4}
    let x{l} = (x{l}b | (x{l}b << 2)) & {S2}
    store(w[jj, 32 * gl + {o} :+ 8], byteperm(lut{l}, lut{l}, x{l}), byteperm(lut{l}, lut{l}, x{l} >> 16))
",
            b = 8 * l,
            m = 8 * (l / 2) + 4 * (l % 2),
            n = 8 * (l / 2) + 4 * (l % 2) + 3,
            o = 8 * l,
            S8 = 0x00FF_00FFu32,
            S4 = 0x0F0F_0F0Fu32,
            S2 = 0x3333_3333u32,
        );
    }
    Decode {
        fetch: vec![
            ("qs", "word(QB[jr, BO + 4 * GG :+ 4], 0)".into()),
            ("qa", "elem(QB[jr, BO + 32 + 2 * GG :+ 2], 0)".into()),
            ("qb", "elem(QB[jr, BO + 32 + 2 * GG :+ 2], 1)".into()),
            ("sa", "elem(QB[jr, BO + 48 + 2 * (GG / 2) :+ 2], 0)".into()),
            ("sb", "elem(QB[jr, BO + 48 + 2 * (GG / 2) :+ 2], 1)".into()),
        ],
        now: Vec::new(),
        body,
        pipelined: false,
        halves: true,
        mins: false,
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
        mins: false,
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
        mins: false,
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
        mins: false,
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
    // A minimum needs each row's sum over each group of the stage: 4 TM of
    // them, eight `dp4a` against ones each over two granule-sized runs.
    let (mins_decl, sums) = if decode.mins {
        let mut sums = String::new();
        for i in 0..QGEMM_TM * 4 / QGEMM_CTA {
            let _ = writeln!(
                sums,
                "    let r{i} = (ctx.thread + {o}) / 4
    let g{i} = (ctx.thread + {o}) % 4",
                o = i * QGEMM_CTA
            );
            let mut acc = "0".to_string();
            for h in 0..2 {
                for t in 0..4 {
                    let _ = writeln!(
                        sums,
                        "    let u{i}{h}{t} = dp4a(word(at[r{i}, 32 * g{i} + {o} :+ 16], {t}), {ONES}, {acc})",
                        o = 16 * h,
                        ONES = 0x0101_0101u32
                    );
                    acc = format!("u{i}{h}{t}");
                }
            }
            let _ = writeln!(sums, "    rsum[r{i}, g{i}] = f32({acc}) * das[r{i}, g{i}]");
        }
        sums.push_str("    barrier()\n");
        ("  var mw: tile<f32>[4, TN]\n  var rsum: tile<f32>[TM, 4]\n".to_string(), sums)
    } else {
        (String::new(), String::new())
    };
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
        if decode.mins {
            let _ = write!(
                contract,
                "    let rs{s} = rsum[0 :+ TM, {s} :+ 1]
    let mw{s} = mw[{s} :+ 1, 0 :+ TN]
    acc -= rs{s} * mw{s}
"
            );
        }
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
{mins_decl}  var acc: tile<f32>[TM, TN] = 0.0
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
{sums}{contract}  }}
  {store}
}}
",
        body = decode.body,
        scales = 4 * per_group,
        pipeline = if decode.pipelined { "@pipeline\n" } else { "" },
    )
}
