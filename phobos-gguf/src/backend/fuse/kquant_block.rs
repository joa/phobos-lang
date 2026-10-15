// The K-quants' block decodes in lanes, shared by every kernel that maps
// lanes as the decode matvecs do: four lanes a column, each a quarter of
// every block, the weight grouped by eight columns as the device holds it.
// A decode is a `fn` the kernel's source carries and calls once a block.
// Q8_0 joins them through the expert slabs' unit of eight file blocks,
// which is a block of 256 in all but name.

use std::fmt::Write as _;

/// A K-quant's block decode: the `fn` to put in a kernel's source, the device
/// block's bytes, and the lets its call reads besides `q`, `g` and `c8`.
/// `name` is the format's prefix.
pub(crate) fn kquant_lanes_block(name: &str) -> (String, usize, &'static str) {
    match name {
        "q4k" => (q4k_block_src(), 144, ""),
        "q5k" => (q5k_block_src(), 176, ""),
        "q6k" => (q6k_block_src(), 208, "  let h = q / 2\n  let l0 = 16 * (q % 2)\n"),
        "q80" => (q80_block_src(), 272, ""),
        other => unreachable!("no lanes block decode for {other}"),
    }
}

/// The call of `name`'s block decode: block `b` of weight `qb` (scale plane
/// `d`) at group row `g`, against row `ar` of `aq` and `asc`, into `acc`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn kquant_block_call(name: &str, qb: &str, d: &str, aq: &str, asc: &str, ar: &str, g: &str, acc: &str, b: &str) -> String {
    match name {
        "q6k" => format!("q6k_block({qb}, {d}, {aq}, {asc}, {ar}, {g}, c8, q, h, l0, {acc}, {b})"),
        _ => format!("{name}_block({qb}, {aq}, {asc}, {ar}, {g}, c8, q, {acc}, {b})"),
    }
}

/// Q4_K's block decode, `fn q4k_block(QB, AQ, AS, ar, g, c8, q, acc, b)`:
/// block `b` of the column at `c8` in group row `g` of the grouped weight
/// `QB`, against row `ar` of `AQ` and `AS`, the lane's quarter `q`, into
/// `acc`.
///
/// A group's blocks are eight columns' 144 bytes each (`d`, `dmin`, twelve
/// bytes of packed 6-bit scales and mins, 128 of nibbles). A lane's quarter
/// `q` is 32 contiguous bytes of nibbles: the low ones are run `2q`, the high
/// ones run `2q + 1`, each against 32 activations and one activation scale.
pub(crate) fn q4k_block_src() -> String {
    "// Block `b` of the lane's column into `acc`. A run's 6-bit scale and min
// come from bytes `j`, `j + 4` and `j - 4` of the twelve, by whether the run
// is in the first four; both forms are read and one kept, so the warp does
// not split.
fn q4k_block(QB, AQ, AS, ar, g, c8, q, acc, b) {
  let blk = b * 1152 + c8 * 144
  let d = f32(f16bits((QB[g, blk] & 255) | ((QB[g, blk + 1] & 255) << 8)))
  let dmin = f32(f16bits((QB[g, blk + 2] & 255) | ((QB[g, blk + 3] & 255) << 8)))
  let s = blk + 4
  let j = 2 * q
  let x0 = QB[g, s + j] & 255
  let x1 = QB[g, s + j + 1] & 255
  let y0 = QB[g, s + j + 4] & 255
  let y1 = QB[g, s + j + 5] & 255
  let z0 = QB[g, s + max(j - 4, 0)] & 255
  let z1 = QB[g, s + max(j - 3, 0)] & 255
  let first = q < 2
  let sc0 = where(first, x0 & 63, (y0 & 15) | ((z0 >> 6) << 4))
  let m0 = where(first, y0 & 63, (y0 >> 4) | ((x0 >> 6) << 4))
  let sc1 = where(first, x1 & 63, (y1 & 15) | ((z1 >> 6) << 4))
  let m1 = where(first, y1 & 63, (y1 >> 4) | ((x1 >> 6) << 4))
  let a = b * 256 + q * 64
  let o = blk + 16 + q * 32
  let lo0 = sum(i32(QB[g, o :+ 16] & 15) * i32(AQ[ar, a :+ 16]))
  let lo1 = sum(i32(QB[g, o + 16 :+ 16] & 15) * i32(AQ[ar, a + 16 :+ 16]))
  let hi0 = sum(i32((QB[g, o :+ 16] >> 4) & 15) * i32(AQ[ar, a + 32 :+ 16]))
  let hi1 = sum(i32((QB[g, o + 16 :+ 16] >> 4) & 15) * i32(AQ[ar, a + 48 :+ 16]))
  let s0 = sum(i32(AQ[ar, a :+ 16])) + sum(i32(AQ[ar, a + 16 :+ 16]))
  let s1 = sum(i32(AQ[ar, a + 32 :+ 16])) + sum(i32(AQ[ar, a + 48 :+ 16]))
  let r0 = d * f32(sc0) * f32(lo0 + lo1) - dmin * f32(m0) * f32(s0)
  let r1 = d * f32(sc1) * f32(hi0 + hi1) - dmin * f32(m1) * f32(s1)
  acc += AS[ar, a / 32] * r0 + AS[ar, a / 32 + 1] * r1
}
"
    .to_string()
}

/// Q8_0's unit decode, `fn q80_block(QB, AQ, AS, ar, g, c8, q, acc, b)`,
/// with [`q4k_block_src`]'s operands over the 272-byte unit the expert
/// slabs hold (`quant::q8_0_units`): eight `f16` scales, then 256 signed
/// quants. Lane quarter `q` takes runs `2q` and `2q + 1`, 32 quants under
/// one scale each, against the activation's matching run and scale.
pub(crate) fn q80_block_src() -> String {
    "// Block `b` of the lane's column into `acc`.
fn q80_block(QB, AQ, AS, ar, g, c8, q, acc, b) {
  let blk = b * 2176 + c8 * 272
  let s = blk + 4 * q
  let d0 = f32(f16bits((QB[g, s] & 255) | ((QB[g, s + 1] & 255) << 8)))
  let d1 = f32(f16bits((QB[g, s + 2] & 255) | ((QB[g, s + 3] & 255) << 8)))
  let a = b * 256 + q * 64
  let o = blk + 16 + q * 64
  let r0 = sum(i32(QB[g, o :+ 16]) * i32(AQ[ar, a :+ 16])) + sum(i32(QB[g, o + 16 :+ 16]) * i32(AQ[ar, a + 16 :+ 16]))
  let r1 = sum(i32(QB[g, o + 32 :+ 16]) * i32(AQ[ar, a + 32 :+ 16])) + sum(i32(QB[g, o + 48 :+ 16]) * i32(AQ[ar, a + 48 :+ 16]))
  acc += AS[ar, a / 32] * d0 * f32(r0) + AS[ar, a / 32 + 1] * d1 * f32(r1)
}
"
    .to_string()
}

/// `0x0F0F_0F0F` and `0x0101_0101`: a nibble and a bit of every byte of a
/// word, in the decimal `.ph` spells them in.
const NIBBLES: u32 = 0x0F0F_0F0F;
const LOW_BITS: u32 = 0x0101_0101;

/// Q5_K's block decode, `fn q5k_block(QB, AQ, AS, ar, g, c8, q, acc, b)`,
/// with [`q4k_block_src`]'s operands and mapping over the 176-byte block:
/// `d`, `dmin`, the twelve scale bytes, 32 bytes of `qh` and 128 of nibbles.
/// An element of run `2q` is its low nibble plus 16 times bit `2q` of its
/// `qh` byte, and of run `2q + 1` its high nibble and bit `2q + 1`. Each
/// word's four elements are built on the packed word with `word` and fed to
/// one `dp4a`.
pub(crate) fn q5k_block_src() -> String {
    let mut words = String::new();
    for half in 0..2 {
        for t in 0..4 {
            let n = half * 4 + t;
            let at = 16 * half;
            let _ = writeln!(
                words,
                "  let qw{n} = word(QB[g, o + {at} :+ 16], {t})
  let hw{n} = word(QB[g, h + {at} :+ 16], {t})
  lo = dp4a((qw{n} & {NIBBLES}) | (((hw{n} >> (2 * q)) & {LOW_BITS}) << 4), word(AQ[ar, a + {at} :+ 16], {t}), lo)
  hi = dp4a(((qw{n} >> 4) & {NIBBLES}) | (((hw{n} >> (2 * q + 1)) & {LOW_BITS}) << 4), word(AQ[ar, a + {lo_at} :+ 16], {t}), hi)",
                lo_at = 32 + at
            );
        }
    }
    format!(
        "// Block `b` of the lane's column into `acc`; the scales as in Q4_K.
fn q5k_block(QB, AQ, AS, ar, g, c8, q, acc, b) {{
  let blk = b * 1408 + c8 * 176
  let d = f32(f16bits((QB[g, blk] & 255) | ((QB[g, blk + 1] & 255) << 8)))
  let dmin = f32(f16bits((QB[g, blk + 2] & 255) | ((QB[g, blk + 3] & 255) << 8)))
  let s = blk + 4
  let j = 2 * q
  let x0 = QB[g, s + j] & 255
  let x1 = QB[g, s + j + 1] & 255
  let y0 = QB[g, s + j + 4] & 255
  let y1 = QB[g, s + j + 5] & 255
  let z0 = QB[g, s + max(j - 4, 0)] & 255
  let z1 = QB[g, s + max(j - 3, 0)] & 255
  let first = q < 2
  let sc0 = where(first, x0 & 63, (y0 & 15) | ((z0 >> 6) << 4))
  let m0 = where(first, y0 & 63, (y0 >> 4) | ((x0 >> 6) << 4))
  let sc1 = where(first, x1 & 63, (y1 & 15) | ((z1 >> 6) << 4))
  let m1 = where(first, y1 & 63, (y1 >> 4) | ((x1 >> 6) << 4))
  let a = b * 256 + q * 64
  let o = blk + 48 + q * 32
  let h = blk + 16
  var lo = 0
  var hi = 0
{words}  let s0 = sum(i32(AQ[ar, a :+ 16])) + sum(i32(AQ[ar, a + 16 :+ 16]))
  let s1 = sum(i32(AQ[ar, a + 32 :+ 16])) + sum(i32(AQ[ar, a + 48 :+ 16]))
  let r0 = d * f32(sc0) * f32(lo) - dmin * f32(m0) * f32(s0)
  let r1 = d * f32(sc1) * f32(hi) - dmin * f32(m1) * f32(s1)
  acc += AS[ar, a / 32] * r0 + AS[ar, a / 32 + 1] * r1
}}
"
    )
}

/// `0x0303_0303`: two bits of every byte of a word.
const TWO_BITS: u32 = 0x0303_0303;

/// Q6_K's block decode, `fn q6k_block(QB, D, AQ, AS, ar, g, c8, q, h, l0,
/// acc, b)`: [`q4k_block_src`]'s operands, the scale plane `D` grouped as
/// `QB` is, and the lane's half `h` and offset `l0`.
///
/// The device block is 208 bytes: 128 of `ql`, 64 of `qh`, sixteen signed
/// scales. Lane `q` takes half `h = q / 2` at `l0 = 16 (q % 2)`: sixteen
/// elements of each of the half's four runs `r`, `128 h + 32 r + l0 + i`.
/// Run `r` reads the low (`r < 2`) or high nibble of
/// `ql[64 h + 32 (r % 2) + l0 + i]` and bits `2r, 2r + 1` of
/// `qh[32 h + l0 + i]`, less 32, under scale `8 h + q % 2 + 2r`. A word of
/// four is one `dp4a`; the 32 comes off as 32 times the run's activation sum.
pub(crate) fn q6k_block_src() -> String {
    let mut runs = String::new();
    for r in 0..4 {
        let ql = if r % 2 == 0 { "ql0" } else { "ql1" };
        let _ = writeln!(runs, "  var dot{r} = 0");
        for t in 0..4 {
            let nibble = if r < 2 {
                format!("(word(QB[g, {ql} :+ 16], {t}) & {NIBBLES})")
            } else {
                format!("((word(QB[g, {ql} :+ 16], {t}) >> 4) & {NIBBLES})")
            };
            let _ = writeln!(
                runs,
                "  dot{r} = dp4a({nibble} | (((word(QB[g, qh :+ 16], {t}) >> {sh}) & {TWO_BITS}) << 4), word(AQ[ar, a + {at} :+ 16], {t}), dot{r})",
                sh = 2 * r,
                at = 32 * r
            );
        }
        let _ = writeln!(
            runs,
            "  let sum{r} = sum(i32(AQ[ar, a + {at} :+ 16]))
  acc += d * f32(QB[g, sc + {sat}]) * AS[ar, b * 8 + 4 * h + {r}] * f32(dot{r} - 32 * sum{r})",
            at = 32 * r,
            sat = 2 * r
        );
    }
    format!(
        "// Block `b` of the lane's column into `acc`.
fn q6k_block(QB, D, AQ, AS, ar, g, c8, q, h, l0, acc, b) {{
  let blk = b * 1664 + c8 * 208
  let d = f32(D[g, b * 8 + c8])
  let ql0 = blk + 64 * h + l0
  let ql1 = ql0 + 32
  let qh = blk + 128 + 32 * h + l0
  let sc = blk + 192 + 8 * h + q % 2
  let a = b * 256 + 128 * h + l0
{runs}}}
"
    )
}
