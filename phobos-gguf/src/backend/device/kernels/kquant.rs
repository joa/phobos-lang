// The table-free decode matvecs for Q4_K, Q5_K, Q6_K and PTQ1_0, each one
// `<fmt>_qdot_i8_t` call with no table operands.
//
// The decode lives in the compiler, in `phobos-lang`'s
// `codegen/tile/kquant_qdot.rs`. The source only names the kernel and tile.
//
// Q4_K and Q5_K read `d` and `dmin` from the block header, so the intrinsic
// takes the `D` plane but never reads it. Q6_K's `d` lives only in the
// plane (see `Quant::device_block`). PTQ1_0 keeps both scales of a block
// pair in the re-laid block itself (`quant/ptq1_0.rs`).

use std::fmt::Write as _;
use super::quant::qdot_i8_cta;
use crate::quant::Quant;
use phobos_kernels::launch::WARP_THREADS;

/// Output tile for the dp4a matvecs. A warp owns eight columns, so 64
/// columns fill a 256-thread CTA.
///
/// PTQ1_0 halves both tile and CTA, keeping eight columns per warp across
/// twice the programs. Otherwise its narrow projections launch fewer CTAs
/// than the card holds.
pub(crate) const Q4K_I8_TN: usize = 64;
pub(crate) const Q5K_I8_TN: usize = 64;
pub(crate) const Q6K_I8_TN: usize = 64;
pub(crate) const PTQ1_I8_TN: usize = 32;

/// The tile for an `n` that 64 does not divide. `project_raw` pads an `n`
/// that this does not divide either into a scratch.
pub(crate) const Q4K_I8_NARROW_TN: usize = 16;
pub(crate) const Q5K_I8_NARROW_TN: usize = 16;
pub(crate) const Q6K_I8_NARROW_TN: usize = 16;
pub(crate) const PTQ1_I8_NARROW_TN: usize = 16;

/// `resident` is the CTA count per multiprocessor that sets the register
/// budget. Q4_K fits four (64 registers). Q5_K and Q6_K hold more planes in
/// flight and would spill at four, so they take three (80 registers).
fn kquant_qdot_i8_matvec_src(name: &str, tn: usize, resident: usize) -> String {
    kquant_qdot_i8_matvec_cta_src(name, tn, qdot_i8_cta(tn), resident)
}

/// [`kquant_qdot_i8_matvec_src`] at a CTA of `cta` threads.
fn kquant_qdot_i8_matvec_cta_src(name: &str, tn: usize, cta: usize, resident: usize) -> String {
    let min_blocks = (1024 / cta * resident / 4).max(1);
    format!(
        "@launch({cta}, {min_blocks})
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel {name}_qdot_i8_matvec(AQ: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                          QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                          C: tensor<f32>[M, N]) {{
  let pn = ctx.block.x
  C[0 :+ 1, pn * TN :+ TN] = {name}_qdot_i8_t(AQ[0 :+ 1, :], AS[0 :+ 1, :],
                                           QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :])
}}
"
    )
}

/// Columns a CTA of [`q4k_qdot_lanes_src`] covers: eight a warp.
pub(crate) const Q4K_LANES_TN: usize = 64;

/// The Q4_K decode matvec written with lanes, `q4k_qdot_i8_t`'s mapping in
/// `.ph`: four lanes a column, each a quarter of every block.
///
/// `QB` is the grouped weight as the device holds it, `[N/8, NB * 8 * 144]`:
/// a group's blocks, each eight columns' 144 bytes (`d`, `dmin`, twelve
/// bytes of packed 6-bit scales and mins, 128 of nibbles). A lane's quarter
/// `q` is 32 contiguous bytes of nibbles: the low ones are run `2q`, the high
/// ones run `2q + 1`, each against 32 activations and one activation scale.
pub(crate) fn q4k_qdot_lanes_src() -> String {
    "// Block `b` of the lane's column into `acc`. A run's 6-bit scale and min
// come from bytes `j`, `j + 4` and `j - 4` of the twelve, by whether the run
// is in the first four; both forms are read and one kept, so the warp does
// not split.
fn q4k_block(b) {
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
  let lo0 = sum(i32(QB[g, o :+ 16] & 15) * i32(AQ[0, a :+ 16]))
  let lo1 = sum(i32(QB[g, o + 16 :+ 16] & 15) * i32(AQ[0, a + 16 :+ 16]))
  let hi0 = sum(i32((QB[g, o :+ 16] >> 4) & 15) * i32(AQ[0, a + 32 :+ 16]))
  let hi1 = sum(i32((QB[g, o + 16 :+ 16] >> 4) & 15) * i32(AQ[0, a + 48 :+ 16]))
  let s0 = sum(i32(AQ[0, a :+ 16])) + sum(i32(AQ[0, a + 16 :+ 16]))
  let s1 = sum(i32(AQ[0, a + 32 :+ 16])) + sum(i32(AQ[0, a + 48 :+ 16]))
  let r0 = d * f32(sc0) * f32(lo0 + lo1) - dmin * f32(m0) * f32(s0)
  let r1 = d * f32(sc1) * f32(hi0 + hi1) - dmin * f32(m1) * f32(s1)
  acc += AS[0, a / 32] * r0 + AS[0, a / 32 + 1] * r1
}

@launch(256)
@aligned(K = 256, GW = 1152)
kernel q4k_qdot_lanes(AQ: tensor<i8>[1, K], AS: tensor<f32>[1, KB],
                      QB: tensor<i8>[G, GW], C: tensor<f32>[1, N]) {
  let col = ctx.block.x * 64 + ctx.warp * 8 + ctx.lane / 4
  let q = ctx.lane % 4
  let g = col / 8
  let c8 = col % 8
  var acc: f32 = 0.0
  for b in range(0, K / 256, 1) {
    q4k_block(b)
  }
  let total = sum<over ctx.lane, width 4>(acc)
  if q == 0 && col < N {
    C[0, col] = total
  }
}
"
    .to_string()
}

/// `0x0F0F_0F0F` and `0x0101_0101`: a nibble and a bit of every byte of a
/// word, in the decimal `.ph` spells them in.
const NIBBLES: u32 = 0x0F0F_0F0F;
const LOW_BITS: u32 = 0x0101_0101;

/// The Q5_K decode matvec written with lanes, Q4_K's mapping
/// ([`q4k_qdot_lanes_src`]) over the 176-byte block: `d`, `dmin`, the twelve
/// scale bytes, 32 bytes of `qh` and 128 of nibbles. An element of run `2q`
/// is its low nibble plus 16 times bit `2q` of its `qh` byte, and of run
/// `2q + 1` its high nibble and bit `2q + 1`. Each word's four elements are
/// built on the packed word with `word` and fed to one `dp4a`.
pub(crate) fn q5k_qdot_lanes_src() -> String {
    let mut words = String::new();
    for half in 0..2 {
        for t in 0..4 {
            let n = half * 4 + t;
            let at = 16 * half;
            let _ = writeln!(
                words,
                "  let qw{n} = word(QB[g, o + {at} :+ 16], {t})
  let hw{n} = word(QB[g, h + {at} :+ 16], {t})
  lo = dp4a((qw{n} & {NIBBLES}) | (((hw{n} >> (2 * q)) & {LOW_BITS}) << 4), word(AQ[0, a + {at} :+ 16], {t}), lo)
  hi = dp4a(((qw{n} >> 4) & {NIBBLES}) | (((hw{n} >> (2 * q + 1)) & {LOW_BITS}) << 4), word(AQ[0, a + {lo_at} :+ 16], {t}), hi)",
                lo_at = 32 + at
            );
        }
    }
    format!(
        "// Block `b` of the lane's column into `acc`; the scales as in Q4_K.
fn q5k_block(b) {{
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
{words}  let s0 = sum(i32(AQ[0, a :+ 16])) + sum(i32(AQ[0, a + 16 :+ 16]))
  let s1 = sum(i32(AQ[0, a + 32 :+ 16])) + sum(i32(AQ[0, a + 48 :+ 16]))
  let r0 = d * f32(sc0) * f32(lo) - dmin * f32(m0) * f32(s0)
  let r1 = d * f32(sc1) * f32(hi) - dmin * f32(m1) * f32(s1)
  acc += AS[0, a / 32] * r0 + AS[0, a / 32 + 1] * r1
}}

@launch(256)
@aligned(K = 256, GW = 1408)
kernel q5k_qdot_lanes(AQ: tensor<i8>[1, K], AS: tensor<f32>[1, KB],
                      QB: tensor<i8>[G, GW], C: tensor<f32>[1, N]) {{
  let col = ctx.block.x * 64 + ctx.warp * 8 + ctx.lane / 4
  let q = ctx.lane % 4
  let g = col / 8
  let c8 = col % 8
  var acc: f32 = 0.0
  for b in range(0, K / 256, 1) {{
    q5k_block(b)
  }}
  let total = sum<over ctx.lane, width 4>(acc)
  if q == 0 && col < N {{
    C[0, col] = total
  }}
}}
"
    )
}

/// `0x0303_0303`: two bits of every byte of a word.
const TWO_BITS: u32 = 0x0303_0303;

/// The Q6_K decode matvec written with lanes, `q6k_qdot_i8_t`'s mapping.
///
/// The device block is 208 bytes: 128 of `ql`, 64 of `qh`, sixteen signed
/// scales; its `d` is in the grouped plane `D`, `[N/8, NB * 8]`. Lane `q`
/// takes half `h = q / 2` at `l0 = 16 (q % 2)`: sixteen elements of each of
/// the half's four runs `r`, `128 h + 32 r + l0 + i`. Run `r` reads the low
/// (`r < 2`) or high nibble of `ql[64 h + 32 (r % 2) + l0 + i]` and bits
/// `2r, 2r + 1` of `qh[32 h + l0 + i]`, less 32, under scale
/// `8 h + q % 2 + 2r`. A word of four is one `dp4a`; the 32 comes off as 32
/// times the run's activation sum.
pub(crate) fn q6k_qdot_lanes_src() -> String {
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
                "  dot{r} = dp4a({nibble} | (((word(QB[g, qh :+ 16], {t}) >> {sh}) & {TWO_BITS}) << 4), word(AQ[0, a + {at} :+ 16], {t}), dot{r})",
                sh = 2 * r,
                at = 32 * r
            );
        }
        let _ = writeln!(
            runs,
            "  let sum{r} = sum(i32(AQ[0, a + {at} :+ 16]))
  acc += d * f32(QB[g, sc + {sat}]) * AS[0, b * 8 + 4 * h + {r}] * f32(dot{r} - 32 * sum{r})",
            at = 32 * r,
            sat = 2 * r
        );
    }
    format!(
        "// Block `b` of the lane's column into `acc`.
fn q6k_block(b) {{
  let blk = b * 1664 + c8 * 208
  let d = f32(D[g, b * 8 + c8])
  let ql0 = blk + 64 * h + l0
  let ql1 = ql0 + 32
  let qh = blk + 128 + 32 * h + l0
  let sc = blk + 192 + 8 * h + q % 2
  let a = b * 256 + 128 * h + l0
{runs}}}

@launch(256)
@aligned(K = 256, GW = 1664, DW = 8)
kernel q6k_qdot_lanes(AQ: tensor<i8>[1, K], AS: tensor<f32>[1, KB],
                      QB: tensor<i8>[G, GW], D: tensor<f16>[G, DW], C: tensor<f32>[1, N]) {{
  let col = ctx.block.x * 64 + ctx.warp * 8 + ctx.lane / 4
  let q = ctx.lane % 4
  let h = q / 2
  let l0 = 16 * (q % 2)
  let g = col / 8
  let c8 = col % 8
  var acc: f32 = 0.0
  for b in range(0, K / 256, 1) {{
    q6k_block(b)
  }}
  let total = sum<over ctx.lane, width 4>(acc)
  if q == 0 && col < N {{
    C[0, col] = total
  }}
}}
"
    )
}

pub(crate) fn q4k_qdot_i8_matvec_src(tn: usize) -> String {
    kquant_qdot_i8_matvec_src("q4k", tn, 4)
}

pub(crate) fn q5k_qdot_i8_matvec_src(tn: usize) -> String {
    kquant_qdot_i8_matvec_src("q5k", tn, 3)
}

pub(crate) fn q6k_qdot_i8_matvec_src(tn: usize) -> String {
    kquant_qdot_i8_matvec_src("q6k", tn, 3)
}

/// The name and register budget of a K-quant's split matvec, for the formats
/// that have one.
fn kquant_split(quant: Quant) -> Option<(&'static str, usize)> {
    match quant {
        Quant::Q4_K => Some(("q4k", 4)),
        Quant::Q5_K => Some(("q5k", 3)),
        Quant::Q6_K => Some(("q6k", 3)),
        _ => None,
    }
}

/// The kernel [`kquant_qdot_i8_split_src`] names.
pub(crate) fn kquant_qdot_i8_split_name(quant: Quant) -> &'static str {
    match quant {
        Quant::Q4_K => "q4k_qdot_i8_split",
        Quant::Q5_K => "q5k_qdot_i8_split",
        _ => "q6k_qdot_i8_split",
    }
}

/// The split-K form of a K-quant matvec, `s` slices of `k`.
///
/// Program `(pn, ps)` contracts slice `ps` of tile `pn` into column
/// `ps * n + pn * TN` of a `[1, s * n]` partial row, which `qgemm_reduce`
/// sums. The weight is grouped by eight rows, so a slice of its bytes and
/// scales starts at eight times its block offset, as in `qgemm_split_src`.
pub(crate) fn kquant_qdot_i8_split_src(quant: Quant, n: usize, k: usize, s: usize) -> Option<String> {
    let (name, resident) = kquant_split(quant)?;
    let tn = Q4K_I8_TN;
    let cta = qdot_i8_cta(tn);
    let min_blocks = (1024 / cta * resident / 4).max(1);
    let sb = k / 256 / s;
    let (sk, skb, srb) = (sb * 256, sb * 256 / 32, sb * quant.device_block_bytes());
    let (gk, gb, gn) = (8 * srb, 8 * sb, n / tn);
    Some(format!(
        "@launch({cta}, {min_blocks})
@autotune(TN in [{tn}])
@aligned(N = TN, K = {sk}, KB = {skb}, RB = {srb}, NB = {sb}, NP = TN)
kernel {name}_qdot_i8_split(AQ: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                          QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                          P: tensor<f32>[M, NP]) {{
  let pn = ctx.block.x
  let ps = ctx.block.y
  P[0 :+ 1, (ps * {gn} + pn) * TN :+ TN] = {name}_qdot_i8_t(AQ[0 :+ 1, ps * {sk} :+ {sk}],
      AS[0 :+ 1, ps * {skb} :+ {skb}], QB[pn * TN :+ TN, ps * {gk} :+ {srb}],
      D[pn * TN :+ TN, ps * {gb} :+ {sb}])
}}
"
    ))
}

/// Eight columns per warp at any tile; see [`PTQ1_I8_TN`].
pub(crate) fn ptq1_qdot_i8_matvec_src(tn: usize) -> String {
    kquant_qdot_i8_matvec_cta_src("ptq1", tn, (tn * 4).max(WARP_THREADS), 4)
}

