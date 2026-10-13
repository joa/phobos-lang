// The IQ formats' f32 decode written with lanes: `qdecode`, which stores the
// decoded weight as a `[K, N]` strip, and the f32 decode matvec, which
// contracts it against one activation row. Both decode one octet of one
// column's block per thread: elements `8o .. 8o + 8` of its 256. The weight
// is grouped: column `j`'s block `kb` sits in row `8 (j / 8)` at block
// `8 kb + j % 8`, and so does its scale.

use std::fmt::Write as _;

use super::{
    IQ1S_GRID_LEN, IQ2S_GRID_LEN, IQ2S_SIGNS_LEN, IQ2XS_GRID_LEN, IQ2XXS_GRID_LEN, IQ2XXS_SIGNS_LEN,
    IQ3S_GRID_LEN, IQ3XXS_GRID_LEN,
};

/// An IQ format with an f32 decode.
#[derive(Clone, Copy)]
pub(crate) enum IqFmt {
    Iq1s,
    Iq1m,
    Iq2xxs,
    Iq2xs,
    Iq2s,
    Iq3xxs,
    Iq3s,
}

impl IqFmt {
    fn name(self) -> &'static str {
        match self {
            IqFmt::Iq1s => "iq1s",
            IqFmt::Iq1m => "iq1m",
            IqFmt::Iq2xxs => "iq2xxs",
            IqFmt::Iq2xs => "iq2xs",
            IqFmt::Iq2s => "iq2s",
            IqFmt::Iq3xxs => "iq3xxs",
            IqFmt::Iq3s => "iq3s",
        }
    }

    /// Device bytes of a block, its `f16` scale held apart in `D`.
    fn bytes(self) -> usize {
        match self {
            IqFmt::Iq1s => 48,
            IqFmt::Iq1m => 56,
            IqFmt::Iq2xxs => 64,
            IqFmt::Iq2xs => 72,
            IqFmt::Iq2s => 80,
            IqFmt::Iq3xxs => 96,
            IqFmt::Iq3s => 108,
        }
    }

    /// The grid's length and, for a format that has one, the sign table's.
    /// The decode reads only the grid: signs come from their bits, an eighth
    /// as the parity of seven.
    fn tables(self) -> (usize, Option<usize>) {
        match self {
            IqFmt::Iq1s | IqFmt::Iq1m => (IQ1S_GRID_LEN, None),
            IqFmt::Iq2xxs => (IQ2XXS_GRID_LEN, Some(IQ2XXS_SIGNS_LEN)),
            IqFmt::Iq2xs => (IQ2XS_GRID_LEN, Some(IQ2XXS_SIGNS_LEN)),
            IqFmt::Iq2s => (IQ2S_GRID_LEN, Some(IQ2S_SIGNS_LEN)),
            IqFmt::Iq3xxs => (IQ3XXS_GRID_LEN, Some(IQ2XXS_SIGNS_LEN)),
            IqFmt::Iq3s => (IQ3S_GRID_LEN, Some(IQ2S_SIGNS_LEN)),
        }
    }

    /// The decode of octet `o` of the block at byte `bo` of row `jr`, with `d`
    /// its scale: statements, then the octet's eight values. Group `G = o / 4`
    /// is the octet's 32 elements and `l = o % 4` its place in them.
    fn octet(self) -> (String, Vec<String>) {
        let ub = |off: &str| format!("(QB[jr, bo + {off}] & 255)");
        let u16 = |off: &str| format!("({} | ({} << 8))", ub(off), ub(&format!("{off} + 1")));
        let mut pre = String::from("      let gg = o / 4\n      let l = o % 4\n");
        // A grid entry of eight at `gi`, or two of four at `gi` and `gj`,
        // signed by the eight bits `sb`.
        let signed = |pair: bool| {
            (0..8)
                .map(|e| {
                    let (at, i) = match (pair, e < 4) {
                        (false, _) => ("gi :+ 8", e),
                        (true, true) => ("gi :+ 4", e),
                        (true, false) => ("gj :+ 4", e - 4),
                    };
                    format!("sc * f32(elem(grid[0, {at}], {i})) * f32(1 - 2 * ((sb >> {e}) & 1))")
                })
                .collect::<Vec<_>>()
        };
        let parity = "      let sb = sv | ((popcount(sv) & 1) << 7)\n";
        let values = match self {
            IqFmt::Iq2xxs => {
                let _ = write!(
                    pre,
                    "      let lo = word(QB[jr, bo + 8 * gg :+ 4], 0)
      let hi = word(QB[jr, bo + 8 * gg + 4 :+ 4], 0)
      let gi = ((lo >> (8 * l)) & 255) * 8
      let sv = (hi >> (7 * l)) & 127
{parity}      let sc = d * 0.125 * f32(2 * ((hi >> 28) & 15) + 1)
"
                );
                signed(false)
            }
            IqFmt::Iq2xs => {
                let _ = write!(
                    pre,
                    "      let u = {code}
      let gi = (u & 511) * 8
      let sv = (u >> 9) & 127
{parity}      let sc = d * 0.125 * f32(2 * (({scale} >> (4 * (l / 2))) & 15) + 1)
",
                    code = u16("8 * gg + 2 * l"),
                    scale = ub("64 + gg")
                );
                signed(false)
            }
            IqFmt::Iq2s => {
                let _ = write!(
                    pre,
                    "      let gi = ({index} | ((({qh} >> (2 * l)) & 3) << 8)) * 8
      let sb = {signs}
      let sc = d * 0.125 * f32(2 * (({scale} >> (4 * (l / 2))) & 15) + 1)
",
                    index = ub("o"),
                    qh = ub("64 + gg"),
                    signs = ub("32 + o"),
                    scale = ub("72 + gg")
                );
                signed(false)
            }
            IqFmt::Iq3xxs => {
                let _ = write!(
                    pre,
                    "      let gi = {first} * 4
      let gj = {second} * 4
      let aux = word(QB[jr, bo + 64 + 4 * gg :+ 4], 0)
      let sv = (aux >> (7 * l)) & 127
{parity}      let sc = d * 0.25 * f32(2 * ((aux >> 28) & 15) + 1)
",
                    first = ub("2 * o"),
                    second = ub("2 * o + 1")
                );
                signed(true)
            }
            IqFmt::Iq3s => {
                let _ = write!(
                    pre,
                    "      let qh = {qh}
      let gi = ({first} | (((qh >> (2 * l)) & 1) << 8)) * 4
      let gj = ({second} | (((qh >> (2 * l + 1)) & 1) << 8)) * 4
      let sb = {signs}
      let sc = d * f32(2 * (({scale} >> (4 * (gg % 2))) & 15) + 1)
",
                    qh = ub("64 + gg"),
                    first = ub("2 * o"),
                    second = ub("2 * o + 1"),
                    signs = ub("72 + o"),
                    scale = ub("104 + gg / 2")
                );
                signed(true)
            }
            IqFmt::Iq1s | IqFmt::Iq1m => {
                // A weight is `d (2n + 1) (g +- 1/8)` for a grid byte `g`.
                let (index_bits, neg, n) = match self {
                    IqFmt::Iq1s => ("(qh >> (3 * l)) & 7", "(qh >> 15) & 1", "(qh >> 12) & 7".to_string()),
                    _ => (
                        "(qh >> (4 * l)) & 7",
                        "(qh >> (4 * l + 3)) & 1",
                        format!(
                            "({} >> (3 * (2 * (gg % 2) + l / 2))) & 7",
                            u16("48 + 2 * (gg / 2)")
                        ),
                    ),
                };
                let _ = write!(
                    pre,
                    "      let qh = {qh}
      let gi = ({index} | (({index_bits}) << 8)) * 8
      let delta = 1 - 2 * ({neg})
      let sc = d * 0.125 * f32(2 * ({n}) + 1)
",
                    qh = u16("32 + 2 * gg"),
                    index = ub("o")
                );
                (0..8).map(|e| format!("sc * f32(8 * elem(grid[0, gi :+ 8], {e}) + delta)")).collect()
            }
        };
        (pre, values)
    }

    /// The kernel's table parameters and the staged grid.
    fn table_params(self) -> (String, String) {
        let (grid, signs) = self.tables();
        let mut params = format!("GRID: tensor<i8>[1, {grid}]");
        if let Some(signs) = signs {
            let _ = write!(params, ", SIGNS: tensor<i8>[1, {signs}]");
        }
        (params, format!("  var grid = GRID[0 :+ 1, 0 :+ {grid}]\n"))
    }
}

/// The decoded weight of `TN` columns stored as a `[K, TN]` strip of
/// `SCRATCH`. Consecutive threads take consecutive columns, so a warp's
/// stores fill whole sectors.
pub(crate) fn qdecode_src(fmt: IqFmt, tn: usize) -> String {
    let (name, bytes) = (fmt.name(), fmt.bytes());
    let (tables, stage) = fmt.table_params();
    let (pre, values) = fmt.octet();
    let stores: String = values
        .iter()
        .enumerate()
        .map(|(e, v)| format!("      SCRATCH[kb * 256 + 8 * o + {e}, j] = {v}\n"))
        .collect();
    format!(
        "@launch(256)
@autotune(TN in [{tn}])
@aligned(N = TN)
kernel {name}_qdecode(QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB], {tables},
                      SCRATCH: tensor<f32>[K, N]) {{
{stage}  let pn = ctx.block.x
  for li in range(ctx.thread, TN * 32, ctx.threads) {{
    let j = pn * TN + li % TN
    let jr = j / 8 * 8
    let o = li / TN
    for kb in range(0, NB, 1) {{
      let bo = (kb * 8 + j % 8) * {bytes}
      let d = f32(D[jr, kb * 8 + j % 8])
{pre}{stores}    }}
  }}
}}
"
    )
}

/// The f32 decode matvec: a warp per column, lane `o` taking octet `o` of
/// every block against the activation row.
pub(crate) fn qdot_matvec_src(fmt: IqFmt, tn: usize) -> String {
    let (name, bytes) = (fmt.name(), fmt.bytes());
    let (tables, stage) = fmt.table_params();
    let (pre, values) = fmt.octet();
    let dot: String = values
        .iter()
        .enumerate()
        .map(|(e, v)| format!("      acc += {v} * elem(A[0, ka + {} :+ 4], {})\n", e / 4 * 4, e % 4))
        .collect();
    format!(
        "@launch(256)
@autotune(TN in [{tn}])
@aligned(N = TN, K = 256)
kernel {name}_qdot_matvec(A: tensor<f32>[M, K], QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                          {tables}, C: tensor<f32>[M, N]) {{
{stage}  let pn = ctx.block.x
  let o = ctx.lane
  for c in range(ctx.warp, TN, ctx.warps) {{
    let j = pn * TN + c
    let jr = j / 8 * 8
    var acc: f32 = 0.0
    for kb in range(0, NB, 1) {{
      let bo = (kb * 8 + j % 8) * {bytes}
      let d = f32(D[jr, kb * 8 + j % 8])
      let ka = kb * 256 + 8 * o
{pre}{dot}    }}
    let total = sum<over ctx.lane>(acc)
    if ctx.lane == 0 {{
      C[0, j] = total
    }}
  }}
}}
"
    )
}
