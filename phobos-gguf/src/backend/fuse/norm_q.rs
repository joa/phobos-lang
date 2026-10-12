// The quantizing RMS norm written with lanes, shared by the standalone norm
// kernels and the fused decode kernel.

use std::fmt::Write as _;

/// Where [`rms_norm_q_lanes`] reads and writes: `x` (and the gate `z`) at
/// row `row0` of `[rows, 32]` views, the gain `g` from row 0, and the
/// quantized copy `q` and its scales `s` at `row0`; `out`, if given, takes
/// the normalized row. Each is a tensor or a whole shared tile.
pub(crate) struct NormOperands<'a> {
    pub(crate) x: &'a str,
    pub(crate) g: &'a str,
    pub(crate) z: Option<&'a str>,
    pub(crate) out: Option<&'a str>,
    pub(crate) q: &'a str,
    pub(crate) s: &'a str,
    pub(crate) row0: &'a str,
}

/// One row's RMS normalization, its gain (and `silu(z)`), and its Q8_0 copy,
/// as statements written with lanes. `nb` blocks of 32 make the row, a `.ph`
/// constant; `tag` keeps the locals of two norms in one kernel apart.
///
/// A thread owns four contiguous elements of every `4 * ctx.threads`: one
/// pass sums their squares into a CTA total, a second normalizes them, and
/// a block's 32 elements, eight lanes, fold their absolute maximum for its
/// scale. Only a partial last stripe's loads and stores are gated, so the
/// shuffles stay under uniform control. A `barrier()` publishes the row.
pub(crate) fn rms_norm_q_lanes(o: &NormOperands<'_>, nb: &str, eps: f32, tag: &str) -> String {
    let NormOperands { x, g, z, out, q, s, row0 } = o;
    let width = format!("({nb} * 32)");
    let stripes = format!("({width} + 4 * ctx.threads - 1) / (4 * ctx.threads)");
    let mut norm = String::new();
    for k in 0..4 {
        let mut value = format!("elem({x}[row_{tag}, col_{tag} :+ 4], {k}) * inv_{tag} * elem({g}[e_{tag} / 32, col_{tag} :+ 4], {k})");
        if let Some(z) = z {
            // silu(z) = z / (1 + exp(-z)), as the tile form computes it.
            let _ = writeln!(
                norm,
                "      let z{k}_{tag} = elem({z}[row_{tag}, col_{tag} :+ 4], {k})"
            );
            value = format!("{value} * (z{k}_{tag} / (1.0 + exp(-z{k}_{tag})))");
        }
        let _ = writeln!(norm, "      n{k}_{tag} = {value}");
    }
    let quant = |k: usize| format!("i8(i32(round(n{k}_{tag} * iq_{tag})))");
    let out_store = out.map_or(String::new(), |out| {
        format!("      store({out}[row_{tag}, col_{tag} :+ 4], n0_{tag}, n1_{tag}, n2_{tag}, n3_{tag})\n")
    });
    format!(
        "  var ss_{tag}: f32 = 0.0
  for i_{tag} in range(0, {stripes}, 1) {{
    let e_{tag} = 4 * ctx.thread + i_{tag} * 4 * ctx.threads
    if e_{tag} < {width} {{
      ss_{tag} += sum({x}[{row0} + e_{tag} / 32, e_{tag} % 32 :+ 4] * {x}[{row0} + e_{tag} / 32, e_{tag} % 32 :+ 4])
    }}
  }}
  let inv_{tag} = 1.0 / sqrt(sum<over ctx.thread>(ss_{tag}) / f32({width}) + {eps:.12})
  for i_{tag} in range(0, {stripes}, 1) {{
    let e_{tag} = 4 * ctx.thread + i_{tag} * 4 * ctx.threads
    let row_{tag} = {row0} + e_{tag} / 32
    let col_{tag} = e_{tag} % 32
    var n0_{tag}: f32 = 0.0
    var n1_{tag}: f32 = 0.0
    var n2_{tag}: f32 = 0.0
    var n3_{tag}: f32 = 0.0
    if e_{tag} < {width} {{
{norm}    }}
    let mx_{tag} = max<over ctx.lane, width 8>(max(max(abs(n0_{tag}), abs(n1_{tag})), max(abs(n2_{tag}), abs(n3_{tag}))))
    let iq_{tag} = 127.0 / (mx_{tag} + 0.00000001)
    if e_{tag} < {width} {{
{out_store}      store({q}[row_{tag}, col_{tag} :+ 4], {q0}, {q1}, {q2}, {q3})
      if ctx.lane % 8 == 0 {{
        {s}[row_{tag}, 0] = mx_{tag} / 127.0
      }}
    }}
  }}
  barrier()
",
        q0 = quant(0),
        q1 = quant(1),
        q2 = quant(2),
        q3 = quant(3),
    )
}
