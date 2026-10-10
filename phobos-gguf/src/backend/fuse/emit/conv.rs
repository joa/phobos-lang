// The delta net's causal convolution as a fused stage.

use super::*;

impl Emit {
    /// One (plane, head) pair of the causal depthwise convolution per unit.
    ///
    /// The epilogue branches on the plane, which is uniform across the CTA.
    /// It must be: the gain is a CTA-wide reduction, which would hang in a
    /// divergent branch.
    pub(super) fn conv(&mut self, key: &ChainKey, s: usize, unit: &str) -> Result<()> {
        let Stage::Conv {
            history,
            taps,
            out,
            heads,
            kv_heads,
            head_dim,
            kernel,
            channels,
            plane_base,
            plane_stride,
            head_stride,
            normalize,
            scale_bits,
            shift,
            ..
        } = key.stages[s]
        else {
            bail!("only a convolution emits a convolution");
        };
        let hist = self.given(history, key.len_of(history), View::Grid(channels));
        let taps = self.given(taps, key.len_of(taps), View::Grid(channels));
        let dst = self.given(out, key.len_of(out), View::Flat);
        let ks = self.tune_const(format!("KS{s}"), kernel);
        let d = self.tune_const(format!("HD{s}"), head_dim);
        let nh = self.tune_const(format!("NH{s}"), heads);
        let ps = self.tune_const(format!("PS{s}"), plane_stride);
        let st = self.tune_const(format!("ST{s}"), head_stride);
        let base = match plane_base {
            0 => String::new(),
            at => format!("{} + ", self.tune_const(format!("PB{s}"), at)),
        };
        // Only the query carries the readout scale. Only the query and key are
        // normalized; the value leaves the convolution unscaled.
        let scale = f32::from_bits(scale_bits);
        let norm = format!("sqrt(rowsum(y{s} * y{s}) + {L2_EPS})");
        let query = match normalize {
            true => format!("{scale:.9} / {norm}"),
            false => format!("{scale:.9}"),
        };
        let gains: String = [Some(query), normalize.then(|| format!("1.0 / {norm}"))]
            .into_iter()
            .enumerate()
            .filter_map(|(plane, gain)| Some((plane, gain?)))
            .map(|(plane, g)| {
                format!("      if pl{s} == {plane} {{\n        g{s} = {g}\n      }}\n")
            })
            .collect();

        // The unit owns its channels' columns, so it can move them up a
        // position in place once the taps above have read them.
        let shifted = match shift {
            true => format!(
                "      for sh{s} in range(0, {}) {{\n        {hist}[sh{s} :+ 1, cb{s} :+ {d}] = {hist}[sh{s} + 1 :+ 1, cb{s} :+ {d}]\n      }}\n",
                kernel - 1
            ),
            false => String::new(),
        };
        // A grouped net's units run over the query heads, the key heads, then
        // the value heads; each query and key head is stored to every value
        // head it serves.
        let (place, store) = match kv_heads == heads {
            true => (
                format!("      let pl{s} = {unit} / {nh}
      let hd{s} = {unit} % {nh}
"),
                format!("      {dst}[0 :+ 1, {unit} * {head_dim} :+ {d}] = y{s} * g{s}
"),
            ),
            false => {
                let (kv, group) = (kv_heads, heads / kv_heads);
                (
                    format!(
                        "      var pl{s}: i32 = 2
      var hd{s}: i32 = {unit} - {}
      if {unit} < {} {{
        pl{s} = 1
        hd{s} = {unit} - {kv}
      }}
      if {unit} < {kv} {{
        pl{s} = 0
        hd{s} = {unit}
      }}
",
                        2 * kv,
                        2 * kv
                    ),
                    format!(
                        "      var o{s} = y{s} * g{s}
      if pl{s} == 2 {{
        {dst}[0 :+ 1, ({} + hd{s}) * {head_dim} :+ {d}] = o{s}
      }} else {{
        for gi{s} in range(0, {group}) {{
          {dst}[0 :+ 1, (pl{s} * {nh} + hd{s} + gi{s} * {kv}) * {head_dim} :+ {d}] = o{s}
        }}
      }}
",
                        2 * heads
                    ),
                )
            }
        };
        let _ = write!(
            self.body,
            "{place}      let cb{s} = {base}pl{s} * {ps} + hd{s} * {st}
      var acc{s}: tile<f32>[1, {d}] = 0.0
      for k{s} in range(0, {ks}) {{
        acc{s} = acc{s} + {hist}[k{s} :+ 1, cb{s} :+ {d}] * {taps}[k{s} :+ 1, cb{s} :+ {d}]
      }}
{shifted}      var y{s}: tile<f32>[1, {d}] = acc{s} / (1.0 + exp(-acc{s}))
      var g{s}: tile<f32>[1, 1] = 1.0
{gains}{store}"
        );
        Ok(())
    }
}
