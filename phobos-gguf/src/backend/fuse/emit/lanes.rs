// The raw-format contractions in lanes, four to a column and eight columns
// a warp, as the decode matvecs map them: a unit's 64 columns are one CTA's
// eight warps. A column's sum is a per-lane value that every lane of its four
// holds, and goes into a tile only when a tile stage reads it.

use super::*;
use super::super::{kquant_block_call, kquant_lanes_block};

impl Emit {
    /// Blocks `b0 .. b0 + nb` of the 64 weight rows from `at` against the
    /// quantized activation `aq`, `asc`, into `tot{s}`, which it returns.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn raw_lanes(&mut self, s: usize, fmt: &'static str, rq: &str, rd: &str, (aq, asc): (&str, &str), at: &str, b0: &str, nb: usize) -> String {
        self.lane_decls(fmt);
        let range = match b0 {
            "0" => format!("0, {nb}"),
            b0 => format!("{b0}, {b0} + {nb}"),
        };
        let call = kquant_block_call(fmt, rq, rd, aq, asc, "0", &format!("g{s}"), &format!("acc{s}"), &format!("b{s}"));
        let _ = write!(
            self.body,
            "      var acc{s}: f32 = 0.0
      let g{s} = {at} + ctx.warp * 8
      for b{s} in range({range}) {{
        {call}
      }}
      let tot{s} = sum<over ctx.lane, width 4>(acc{s})
"
        );
        format!("tot{s}")
    }

    /// Stores `value`, a per-lane column sum, at column `cu` of `dst`'s row
    /// from `at`, for columns below `width`.
    pub(super) fn store_lanes(&mut self, dst: &str, at: &str, value: &str, width: usize) {
        let at = match at {
            "0" => "cu".to_string(),
            at => format!("{at} + cu"),
        };
        let guard = match width {
            RAW_UNIT => String::new(),
            width => format!(" && cu < {width}"),
        };
        let _ = write!(
            self.body,
            "      if q == 0{guard} {{
        {dst}[0, {at}] = {value}
      }}
"
        );
        self.lane_stores = true;
    }

    /// A CTA barrier after per-lane stores, ahead of a nest whose tile ops may
    /// touch the same buffer: the language orders a tile op after per-thread
    /// stores by buffer, not by the columns they wrote.
    pub(super) fn land_lane_stores(&mut self) {
        if std::mem::take(&mut self.lane_stores) {
            let _ = writeln!(self.body, "\n  barrier()");
        }
    }

    /// `val`'s per-lane sums as a tile, for a tile stage to read. The barrier
    /// lands the other warps' columns.
    pub(super) fn lanes_tile(&mut self, val: Val, lane: &str) -> String {
        let name = format!("r{}", val.0);
        let _ = writeln!(self.body, "      var {name}: tile<f32>[1, {RAW_UNIT}]");
        self.store_lanes(&name, "0", lane, RAW_UNIT);
        let _ = writeln!(self.body, "      barrier()");
        self.lane_stores = false;
        self.regs.insert(val, name.clone());
        name
    }

    /// The lane coordinates every raw contraction reads, and `fmt`'s block
    /// decode, each declared once.
    fn lane_decls(&mut self, fmt: &'static str) {
        if self.lane_fns.is_empty() {
            self.decls.push_str("  let q = ctx.lane % 4\n  let c8 = ctx.lane / 4\n  let cu = ctx.warp * 8 + c8\n");
        }
        if !self.lane_fns.contains_key(fmt) {
            let (block, _, setup) = kquant_lanes_block(fmt);
            self.decls.push_str(setup);
            self.lane_fns.insert(fmt, block);
        }
    }
}
