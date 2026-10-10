// Plans a chain into loop nests and writes the fused kernel source for it.
// See [`Emit`].

use super::*;

/// A run of stages sharing a partition, and whether a barrier has to precede it.
struct Nest {
    part: Part,
    stages: Vec<usize>,
    barrier_before: bool,
}

impl ChainKey {
    /// Number of stages in the chain.
    pub(crate) fn stages(&self) -> usize {
        self.stages.len()
    }

    /// Groups the stages into nests, places the barriers, and emits the
    /// kernel.
    ///
    /// `Ok(None)` means the chain has no fused form and the caller should
    /// launch the stages separately. An `Err` means the chain was recorded
    /// wrong.
    pub(crate) fn plan(&self) -> Result<Option<Plan>> {
        let mut nests = self.nests()?;
        let mut nest_of = vec![usize::MAX; self.stages.len()];
        for (n, nest) in nests.iter().enumerate() {
            for &s in &nest.stages {
                nest_of[s] = n;
            }
        }

        // A value read outside the nest that wrote it must be in memory. It
        // needs a barrier unless the reader only reads its own block's work.
        // Writes are recorded after their stage's reads, so a value read
        // early and written late, like the residual, is not a dependency of
        // its first reader.
        let mut redundant: HashSet<Val> = HashSet::new();
        let mut writer: HashMap<Val, usize> = HashMap::new();
        for (s, stage) in self.stages.iter().enumerate() {
            let n = nest_of[s];
            for (val, read) in stage.reads() {
                let Some(&w) = writer.get(&val) else { continue };
                if nest_of[w] == n {
                    continue;
                }
                if matches!(self.vals[val.0], Kind::Temp { .. }) {
                    // There is no f32 scratch, so a register value cannot
                    // cross a nest.
                    return Ok(None);
                }
                // The barrier can be skipped only when the reading block is
                // the writing block. That holds for a redundant writer, where
                // every block wrote its own copy, and for a local read between
                // nests with the same partition, which give unit `i` to the
                // same block.
                let part = nests[nest_of[w]].part;
                if matches!(part, Part::Whole(_)) {
                    redundant.insert(val);
                } else if read == Read::All || part != nests[n].part {
                    nests[n].barrier_before = true;
                }
            }
            writer.insert(stage.writes(), s);
        }

        let mut emit = Emit::new(self.blocks, redundant);
        for (n, nest) in nests.iter().enumerate() {
            if nest.barrier_before {
                emit.barrier();
            }
            if !emit.nest(self, nest, n)? {
                return Ok(None);
            }
        }
        Ok(Some(emit.finish()))
    }

    fn nests(&self) -> Result<Vec<Nest>> {
        let mut nests: Vec<Nest> = Vec::new();
        for (s, stage) in self.stages.iter().enumerate() {
            let part = stage.part();
            // A whole-row stage never shares a nest. An elementwise stage
            // always joins the current one.
            let joins = match part {
                Part::Whole(_) => false,
                Part::Inherit => true,
                Part::Units(n) => nests.last().is_some_and(|l| l.part == Part::Units(n)),
            };
            match (joins, nests.last_mut()) {
                (true, Some(last)) => last.stages.push(s),
                (true, None) => bail!("a chain cannot start with an elementwise stage"),
                (false, _) => nests.push(Nest {
                    part,
                    stages: vec![s],
                    barrier_before: false,
                }),
            }
        }
        Ok(nests)
    }

    fn len_of(&self, val: Val) -> usize {
        match self.vals[val.0] {
            Kind::Given { len } | Kind::Quant { len } | Kind::Temp { len } => len,
            Kind::Weight { rows, k } | Kind::Raw { rows, k, .. } => rows * k,
        }
    }
}

/// The source being built, and the operand list it implies.
struct Emit {
    blocks: u32,
    tune: Vec<(String, usize)>,
    params: Vec<String>,
    slots: Vec<Slot>,
    scratch: Vec<Scratch>,
    /// Tile declarations and their flat views. They go before all loops.
    decls: String,
    body: String,
    barriers: usize,
    /// Parameters already declared, keyed by value and view, so a repeat use
    /// reuses the operand. `pairs` holds the bytes and scales of quantized
    /// values and weights.
    views: HashMap<(Val, View), String>,
    pairs: HashMap<(Val, View), (String, String)>,
    /// Scratch already claimed, by value.
    stored: HashMap<Val, usize>,
    /// Values every block wrote its own copy of. They live in shared memory
    /// rather than in scratch.
    redundant: HashSet<Val>,
    /// The redundant values already declared in [`Emit::decls`].
    shared: HashSet<Val>,
    /// Tile variables holding a value that never leaves its nest.
    regs: HashMap<Val, String>,
    /// The barrier state's parameter, once a nest needs one.
    bar: Option<String>,
    /// Whether the kernel has a raw-format contraction, which sets its
    /// launch bound.
    raw: bool,
    /// Units the nests since the last barrier hand out, so the next
    /// independent nest starts on the blocks they leave idle.
    rot: usize,
}

/// Most slices [`Emit::proj_add_raw_split`] cuts `k` into.
const SPLIT_MAX: usize = 8;

/// Shortest `k` [`Emit::proj_add_raw_split`] splits. A shorter contraction
/// keeps its blocks brief, and the barrier would cost more than it saves.
const SPLIT_MIN_K: usize = 4096;

/// The shape a value is seen under.
///
/// A normalization writes rows of [`Q8_BLOCK`] and a contraction reads one
/// flat row of the same bytes. The CTA barrier after every tile store orders
/// the two.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum View {
    Folded,
    Flat,
    /// `[len / width, width]`, one stream position per row, so the
    /// convolution indexes a tap by row.
    Grid(usize),
}

impl Emit {
    fn new(blocks: u32, redundant: HashSet<Val>) -> Emit {
        Emit {
            blocks,
            tune: vec![("BLOCKS".into(), blocks as usize)],
            params: Vec::new(),
            slots: Vec::new(),
            scratch: Vec::new(),
            decls: "  let p = program_id(0)\n".into(),
            body: String::new(),
            barriers: 0,
            views: HashMap::new(),
            pairs: HashMap::new(),
            stored: HashMap::new(),
            redundant,
            shared: HashSet::new(),
            regs: HashMap::new(),
            bar: None,
            raw: false,
            rot: 0,
        }
    }

    /// Emits one nest. Returns `false` to decline the chain.
    fn nest(&mut self, key: &ChainKey, nest: &Nest, n: usize) -> Result<bool> {
        match nest.part {
            Part::Whole(width) => {
                let [only] = nest.stages[..] else {
                    bail!("a whole-row nest holds exactly one stage");
                };
                self.whole(key, only, width)
            }
            Part::Units(_) if let [only] = nest.stages[..]
                && let Some(done) = self.proj_add_raw_split(key, only)? =>
            {
                Ok(done)
            }
            Part::Units(units) => self.units(key, nest, n, units),
            Part::Inherit => bail!("a nest cannot itself be elementwise"),
        }
    }

    /// A raw-format [`Stage::ProjAddRaw`] split along `k`, when its tiles
    /// alone would leave most of the grid idle: each block walks a slice of
    /// `k` for one tile into f32 partials, then after a grid barrier the
    /// tiles add their partials onto `y`. `None` when the stage is not one,
    /// or its shape or the grid does not call for it.
    ///
    /// The partials take the scales plane of a scratch entry, the one f32
    /// scratch the pass allocates.
    fn proj_add_raw_split(&mut self, key: &ChainKey, s: usize) -> Result<Option<bool>> {
        let Stage::ProjAddRaw { a, w, y, width } = key.stages[s] else {
            return Ok(None);
        };
        let Kind::Raw { quant, .. } = key.vals[w.0] else {
            return Ok(None);
        };
        let (k, blocks) = (key.len_of(a), self.blocks as usize);
        let tiles = width / RAW_UNIT;
        if !width.is_multiple_of(RAW_UNIT) || self.shared.contains(&a) || 2 * tiles > blocks {
            return Ok(None);
        }
        if k < SPLIT_MIN_K || !k.is_multiple_of(256) {
            return Ok(None);
        }
        let nb = k / 256;
        let Some(splits) = (2..=SPLIT_MAX).rev().find(|&sp| nb.is_multiple_of(sp) && tiles * sp <= blocks) else {
            return Ok(None);
        };
        let (aq, asc) = self.quant(a, k, false);
        let (rq, rd, fmt) = self.raw_weight(key, w)?;
        let yn = self.given(y, key.len_of(y), View::Flat);
        self.scratch.push(Scratch {
            bytes: Q8_BLOCK,
            scales: splits * width,
        });
        let at = self.scratch.len() - 1;
        let pp = self.slot(format!("P{s}"), "f32", [1, (splits * width) as i64], Bound::ScratchScales(at));
        let rt = self.tune_const("RT".into(), RAW_UNIT);
        // A slice's bytes and scales start at eight times its block offset,
        // since the weight is grouped by eight rows.
        let sb = nb / splits;
        let (sk, skb, srb) = (sb * 256, sb * 256 / Q8_BLOCK, sb * quant.device_block_bytes());
        let (gk, gb) = (8 * srb, 8 * sb);
        let units = tiles * splits;
        let iters = units.div_ceil(blocks);
        let _ = write!(
            self.body,
            "
  for k{s} in range(0, {iters}) {{
    let u{s} = p + k{s} * BLOCKS
    if u{s} < {units} {{
      let ps{s} = u{s} / {tiles}
      let t{s} = (u{s} - ps{s} * {tiles}) * {rt}
      {pp}[0 :+ 1, ps{s} * {width} + t{s} :+ {rt}] = {fmt}_qdot_i8_t({aq}[0 :+ 1, ps{s} * {sk} :+ {sk}],
          {asc}[0 :+ 1, ps{s} * {skb} :+ {skb}], {rq}[t{s} :+ {rt}, ps{s} * {gk} :+ {srb}],
          {rd}[t{s} :+ {rt}, ps{s} * {gb} :+ {sb}])
    }}
  }}
"
        );
        self.barrier();
        let sum = (0..splits)
            .map(|sp| format!("{pp}[0 :+ 1, {} + r{s} :+ {rt}]", sp * width))
            .collect::<Vec<_>>()
            .join(" + ");
        let _ = write!(
            self.body,
            "
  for j{s} in range(0, {}) {{
    let v{s} = p + j{s} * BLOCKS
    if v{s} < {tiles} {{
      let r{s} = v{s} * {rt}
      {yn}[0 :+ 1, r{s} :+ {rt}] += {sum}
    }}
  }}
",
            tiles.div_ceil(blocks)
        );
        Ok(Some(true))
    }

    /// The redundant sweep: every block normalizes and quantizes the whole
    /// row into its own copy, so no barrier precedes the first contraction.
    fn whole(&mut self, key: &ChainKey, s: usize, width: usize) -> Result<bool> {
        let Stage::NormQ {
            x,
            gain,
            out,
            eps_bits,
            ..
        } = key.stages[s]
        else {
            bail!("only a normalization partitions as a whole row");
        };
        if !width.is_multiple_of(Q8_BLOCK) {
            return Ok(false);
        }

        let rows = width / Q8_BLOCK;
        let nb = self.tune_const(format!("NB{s}"), rows);
        let xf = self.given(x, key.len_of(x), View::Folded);
        let gf = self.given(gain, key.len_of(gain), View::Folded);
        let (qs, sc) = self.quant_rows(out, key.len_of(out));
        let eps = f32::from_bits(eps_bits);
        let q8 = Q8_BLOCK;
        let _ = write!(
            self.body,
            "
  rms_norm_q_t({xf}[0 :+ {nb}, 0 :+ {q8}], {gf}[0 :+ {nb}, 0 :+ {q8}], {eps:.12},
               {qs}[0 :+ {nb}, 0 :+ {q8}], {sc}[0 :+ {nb}, 0 :+ 1])
"
        );
        Ok(true)
    }

    /// A grid-strided nest, one unit of work per block per iteration.
    ///
    /// The iteration count is a compile-time constant and the tail is
    /// guarded, so the loop keeps a static shape.
    fn units(&mut self, key: &ChainKey, nest: &Nest, n: usize, units: usize) -> Result<bool> {
        let iters = units.div_ceil(self.blocks as usize);
        let it = self.tune_const(format!("IT{n}"), iters);
        let un = self.tune_const(format!("UN{n}"), units);
        let unit = format!("u{n}");
        // Nests between two barriers are independent, so each starts its
        // units on the blocks the ones before it left idle.
        if nest.barrier_before {
            self.rot = 0;
        }
        let rot = self.rot % self.blocks as usize;
        self.rot += units;
        let base = match rot {
            0 => "p".to_string(),
            rot => format!("(p + {}) % BLOCKS", self.blocks as usize - rot),
        };
        let _ = write!(
            self.body,
            "
  for i{n} in range(0, {it}) {{
    let {unit} = {base} + i{n} * BLOCKS
    if {unit} < {un} {{
"
        );
        for &s in &nest.stages {
            if !self.unit_stage(key, s, &unit)? {
                return Ok(false);
            }
        }
        let _ = write!(self.body, "    }}\n  }}\n");
        Ok(true)
    }

    fn unit_stage(&mut self, key: &ChainKey, s: usize, unit: &str) -> Result<bool> {
        let q8 = Q8_BLOCK;
        match key.stages[s] {
            Stage::ProjQ {
                a, w, out, row_off, ..
            } => {
                let (aq, asc) = self.quant_row(a, key.len_of(a));
                let (wq, wsc) = self.weight(key, w)?;
                let at = self.strided(format!("OF{s}"), row_off, unit, q8);
                let name = self.reg(out, s);
                let head = format!("      var {name}: tile<f32>[1, {q8}] = qdot_t(");
                let pad = " ".repeat(head.len());
                let _ = write!(
                    self.body,
                    "      let at{s} = {at}
{head}{aq}, {asc},
{pad}{wq}[at{s} :+ {q8}, :], {wsc}[at{s} :+ {q8}, :])
"
                );
            }
            Stage::ProjF {
                a,
                w,
                out,
                out_off,
                row_off,
                ..
            } => {
                let (aq, asc) = self.quant_row(a, key.len_of(a));
                let (wq, wsc) = self.weight(key, w)?;
                let dst = self.given(out, key.len_of(out), View::Flat);
                let at = self.strided(format!("OF{s}"), row_off, unit, q8);
                let to = self.strided(format!("DO{s}"), out_off, unit, q8);
                let head = format!("      {dst}[0 :+ 1, to{s} :+ {q8}] = qdot_t(");
                let pad = " ".repeat(head.len());
                let _ = write!(
                    self.body,
                    "      let at{s} = {at}
      let to{s} = {to}
{head}{aq}, {asc},
{pad}{wq}[at{s} :+ {q8}, :], {wsc}[at{s} :+ {q8}, :])
"
                );
            }
            Stage::ProjRaw {
                a, w, out, row_off, ..
            } => {
                let (aq, asc) = self.quant_row(a, key.len_of(a));
                let (rq, rd, fmt) = self.raw_weight(key, w)?;
                let at = self.strided(format!("OF{s}"), row_off, unit, RAW_UNIT);
                let name = self.reg(out, s);
                let head = format!("      var {name}: tile<f32>[1, {RAW_UNIT}] = {fmt}_qdot_i8_t(");
                let pad = " ".repeat(head.len());
                let _ = write!(
                    self.body,
                    "      let at{s} = {at}
{head}{aq}, {asc},
{pad}{rq}[at{s} :+ {RAW_UNIT}, :], {rd}[at{s} :+ {RAW_UNIT}, :])
"
                );
            }
            Stage::ProjRawF {
                a,
                w,
                out,
                out_off,
                width,
                row_off,
                ..
            } => {
                let (aq, asc) = self.quant_row(a, key.len_of(a));
                let (rq, rd, fmt) = self.raw_weight(key, w)?;
                let dst = self.given(out, key.len_of(out), View::Flat);
                let at = self.strided(format!("OF{s}"), row_off, unit, RAW_UNIT);
                let to = self.strided(format!("DO{s}"), out_off, unit, RAW_UNIT);
                let _ = writeln!(self.body, "      let at{s} = {at}\n      let to{s} = {to}");
                // A whole unit stores straight from the intrinsic. A
                // remainder decodes a whole unit and stores only its head.
                let head = match width == RAW_UNIT {
                    true => format!("      {dst}[0 :+ 1, to{s} :+ {RAW_UNIT}] = {fmt}_qdot_i8_t("),
                    false => format!("      var v{s}: tile<f32>[1, {RAW_UNIT}] = {fmt}_qdot_i8_t("),
                };
                let pad = " ".repeat(head.len());
                let _ = write!(
                    self.body,
                    "{head}{aq}, {asc},
{pad}{rq}[at{s} :+ {RAW_UNIT}, :], {rd}[at{s} :+ {RAW_UNIT}, :])
"
                );
                if width != RAW_UNIT {
                    let _ = writeln!(self.body, "      {dst}[0 :+ 1, to{s} :+ {width}] = v{s}[0 :+ 1, 0 :+ {width}]");
                }
            }
            Stage::Swiglu { g, u, out } => {
                let (gn, un) = (self.reg_of(g)?, self.reg_of(u)?);
                let width = key.len_of(out);
                let name = self.reg(out, s);
                let _ = writeln!(
                    self.body,
                    "      var {name}: tile<f32>[1, {width}] = ({gn} / (1.0 + exp(-{gn}))) * {un}"
                );
            }
            Stage::QuantQ { h, out, blocks, .. } if blocks > 1 => {
                // A raw unit spans several Q8_0 blocks. Each is sliced out,
                // quantized on its own, and stored to its own scratch row.
                let hn = self.reg_of(h)?;
                let (qs, sc) = self.quant_rows(out, key.len_of(out));
                for b in 0..blocks {
                    let (off, row) = (b * q8, format!("{unit} * {blocks} + {b}"));
                    let _ = write!(
                        self.body,
                        "      var h{s}_{b}: tile<f32>[1, {q8}] = {hn}[0 :+ 1, {off} :+ {q8}]
      var m{s}_{b}: tile<f32>[1, 1] = rowmax(tmax(h{s}_{b}, -h{s}_{b}))
      var q{s}_{b} = h{s}_{b} * (127.0 / (m{s}_{b} + {QUANT_EPS}))
      {qs}[{row} :+ 1, 0 :+ {q8}] = i8(i32(round(q{s}_{b})))
      {sc}[{row} :+ 1, 0 :+ 1] = m{s}_{b} / 127.0
"
                    );
                }
            }
            Stage::QuantQ { h, out, .. } => {
                // `h` is either a register from this nest, like the SwiGLU
                // output, or a caller buffer, like attention's mixed heads.
                // The two are read differently.
                let hn = match self.regs.get(&h) {
                    Some(reg) => reg.clone(),
                    None => {
                        let flat = self.given(h, key.len_of(h), View::Flat);
                        format!("{flat}[0 :+ 1, {unit} * {q8} :+ {q8}]")
                    }
                };
                let (qs, sc) = self.quant_rows(out, key.len_of(out));
                // Bind the scale to a named tile before the store. Elementwise
                // fusion needs a named tile, not an expression.
                let _ = write!(
                    self.body,
                    "      var m{s}: tile<f32>[1, 1] = rowmax(tmax({hn}, -{hn}))
      var q{s} = {hn} * (127.0 / (m{s} + {QUANT_EPS}))
      {qs}[{unit} :+ 1, 0 :+ {q8}] = i8(i32(round(q{s})))
      {sc}[{unit} :+ 1, 0 :+ 1] = m{s} / 127.0
"
                );
            }
            Stage::ProjAdd { a, w, y, width } => {
                if !width.is_multiple_of(OUT_TILE) {
                    return Ok(false);
                }
                let (aq, asc) = self.quant_row(a, key.len_of(a));
                let (wq, wsc) = self.weight(key, w)?;
                let yn = self.given(y, key.len_of(y), View::Flat);
                let tn = self.tune_const("TN".into(), OUT_TILE);
                let head = format!("      {yn}[0 :+ 1, t{s} :+ {tn}] += qdot_t(");
                let pad = " ".repeat(head.len());
                let _ = write!(
                    self.body,
                    "      let t{s} = {unit} * {tn}
{head}{aq}, {asc},
{pad}{wq}[t{s} :+ {tn}, :], {wsc}[t{s} :+ {tn}, :])
"
                );
            }
            Stage::ProjAddRaw { a, w, y, width } => {
                if !width.is_multiple_of(RAW_UNIT) {
                    return Ok(false);
                }
                let (aq, asc) = self.quant_row(a, key.len_of(a));
                let (rq, rd, fmt) = self.raw_weight(key, w)?;
                let yn = self.given(y, key.len_of(y), View::Flat);
                let rt = self.tune_const("RT".into(), RAW_UNIT);
                let head = format!("      {yn}[0 :+ 1, t{s} :+ {rt}] += {fmt}_qdot_i8_t(");
                let pad = " ".repeat(head.len());
                let _ = write!(
                    self.body,
                    "      let t{s} = {unit} * {rt}
{head}{aq}, {asc},
{pad}{rq}[t{s} :+ {rt}, :], {rd}[t{s} :+ {rt}, :])
"
                );
            }
            Stage::Conv { .. } => self.conv(key, s, unit)?,
            Stage::Gates {
                raw,
                decay_at,
                beta_at,
                rate,
                bias,
                out,
                heads,
                span,
                ..
            } => {
                let src = self.given(raw, key.len_of(raw), View::Flat);
                let rt = self.given(rate, key.len_of(rate), View::Flat);
                let bs = self.given(bias, key.len_of(bias), View::Flat);
                let dst = self.given(out, key.len_of(out), View::Flat);
                let (da, ba) = (decay_at, beta_at);
                let (dec, bet) = (3 * span, 3 * span + heads);
                let nh = self.tune_const(format!("GH{s}"), heads);
                // Only the first `heads` units do work. The softplus is
                // max(x, 0) + log(1 + exp(-|x|)), since log(1 + exp(x))
                // overflows in the range the decay projection reaches.
                let _ = write!(
                    self.body,
                    "      if {unit} < {nh} {{
        var a{s}: tile<f32>[1, 1] = {src}[0 :+ 1, {da} + {unit} :+ 1] \
+ {bs}[0 :+ 1, {unit} :+ 1]
        var z{s}: tile<f32>[1, 1] = 0.0
        var sp{s}: tile<f32>[1, 1] = tmax(a{s}, z{s}) + log(1.0 + exp(-tmax(a{s}, -a{s})))
        {dst}[0 :+ 1, {dec} + {unit} :+ 1] = exp({rt}[0 :+ 1, {unit} :+ 1] * sp{s})
        var b{s}: tile<f32>[1, 1] = {src}[0 :+ 1, {ba} + {unit} :+ 1]
        {dst}[0 :+ 1, {bet} + {unit} :+ 1] = 1.0 / (1.0 + exp(-b{s}))
      }}
"
                );
            }
            Stage::NormQ { .. } => bail!("a normalization does not partition into units"),
        }
        Ok(true)
    }

    /// One (plane, head) pair of the causal depthwise convolution per unit.
    ///
    /// The epilogue branches on the plane, which is uniform across the CTA.
    /// It must be: the gain is a CTA-wide reduction, which would hang in a
    /// divergent branch.
    fn conv(&mut self, key: &ChainKey, s: usize, unit: &str) -> Result<()> {
        let Stage::Conv {
            history,
            taps,
            out,
            heads,
            head_dim,
            kernel,
            channels,
            plane_base,
            plane_stride,
            head_stride,
            normalize,
            scale_bits,
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

        let _ = write!(
            self.body,
            "      let pl{s} = {unit} / {nh}
      let hd{s} = {unit} % {nh}
      let cb{s} = {base}pl{s} * {ps} + hd{s} * {st}
      var acc{s}: tile<f32>[1, {d}] = 0.0
      for k{s} in range(0, {ks}) {{
        acc{s} = acc{s} + {hist}[k{s} :+ 1, cb{s} :+ {d}] * {taps}[k{s} :+ 1, cb{s} :+ {d}]
      }}
      var y{s}: tile<f32>[1, {d}] = acc{s} / (1.0 + exp(-acc{s}))
      var g{s}: tile<f32>[1, 1] = 1.0
{gains}      {dst}[0 :+ 1, {unit} * {head_dim} :+ {d}] = y{s} * g{s}
"
        );
        Ok(())
    }

    /// Where a unit's run of `stride` elements starts, as source text. A zero
    /// offset is omitted rather than emitted as a tune constant.
    fn strided(&mut self, name: String, offset: usize, unit: &str, stride: usize) -> String {
        if offset == 0 {
            return format!("{unit} * {stride}");
        }
        let off = self.tune_const(name, offset);
        format!("{off} + {unit} * {stride}")
    }

    fn reg(&mut self, val: Val, s: usize) -> String {
        let name = format!("v{s}");
        self.regs.insert(val, name.clone());
        name
    }

    fn reg_of(&self, val: Val) -> Result<String> {
        self.regs
            .get(&val)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("value {} is read before its nest writes it", val.0))
    }

    /// Emits a grid barrier. One counter and generation pair serves every
    /// barrier, since each barrier leaves both as it found them.
    fn barrier(&mut self) {
        let bar = match &self.bar {
            Some(name) => name.clone(),
            None => {
                let name = self.slot("BAR".into(), "i32", [2, 1], Bound::Barrier);
                self.bar = Some(name.clone());
                name
            }
        };
        let _ = writeln!(self.body, "\n  grid_barrier({bar})");
        self.barriers += 1;
    }

    /// Declares a parameter and records what the backend must bind to it.
    fn slot(&mut self, name: String, ty: &str, dims: [i64; 2], bound: Bound) -> String {
        self.params
            .push(format!("{name}: tensor<{ty}>[{}, {}]", dims[0], dims[1]));
        self.slots.push(Slot { bound, dims });
        name
    }

    fn tune_const(&mut self, name: String, value: usize) -> String {
        if !self.tune.iter().any(|(n, _)| *n == name) {
            self.tune.push((name.clone(), value));
        }
        name
    }

    /// The parameter a caller buffer is read through, under `view`.
    fn given(&mut self, val: Val, len: usize, view: View) -> String {
        if let Some(name) = self.views.get(&(val, view)) {
            return name.clone();
        }
        let (dims, suffix) = match view {
            View::Folded => ([(len / Q8_BLOCK) as i64, Q8_BLOCK as i64], "F"),
            View::Flat => ([1, len as i64], ""),
            View::Grid(width) => ([(len / width) as i64, width as i64], "G"),
        };
        let name = self.slot(
            format!("X{}{suffix}", val.0),
            "f32",
            dims,
            Bound::Given(val),
        );
        self.views.insert((val, view), name.clone());
        name
    }

    fn weight(&mut self, key: &ChainKey, val: Val) -> Result<(String, String)> {
        if let Some(pair) = self.pairs.get(&(val, View::Flat)) {
            return Ok(pair.clone());
        }
        let Kind::Weight { rows, k } = key.vals[val.0] else {
            bail!(
                "value {} is used as a weight but was not recorded as one",
                val.0
            );
        };
        let qs = self.slot(
            format!("Wq{}", val.0),
            "i8",
            [rows as i64, k as i64],
            Bound::WeightQs(val),
        );
        let scales = self.slot(
            format!("Ws{}", val.0),
            "f32",
            [rows as i64, (k / Q8_BLOCK) as i64],
            Bound::WeightScales(val),
        );
        let pair = (qs, scales);
        self.pairs.insert((val, View::Flat), pair.clone());
        Ok(pair)
    }

    /// A raw weight's block bytes and `d` plane, and the intrinsic that
    /// decodes them.
    fn raw_weight(&mut self, key: &ChainKey, val: Val) -> Result<(String, String, &'static str)> {
        let Kind::Raw { rows, k, quant } = key.vals[val.0] else {
            bail!("value {} is used as a raw weight but was not recorded as one", val.0);
        };
        let fmt = match quant {
            Quant::Q4_K => "q4k",
            Quant::Q5_K => "q5k",
            Quant::Q6_K => "q6k",
            other => bail!("no fused decode for {}", other.name()),
        };
        self.raw = true;
        if let Some(pair) = self.pairs.get(&(val, View::Flat)) {
            return Ok((pair.0.clone(), pair.1.clone(), fmt));
        }
        // The upload pads rows to a whole unit and a remainder reads into the
        // padding. The binding checks the two agree.
        let nb = k / 256;
        let rows = rows.next_multiple_of(RAW_UNIT);
        let qs = self.slot(
            format!("Rq{}", val.0),
            "i8",
            [rows as i64, (nb * quant.device_block_bytes()) as i64],
            Bound::RawBytes(val),
        );
        let d = self.slot(format!("Rd{}", val.0), "f16", [rows as i64, nb as i64], Bound::RawD(val));
        self.pairs.insert((val, View::Flat), (qs.clone(), d.clone()));
        Ok((qs, d, fmt))
    }

    /// The bytes and scales of a quantized activation as rows of
    /// [`Q8_BLOCK`], the shape stages store through. The caller adds the row
    /// subscript.
    fn quant_rows(&mut self, val: Val, len: usize) -> (String, String) {
        self.quant(val, len, true)
    }

    /// The same value as one row, ready to pass to a contraction.
    ///
    /// Returns a complete operand. A shared value's flat view is already a
    /// one-row tile; a global one needs a slice.
    fn quant_row(&mut self, val: Val, len: usize) -> (String, String) {
        let (qs, scales) = self.quant(val, len, false);
        match self.shared.contains(&val) {
            true => (qs, scales),
            false => (format!("{qs}[0 :+ 1, :]"), format!("{scales}[0 :+ 1, :]")),
        }
    }

    /// Claims a quantized activation's storage on first use under either
    /// view. A redundantly written value goes in shared memory, since only its
    /// writer reads it. Anything else goes in global scratch, behind a
    /// barrier.
    ///
    /// `folded` picks [`View::Folded`] over [`View::Flat`].
    fn quant(&mut self, val: Val, len: usize, folded: bool) -> (String, String) {
        let view = if folded { View::Folded } else { View::Flat };
        if let Some(pair) = self.pairs.get(&(val, view)) {
            return pair.clone();
        }
        if self.redundant.contains(&val) {
            return self.quant_shared(val, len, folded);
        }
        let at = match self.stored.get(&val) {
            Some(&at) => at,
            None => {
                self.scratch.push(Scratch {
                    bytes: len,
                    scales: len / Q8_BLOCK,
                });
                let at = self.scratch.len() - 1;
                self.stored.insert(val, at);
                at
            }
        };
        let rows = (len / Q8_BLOCK) as i64;
        let (dims_q, dims_s, suffix) = match folded {
            true => ([rows, Q8_BLOCK as i64], [rows, 1], "F"),
            false => ([1, len as i64], [1, rows], ""),
        };
        let qs = self.slot(
            format!("Vq{}{suffix}", val.0),
            "i8",
            dims_q,
            Bound::ScratchQs(at),
        );
        let scales = self.slot(
            format!("Vs{}{suffix}", val.0),
            "f32",
            dims_s,
            Bound::ScratchScales(at),
        );
        let pair = (qs, scales);
        self.pairs.insert((val, view), pair.clone());
        pair
    }

    /// A redundantly written activation, in shared memory.
    ///
    /// The tile has no initializer, since the sweep writes every element. Its
    /// flat view is declared alongside it, before any loop. Both views are
    /// the same bytes: the reduction writes the folded one and the
    /// contraction reads the flat one.
    fn quant_shared(&mut self, val: Val, len: usize, folded: bool) -> (String, String) {
        let rows = len / Q8_BLOCK;
        let (qs, scales) = (format!("Aq{}", val.0), format!("As{}", val.0));
        if self.shared.insert(val) {
            let q8 = Q8_BLOCK;
            let _ = write!(
                self.decls,
                "  var {qs}: tile<i8>[{rows}, {q8}]
  var {scales}: tile<f32>[{rows}, 1]
  let {qs}L = flat({qs})
  let {scales}L = flat({scales})
"
            );
        }
        let (pair, view) = match folded {
            true => ((qs, scales), View::Folded),
            false => ((format!("{qs}L"), format!("{scales}L")), View::Flat),
        };
        self.pairs.insert((val, view), pair.clone());
        pair
    }

    fn finish(self) -> Plan {
        let tune = self
            .tune
            .iter()
            .map(|(n, v)| format!("{n} in [{v}]"))
            .collect::<Vec<_>>()
            .join(", ");
        let params = self.params.join(",\n             ");
        // Two CTAs per SM with a raw decode in the kernel. Three would spill.
        let launch = match self.raw {
            true => format!("{CTA}, 2"),
            false => CTA.to_string(),
        };
        // Shared declarations go first, so every tile is in scope before the
        // loop that fills it.
        let source = format!(
            "@launch({launch})\n@persistent\n@autotune({tune})\nkernel fused({params}) {{\n{}{}}}\n",
            self.decls, self.body
        );
        Plan {
            source,
            held: self.shared.len(),
            slots: self.slots,
            scratch: self.scratch,
            blocks: self.blocks,
            barriers: self.barriers,
        }
    }
}
