// Planning a chain into a nest of loops, and writing the fused kernel
// source for it. One struct does the whole job; see [`Emit`].

use super::*;

/// A run of stages sharing a partition, and whether a barrier has to precede it.
struct Nest {
    part: Part,
    stages: Vec<usize>,
    barrier_before: bool,
}

impl ChainKey {
    /// Stages the chain records, which is what one fused launch replaces.
    pub(crate) fn stages(&self) -> usize {
        self.stages.len()
    }

    /// Groups the stages into nests, decides where a barrier is genuinely
    /// required, and emits the kernel.
    ///
    /// `Ok(None)` means the pass has no fused form for this chain and the caller
    /// should run the stages as separate launches. An `Err` is a malformed
    /// chain, which is a bug in whoever recorded it.
    pub(crate) fn plan(&self) -> Result<Option<Plan>> {
        let mut nests = self.nests()?;
        let mut nest_of = vec![usize::MAX; self.stages.len()];
        for (n, nest) in nests.iter().enumerate() {
            for &s in &nest.stages {
                nest_of[s] = n;
            }
        }

        // The whole of the pass's judgement is here. A value read outside the
        // nest that wrote it has to be in memory, and it needs a barrier unless
        // the reader can prove it is reading its own block's work.
        //
        // The walk is in chain order and the write is recorded after the reads,
        // so a value the chain both reads early and writes late, which is what
        // accumulating into the residual is, does not look like a dependency of
        // the stage that read it first.
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
                    // No f32 scratch path, so a register value crossing a nest
                    // is a chain this pass cannot fuse.
                    return Ok(None);
                }
                // A barrier costs nothing to skip only when the block that reads
                // is the block that wrote: a redundant writer means every block
                // wrote its own copy, and two nests striding the same unit count
                // over the same grid hand unit `i` to the same block in both.
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
            // A whole-row stage emits a sweep of its own, so it never shares a
            // nest. An elementwise one always joins the nest it is found in.
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
    /// Tile declarations and their flat views, which have to precede the loops
    /// that fill them however late the stage wanting one is emitted.
    decls: String,
    body: String,
    barriers: usize,
    /// Parameters already declared, by value and the shape it is seen under, so
    /// a second use under the same shape reuses the operand. A quantized value
    /// and a weight name two of them, the bytes and the scales.
    views: HashMap<(Val, View), String>,
    pairs: HashMap<(Val, View), (String, String)>,
    /// Scratch already claimed, by value.
    stored: HashMap<Val, usize>,
    /// Values every block wrote its own copy of, so a reader is reading its own
    /// work and the value belongs in shared memory rather than in scratch.
    redundant: HashSet<Val>,
    /// Those of them a stage has reached, so their tiles are declared exactly
    /// once in [`Emit::decls`].
    shared: HashSet<Val>,
    /// Tile variables holding a value that never leaves its nest.
    regs: HashMap<Val, String>,
    /// The barrier state's parameter, once some nest has needed one.
    bar: Option<String>,
    /// Whether a raw-format contraction is in the kernel, which sets its
    /// launch bound.
    raw: bool,
}

/// The shape a value is seen under.
///
/// A normalization writes rows of [`Q8_BLOCK`] and a contraction reads one flat
/// row, and both are the same bytes: the CTA barrier that trails every tile
/// store is what orders the one against the other.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum View {
    Folded,
    Flat,
    /// `[len / width, width]`, which is what the convolution walks: one position
    /// of the stream a row, so a tap is a row subscript rather than arithmetic
    /// on a flat offset.
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
        }
    }

    /// Emit one nest, or decline the chain by returning `false`.
    fn nest(&mut self, key: &ChainKey, nest: &Nest, n: usize) -> Result<bool> {
        match nest.part {
            Part::Whole(width) => {
                let [only] = nest.stages[..] else {
                    bail!("a whole-row nest holds exactly one stage");
                };
                self.whole(key, only, width)
            }
            Part::Units(units) => self.units(key, nest, n, units),
            Part::Inherit => bail!("a nest cannot itself be elementwise"),
        }
    }

    /// The redundant sweep: every block normalizes and quantizes the whole row
    /// into its own copy, so nothing has to be published before the first
    /// contraction.
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
        // The row is swept NORM_ROWS rows at a time and a row is Q8_BLOCK wide,
        // so a width that does not divide takes the unfused path.
        if !width.is_multiple_of(Q8_BLOCK * NORM_ROWS) {
            return Ok(false);
        }

        let rows = width / Q8_BLOCK;
        let nb = self.tune_const(format!("NB{s}"), rows);
        let sb = self.tune_const("SB".into(), NORM_ROWS);
        let xf = self.given(x, key.len_of(x), View::Folded);
        let gf = self.given(gain, key.len_of(gain), View::Folded);
        let (qs, sc) = self.quant_rows(out, key.len_of(out));
        let eps = f32::from_bits(eps_bits);
        let q8 = Q8_BLOCK;

        let _ = write!(
            self.body,
            "
  var acc{s}: tile<f32>[{sb}, 1] = 0.0
  for b{s} in range(0, {nb}, {sb}) {{
    let xb{s} = {xf}[b{s} :+ {sb}, 0 :+ {q8}]
    acc{s} = acc{s} + rowsum(xb{s} * xb{s})
  }}
  var tot{s}: tile<f32>[1, 1] = rowsum(transpose(acc{s}))
  var inv{s}: tile<f32>[1, 1] = 1.0 / sqrt(tot{s} / {width}.0 + {eps:.12})
  for b{s} in range(0, {nb}, {sb}) {{
    var y{s}: tile<f32>[{sb}, {q8}] = {xf}[b{s} :+ {sb}, 0 :+ {q8}] * inv{s} \
* {gf}[b{s} :+ {sb}, 0 :+ {q8}]
    var mx{s}: tile<f32>[{sb}, 1] = rowmax(tmax(y{s}, -y{s}))
    var q{s} = y{s} * (127.0 / (mx{s} + {QUANT_EPS}))
    {qs}[b{s} :+ {sb}, 0 :+ {q8}] = i8(i32(round(q{s})))
    {sc}[b{s} :+ {sb}, 0 :+ 1] = mx{s} / 127.0
  }}
"
        );
        Ok(true)
    }

    /// A grid-strided nest, one unit of work per block per turn.
    ///
    /// The iteration count is compiled in and the tail guarded, because a
    /// grid-stride loop over a dynamic extent splits and the masked remainder
    /// then wants a static shape.
    fn units(&mut self, key: &ChainKey, nest: &Nest, n: usize, units: usize) -> Result<bool> {
        let iters = units.div_ceil(self.blocks as usize);
        let it = self.tune_const(format!("IT{n}"), iters);
        let un = self.tune_const(format!("UN{n}"), units);
        let unit = format!("u{n}");
        let _ = write!(
            self.body,
            "
  for i{n} in range(0, {it}) {{
    let {unit} = p + i{n} * BLOCKS
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
                // A raw unit's run is several Q8_0 blocks: each is sliced out
                // of the register tile and quantized on its own, and lands on
                // its own row of the scratch.
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
                // A register when `h` is a nest-computed temp (the MLP's own
                // SwiGLU output), a window of a caller buffer when it is not
                // (attention's mixed heads, already resident before this
                // chain runs): the two storage classes read differently, and
                // this is the one place a stage has to tell them apart itself
                // rather than through `quant_row`'s [`View`].
                let hn = match self.regs.get(&h) {
                    Some(reg) => reg.clone(),
                    None => {
                        let flat = self.given(h, key.len_of(h), View::Flat);
                        format!("{flat}[0 :+ 1, {unit} * {q8} :+ {q8}]")
                    }
                };
                let (qs, sc) = self.quant_rows(out, key.len_of(out));
                // The scale has to be bound before the store so the elementwise
                // chain fuses into one sweep: the fusion fires on a named tile
                // and not on an expression.
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
                // The nest is wider than the gates, which cover a head each, so
                // the tail of it sits this out. The softplus is
                // max(x, 0) + log(1 + exp(-|x|)) rather than the direct
                // log(1 + exp(x)), which overflows well inside the range the
                // decay projection reaches.
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

    /// One (plane, head) pair of the causal depthwise convolution.
    ///
    /// The plane rides the unit index rather than being unrolled inside it: a
    /// block does one plane of one head, which is the parallelism the launched
    /// kernel had on its third grid axis. Unrolling instead measured 13.5
    /// microseconds a layer against the 9.5 the two launches it replaced cost.
    ///
    /// The epilogue tests the plane, which is derived from the block index and so
    /// is uniform across the CTA. That matters: the gain reduces the row, and a
    /// CTA-wide reduction inside a divergent branch would hang.
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
        // Only the query carries the readout scale, and only the query and key
        // are normalized: the value is written into the recurrent state rather
        // than matched against it, so it leaves the convolution as it is.
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
    /// offset is left out rather than compiled in as a constant nobody reads,
    /// which keeps the emitted source of a projection starting at row zero the
    /// shape it had before offsets existed.
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

    /// One counter and generation pair serves every barrier of the kernel, since
    /// the expansion leaves both as it found them.
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

    /// Declare a parameter and record what the backend must bind to it.
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

    /// The parameter a caller buffer is seen through, folded into rows of
    /// [`Q8_BLOCK`] or flat.
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
        let nb = k / 256;
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

    /// The bytes and scales of a quantized activation as rows of [`Q8_BLOCK`],
    /// which is the shape a per-block reduction produces and so what a stage
    /// stores through. The caller adds the row subscript.
    fn quant_rows(&mut self, val: Val, len: usize) -> (String, String) {
        self.quant(val, len, true)
    }

    /// The same value as one row, ready to pass to a contraction.
    ///
    /// Unlike [`Self::quant_rows`] this is a whole operand rather than a name to
    /// subscript, because the two storage classes reach the contraction
    /// differently: a shared value's flat view already *is* a one-row tile, where
    /// a global one is a tensor parameter that a slice has to turn into one.
    fn quant_row(&mut self, val: Val, len: usize) -> (String, String) {
        let (qs, scales) = self.quant(val, len, false);
        match self.shared.contains(&val) {
            true => (qs, scales),
            false => (format!("{qs}[0 :+ 1, :]"), format!("{scales}[0 :+ 1, :]")),
        }
    }

    /// Claims a quantized activation's storage on first sight under either shape.
    ///
    /// Where that storage is, is the one decision here. A value every block wrote
    /// its own copy of is read back only by the block that wrote it, so it goes
    /// in shared and the grid's worth of copies a redundant stage would otherwise
    /// publish never happens. A value one block wrote and another reads has to be
    /// global, and that is exactly the case a barrier already precedes.
    ///
    /// `folded` is a flag rather than a [`View`] because those two shapes are the
    /// only ones a quantized row has any meaning under.
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
    /// The tile is declared without an initializer, since the sweep that follows
    /// writes every element of it, and the flat view is bound next to the
    /// declaration so both are in scope before any loop. The two views are the
    /// same bytes: the folded one is what the per-block reduction produces, the
    /// flat one what the contraction reads.
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
        // Two CTAs of 256 an SM with a raw decode in the kernel: the
        // intrinsics' own bound of three spills there.
        let launch = match self.raw {
            true => format!("{CTA}, 2"),
            false => CTA.to_string(),
        };
        // The shared declarations go first whatever order the stages wanted them
        // in, since a tile has to be in scope before the loop that fills it.
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
