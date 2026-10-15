// Plans a chain into loop nests and writes the fused kernel source for it.
// See [`Emit`].

use super::*;

mod conv;
mod lanes;
mod split;

/// A run of stages sharing a partition, and whether it starts its units on
/// block 0 again.
struct Nest {
    part: Part,
    stages: Vec<usize>,
    fresh: bool,
}

impl ChainKey {
    /// Number of stages in the chain.
    pub(crate) fn stages(&self) -> usize {
        self.stages.len()
    }

    /// Groups the stages into nests and writes the kernel. The compiler
    /// places its grid barriers and decides which scratch it holds.
    ///
    /// `Ok(None)` means the chain has no fused form and the caller should
    /// launch the stages separately. An `Err` means the chain was recorded
    /// wrong.
    pub(crate) fn plan(&self) -> Result<Option<Plan>> {
        let mut nests = self.nests()?;
        // A whole-row nest's output is every block's own copy. A nest that
        // reads what a unit nest wrote since the last fresh one waits at a
        // grid barrier, so it starts on block 0 rather than on the blocks
        // the nests before it left idle. Writes count after their stage's
        // reads, so the residual, read first and written last, is no
        // dependency of its first reader.
        let mut redundant: HashSet<Val> = HashSet::new();
        let mut since: HashSet<Val> = HashSet::new();
        for nest in &mut nests {
            let stages = || nest.stages.iter().map(|&s| &self.stages[s]);
            nest.fresh = stages().any(|stage| stage.reads().iter().any(|v| since.contains(v)));
            if nest.fresh {
                since.clear();
            }
            let written = stages().map(Stage::writes).collect::<Vec<_>>();
            match nest.part {
                Part::Whole(_) => redundant.extend(written),
                _ => since.extend(written),
            }
        }

        // Written twice: once to learn the views each caller buffer is read
        // under, then with each buffer one parameter, so the compiler sees
        // every access to it.
        let write = |shapes: Option<HashMap<Val, View>>| -> Result<Option<Emit>> {
            let mut emit = Emit::new(self.blocks, redundant.clone(), shapes);
            for (n, nest) in nests.iter().enumerate() {
                emit.land_lane_stores();
                if !emit.nest(self, nest, n)? {
                    return Ok(None);
                }
            }
            Ok(Some(emit))
        };
        let Some(shapes) = write(None)?.and_then(|dry| dry.shapes()) else {
            return Ok(None);
        };
        Ok(write(Some(shapes))?.map(Emit::finish))
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
                    fresh: false,
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
    scratch_args: Vec<ScratchArg>,
    /// Tile declarations and their flat views. They go before all loops.
    decls: String,
    body: String,
    /// The shape each caller buffer is declared under, None while learning
    /// them, and every view asked for.
    shapes: Option<HashMap<Val, View>>,
    wanted: HashSet<(Val, View)>,
    /// Parameters already declared, keyed by value and view, so a repeat use
    /// reuses the operand. `pairs` holds the bytes and scales of quantized
    /// values and weights.
    views: HashMap<(Val, View), String>,
    pairs: HashMap<(Val, View), (String, String)>,
    /// Scratch already claimed, by value.
    stored: HashMap<Val, usize>,
    /// Values every block wrote its own copy of, which the compiler holds in
    /// shared memory.
    redundant: HashSet<Val>,
    /// The redundant values already declared in [`Emit::decls`].
    shared: HashSet<Val>,
    /// Their declared names, which the kernel asserts `@shared`.
    held: Vec<String>,
    /// Tile variables holding a value that never leaves its nest.
    regs: HashMap<Val, String>,
    /// Per-lane column sums of a raw contraction, by value, until a tile
    /// stage reads one.
    lanes: HashMap<Val, String>,
    /// The block decodes the raw contractions call, by format.
    lane_fns: std::collections::BTreeMap<&'static str, String>,
    /// Whether a per-lane store has had no barrier since.
    lane_stores: bool,
    /// Whether the kernel has a raw-format contraction, which sets its
    /// launch bound.
    raw: bool,
    /// Units the nests since the last barrier hand out, so the next
    /// independent nest starts on the blocks they leave idle.
    rot: usize,
}

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
    fn new(blocks: u32, redundant: HashSet<Val>, shapes: Option<HashMap<Val, View>>) -> Emit {
        Emit {
            shapes,
            wanted: HashSet::new(),
            blocks,
            tune: vec![("BLOCKS".into(), blocks as usize)],
            params: Vec::new(),
            slots: Vec::new(),
            scratch: Vec::new(),
            scratch_args: Vec::new(),
            decls: "  let p = ctx.block.x\n".into(),
            body: String::new(),
            views: HashMap::new(),
            pairs: HashMap::new(),
            stored: HashMap::new(),
            redundant,
            shared: HashSet::new(),
            held: Vec::new(),
            regs: HashMap::new(),
            lanes: HashMap::new(),
            lane_fns: Default::default(),
            lane_stores: false,
            raw: false,
            rot: 0,
        }
    }

    /// Emits one nest. Returns `false` to decline the chain.
    fn nest(&mut self, key: &ChainKey, nest: &Nest, n: usize) -> Result<bool> {
        // A register value is in scope only in the nest that writes it.
        self.regs.clear();
        self.lanes.clear();
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
        let operands = NormOperands { x: &xf, g: &gf, z: None, out: None, q: &qs, s: &sc, row0: "0" };
        let _ = write!(self.body, "\n{}", rms_norm_q_lanes(&operands, &nb, eps, &format!("s{s}")));
        Ok(true)
    }

    /// A grid-strided nest, one unit of work per block per iteration.
    fn units(&mut self, key: &ChainKey, nest: &Nest, n: usize, units: usize) -> Result<bool> {
        let un = self.tune_const(format!("UN{n}"), units);
        let unit = format!("u{n}");
        // Nests between two barriers are independent, so each starts its
        // units on the blocks the ones before it left idle.
        if nest.fresh {
            self.rot = 0;
        }
        let rot = self.rot % self.blocks as usize;
        self.rot += units;
        let base = match rot {
            0 => "p".to_string(),
            rot => format!("(p + {}) % ctx.blocks.x", self.blocks as usize - rot),
        };
        let _ = write!(self.body, "\n  for {unit} in range({base}, {un}, ctx.blocks.x) {{\n");
        for &s in &nest.stages {
            if !self.unit_stage(key, s, &unit)? {
                return Ok(false);
            }
        }
        let _ = writeln!(self.body, "  }}");
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
                let act = self.quant(a, key.len_of(a), false);
                let (rq, rd, fmt) = self.raw_weight(key, w)?;
                let at = self.strided(format!("OF{s}"), row_off, unit, RAW_UNIT);
                let _ = writeln!(self.body, "      let at{s} = {at}");
                let nb = key.len_of(a) / 256;
                let tot = self.raw_lanes(s, fmt, &rq, &rd, (&act.0, &act.1), &format!("at{s}"), "0", nb);
                self.lanes.insert(out, tot);
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
                let act = self.quant(a, key.len_of(a), false);
                let (rq, rd, fmt) = self.raw_weight(key, w)?;
                let dst = self.given(out, key.len_of(out), View::Flat);
                let at = self.strided(format!("OF{s}"), row_off, unit, RAW_UNIT);
                let to = self.strided(format!("DO{s}"), out_off, unit, RAW_UNIT);
                let _ = writeln!(self.body, "      let at{s} = {at}\n      let to{s} = {to}");
                // A remainder decodes a whole unit and stores only its head.
                let nb = key.len_of(a) / 256;
                let tot = self.raw_lanes(s, fmt, &rq, &rd, (&act.0, &act.1), &format!("at{s}"), "0", nb);
                self.store_lanes(&dst, &format!("to{s}"), &tot, width);
            }
            Stage::Swiglu { g, u, out } if self.lanes.contains_key(&g) && self.lanes.contains_key(&u) => {
                let (gn, un) = (self.lanes[&g].clone(), self.lanes[&u].clone());
                let _ = writeln!(self.body, "      let sw{s} = {gn} / (1.0 + exp(0.0 - {gn})) * {un}");
                self.lanes.insert(out, format!("sw{s}"));
            }
            Stage::Swiglu { g, u, out } => {
                let (Some(gn), Some(un)) = (self.reg_of(g), self.reg_of(u)) else {
                    return Ok(false);
                };
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
                let Some(hn) = self.reg_of(h) else {
                    return Ok(false);
                };
                let (qs, sc) = self.quant_rows(out, key.len_of(out));
                for b in 0..blocks {
                    let (off, row) = (b * q8, format!("{unit} * {blocks} + {b}"));
                    let _ = write!(
                        self.body,
                        "      var h{s}_{b}: tile<f32>[1, {q8}] = {hn}[0 :+ 1, {off} :+ {q8}]
      var m{s}_{b}: tile<f32>[1, 1] = rowmax(abs(h{s}_{b}))
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
                let hn = match self.reg_of(h) {
                    Some(reg) => reg,
                    None if matches!(key.vals[h.0], Kind::Temp { .. }) => return Ok(false),
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
                    "      var m{s}: tile<f32>[1, 1] = rowmax(abs({hn}))
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
                let act = self.quant(a, key.len_of(a), false);
                let (rq, rd, fmt) = self.raw_weight(key, w)?;
                let yn = self.given(y, key.len_of(y), View::Flat);
                let _ = writeln!(self.body, "      let t{s} = {unit} * {RAW_UNIT}");
                let nb = key.len_of(a) / 256;
                let tot = self.raw_lanes(s, fmt, &rq, &rd, (&act.0, &act.1), &format!("t{s}"), "0", nb);
                self.store_lanes(&yn, &format!("t{s}"), &format!("{yn}[0, t{s} + cu] + {tot}"), RAW_UNIT);
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
        var sp{s}: tile<f32>[1, 1] = tmax(a{s}, z{s}) + log(1.0 + exp(-abs(a{s})))
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

    /// `val`'s register, or None when no stage of this nest wrote it. There
    /// is no f32 scratch, so a register value cannot cross a nest.
    fn reg_of(&mut self, val: Val) -> Option<String> {
        if let Some(reg) = self.regs.get(&val) {
            return Some(reg.clone());
        }
        let lane = self.lanes.get(&val).cloned()?;
        Some(self.lanes_tile(val, &lane))
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

    /// Each caller buffer's declared shape: the one view other than flat it
    /// is read under, if any. None when a buffer is read under two.
    fn shapes(&self) -> Option<HashMap<Val, View>> {
        let mut shapes = HashMap::new();
        for &(val, view) in &self.wanted {
            let shape = shapes.entry(val).or_insert(View::Flat);
            match (*shape, view) {
                (_, View::Flat) => {}
                (View::Flat, _) => *shape = view,
                (shape, view) if shape == view => {}
                _ => return None,
            }
        }
        Some(shapes)
    }

    /// A caller buffer under `view`: its one parameter, or that
    /// parameter's flat view.
    fn given(&mut self, val: Val, len: usize, view: View) -> String {
        if let Some(name) = self.views.get(&(val, view)) {
            return name.clone();
        }
        self.wanted.insert((val, view));
        let shape = self.shapes.as_ref().and_then(|s| s.get(&val)).copied().unwrap_or(view);
        if shape != view {
            let base = self.given(val, len, shape);
            let name = format!("X{}", val.0);
            let _ = writeln!(self.decls, "  let {name} = flat({base})");
            self.views.insert((val, view), name.clone());
            return name;
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
        // Scratch the launch supplies, declared in the kernel as rows of
        // `Q8_BLOCK`, with its one-row view beside it.
        if !self.stored.contains_key(&val) {
            let at = self.claim_scratch(Scratch {
                bytes: len,
                scales: len / Q8_BLOCK,
            });
            self.stored.insert(val, at);
            let (rows, q8, v) = (len / Q8_BLOCK, Q8_BLOCK, val.0);
            self.declare_scratch(at, false, &format!("Vq{v}F"), "i8", [rows as i64, q8 as i64]);
            self.declare_scratch(at, true, &format!("Vs{v}F"), "f32", [rows as i64, 1]);
            let _ = write!(self.decls, "  let Vq{v} = flat(Vq{v}F)\n  let Vs{v} = flat(Vs{v}F)\n");
        }
        let suffix = if folded { "F" } else { "" };
        let pair = (format!("Vq{}{suffix}", val.0), format!("Vs{}{suffix}", val.0));
        self.pairs.insert((val, view), pair.clone());
        pair
    }

    /// A scratch pair the launch allocates, by index.
    pub(super) fn claim_scratch(&mut self, need: Scratch) -> usize {
        self.scratch.push(need);
        self.scratch.len() - 1
    }

    /// Declares one buffer of scratch pair `at` as the kernel's tensor `name`.
    pub(super) fn declare_scratch(&mut self, at: usize, scales: bool, name: &str, elem: &str, dims: [i64; 2]) {
        let _ = writeln!(self.decls, "  var {name}: tensor<{elem}>[{}, {}]", dims[0], dims[1]);
        self.scratch_args.push(ScratchArg {
            name: name.to_string(),
            at,
            scales,
            dims,
        });
    }

    /// A redundantly written activation, which the compiler holds in shared
    /// memory since every block writes all of it.
    ///
    /// Its flat view is declared alongside it, before any loop. Both views
    /// are the same bytes: the reduction writes the folded one and the
    /// contraction reads the flat one.
    fn quant_shared(&mut self, val: Val, len: usize, folded: bool) -> (String, String) {
        let rows = len / Q8_BLOCK;
        let (qs, scales) = (format!("Aq{}", val.0), format!("As{}", val.0));
        if self.shared.insert(val) {
            self.held.extend([qs.clone(), scales.clone()]);
            let q8 = Q8_BLOCK;
            let _ = write!(
                self.decls,
                "  var {qs}: tensor<i8>[{rows}, {q8}]
  var {scales}: tensor<f32>[{rows}, 1]
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
        let fns: String = self.lane_fns.values().map(|f| format!("{f}\n")).collect();
        let shared = match self.held.is_empty() {
            true => String::new(),
            false => format!("@shared({})\n", self.held.join(", ")),
        };
        let source = format!(
            "{fns}@launch({launch})\n@persistent(BLOCKS)\n{shared}@autotune({tune})\nkernel fused({params}) {{\n{}{}}}\n",
            self.decls, self.body
        );
        Plan {
            source,
            slots: self.slots,
            scratch: self.scratch,
            scratch_args: self.scratch_args,
            blocks: self.blocks,
        }
    }
}
