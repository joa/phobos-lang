// Runs one CTA of a kernel's emitted MLIR, every thread side by side with an
// active mask, so per-thread loops and branches run as the hardware runs
// them. It computes integers and memref layouts, never data: a load from a
// tensor or a tile yields nothing, and control that needs such a value stops
// the run as untraceable. Shared-memory accesses go to [`Memory`].
//
// A value the same in every thread is held once, and a loop or branch whose
// bounds or condition are runs once for the whole mask, which is most of a
// kernel's index arithmetic.

use std::collections::HashMap;
use std::rc::Rc;

use super::arith::{arith, compare, extent, linear, row_major, unsigned_of, wrap};
use super::memory::{Memory, Touch};
use super::program::{Kind, MemTy, Op, Program, Region, Space, Ty, Vid};

/// Op executions, uniform ones counted once, before a run gives up.
const BUDGET: u64 = 200_000_000;

const WARP: usize = 32;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Mem {
    pub buf: Buf,
    /// The byte the view starts from in its buffer.
    pub base: i64,
    /// Elements from `base` to element zero.
    pub offset: Option<i64>,
    pub sizes: Vec<Option<i64>>,
    pub strides: Vec<Option<i64>>,
    pub elem: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Buf {
    Shared(usize),
    Global,
    /// A thread's own buffer, by the op that made it.
    Private(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum V {
    Unknown,
    Int(i64),
    Mem(Rc<Mem>),
}

/// A value in every thread: held once when it is the same in all of them.
#[derive(Clone, Debug)]
pub(super) enum Lanes {
    Same(V),
    Each(Vec<V>),
}

/// Why a run ended before the kernel did.
pub(super) enum Stop {
    Trap(usize),
    Untraceable(String),
}

pub(super) struct Inputs {
    pub block: [i64; 3],
    pub grid: [i64; 3],
    /// Every dynamic tensor extent, and every integer scalar parameter.
    pub extent: i64,
    pub scalar: i64,
}

pub(super) struct Exec<'p> {
    pub(super) p: &'p Program,
    pub(super) n: usize,
    inputs: Inputs,
    pub(super) vals: Vec<Lanes>,
    private: Vec<HashMap<(usize, i64), i64>>,
    stamps: Vec<u64>,
    shared: HashMap<String, usize>,
    dynamic: usize,
    dynamic_bytes: i64,
    pub mem: Memory,
    pub divergent: Vec<(usize, usize)>,
    /// Shared accesses through a memref the run could not follow.
    pub unfollowed: u64,
    /// Loops that waited on data, taken to end.
    pub waits: u64,
    spent: u64,
}

impl<'p> Exec<'p> {
    pub(super) fn new(p: &'p Program, inputs: Inputs, dynamic_bytes: i64) -> Exec<'p> {
        let n = p.threads as usize;
        let mut buffers: Vec<(String, i64)> =
            p.globals.iter().map(|(k, &v)| (k.clone(), v)).collect();
        buffers.sort();
        let shared = buffers
            .iter()
            .enumerate()
            .map(|(i, (k, _))| (k.clone(), i))
            .collect();
        let dynamic = buffers.len();
        buffers.push(("dynamic shared memory".to_string(), dynamic_bytes));
        Exec {
            p,
            n,
            inputs,
            vals: vec![Lanes::Same(V::Unknown); p.types.len()],
            private: vec![HashMap::new(); n],
            stamps: vec![0; n.div_ceil(WARP)],
            shared,
            dynamic,
            dynamic_bytes,
            mem: Memory::new(buffers),
            divergent: Vec::new(),
            unfollowed: 0,
            waits: 0,
            spent: 0,
        }
    }

    pub(super) fn run(&mut self) -> Result<(), Stop> {
        for (v, ty) in self.p.params.clone() {
            let val = match ty {
                Ty::Mem(m) => {
                    // Each dynamic extent a multiple of what an `@aligned`
                    // check at entry asks of it, so the run does not trap.
                    let sizes: Vec<Option<i64>> = (0..m.shape.len())
                        .filter(|&d| m.shape[d].is_none())
                        .map(|d| {
                            let unit = self.p.aligned.get(&(v, d)).copied().unwrap_or(1);
                            Some((self.inputs.extent + unit - 1) / unit * unit)
                        })
                        .collect();
                    V::Mem(Rc::new(self.fresh(Buf::Global, &m, &sizes)))
                }
                Ty::Int(_) => V::Int(self.inputs.scalar),
                _ => V::Unknown,
            };
            self.vals[v] = Lanes::Same(val);
        }
        let all = vec![true; self.n];
        let p = self.p;
        self.region(&p.body, &all)
    }

    /// A row-major view of a whole `m`, its dynamic sizes from `sizes` or
    /// the inputs' extent.
    fn fresh(&self, buf: Buf, m: &MemTy, sizes: &[Option<i64>]) -> Mem {
        let mut dynamic = sizes.iter();
        let sizes: Vec<Option<i64>> = m
            .shape
            .iter()
            .map(|d| {
                d.or_else(|| {
                    dynamic
                        .next()
                        .copied()
                        .flatten()
                        .or(Some(self.inputs.extent))
                })
            })
            .collect();
        let strides = match &m.strides {
            Some(s) => s.clone(),
            None => row_major(&sizes),
        };
        Mem {
            buf,
            base: 0,
            offset: m.offset.unwrap_or(Some(0)),
            sizes,
            strides,
            elem: m.elem,
        }
    }

    pub(super) fn get(&self, v: Vid, t: usize) -> V {
        match &self.vals[v] {
            Lanes::Same(x) => x.clone(),
            Lanes::Each(xs) => xs.get(t).cloned().unwrap_or(V::Unknown),
        }
    }

    pub(super) fn int(&self, v: Vid, t: usize) -> Option<i64> {
        match &self.vals[v] {
            Lanes::Same(V::Int(x)) => Some(*x),
            Lanes::Each(xs) => match xs.get(t) {
                Some(V::Int(x)) => Some(*x),
                _ => None,
            },
            _ => None,
        }
    }

    fn mem(&self, v: Vid, t: usize) -> Option<Rc<Mem>> {
        match self.get(v, t) {
            V::Mem(m) => Some(m),
            _ => None,
        }
    }

    pub(super) fn same(&self, v: Vid) -> bool {
        matches!(self.vals[v], Lanes::Same(_))
    }

    /// Sets `result` to `f` of each thread, once when every operand is the
    /// same in all of them.
    fn map(&mut self, operands: &[Vid], result: Vid, f: impl Fn(&Self, usize) -> V) {
        self.vals[result] = if operands.iter().all(|&v| self.same(v)) {
            Lanes::Same(f(self, 0))
        } else {
            Lanes::Each((0..self.n).map(|t| f(self, t)).collect())
        };
    }

    fn width(&self, v: Vid) -> u32 {
        match self.p.types[v] {
            Ty::Int(bits) => bits,
            _ => 64,
        }
    }

    pub(super) fn region(&mut self, r: &Region, mask: &[bool]) -> Result<(), Stop> {
        if !mask.iter().any(|&m| m) {
            return Ok(());
        }
        for &op in &r.ops {
            self.op(op, mask)?;
        }
        Ok(())
    }

    pub(super) fn threads(&self, mask: &[bool]) -> Vec<usize> {
        (0..self.n).filter(|&t| mask[t]).collect()
    }

    fn op(&mut self, i: usize, mask: &[bool]) -> Result<(), Stop> {
        let p = self.p;
        let op = &p.ops[i];
        self.spent += 1;
        if self.spent > BUDGET {
            return Err(Stop::Untraceable("the run took too long".to_string()));
        }
        let n = self.n;
        match &op.kind {
            Kind::Const(c) => self.vals[op.results[0]] = Lanes::Same(c.map_or(V::Unknown, V::Int)),
            Kind::Arith(a) => {
                let bits = self.width(op.results[0]);
                let (x, y) = (op.operands[0], op.operands[1]);
                self.map(&op.operands, op.results[0], |s, t| {
                    match (s.int(x, t), s.int(y, t)) {
                        (Some(x), Some(y)) => arith(*a, x, y, bits).map_or(V::Unknown, V::Int),
                        _ => V::Unknown,
                    }
                });
            }
            Kind::Cmp(pred) => {
                let bits = self.width(op.operands[0]);
                let (x, y) = (op.operands[0], op.operands[1]);
                self.map(&op.operands, op.results[0], |s, t| {
                    match (s.int(x, t), s.int(y, t)) {
                        (Some(x), Some(y)) => {
                            compare(*pred, x, y, bits).map_or(V::Unknown, |b| V::Int(b as i64))
                        }
                        _ => V::Unknown,
                    }
                });
            }
            Kind::Select => {
                let [c, a, b] = [op.operands[0], op.operands[1], op.operands[2]];
                self.map(&op.operands, op.results[0], |s, t| match s.int(c, t) {
                    Some(0) => s.get(b, t),
                    Some(_) => s.get(a, t),
                    None => V::Unknown,
                });
            }
            Kind::Convert(unsigned) => {
                let (from, to) = (self.width(op.operands[0]), self.width(op.results[0]));
                let x = op.operands[0];
                self.map(&op.operands, op.results[0], |s, t| match s.int(x, t) {
                    Some(x) => {
                        let x = if *unsigned {
                            unsigned_of(x, from) as i64
                        } else {
                            x
                        };
                        V::Int(wrap(x, to))
                    }
                    None => V::Unknown,
                });
            }
            Kind::Thread => {
                self.vals[op.results[0]] = Lanes::Each((0..n).map(|t| V::Int(t as i64)).collect())
            }
            Kind::BlockDim => self.vals[op.results[0]] = Lanes::Same(V::Int(n as i64)),
            Kind::Block(d) => self.vals[op.results[0]] = Lanes::Same(V::Int(self.inputs.block[*d])),
            Kind::Grid(d) => self.vals[op.results[0]] = Lanes::Same(V::Int(self.inputs.grid[*d])),
            Kind::Lane => {
                self.vals[op.results[0]] =
                    Lanes::Each((0..n).map(|t| V::Int((t % WARP) as i64)).collect())
            }
            Kind::Dim => {
                let (m, d) = (op.operands[0], op.operands[1]);
                self.map(&op.operands, op.results[0], |s, t| {
                    match (s.mem(m, t), s.int(d, t)) {
                        (Some(m), Some(d)) => m
                            .sizes
                            .get(d as usize)
                            .copied()
                            .flatten()
                            .map_or(V::Unknown, V::Int),
                        _ => V::Unknown,
                    }
                });
            }
            Kind::View | Kind::Subview(..) | Kind::Reinterpret(..) | Kind::Alias => {
                self.map(&op.operands, op.results[0], |s, t| s.view(op, t));
            }
            Kind::Global(name) => {
                let v = match (self.shared.get(name), &p.types[op.results[0]]) {
                    (Some(&b), Ty::Mem(m)) => V::Mem(Rc::new(self.fresh(Buf::Shared(b), m, &[]))),
                    _ => V::Unknown,
                };
                self.vals[op.results[0]] = Lanes::Same(v);
            }
            Kind::DynShared | Kind::Alloca => {
                let buf = if op.kind == Kind::Alloca {
                    Buf::Private(i)
                } else {
                    Buf::Shared(self.dynamic)
                };
                let v = match &p.types[op.results[0]] {
                    Ty::Mem(m) => {
                        let sizes: Vec<Option<i64>> = if op.kind == Kind::DynShared {
                            vec![Some(self.dynamic_bytes)]
                        } else {
                            op.operands.iter().map(|_| None).collect()
                        };
                        V::Mem(Rc::new(self.fresh(buf, m, &sizes)))
                    }
                    _ => V::Unknown,
                };
                self.vals[op.results[0]] = Lanes::Same(v);
            }
            Kind::Load | Kind::VLoad => {
                let bytes = self.bytes_of(op.results[0]);
                let mut out = vec![V::Unknown; n];
                let mut any = false;
                for t in self.threads(mask) {
                    if let Some(v) =
                        self.access(i, t, op.operands[0], &op.operands[1..], bytes, false, None)
                    {
                        out[t] = V::Int(v);
                        any = true;
                    }
                }
                self.vals[op.results[0]] = if any {
                    Lanes::Each(out)
                } else {
                    Lanes::Same(V::Unknown)
                };
            }
            Kind::Store | Kind::VStore => {
                let bytes = self.bytes_of(op.operands[0]);
                for t in self.threads(mask) {
                    let value = self.int(op.operands[0], t);
                    self.access(
                        i,
                        t,
                        op.operands[1],
                        &op.operands[2..],
                        bytes,
                        true,
                        Some(value),
                    );
                }
            }
            Kind::LdMatrix(tiles) => {
                // Each of the first 8 lanes per matrix names one 16-byte row.
                for t in self
                    .threads(mask)
                    .into_iter()
                    .filter(|t| ((t % WARP) as i64) < 8 * tiles)
                {
                    self.access(i, t, op.operands[0], &op.operands[1..], 16, false, None);
                }
                self.warp_sync(mask);
                self.unknown_results(op);
            }
            Kind::AsyncCopy([d, di, _, _], elems) => {
                let (dst, idx) = (op.operands[0], &op.operands[*d..*d + *di]);
                for t in self.threads(mask) {
                    if let Some((buffer, lo, hi, touch)) = self.locate(i, t, dst, idx, None, true) {
                        let bytes = elems * self.mem(dst, t).map_or(1, |m| m.elem);
                        self.mem
                            .copy_async(buffer, lo, lo + bytes.max(hi - lo), touch);
                    }
                }
                self.unknown_results(op);
            }
            Kind::AsyncGroup => {
                for t in self.threads(mask) {
                    self.mem.commit(t as u32);
                }
                self.unknown_results(op);
            }
            Kind::AsyncWait(k) => {
                for t in self.threads(mask) {
                    self.mem.wait(t as u32, *k);
                }
            }
            Kind::MmaLoad(ld) | Kind::MmaStore(ld) => {
                let load = matches!(op.kind, Kind::MmaLoad(_));
                let (frag, mem, idx) = if load {
                    (op.results[0], op.operands[0], &op.operands[1..])
                } else {
                    (op.operands[0], op.operands[1], &op.operands[2..])
                };
                let (Ty::Frag(rows, cols, _), Ty::Mem(m)) = (&p.types[frag], &p.types[mem]) else {
                    return Err(Stop::Untraceable(format!(
                        "a fragment of type {:?}",
                        p.types[frag]
                    )));
                };
                let (rows, row_bytes) = (*rows, cols * m.elem);
                // The warp moves the fragment together; its first active
                // lane stands for it.
                for w in 0..n.div_ceil(WARP) {
                    let Some(t) = (w * WARP..((w + 1) * WARP).min(n)).find(|&t| mask[t]) else {
                        continue;
                    };
                    for r in 0..rows {
                        self.access(i, t, mem, idx, row_bytes, !load, Some(Some(r * ld)));
                    }
                }
                self.warp_sync(mask);
                self.unknown_results(op);
            }
            Kind::Barrier => {
                let arrived = self.threads(mask).len();
                if arrived != n {
                    self.divergent.push((i, arrived));
                }
                self.mem.barrier();
            }
            Kind::WarpSync => {
                self.warp_sync(mask);
                self.unknown_results(op);
            }
            Kind::For => self.for_loop(op, mask)?,
            Kind::AffineFor(lo, hi, step) => {
                let body = &op.regions[0];
                let mut iv = *lo;
                while iv < *hi {
                    self.vals[body.args[0]] = Lanes::Same(V::Int(iv));
                    self.region(body, mask)?;
                    iv += step;
                }
            }
            Kind::If => self.branch(op, mask)?,
            Kind::While => self.while_loop(op, mask)?,
            Kind::Trap => {
                if mask.iter().any(|&m| m) {
                    return Err(Stop::Trap(i));
                }
            }
            Kind::Yield | Kind::Condition | Kind::Return => {}
            Kind::Opaque => self.unknown_results(op),
        }
        Ok(())
    }

    pub(super) fn unknown_results(&mut self, op: &Op) {
        for &r in &op.results {
            self.vals[r] = Lanes::Same(V::Unknown);
        }
    }

    fn warp_sync(&mut self, mask: &[bool]) {
        for w in 0..self.stamps.len() {
            if (w * WARP..((w + 1) * WARP).min(self.n)).any(|t| mask[t]) {
                self.stamps[w] += 1;
            }
        }
    }

    /// The bytes a vector moves, or zero for a scalar, which moves one
    /// element of the memref it goes through.
    fn bytes_of(&self, v: Vid) -> i64 {
        match &self.p.types[v] {
            Ty::Vector(b) => *b,
            _ => 0,
        }
    }

    /// Where one thread's access lands: its buffer, its byte range, and the
    /// touch to record. `None` for global or private memory, or a memref the
    /// run could not follow. `extra` adds elements past the indexed one.
    fn locate(
        &mut self,
        site: usize,
        t: usize,
        mem: Vid,
        idx: &[Vid],
        extra: Option<i64>,
        write: bool,
    ) -> Option<(usize, i64, i64, Touch)> {
        let Some(m) = self.mem(mem, t) else {
            if matches!(&self.p.types[mem], Ty::Mem(m) if m.space == Space::Shared) {
                self.unfollowed += 1;
            }
            return None;
        };
        let Buf::Shared(buffer) = m.buf else {
            return None;
        };
        let at = linear(&m, &idx.iter().map(|&v| self.int(v, t)).collect::<Vec<_>>());
        let touch = Touch {
            thread: t as u32,
            site: site as u32,
            write,
            stamp: self.stamps[t / WARP],
            collective: false,
            exact: at.is_some(),
        };
        match at {
            Some(at) => {
                let lo = m.base + (at + extra.unwrap_or(0)) * m.elem;
                Some((buffer, lo, lo + m.elem, touch))
            }
            None => {
                let (lo, hi) = extent(&m).unwrap_or((0, i64::MAX));
                Some((buffer, lo, hi, touch))
            }
        }
    }

    /// One thread's load or store of `bytes` bytes, zero for one element. A
    /// private buffer keeps integer values, and a load from it returns what
    /// was stored.
    #[allow(clippy::too_many_arguments)]
    fn access(
        &mut self,
        site: usize,
        t: usize,
        mem: Vid,
        idx: &[Vid],
        bytes: i64,
        write: bool,
        value: Option<Option<i64>>,
    ) -> Option<i64> {
        if let Some(m) = self.mem(mem, t)
            && let Buf::Private(a) = m.buf
        {
            let at = linear(&m, &idx.iter().map(|&v| self.int(v, t)).collect::<Vec<_>>())?;
            if write {
                match value.flatten() {
                    Some(v) => self.private[t].insert((a, at), v),
                    None => self.private[t].remove(&(a, at)),
                };
                return None;
            }
            return self.private[t].get(&(a, at)).copied();
        }
        // A fragment row: `value` carries the row's element offset.
        let extra = if matches!(self.p.ops[site].kind, Kind::MmaLoad(_) | Kind::MmaStore(_)) {
            value.flatten()
        } else {
            None
        };
        let (buffer, lo, hi, mut touch) = self.locate(site, t, mem, idx, extra, write)?;
        touch.collective = extra.is_some();
        let hi = if touch.exact && bytes > 0 {
            lo + bytes
        } else {
            hi
        };
        self.mem.touch(buffer, lo, hi, touch);
        None
    }

    /// One thread's view: a byte shift, a subview, a reinterpretation or an
    /// alias of its source.
    fn view(&self, op: &Op, t: usize) -> V {
        let Some(src) = self.mem(op.operands[0], t) else {
            return V::Unknown;
        };
        let ints: Vec<Option<i64>> = op.operands[1..].iter().map(|&v| self.int(v, t)).collect();
        let result = match &self.p.types[op.results[0]] {
            Ty::Mem(m) => m.clone(),
            _ => return V::Unknown,
        };
        let m = match &op.kind {
            Kind::Alias => return V::Mem(src),
            Kind::View => {
                let shift = ints.first().copied().flatten();
                let mut m = self.fresh(src.buf, &result, &ints[1.min(ints.len())..]);
                m.base = src.base + src.offset.unwrap_or(0) * src.elem + shift.unwrap_or(0);
                if shift.is_none() {
                    m.offset = None;
                }
                m
            }
            Kind::Subview(off, size, stride) | Kind::Reinterpret(off, size, stride) => {
                let mut dynamic = ints.iter().copied();
                let mut fill = |v: &Vec<Option<i64>>| -> Vec<Option<i64>> {
                    v.iter()
                        .map(|s| s.or_else(|| dynamic.next().flatten()))
                        .collect()
                };
                let (off, size, stride) = (fill(off), fill(size), fill(stride));
                let mut m = (*src).clone();
                if matches!(op.kind, Kind::Reinterpret(..)) {
                    m.offset = off.first().copied().flatten();
                    m.sizes = size;
                    m.strides = stride;
                } else {
                    let shift: Option<i64> = off
                        .iter()
                        .zip(&src.strides)
                        .map(|(o, s)| Some(o.as_ref()? * s.as_ref()?))
                        .sum();
                    m.offset = src.offset.zip(shift).map(|(a, b)| a + b);
                    let mut strides: Vec<Option<i64>> = stride
                        .iter()
                        .zip(&src.strides)
                        .map(|(a, b)| a.zip(*b).map(|(a, b)| a * b))
                        .collect();
                    let mut sizes = size;
                    // A rank-reducing subview drops unit dimensions.
                    while sizes.len() > result.shape.len() {
                        let Some(at) = sizes.iter().position(|s| *s == Some(1)) else {
                            break;
                        };
                        sizes.remove(at);
                        strides.remove(at);
                    }
                    m.sizes = sizes;
                    m.strides = strides;
                }
                m
            }
            _ => return V::Unknown,
        };
        V::Mem(Rc::new(m))
    }
}
