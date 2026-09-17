use std::fmt::{self, Write as _};

use super::{BlockId, Ir, OpId, OpKind, ValueId};

/// One op per line: results first, then the kind and its attributes, then
/// the operands, with blocks indented under the op that owns them. The
/// order is deterministic, and what the tests grep.
///
/// ```text
/// kernel matmul(%A.0: tensor<f32>[?, ?], %n.1: index) {
///   %2: index = const 0
///   %acc.3: tile<f32>[64, 64]@shared = alloc
///   for %2, %n.1, %2 {
///   ^(%i.4: index)
///     %5: tile<f32>[64, 64]@global = slice [64, 64] %A.0, %i.4, %2
///     yield
///   }
/// }
/// ```
impl fmt::Display for Ir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut p = Printer {
            ir: self,
            out: String::new(),
            depth: 0,
        };
        p.kernel();
        f.write_str(&p.out)
    }
}

struct Printer<'a> {
    ir: &'a Ir,
    out: String,
    depth: usize,
}

impl Printer<'_> {
    fn kernel(&mut self) {
        let entry = self.ir.entry();
        let _ = write!(self.out, "kernel {}(", self.ir.kernel.name);
        self.args(entry);
        self.out.push_str(") {\n");
        self.depth = 1;
        self.ops(entry);
        self.out.push_str("}\n");
    }

    fn args(&mut self, block: BlockId) {
        for (i, &arg) in self.ir.args(block).iter().enumerate() {
            if i > 0 {
                self.out.push_str(", ");
            }
            let _ = write!(self.out, "{}: {}", self.value(arg), self.ir.ty(arg));
        }
    }

    fn value(&self, v: ValueId) -> String {
        match self.ir.name(v) {
            Some(name) => format!("%{name}.{}", v.index()),
            None => v.to_string(),
        }
    }

    fn indent(&mut self) {
        for _ in 0..self.depth {
            self.out.push_str("  ");
        }
    }

    fn ops(&mut self, block: BlockId) {
        for &op in self.ir.ops(block) {
            self.op(op);
        }
    }

    fn op(&mut self, op: OpId) {
        self.indent();
        let results = self.ir.results(op);
        if !results.is_empty() {
            let rs: Vec<String> = results
                .iter()
                .map(|&r| format!("{}: {}", self.value(r), self.ir.ty(r)))
                .collect();
            let _ = write!(self.out, "{} = ", rs.join(", "));
        }
        let _ = write!(self.out, "{}", self.ir.kind(op));
        let operands = self.ir.operands(op);
        if !operands.is_empty() {
            let os: Vec<String> = operands.iter().map(|&v| self.value(v)).collect();
            let _ = write!(self.out, " {}", os.join(", "));
        }
        let blocks = self.ir.blocks_of(op).to_vec();
        if blocks.is_empty() {
            self.out.push('\n');
            return;
        }
        for (i, &block) in blocks.iter().enumerate() {
            if i > 0 {
                self.indent();
                self.out.push_str(match self.ir.kind(op) {
                    OpKind::If => "} else {\n",
                    OpKind::For(_) => "} ragged {\n",
                    OpKind::While => "} do {\n",
                    _ => "} {\n",
                });
            } else {
                self.out.push_str(" {\n");
            }
            if !self.ir.args(block).is_empty() {
                self.indent();
                self.out.push_str("^(");
                self.args(block);
                self.out.push_str(")\n");
            }
            self.depth += 1;
            self.ops(block);
            self.depth -= 1;
        }
        self.indent();
        self.out.push_str("}\n");
    }
}
