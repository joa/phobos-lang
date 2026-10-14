// `let w: tile<T>[r, c] = generate(f)`: a tile with no storage, each element
// block computed where it is read.
//
// When every later read of `w` is the weight of a fragment accumulator's
// scaled `dot_t`, `w` stays unmaterialized and each such read emits a
// `Generate` op right before it, whose region the dot's lowering evaluates
// once per fragment word. Program order is then the order the reads run in,
// which the barrier and race analyses rely on. Any other read materializes
// `w` into a shared tile where it is declared: each thread evaluates `f` on
// a strided share of its blocks, stores them, and a barrier follows.

use std::rc::Rc;

use anyhow::{Result, bail};

use super::{Binding, Build};
use crate::ast::{
    self, AssignOp, BinOp, Expr, ExprKind, Generated, Scalar as AstScalar, Stmt, Sub, Type as AstType, ctx,
};
use crate::diag::Span;
use crate::ir::{DistType, OpKind, RegLayout, Scalar, Type, ValueId};

/// A generated tile's type and its element block.
#[derive(Debug)]
pub(crate) struct GenTile {
    pub(crate) elem: Scalar,
    pub(crate) shape: [i64; 2],
    pub(crate) src: Generated,
}

impl GenTile {
    /// Elements in one block: a 4-byte word of an 8-bit tile, else one.
    fn unit(&self) -> i64 {
        if self.elem.bytes() == Some(1) { 4 } else { 1 }
    }
}

impl Build {
    /// Binds a `let` of `generate(f)` to a typed tile, given the statements
    /// after it. `None` when `stmt` is no such `let`.
    pub(crate) fn generated_let(&mut self, stmt: &Stmt, rest: &[Stmt]) -> Result<Option<()>> {
        let Stmt::Let {
            name,
            ty: Some(AstType::Tile(sc, dims)),
            value: Expr { kind: ExprKind::Generate(src), .. },
            ..
        } = stmt
        else {
            return Ok(None);
        };
        let shape = self.tile_shape(dims)?;
        let &[rows, cols] = &shape[..] else {
            bail!("a generated tile is rank 2, `{name}` is rank {}", shape.len());
        };
        let tile = GenTile {
            elem: Scalar::from_ast(*sc),
            shape: [rows, cols],
            src: (**src).clone(),
        };
        if cols % tile.unit() != 0 {
            bail!("generated `{name}`: its {cols} columns are not whole 4-byte words");
        }
        if let Some(at) = first_write(&tile.src.body) {
            return self.at_span(at, |_| {
                bail!("`{name}` is generated afresh at each read, so its function may read memory but not write it")
            });
        }
        if self.only_dot_weight(name, rest) {
            self.bind(name, Binding::Generated(Rc::new(tile)));
        } else {
            let t = self.materialize_generated(name, *sc, dims, &tile)?;
            self.bind(name, Binding::View(t));
        }
        Ok(Some(()))
    }

    /// Whether every read of `name` in `rest` is the weight of a scaled
    /// `dot_t` updating a fragment accumulator, each a statement of `rest`
    /// itself.
    fn only_dot_weight(&self, name: &str, rest: &[Stmt]) -> bool {
        let mut any = false;
        for stmt in rest {
            if let Stmt::Assign {
                target: Expr { kind: ExprKind::Var(acc), .. },
                op: AssignOp::Add,
                value,
                ..
            } = stmt
                && matches!(self.lookup(acc), Some(Binding::Frags(_)))
                && let Some([a, w, s1, s2]) = super::frag::scaled_dot_parts(value)
                && w == name
                && ![a, s1, s2].contains(&name)
            {
                any = true;
                continue;
            }
            if stmt.uses_name(name) {
                return false;
            }
        }
        any
    }

    /// `Generate` for a read of `tile`: a region over `(row, unit)` that
    /// yields its block.
    pub(crate) fn emit_generate(&mut self, tile: &GenTile) -> Result<ValueId> {
        let region = self.ir.new_block(&[Type::INDEX, Type::INDEX]);
        let [row, unit] = self.ir.args(region)[..] else {
            unreachable!("a generate region takes a row and a unit");
        };
        let [row_name, unit_name] = &tile.src.params;
        self.in_block(region, |b| -> Result<()> {
            b.push_scope();
            b.bind(row_name, Binding::Let { value: row, div: 1 });
            b.bind(unit_name, Binding::Let { value: unit, div: 1 });
            let value = b.emit_stmts(&tile.src.body).and_then(|()| match b.emit_expr(&tile.src.value)? {
                super::Rv::Scalar(v) => Ok(v),
                super::Rv::Tile(_) => bail!("a generated block is a scalar, not a tile"),
            });
            b.pop_scope();
            b.stmt(OpKind::Yield, &[value?]);
            Ok(())
        })?;
        let ty = Type::Dist(DistType {
            elem: tile.elem,
            shape: tile.shape.to_vec(),
            layout: RegLayout::Generated,
        });
        let op = self.op(OpKind::Generate, &[], vec![ty], vec![region]);
        Ok(self.ir.result(op))
    }

    /// `name` as a shared tile: each thread evaluates `f` on blocks
    /// `ctx.thread`, `ctx.thread + ctx.threads`, .. and stores them.
    fn materialize_generated(
        &mut self,
        name: &str,
        sc: AstScalar,
        dims: &[ast::Dim],
        tile: &GenTile,
    ) -> Result<ValueId> {
        let t = self.alloc_tile(sc, dims)?;
        let units = tile.shape[1] / tile.unit();
        let [row_name, unit_name] = &tile.src.params;
        let u = format!("{name}#unit");
        let mut body = vec![
            let_stmt(row_name, Expr::binary(BinOp::Div, Expr::var(&u), Expr::int(units))),
            let_stmt(unit_name, Expr::binary(BinOp::Rem, Expr::var(&u), Expr::int(units))),
        ];
        body.extend(tile.src.body.iter().cloned());
        let at = Expr::index(
            Expr::var(format!("{name}#tile")),
            vec![
                Sub::Point(Expr::var(row_name)),
                Sub::Span {
                    start: Expr::binary(BinOp::Mul, Expr::var(unit_name), Expr::int(tile.unit())),
                    len: Expr::int(tile.unit()),
                },
            ],
        );
        body.push(Stmt::Expr(Expr::call("store", vec![at, tile.src.value.clone()]), Span::default()));
        let fill = Stmt::For {
            var: u,
            var_span: Span::default(),
            start: Expr::call(ctx::THREAD, Vec::new()),
            end: Expr::int(tile.shape[0] * units),
            step: Some(Expr::call(ctx::THREADS, Vec::new())),
            unroll: None,
            body,
            span: Span::default(),
        };
        self.push_scope();
        self.bind(&format!("{name}#tile"), Binding::Tile(t));
        let filled = self.emit_stmts(&[fill, Stmt::Expr(Expr::call("barrier", Vec::new()), Span::default())]);
        self.pop_scope();
        filled?;
        Ok(t)
    }
}

/// Where `stmts` first write memory: an assignment into a slice, or a
/// `store`.
fn first_write(stmts: &[Stmt]) -> Option<Span> {
    stmts.iter().find_map(|stmt| match stmt {
        Stmt::Assign {
            target: Expr { kind: ExprKind::Index { .. }, .. },
            span,
            ..
        } => Some(*span),
        Stmt::Expr(Expr { kind: ExprKind::Call { callee, .. }, .. }, span) if callee == "store" => Some(*span),
        Stmt::For { body, .. } | Stmt::While { body, .. } => first_write(body),
        Stmt::If { then, r#else, .. } => first_write(then).or_else(|| first_write(r#else.as_deref().unwrap_or_default())),
        _ => None,
    })
}

fn let_stmt(name: &str, value: Expr) -> Stmt {
    Stmt::Let {
        name: name.to_string(),
        name_span: Span::default(),
        ty: None,
        ty_span: Span::default(),
        value,
        span: Span::default(),
    }
}
