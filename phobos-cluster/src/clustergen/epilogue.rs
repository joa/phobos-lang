// Peeling an accumulator or a previous-value read out of a store, which
// is what makes a step an epilogue rather than a fresh write.

use super::*;

impl<'a> Analyzer<'a> {
    /// Parse an accumulator epilogue [alpha *] acc [+ [beta *] c_old], where
    /// c_old is a prior load of the output supertile target. The store's
    /// scalars stay out of the cluster IR: they only surface in the leaves.
    pub(super) fn parse_epilogue(&self, value: &Expr, target: &SuperTile) -> Result<Epilogue> {
        if let Expr::Binary {
            op: BinOp::Add,
            lhs,
            rhs,
        } = value
        {
            // one side carries acc, the other the prior-C load
            let (acc_side, prev_side) = if self.uses_scratch(lhs) {
                (lhs.as_ref(), rhs.as_ref())
            } else {
                (rhs.as_ref(), lhs.as_ref())
            };
            let (alpha, acc) = self.peel_acc(acc_side)?;
            let (beta, c_old) = self.peel_prev(prev_side, target)?;
            return Ok(Epilogue {
                acc,
                alpha,
                prev: Some((beta, c_old)),
            });
        }
        let (alpha, acc) = self.peel_acc(value)?;
        Ok(Epilogue {
            acc,
            alpha,
            prev: None,
        })
    }

    /// Peel acc or SCALAR * acc into (coefficient, acc name).
    pub(super) fn peel_acc(&self, e: &Expr) -> Result<(Option<Expr>, String)> {
        if let Expr::Var(n) = e
            && matches!(self.symbols.get(n), Some(Binding::Scratch))
        {
            return Ok((None, n.clone()));
        }
        if let Expr::Binary {
            op: BinOp::Mul,
            lhs,
            rhs,
        } = e
        {
            for (atom, coeff) in [(lhs, rhs), (rhs, lhs)] {
                if let Expr::Var(n) = atom.as_ref()
                    && matches!(self.symbols.get(n), Some(Binding::Scratch))
                {
                    self.check_invariant(coeff)?;
                    return Ok((Some((**coeff).clone()), n.clone()));
                }
            }
        }
        bail!("the accumulator term of a GEMM epilogue must be `acc` or `SCALAR * acc`");
    }

    /// Peel c_old or SCALAR * c_old into (coefficient, c_old name), checking
    /// that c_old is a prior load of the same supertile the store targets.
    pub(super) fn peel_prev(&self, e: &Expr, target: &SuperTile) -> Result<(Option<Expr>, String)> {
        let (coeff, name) = match e {
            Expr::Var(n) => (None, n.clone()),
            Expr::Binary {
                op: BinOp::Mul,
                lhs,
                rhs,
            } => {
                let mut found = None;
                for (atom, coeff) in [(lhs, rhs), (rhs, lhs)] {
                    if let Expr::Var(n) = atom.as_ref()
                        && matches!(self.symbols.get(n), Some(Binding::Ref(_)))
                    {
                        self.check_invariant(coeff)?;
                        found = Some((Some((**coeff).clone()), n.clone()));
                        break;
                    }
                }
                found.ok_or_else(|| {
                    anyhow::anyhow!(
                        "the second term of a GEMM epilogue must be `c_old` or `SCALAR * c_old`"
                    )
                })?
            }
            _ => bail!("the second term of a GEMM epilogue must be `c_old` or `SCALAR * c_old`"),
        };
        match self.symbols.get(&name) {
            Some(Binding::Ref(r)) if r == target => Ok((coeff, name)),
            Some(Binding::Ref(_)) => bail!(
                "GEMM epilogue reads '{name}' at a different supertile than it writes \
                 (cross-supertile access needs halo exchange, unsupported)"
            ),
            _ => bail!("GEMM epilogue term '{name}' is not a prior load of the output supertile"),
        }
    }
}
