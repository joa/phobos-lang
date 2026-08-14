// Rewriting a step into its leaf kernels once the shape is settled.

use super::*;

/// Rewrite the step leaf's store so each launch contributes one k-chunk.
///
/// A plain C[..] = acc flips to += acc. A GEMM epilogue keeps the fused
/// register-accumulator store shape C[..] = alpha*acc + c_old (an implicit
/// beta of 1); the real beta is applied once by the init leaf. alpha*acc
/// with no prior-C term accumulates as += alpha*acc.
pub(super) fn rewrite_step_store(step: &mut Kernel, d: &Define) {
    let Stmt::Assign { op, value, .. } = &mut step.body[d.stmt_idx] else {
        unreachable!("define index points at the accumulator store");
    };
    let Some(epi) = &d.epilogue else {
        *op = AssignOp::Add;
        return;
    };
    let acc_term = match &epi.alpha {
        Some(a) => Expr::Binary {
            op: BinOp::Mul,
            lhs: Box::new(a.clone()),
            rhs: Box::new(Expr::Var(epi.acc.clone())),
        },
        None => Expr::Var(epi.acc.clone()),
    };
    match &epi.prev {
        Some((_, c_old)) => {
            *op = AssignOp::Set;
            *value = Expr::Binary {
                op: BinOp::Add,
                lhs: Box::new(acc_term),
                rhs: Box::new(Expr::Var(c_old.clone())),
            };
        }
        None => {
            *op = AssignOp::Add;
            *value = acc_term;
        }
    }
}

pub(super) fn has_cluster_loop(body: &[Stmt], super_set: &HashSet<String>) -> bool {
    body.iter().any(|s| match s {
        Stmt::For { step, body, .. } => {
            matches!(step, Some(Expr::Var(v)) if super_set.contains(v))
                || has_cluster_loop(body, super_set)
        }
        Stmt::While { body, .. } => has_cluster_loop(body, super_set),
        Stmt::If { then, r#else, .. } => {
            has_cluster_loop(then, super_set)
                || r#else
                    .as_deref()
                    .is_some_and(|e| has_cluster_loop(e, super_set))
        }
        _ => false,
    })
}

pub(super) fn find_target(body: &[Statement]) -> Option<(usize, AccessMode)> {
    body.iter().find_map(|b| match b {
        Statement::Compute(p) => p.target.as_ref().map(|(t, _, m)| (*t, *m)),
        Statement::Loop { body, .. } => find_target(body),
        Statement::InitPlaceholder => None,
    })
}

pub(super) fn lower_body(
    body: Vec<Statement>,
    tensors: &[TensorDecl],
    define: Option<&Define>,
    step_scalars: &[usize],
    init: &InitInfo,
) -> Result<Vec<ClusterStmt>> {
    let mut out = Vec::new();
    for b in body {
        match b {
            Statement::InitPlaceholder => {
                if init.skip {
                    continue; // beta identity: C keeps its original value
                }
                let d = define.expect("placeholder implies a define (validated)");
                out.push(ClusterStmt::Compute {
                    leaf: 1,
                    args: vec![(
                        SuperTile {
                            tensor: d.tensor,
                            coords: d.coords.clone(),
                        },
                        init.c_mode,
                    )],
                    scalars: init.scalars.clone(),
                });
            }
            Statement::Compute(p) => out.push(finalize_compute(p, tensors, define, step_scalars)?),
            Statement::Loop {
                var,
                dim,
                super_sym,
                body,
            } => out.push(ClusterStmt::Loop {
                var,
                dim,
                super_sym,
                body: lower_body(body, tensors, define, step_scalars, init)?,
            }),
        }
    }
    Ok(out)
}

pub(super) fn finalize_compute(
    p: Pending,
    tensors: &[TensorDecl],
    define: Option<&Define>,
    step_scalars: &[usize],
) -> Result<ClusterStmt> {
    let mut args = Vec::new();
    for (i, t) in tensors.iter().enumerate() {
        // every read of this tensor in the compute must hit the same supertile
        let mut reads = p.reads.iter().filter(|(ti, _)| *ti == i).map(|(_, r)| r);
        let read = reads.next();
        if let Some(first) = read
            && reads.any(|r| r != first)
        {
            bail!(
                "tensor '{}' is read at multiple supertile coordinates in one \
                 compute (cross-supertile access needs halo exchange, unsupported)",
                t.name
            );
        }

        if let Some((ti, r, mode)) = &p.target
            && *ti == i
        {
            match read {
                Some(rr) if rr != r => bail!(
                    "tensor '{}' is read and written at different supertile \
                     coordinates in one compute",
                    t.name
                ),
                // read+write of the same supertile is read-modify-write
                Some(_) => args.push((r.clone(), AccessMode::RMW)),
                None => args.push((r.clone(), *mode)),
            }
            continue;
        }
        if p.uses_scratch
            && let Some(d) = define
            && d.tensor == i
        {
            // the accumulator, copy-elided onto the output supertile
            args.push((
                SuperTile {
                    tensor: i,
                    coords: d.coords.clone(),
                },
                AccessMode::RMW,
            ));
            continue;
        }
        match read {
            Some(r) => args.push((r.clone(), AccessMode::Read)),
            None => bail!(
                "tensor '{}' is not used in the compute; every kernel parameter \
                 must map to exactly one supertile per compute",
                t.name
            ),
        }
    }
    Ok(ClusterStmt::Compute {
        leaf: 0,
        args,
        scalars: step_scalars.to_vec(),
    })
}
