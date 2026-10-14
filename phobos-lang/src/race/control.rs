// Loops and branches: a uniform loop or branch runs once for the whole
// mask, a per-thread one with a mask per turn or side.

use super::exec::{Exec, Lanes, Stop, V};
use super::program::Op;

/// Turns of a loop with uniform bounds run at each end. A hazard between
/// turns shows between two neighbouring ones, and the last turns see where
/// the loop ends, so the turns between them are skipped.
const END_TURNS: i64 = 2;

impl Exec<'_> {
    pub(super) fn for_loop(&mut self, op: &Op, mask: &[bool]) -> Result<(), Stop> {
        let n = self.n;
        let body = &op.regions[0];
        let [lb, ub, step] = [op.operands[0], op.operands[1], op.operands[2]];
        let inits = &op.operands[3..];

        // The same bounds everywhere: every thread of the mask takes every turn.
        if self.same(lb) && self.same(ub) && self.same(step) {
            let (Some(mut iv), Some(ub), Some(step)) =
                (self.int(lb, 0), self.int(ub, 0), self.int(step, 0))
            else {
                return Err(Stop::Untraceable(format!(
                    "a loop whose bounds depend on data: {}",
                    op.text
                )));
            };
            let mut iters: Vec<Lanes> = inits.iter().map(|&v| self.vals[v].clone()).collect();
            let turns = if step > 0 {
                (ub - iv + step - 1).div_euclid(step)
            } else {
                0
            };
            let mut turn = 0;
            while iv < ub {
                if turn == END_TURNS && turns > 2 * END_TURNS {
                    // Skip to the last turns. What the loop carries is no
                    // longer known.
                    turn = turns - END_TURNS;
                    iv += (turn - END_TURNS) * step;
                    iters
                        .iter_mut()
                        .for_each(|it| *it = Lanes::Same(V::Unknown));
                }
                turn += 1;
                self.vals[body.args[0]] = Lanes::Same(V::Int(iv));
                for (k, it) in iters.iter().enumerate() {
                    self.vals[body.args[k + 1]] = it.clone();
                }
                self.region(body, mask)?;
                if let Some(term) = body.term {
                    iters = self.p.ops[term]
                        .operands
                        .iter()
                        .map(|&y| self.vals[y].clone())
                        .collect();
                }
                iv += step;
            }
            for (k, &r) in op.results.iter().enumerate() {
                self.vals[r] = iters[k].clone();
            }
            return Ok(());
        }

        let mut iv: Vec<V> = (0..n).map(|t| self.get(lb, t)).collect();
        let mut iters: Vec<Vec<V>> = inits
            .iter()
            .map(|&v| (0..n).map(|t| self.get(v, t)).collect())
            .collect();
        loop {
            let mut active = vec![false; n];
            for t in self.threads(mask) {
                match (&iv[t], self.int(ub, t), self.int(step, t)) {
                    (V::Int(i), Some(ub), Some(_)) => active[t] = *i < ub,
                    _ => {
                        return Err(Stop::Untraceable(format!(
                            "a loop whose bounds depend on data: {}",
                            op.text
                        )));
                    }
                }
            }
            if !active.iter().any(|&a| a) {
                break;
            }
            self.vals[body.args[0]] = Lanes::Each(iv.clone());
            for (k, it) in iters.iter().enumerate() {
                self.vals[body.args[k + 1]] = Lanes::Each(it.clone());
            }
            self.region(body, &active)?;
            if let Some(term) = body.term {
                for (k, &y) in self.p.ops[term].operands.iter().enumerate() {
                    for t in (0..n).filter(|&t| active[t]) {
                        iters[k][t] = self.get(y, t);
                    }
                }
            }
            for t in (0..n).filter(|&t| active[t]) {
                if let (V::Int(i), Some(s)) = (&iv[t], self.int(step, t)) {
                    iv[t] = V::Int(i + s);
                }
            }
        }
        for (k, &r) in op.results.iter().enumerate() {
            self.vals[r] = Lanes::Each(iters[k].clone());
        }
        Ok(())
    }

    pub(super) fn branch(&mut self, op: &Op, mask: &[bool]) -> Result<(), Stop> {
        let n = self.n;
        let c = op.operands[0];
        // The same condition everywhere: one side, for the whole mask.
        if self.same(c)
            && let Some(cond) = self.int(c, 0)
        {
            let region = &op.regions[if cond != 0 { 0 } else { 1 }];
            self.region(region, mask)?;
            match region.term {
                Some(term) => {
                    for (k, &y) in self.p.ops[term].operands.iter().enumerate() {
                        self.vals[op.results[k]] = self.vals[y].clone();
                    }
                }
                None => self.unknown_results(op),
            }
            return Ok(());
        }
        let cond: Vec<Option<i64>> = (0..n).map(|t| self.int(c, t)).collect();
        // A thread whose condition is not known takes both sides, so every
        // access either side could make is checked.
        let then: Vec<bool> = (0..n).map(|t| mask[t] && cond[t] != Some(0)).collect();
        let other: Vec<bool> = (0..n)
            .map(|t| mask[t] && !matches!(cond[t], Some(c) if c != 0))
            .collect();
        let mut sides = Vec::new();
        for (side, region) in [(&then, &op.regions[0]), (&other, &op.regions[1])] {
            let mut yielded = vec![vec![V::Unknown; n]; op.results.len()];
            if side.iter().any(|&s| s) {
                self.region(region, side)?;
                if let Some(term) = region.term {
                    for (k, &y) in self.p.ops[term].operands.iter().enumerate() {
                        yielded[k] = (0..n).map(|t| self.get(y, t)).collect();
                    }
                }
            }
            sides.push(yielded);
        }
        for (k, &r) in op.results.iter().enumerate() {
            let out = (0..n)
                .map(|t| match cond[t] {
                    Some(0) => sides[1][k][t].clone(),
                    Some(_) => sides[0][k][t].clone(),
                    // Taken both ways, a result is only known if both agree.
                    None => match (&sides[0][k][t], &sides[1][k][t]) {
                        (V::Int(a), V::Int(b)) if a == b => V::Int(*a),
                        _ => V::Unknown,
                    },
                })
                .collect();
            self.vals[r] = Lanes::Each(out);
        }
        Ok(())
    }

    pub(super) fn while_loop(&mut self, op: &Op, mask: &[bool]) -> Result<(), Stop> {
        let n = self.n;
        let (before, after) = (&op.regions[0], &op.regions[1]);
        let mut args: Vec<Vec<V>> = op
            .operands
            .iter()
            .map(|&v| (0..n).map(|t| self.get(v, t)).collect())
            .collect();
        let mut out = vec![vec![V::Unknown; n]; op.results.len()];
        let mut active = mask.to_vec();
        while active.iter().any(|&a| a) {
            for (k, a) in args.iter().enumerate() {
                self.vals[before.args[k]] = Lanes::Each(a.clone());
            }
            self.region(before, &active)?;
            let Some(term) = before.term else {
                return Err(Stop::Untraceable(
                    "a while loop with no condition".to_string(),
                ));
            };
            let cond_ops = &self.p.ops[term].operands;
            let mut next = vec![false; n];
            let mut once = vec![false; n];
            for t in (0..n).filter(|&t| active[t]) {
                match self.int(cond_ops[0], t) {
                    Some(0) => {
                        for (k, &v) in cond_ops[1..].iter().enumerate() {
                            out[k][t] = self.get(v, t);
                        }
                    }
                    Some(_) => next[t] = true,
                    // A wait on data, such as a flag or a grid barrier:
                    // taken to end after this pass, its body run once.
                    None => {
                        self.waits += 1;
                        once[t] = true;
                        for (k, &v) in cond_ops[1..].iter().enumerate() {
                            out[k][t] = self.get(v, t);
                        }
                    }
                }
            }
            let body: Vec<bool> = (0..n).map(|t| next[t] || once[t]).collect();
            if !body.iter().any(|&a| a) {
                break;
            }
            for (k, &v) in cond_ops[1..].iter().enumerate() {
                self.vals[after.args[k]] = Lanes::Each((0..n).map(|t| self.get(v, t)).collect());
            }
            self.region(after, &body)?;
            if let Some(term) = after.term {
                for (k, &y) in self.p.ops[term].operands.iter().enumerate() {
                    for t in (0..n).filter(|&t| next[t]) {
                        args[k][t] = self.get(y, t);
                    }
                }
            }
            active = next;
        }
        for (k, &r) in op.results.iter().enumerate() {
            self.vals[r] = Lanes::Each(out[k].clone());
        }
        Ok(())
    }
}
