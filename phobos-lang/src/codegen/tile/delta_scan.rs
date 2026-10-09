// `delta_scan_t(q, k, v, dec, bet, st, o)`: the gated delta rule over a
// prompt, the state in registers.
//
// Every operand is a view: `q`, `k` are `[N, D]`, one head's columns; `v`,
// `o` are `[N, C]`, the value columns this CTA owns; `dec`, `bet` are
// `[N, 1]`; `st` is `[D, C]`, the matching state columns. Warp `w` owns
// columns `w * COLS ..`, and lane `l` rows `l * D / 32 ..` of them, so the
// state never leaves the registers between positions. Per position the two
// reductions over rows, `k . S` and `q . S`, are xor-shuffle butterflies.
//
// Per position `t`, with `S` the state:
//
//   S = dec_t * S
//   S = S + k_t^T (bet_t * (v_t - k_t S))
//   o_t = q_t S

use super::*;

/// State columns one warp owns.
const COLS: i64 = 2;

impl<'c> Codegen<'c> {
    /// The scan over resolved views `q, k, v, dec, bet, st, o`.
    pub(in crate::codegen) fn delta_scan_raw(&mut self, block: &Block<'c>, tiles: [&MemVal<'c>; 7]) -> Result<()> {
        let [q, k, v, dec, bet, st, o] = tiles;
        let f32_t = self.f32_t;
        for (t, what) in [(q, "q"), (k, "k"), (v, "v"), (dec, "dec"), (bet, "bet"), (st, "state"), (o, "o")] {
            if t.elem != f32_t || t.shape.len() != 2 {
                bail!("delta_scan_t {what} must be a rank-2 f32 view");
            }
            // A masked slice arrives as a shared copy, and the stores would
            // land in that copy.
            if t.is_masked() || t.global.is_some() {
                bail!("delta_scan_t {what} must be an in-bounds view of a tensor; `@aligned` promises that");
            }
        }
        if st.shape.contains(&DYN) {
            bail!("delta_scan_t state must have a static shape [D, C]");
        }
        let (d, c) = (st.shape[0], st.shape[1]);
        let shapes = [(q, d), (k, d), (v, c), (o, c), (dec, 1), (bet, 1)];
        if shapes.iter().any(|(t, w)| t.shape[1] != *w) {
            bail!("delta_scan_t takes q, k as [N, {d}], v, o as [N, {c}] and dec, bet as [N, 1]");
        }
        if d % WARP != 0 {
            bail!("delta_scan_t needs a key dimension divisible by {WARP}, got {d}");
        }
        let warps = self.cta_threads / WARP;
        if c != warps * COLS {
            bail!(
                "delta_scan_t's state view is {c} columns wide, but this kernel's {warps} warps own \
                 {COLS} each; the two must match"
            );
        }
        let lr = d / WARP;

        let tid = self.thread_id(block)?;
        let warp_w = self.const_index(block, WARP)?;
        let warp = self.divui(block, tid, warp_w)?;
        let lane = self.remui(block, tid, warp_w)?;
        let lr_v = self.const_index(block, lr)?;
        let lane_row = self.muli(block, lane, lr_v)?;
        let cols_v = self.const_index(block, COLS)?;
        let my_col = self.muli(block, warp, cols_v)?;

        // The lane's state, row-major over (row, column).
        let mut inits = Vec::with_capacity((lr * COLS) as usize);
        for r in 0..lr {
            let rr = self.addi(block, lane_row, self.const_index(block, r)?)?;
            for j in 0..COLS {
                let cc = self.addi(block, my_col, self.const_index(block, j)?)?;
                inits.push(self.push(block, memref::load(st.mem, &[rr, cc], self.loc))?);
            }
        }

        // A lane's rows of k and q, as 16-byte loads when they come four at a
        // time from a 16-byte-aligned row.
        let wide = lr % 4 == 0 && [q, k].iter().all(|t| t.align_div == 0 || t.align_div % 4 == 0);
        let vec4 = Type::vector(&[4], f32_t);
        let lane_values = |cg: &mut Self, blk: &Block<'c>, t: &MemVal<'c>, row: Value<'c, 'c>| {
            let mut out = Vec::with_capacity(lr as usize);
            let mut r = 0;
            while r < lr {
                let at = cg.addi(blk, lane_row, cg.const_index(blk, r)?)?;
                if wide {
                    let v4 = cg.vec_load_al(blk, t.mem, &[row, at], vec4, 16)?;
                    for e in 0..4 {
                        out.push(cg.vec_extract(blk, v4, &[e], f32_t)?);
                    }
                    r += 4;
                } else {
                    out.push(cg.push(blk, memref::load(t.mem, &[row, at], cg.loc))?);
                    r += 1;
                }
            }
            Ok::<_, anyhow::Error>(out)
        };
        let butterfly = |cg: &mut Self, blk: &Block<'c>, vals: &mut [Value<'c, 'c>]| -> Result<()> {
            let mut mask = WARP / 2;
            while mask >= 1 {
                for x in vals.iter_mut() {
                    let other = cg.shfl_xor_f32(blk, *x, mask)?;
                    *x = cg.push(blk, arith::addf(*x, other, cg.loc))?;
                }
                mask /= 2;
            }
            Ok(())
        };

        let zero = self.const_index(block, 0)?;
        let one = self.const_index(block, 1)?;
        let rows = self.push(block, memref::dim(q.mem, zero, self.loc))?;
        let last = self.subi(block, rows, one)?;
        let is_lead = self.push(
            block,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, lane, zero, self.loc),
        )?;
        // One position's inputs: the lane's k and q rows, the warp's v
        // columns, the decay and the beta. Position `t + 1`'s are loaded at
        // the top of position `t`, clamped to the last row, so their latency
        // overlaps `t`'s arithmetic.
        let inputs = |cg: &mut Self, blk: &Block<'c>, t: Value<'c, 'c>| -> Result<Vec<Value<'c, 'c>>> {
            let mut out = lane_values(cg, blk, k, t)?;
            out.extend(lane_values(cg, blk, q, t)?);
            for j in 0..COLS {
                let at = cg.addi(blk, my_col, cg.const_index(blk, j)?)?;
                out.push(cg.push(blk, memref::load(v.mem, &[t, at], cg.loc))?);
            }
            out.push(cg.push(blk, memref::load(dec.mem, &[t, zero], cg.loc))?);
            out.push(cg.push(blk, memref::load(bet.mem, &[t, zero], cg.loc))?);
            Ok(out)
        };
        let n_state = inits.len();
        let first = self.minsi(block, zero, last)?;
        inits.extend(inputs(self, block, first)?);
        let finals = self.carry_loop(block, zero, rows, one, &inits, |cg, lblk, t, carried| {
            let (state, cur) = carried.split_at(n_state);
            let next_t = cg.minsi(lblk, cg.addi(lblk, t, one)?, last)?;
            let next = inputs(cg, lblk, next_t)?;
            let (lr_u, cols_u) = (lr as usize, COLS as usize);
            let (kr, rest) = cur.split_at(lr_u);
            let (qr, rest) = rest.split_at(lr_u);
            let (vs, rest) = rest.split_at(cols_u);
            let (decay, beta) = (rest[0], rest[1]);

            let mut s: Vec<Value<'c, 'c>> = Vec::with_capacity(state.len());
            for &x in state {
                s.push(cg.push(lblk, arith::mulf(x, decay, cg.loc))?);
            }
            let zero_f = cg.zero_scalar(lblk, f32_t)?;
            let mut kv = vec![zero_f; cols_u];
            for r in 0..lr_u {
                for j in 0..cols_u {
                    kv[j] = cg.elem_mac(lblk, f32_t, s[r * cols_u + j], kr[r], kv[j])?;
                }
            }
            butterfly(cg, lblk, &mut kv)?;
            for j in 0..cols_u {
                let diff = cg.push(lblk, arith::subf(vs[j], kv[j], cg.loc))?;
                let u = cg.push(lblk, arith::mulf(diff, beta, cg.loc))?;
                for (r, &kv_r) in kr.iter().enumerate() {
                    let i = r * cols_u + j;
                    s[i] = cg.elem_mac(lblk, f32_t, kv_r, u, s[i])?;
                }
            }
            let mut out = vec![zero_f; cols_u];
            for r in 0..lr_u {
                for j in 0..cols_u {
                    out[j] = cg.elem_mac(lblk, f32_t, s[r * cols_u + j], qr[r], out[j])?;
                }
            }
            butterfly(cg, lblk, &mut out)?;
            let lead = Block::new(&[]);
            for (j, val) in out.iter().enumerate() {
                let at = cg.addi(&lead, my_col, cg.const_index(&lead, j as i64)?)?;
                lead.append_operation(memref::store(*val, o.mem, &[t, at], cg.loc));
            }
            lead.append_operation(scf::r#yield(&[], cg.loc));
            let region = Region::new();
            region.append_block(lead);
            lblk.append_operation(scf::r#if(is_lead, &[], region, Region::new(), cg.loc));
            s.extend(next);
            Ok(s)
        })?;

        for r in 0..lr {
            let rr = self.addi(block, lane_row, self.const_index(block, r)?)?;
            for j in 0..COLS {
                let cc = self.addi(block, my_col, self.const_index(block, j)?)?;
                let val = finals[(r * COLS + j) as usize];
                block.append_operation(memref::store(val, st.mem, &[rr, cc], self.loc));
            }
        }
        self.barrier(block)
    }
}
