// `delta_scan_t(q, k, v, dec, bet, st, o)`: the gated delta rule over a
// prompt, the state in registers.
//
// Every operand is a view: `q`, `k` are `[N, D]`, one head's columns; `v`,
// `o` are `[N, C]`, the value columns this CTA owns; `dec`, `bet` are
// `[N, 1]`; `st` is `[D, C]`, the matching state columns. The CTA's warps
// split the columns evenly, `cols` each. Within a warp, lane `l` owns column
// `l % cols` and the `cols * D / 32` rows of row group `l / cols`, so the
// state never leaves the registers between positions. Per position the two
// reductions over rows, `k . S` and `q . S`, are xor-shuffle butterflies
// across the row groups only, which keeps the shuffles per position low:
// they, not the arithmetic, bound the scan on a card with few SMs.
//
// Per position `t`, with `S` the state:
//
//   S = dec_t * S
//   S = S + k_t^T (bet_t * (v_t - k_t S))
//   o_t = q_t S

use super::*;

/// Most state columns one warp may own. A lane's rows grow with it.
const MAX_COLS: i64 = 8;

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
        let warps = self.cta_threads / WARP;
        let cols = c / warps;
        if c % warps != 0 || cols > MAX_COLS || WARP % cols != 0 {
            bail!(
                "delta_scan_t's state view is {c} columns wide; this kernel's {warps} warps need it \
                 to split evenly, a power of two of at most {MAX_COLS} columns each"
            );
        }
        let groups = WARP / cols;
        if d % groups != 0 {
            bail!("delta_scan_t needs a key dimension divisible by {groups}, got {d}");
        }
        let lr = d / groups;

        let tid = self.thread_id(block)?;
        let warp_w = self.const_index(block, WARP)?;
        let warp = self.divui(block, tid, warp_w)?;
        let lane = self.remui(block, tid, warp_w)?;
        let cols_v = self.const_index(block, cols)?;
        let group = self.divui(block, lane, cols_v)?;
        let lane_col = self.remui(block, lane, cols_v)?;
        let lane_row = self.muli(block, group, self.const_index(block, lr)?)?;
        let warp_col = self.muli(block, warp, cols_v)?;
        let my_col = self.addi(block, warp_col, lane_col)?;

        // The lane's rows of its column.
        let mut inits = Vec::with_capacity(lr as usize);
        for r in 0..lr {
            let rr = self.addi(block, lane_row, self.const_index(block, r)?)?;
            inits.push(self.push(block, memref::load(st.mem, &[rr, my_col], self.loc))?);
        }

        // A lane's rows of k and q, as 16-byte loads when they come four at a
        // time from a 16-byte-aligned row. The lanes of a row group load the
        // same rows.
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
        // Sums across the row groups: the lane bits above the column's.
        let butterfly = |cg: &mut Self, blk: &Block<'c>, x: Value<'c, 'c>| -> Result<Value<'c, 'c>> {
            let (mut x, mut mask) = (x, WARP / 2);
            while mask >= cols {
                let other = cg.shfl_xor_f32(blk, x, mask)?;
                x = cg.push(blk, arith::addf(x, other, cg.loc))?;
                mask /= 2;
            }
            Ok(x)
        };

        // A lane's share of a dot product over its rows.
        let dot = |cg: &mut Self, blk: &Block<'c>, a: &[Value<'c, 'c>], b: &[Value<'c, 'c>]| -> Result<Value<'c, 'c>> {
            let mut sum = cg.zero_scalar(blk, f32_t)?;
            for (&x, &y) in a.iter().zip(b) {
                sum = cg.elem_mac(blk, f32_t, x, y, sum)?;
            }
            Ok(sum)
        };

        let zero = self.const_index(block, 0)?;
        let one = self.const_index(block, 1)?;
        let rows = self.push(block, memref::dim(q.mem, zero, self.loc))?;
        let last = self.subi(block, rows, one)?;
        let is_lead = self.push(
            block,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, group, zero, self.loc),
        )?;
        // One position's inputs: the lane's k and q rows, its column of v,
        // the decay and the beta. Position `t + 1`'s are loaded at the top of
        // position `t`, clamped to the last row, so their latency overlaps
        // `t`'s arithmetic.
        let inputs = |cg: &mut Self, blk: &Block<'c>, t: Value<'c, 'c>| -> Result<Vec<Value<'c, 'c>>> {
            let mut out = lane_values(cg, blk, k, t)?;
            out.extend(lane_values(cg, blk, q, t)?);
            out.push(cg.push(blk, memref::load(v.mem, &[t, my_col], cg.loc))?);
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
            let lr_u = lr as usize;
            let (kr, rest) = cur.split_at(lr_u);
            let (qr, rest) = rest.split_at(lr_u);
            let (vt, decay, beta) = (rest[0], rest[1], rest[2]);

            let mut s: Vec<Value<'c, 'c>> = Vec::with_capacity(state.len());
            for &x in state {
                s.push(cg.push(lblk, arith::mulf(x, decay, cg.loc))?);
            }
            let kv = dot(cg, lblk, &s, kr)?;
            let kv = butterfly(cg, lblk, kv)?;
            let diff = cg.push(lblk, arith::subf(vt, kv, cg.loc))?;
            let u = cg.push(lblk, arith::mulf(diff, beta, cg.loc))?;
            for (s_r, &k_r) in s.iter_mut().zip(kr) {
                *s_r = cg.elem_mac(lblk, f32_t, k_r, u, *s_r)?;
            }
            let out = dot(cg, lblk, &s, qr)?;
            let out = butterfly(cg, lblk, out)?;
            let lead = Block::new(&[]);
            lead.append_operation(memref::store(out, o.mem, &[t, my_col], cg.loc));
            lead.append_operation(scf::r#yield(&[], cg.loc));
            let region = Region::new();
            region.append_block(lead);
            lblk.append_operation(scf::r#if(is_lead, &[], region, Region::new(), cg.loc));
            s.extend(next);
            Ok(s)
        })?;

        for (r, &val) in finals[..n_state].iter().enumerate() {
            let rr = self.addi(block, lane_row, self.const_index(block, r as i64)?)?;
            block.append_operation(memref::store(val, st.mem, &[rr, my_col], self.loc));
        }
        self.barrier(block)
    }
}
