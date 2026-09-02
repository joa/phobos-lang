// `rms_norm_q_t(x, g, eps, o, q, s)`: one row's RMS normalization with its
// gain, and the Q8_0 copy with a scale a 32-element block, as one statement.
// A thread owns four contiguous elements of every `4 * threads`; the
// reductions are warp shuffles and one shared word a warp.

use super::*;

impl<'c> Codegen<'c> {
    pub(in crate::codegen) fn emit_rms_norm_q(&mut self, block: &Block<'c>, args: &[Expr]) -> Result<Rv<'c>> {
        let [x, g, eps, o, q, s] = args else {
            bail!("rms_norm_q_t expects (x, gain, eps, out, q, scales)");
        };
        let tile = |cg: &mut Self, e: &Expr, what: &str| match cg.emit_expr(block, e)? {
            Rv::Tile(t) => Ok(t),
            Rv::Scalar(_) => bail!("rms_norm_q_t {what} must be a tile"),
        };
        let x = tile(self, x, "x")?;
        let g = tile(self, g, "gain")?;
        let eps = self.emit_scalar(block, eps)?;
        let o = tile(self, o, "out")?;
        let q = tile(self, q, "q")?;
        let s = tile(self, s, "scales")?;
        let (f32_t, i8_t, i32_t) = (self.f32_t, self.i8_t, self.i32_t);
        for (t, what) in [(&x, "x"), (&g, "gain"), (&o, "out"), (&q, "q"), (&s, "scales")] {
            if t.shape.len() != 2 || t.shape.contains(&DYN) {
                bail!("rms_norm_q_t {what} must be a rank-2 tile of static shape");
            }
            // A masked slice reaches here already copied into a shared tile,
            // and the stores below would land in that copy.
            if t.global.is_some() || t.is_masked() {
                bail!("rms_norm_q_t {what} must be an in-bounds slice of a tensor; `@aligned` promises that");
            }
        }
        if x.elem != f32_t || g.elem != f32_t || o.elem != f32_t || s.elem != f32_t {
            bail!("rms_norm_q_t x, gain, out and scales must be f32");
        }
        if q.elem != i8_t {
            bail!("rms_norm_q_t q must be int8");
        }
        if eps.r#type() != f32_t {
            bail!("rms_norm_q_t eps must be an f32 scalar");
        }
        let (nb, lane_w) = (x.shape[0], x.shape[1]);
        if lane_w != Q8_BLOCK {
            bail!("rms_norm_q_t takes the row as [blocks, {Q8_BLOCK}]");
        }
        self.check_shapes(&x.shape, &g.shape, "rms_norm_q_t gain")?;
        self.check_shapes(&x.shape, &o.shape, "rms_norm_q_t out")?;
        self.check_shapes(&x.shape, &q.shape, "rms_norm_q_t q")?;
        self.check_shapes(&[nb, 1], &s.shape, "rms_norm_q_t scales")?;
        let cta = self.cta_threads;
        if cta % WARP != 0 {
            bail!("rms_norm_q_t needs a CTA that is a whole number of warps");
        }
        let width = nb * lane_w;
        if width % (cta * 4) != 0 {
            bail!("rms_norm_q_t needs the row to be a whole number of {} elements", cta * 4);
        }
        let per = width / (cta * 4);
        let warps = cta / WARP;

        let vec4 = Type::vector(&[4], f32_t);
        let vec4_i8 = Type::vector(&[4], i8_t);
        let tid = self.thread_id(block)?;
        let c = |cg: &mut Self, v: i64| cg.const_index(block, v);
        let warp_w = c(self, WARP)?;
        let lane = self.remui(block, tid, warp_w)?;
        let warp = self.divui(block, tid, warp_w)?;
        let four = c(self, 4)?;
        let thirty_two = c(self, Q8_BLOCK)?;
        let zero = c(self, 0)?;
        let e0 = self.muli(block, tid, four)?;
        // The thread's pieces: element `i * 4 cta + 4 tid`, as (row, col) of
        // the [blocks, 32] view.
        let mut at = Vec::with_capacity(per as usize);
        for i in 0..per {
            let off = c(self, i * cta * 4)?;
            let e = self.addi(block, e0, off)?;
            let row = self.divui(block, e, thirty_two)?;
            let col = self.remui(block, e, thirty_two)?;
            at.push((row, col));
        }

        // Sum of squares: the thread's, the warp's, the CTA's.
        let mut acc = self.zero_scalar(block, f32_t)?;
        for (row, col) in &at {
            let v = self.vec_load_al(block, x.mem, &[*row, *col], vec4, 16)?;
            for k in 0..4 {
                let vk = self.vec_extract(block, v, &[k], f32_t)?;
                acc = self.elem_mac(block, f32_t, vk, vk, acc)?;
            }
        }
        let mut mask = WARP / 2;
        while mask >= 1 {
            let other = self.shfl_xor_f32(block, acc, mask)?;
            acc = self.push(block, arith::addf(acc, other, self.loc))?;
            mask /= 2;
        }
        let parts = self.alloc_tile_shaped(block, f32_t, &[1, warps])?;
        let is_lead = self.push(
            block,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, lane, zero, self.loc),
        )?;
        let store = Block::new(&[]);
        store.append_operation(memref::store(acc, parts.mem, &[zero, warp], self.loc));
        store.append_operation(scf::r#yield(&[], self.loc));
        let sr = Region::new();
        sr.append_block(store);
        block.append_operation(scf::r#if(is_lead, &[], sr, Region::new(), self.loc));
        self.barrier(block)?;
        let mut total = self.zero_scalar(block, f32_t)?;
        for w in 0..warps {
            let wi = c(self, w)?;
            let part = self.push(block, memref::load(parts.mem, &[zero, wi], self.loc))?;
            total = self.push(block, arith::addf(total, part, self.loc))?;
        }
        let width_f = self.const_f32(block, width as f64)?;
        let mean = self.push(block, arith::divf(total, width_f, self.loc))?;
        let mean = self.push(block, arith::addf(mean, eps, self.loc))?;
        let root = self.approx_sqrt(block, mean)?;
        let one_f = self.const_f32(block, 1.0)?;
        let inv = self.push(block, arith::divf(one_f, root, self.loc))?;

        // Normalize, store, and quantize: a block's 32 elements sit in the
        // eight lanes `8 j .. 8 j + 8`, so its absolute maximum is three
        // shuffles away.
        let c127 = self.const_f32(block, 127.0)?;
        let tiny = self.const_f32(block, 1e-8)?;
        let eight = c(self, 8)?;
        let in_block = self.remui(block, lane, eight)?;
        let leads_block = self.push(
            block,
            arith::cmpi(self.ctx, arith::CmpiPredicate::Eq, in_block, zero, self.loc),
        )?;
        for (row, col) in &at {
            let v = self.vec_load_al(block, x.mem, &[*row, *col], vec4, 16)?;
            let gv = self.vec_load_al(block, g.mem, &[*row, *col], vec4, 16)?;
            let mut n = Vec::with_capacity(4);
            let mut mx = self.zero_scalar(block, f32_t)?;
            for k in 0..4 {
                let vk = self.vec_extract(block, v, &[k], f32_t)?;
                let gk = self.vec_extract(block, gv, &[k], f32_t)?;
                let nk = self.push(block, arith::mulf(vk, inv, self.loc))?;
                let nk = self.push(block, arith::mulf(nk, gk, self.loc))?;
                let neg = self.push(block, arith::negf(nk, self.loc))?;
                let ak = self.fmax(block, nk, neg)?;
                mx = self.fmax(block, mx, ak)?;
                n.push(nk);
            }
            let mut out = self.vec_broadcast(block, n[0], vec4)?;
            for (k, nk) in n.iter().enumerate().skip(1) {
                out = self.vec_insert(block, *nk, out, &[k as i64])?;
            }
            self.vec_store_al(block, out, o.mem, &[*row, *col], 16)?;
            for m in [4, 2, 1] {
                let other = self.shfl_xor_f32(block, mx, m)?;
                mx = self.fmax(block, mx, other)?;
            }
            let denom = self.push(block, arith::addf(mx, tiny, self.loc))?;
            let inv_q = self.push(block, arith::divf(c127, denom, self.loc))?;
            let mut bytes = Vec::with_capacity(4);
            for nk in &n {
                let scaled = self.push(block, arith::mulf(*nk, inv_q, self.loc))?;
                let rounded = self.round_even(block, scaled)?;
                let as_i32 = self.push(block, arith::fptosi(rounded, i32_t, self.loc))?;
                bytes.push(self.push(block, arith::trunci(as_i32, i8_t, self.loc))?);
            }
            let mut packed = self.vec_broadcast(block, bytes[0], vec4_i8)?;
            for (k, b) in bytes.iter().enumerate().skip(1) {
                packed = self.vec_insert(block, *b, packed, &[k as i64])?;
            }
            self.vec_store_al(block, packed, q.mem, &[*row, *col], 4)?;
            let scale = self.push(block, arith::divf(mx, c127, self.loc))?;
            let store = Block::new(&[]);
            store.append_operation(memref::store(scale, s.mem, &[*row, zero], self.loc));
            store.append_operation(scf::r#yield(&[], self.loc));
            let sr = Region::new();
            sr.append_block(store);
            block.append_operation(scf::r#if(leads_block, &[], sr, Region::new(), self.loc));
        }
        for t in [&x, &g, &o, &q, &s] {
            self.release(t);
        }
        self.barrier(block)?;
        Ok(Rv::Scalar(inv))
    }
}
