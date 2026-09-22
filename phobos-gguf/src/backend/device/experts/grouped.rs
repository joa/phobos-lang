// The prompt pass's routed feed-forward as grouped GEMMs: the rows sorted
// by expert on the host, a permuted activation whose segments are padded to
// the GEMM's tile, the experts brought in a group at a time (as many as the
// block has slots), each group's segments contracted through a schedule
// table, and each row's eight results gathered back with its weights.
// `PHOBOS_MOE_GROUPED` opts in; see `ENV.md`.

use anyhow::{Result, ensure};
use cust::memory::DeviceBuffer;

use super::super::kernels::{MOE_COMBINE_TN, MOE_USED, QGEMM_TM, QGEMM_TN};
use super::super::{DeviceBackend, Plane};
use super::op::kernel_for;
use super::{Experts, Kind};
use crate::backend::{Backend, Buf, Moe};

/// One expert's rows of the permuted activation: its first padded row and
/// the real rows there.
struct Segment {
    expert: usize,
    start: usize,
    len: usize,
}

/// Columns a permutation program copies.
const PERMUTE_TK: usize = 256;

pub(crate) fn moe_permute_src() -> String {
    format!(
        "@launch(256)
@autotune(TK in [{PERMUTE_TK}])
@aligned(K = TK)
kernel moe_permute(PERM: tensor<i32>[1, M], A: tensor<f32>[R, K], OUT: tensor<f32>[M, K]) {{
  let p = program_id(0)
  let pk = program_id(1)
  let r = PERM[0, p]
  OUT[p :+ 1, pk * TK :+ TK] = A[r :+ 1, pk * TK :+ TK]
}}
"
    )
}

/// The K-quant prompt GEMM over a schedule table: program `p` contracts
/// activation tile `SCHED[1, p]` against slot `SCHED[0, p]`'s rows.
pub(crate) fn moe_qgemm_src(name: &str, ne: usize) -> String {
    format!(
        "@launch(256, 2)
@autotune(TM in [{QGEMM_TM}], TN in [{QGEMM_TN}], NE in [{ne}])
@aligned(M = TM, N = TN, NS = NE, K = 256)
kernel {name}_moe_qgemm(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB], SCHED: tensor<i32>[2, T],
                       QB: tensor<i8>[NS, RB], D: tensor<f16>[NS, NB], C: tensor<f32>[M, N]) {{
  let p = program_id(0)
  let pn = program_id(1)
  let s = SCHED[0, p]
  let t = SCHED[1, p]
  let r = s * NE + pn * TN
  C[t * TM :+ TM, pn * TN :+ TN] = {name}_qgemm_t(A[t * TM :+ TM, :], AS[t * TM :+ TM, :], QB[r :+ TN, :], D[r :+ TN, :])
}}
"
    )
}

/// Each row's eight down rows gathered by position and summed with the
/// router's weights, plus the gated shared expert, into the residual.
pub(crate) fn moe_gather_src() -> String {
    format!(
        "@launch(256)
@autotune(TN in [{MOE_COMBINE_TN}], U in [{MOE_USED}])
@aligned(N = TN)
kernel moe_gather(POS: tensor<i32>[R, U], W: tensor<f32>[R, U], C: tensor<f32>[M, N], G: tensor<f32>[R, 1],
                  S: tensor<f32>[R, N], X: tensor<f32>[R, N]) {{
  let r = program_id(0)
  let pn = program_id(1)
  var acc: tile<f32>[1, TN] = X[r :+ 1, pn * TN :+ TN]
  for j in range(0, U, 1) {{
    let p = POS[r, j]
    let w = W[r, j]
    acc = acc + C[p :+ 1, pn * TN :+ TN] * w
  }}
  var g: tile<f32>[1, 1] = G[r :+ 1, 0 :+ 1]
  g = 1.0 / (1.0 + exp(0.0 - g))
  acc = acc + S[r :+ 1, pn * TN :+ TN] * g
  X[r :+ 1, pn * TN :+ TN] = acc
}}
"
    )
}

impl DeviceBackend {
    /// The rows' feed-forward as grouped GEMMs, after the sync point has
    /// filled the block's `topk_host`. Eager: the caller flushed.
    pub(super) fn run_grouped(&self, experts: &mut Experts, req: &Moe, shared: (Buf, Buf)) -> Result<()> {
        let (d, d_ff, rows, block) = (req.d_model, req.d_ff, req.rows, req.experts.0);
        let quants = {
            let set = &experts.blocks[block].set;
            [set.gate.quant(), set.up.quant(), set.down.quant()]
        };
        ensure!(
            quants[0] == quants[1] && [d, d_ff].iter().all(|w| w.is_multiple_of(256) && w.is_multiple_of(QGEMM_TN)),
            "the grouped prompt path wants gate and up in one format and widths in whole blocks and tiles"
        );

        // Rows by expert, each expert's segment padded to the tile; padding
        // rows read row zero and nothing gathers their outputs.
        let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); req.n_expert];
        for r in 0..rows {
            for (j, e) in experts.blocks[block].chosen(r)?.into_iter().enumerate() {
                by_expert[e].push((r, j));
            }
        }
        let mut segments = Vec::new();
        let mut perm: Vec<i32> = Vec::new();
        let mut pos = vec![0i32; rows * MOE_USED];
        for (expert, users) in by_expert.iter().enumerate().filter(|(_, u)| !u.is_empty()) {
            let start = perm.len();
            for &(r, j) in users {
                pos[r * MOE_USED + j] = perm.len() as i32;
                perm.push(r as i32);
            }
            perm.resize(perm.len().next_multiple_of(QGEMM_TM), 0);
            segments.push(Segment { expert, start, len: users.len() });
        }
        let m_pad = perm.len();

        let perm_dev = DeviceBuffer::from_slice(&perm)?;
        let pos_dev = DeviceBuffer::from_slice(&pos)?;
        let xp = self.alloc(m_pad * d)?;
        self.with_kernel(&self.moe_permute, (), "moe_permute", moe_permute_src, |module| {
            self.launch(
                module,
                "moe_permute",
                &[
                    (perm_dev.as_device_ptr().as_raw(), [1, m_pad as i64]),
                    (self.ptr(req.x, 0)?, [rows as i64, d as i64]),
                    (self.ptr(xp, 0)?, [m_pad as i64, d as i64]),
                ],
                (m_pad as u32, (d / PERMUTE_TK) as u32, 1),
            )
        })?;
        let (qa, das) = self.act_ptrs(self.quantize_act(xp, m_pad, d)?)?;
        let bufs = [self.alloc(m_pad * d_ff)?, self.alloc(m_pad * d_ff)?, self.alloc(m_pad * d_ff)?, self.alloc(m_pad * d)?];
        let [gate_out, up_out, h, down_out] = bufs;

        // Experts a group at a time, as many as the block has slots.
        for group in segments.chunks(experts.per_block) {
            let ids: Vec<usize> = group.iter().map(|s| s.expert).collect();
            let tick = experts.step(block, &ids);
            let mut sched = [Vec::new(), Vec::new()];
            for seg in group {
                let slot = experts
                    .place(block, seg.expert, tick, &self.stream)?
                    .ok_or_else(|| anyhow::anyhow!("a group of {} experts does not fit the block's slots", group.len()))?;
                for t in 0..seg.len.div_ceil(QGEMM_TM) {
                    sched[0].push(slot as i32);
                    sched[1].push((seg.start / QGEMM_TM + t) as i32);
                }
            }
            let tiles = sched[0].len();
            let table = DeviceBuffer::from_slice(&sched.concat())?;
            let table_ptr = table.as_device_ptr().as_raw();
            self.grouped_gemm(experts, block, Kind::Gate, qa, das, m_pad, d, table_ptr, tiles, d_ff, gate_out)?;
            self.grouped_gemm(experts, block, Kind::Up, qa, das, m_pad, d, table_ptr, tiles, d_ff, up_out)?;
            let plane = |buf| Plane { buf, offset: 0, pitch: d_ff };
            self.swiglu_planes(plane(gate_out), plane(up_out), h, m_pad, d_ff)?;
            let (hqa, hdas) = self.act_ptrs(self.quantize_act(h, m_pad, d_ff)?)?;
            self.grouped_gemm(experts, block, Kind::Down, hqa, hdas, m_pad, d_ff, table_ptr, tiles, d, down_out)?;
            // The table is read by launches already issued; it goes once
            // they have run.
            self.stream.synchronize()?;
        }

        let w_ptr = experts.blocks[block].weights.as_device_ptr().as_raw();
        let u = MOE_USED as i64;
        self.with_kernel(&self.moe_gather, (), "moe_gather", moe_gather_src, |module| {
            self.launch(
                module,
                "moe_gather",
                &[
                    (pos_dev.as_device_ptr().as_raw(), [rows as i64, u]),
                    (w_ptr, [rows as i64, u]),
                    (self.ptr(down_out, 0)?, [m_pad as i64, d as i64]),
                    (self.ptr(shared.1, 0)?, [rows as i64, 1]),
                    (self.ptr(shared.0, 0)?, [rows as i64, d as i64]),
                    (self.ptr(req.dest, 0)?, [rows as i64, d as i64]),
                ],
                (rows as u32, (d / MOE_COMBINE_TN) as u32, 1),
            )
        })?;
        self.stream.synchronize()?;
        for buf in [xp].into_iter().chain(bufs) {
            self.release(buf);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn grouped_gemm(&self, experts: &Experts, block: usize, kind: Kind, qa: u64, das: u64, m_pad: usize, k: usize, table: u64, tiles: usize, n: usize, out: Buf) -> Result<()> {
        let slab = experts.slab(block, kind);
        let b = &experts.blocks[block];
        let (name, [_, _, function], _) = kernel_for(kind.stack(&b.set).quant())?;
        let ns = (experts.per_block * slab.n) as i64;
        let (bytes_at, d_at) = slab.at(b.base[kind as usize]);
        self.with_kernel(&self.moe_qgemm, (name, n), "moe_qgemm", || moe_qgemm_src(name, n), |module| {
            self.launch(
                module,
                function,
                &[
                    (qa, [m_pad as i64, k as i64]),
                    (das, [m_pad as i64, (k / 32) as i64]),
                    (table, [2, tiles as i64]),
                    (bytes_at, [ns, slab.rb as i64]),
                    (d_at, [ns, slab.nb as i64]),
                    (self.ptr(out, 0)?, [m_pad as i64, n as i64]),
                ],
                (tiles as u32, (n / QGEMM_TN) as u32, 1),
            )
        })
    }
}
