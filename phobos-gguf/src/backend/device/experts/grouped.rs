// The prompt pass's routed feed-forward as grouped GEMMs: the rows sorted
// by expert on the host, a permuted activation whose segments are padded to
// the GEMM's tile, the experts brought in a group at a time (as many as the
// block has slots), each group's segments contracted through a schedule
// table, and each row's eight results gathered back with its weights.
// `PHOBOS_MOE_GROUPED` opts in; see `ENV.md`.

use anyhow::{Result, bail, ensure};
use cust::memory::DeviceBuffer;
use phobos_kernels::cuda_ok;

use super::super::kernels::{MOE_COMBINE_TN, MOE_USED, QGEMM_TM, QGEMM_TN};
use super::super::{DeviceBackend, Plane};
use super::{Experts, KINDS, Kind, NONE};
use crate::backend::{Backend, Buf, Moe};
use crate::quant::Quant;

/// One expert's rows of the permuted activation.
struct Segment {
    expert: usize,
    /// First padded row, and rows of real activation there.
    start: usize,
    len: usize,
}

/// Rows of the padded activation a permutation kernel copies per program.
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

/// The K-quant prompt GEMM over a schedule table: program `p` of the first
/// grid axis contracts activation tile `SCHED[1, p]` against the slab rows
/// of slot `SCHED[0, p]`, `NE` rows an expert.
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
        let per_block = experts.per_block;
        let quants = {
            let set = &experts.blocks[block].set;
            (set.gate.quant(), set.up.quant(), set.down.quant())
        };
        ensure!(
            quants.0 == quants.1 && d.is_multiple_of(256) && d_ff.is_multiple_of(256) && d_ff.is_multiple_of(QGEMM_TN) && d.is_multiple_of(QGEMM_TN),
            "the grouped prompt path wants gate and up in one format and widths in whole blocks and tiles"
        );

        // The rows by expert, each expert's rows padded to the tile.
        let chosen: Vec<i32> = experts.blocks[block].topk_host.as_slice()[..rows * MOE_USED].to_vec();
        let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); req.n_expert];
        for (r, row) in chosen.chunks_exact(MOE_USED).enumerate() {
            for (j, &e) in row.iter().enumerate() {
                ensure!((0..req.n_expert as i32).contains(&e), "the router chose expert {e}");
                by_expert[e as usize].push((r, j));
            }
        }
        let mut segments = Vec::new();
        let mut perm: Vec<i32> = Vec::new();
        let mut pos = vec![0i32; rows * MOE_USED];
        for (e, users) in by_expert.iter().enumerate() {
            if users.is_empty() {
                continue;
            }
            let start = perm.len();
            for &(r, j) in users {
                pos[r * MOE_USED + j] = perm.len() as i32;
                perm.push(r as i32);
            }
            // Padding rows read row zero; nothing gathers their outputs.
            while !perm.len().is_multiple_of(QGEMM_TM) {
                perm.push(0);
            }
            segments.push(Segment { expert: e, start, len: users.len() });
        }
        let m_pad = perm.len();

        // The permuted activation, quantized once.
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
        let act = self.quantize_act(xp, m_pad, d)?;
        let (qa, das) = self.act_ptrs(act)?;
        let gate_out = self.alloc(m_pad * d_ff)?;
        let up_out = self.alloc(m_pad * d_ff)?;
        let h = self.alloc(m_pad * d_ff)?;
        let down_out = self.alloc(m_pad * d)?;

        // Experts a group at a time, as many as the block has slots.
        for group in segments.chunks(per_block) {
            let slots = self.bring_in(experts, block, group)?;
            let mut sched: Vec<i32> = Vec::new();
            let mut tiles = 0;
            for (seg, &slot) in group.iter().zip(&slots) {
                let count = seg.len.div_ceil(QGEMM_TM);
                for t in 0..count {
                    sched.push(slot as i32);
                    sched.push((seg.start / QGEMM_TM + t) as i32);
                }
                tiles += count;
            }
            // `[2, T]`: slots first, then tiles.
            let (a, b): (Vec<i32>, Vec<i32>) = sched.chunks_exact(2).map(|p| (p[0], p[1])).unzip();
            let table = DeviceBuffer::from_slice(&[a, b].concat())?;
            let table_ptr = table.as_device_ptr().as_raw();
            self.grouped_gemm(experts, block, Kind::Gate, quants.0, qa, das, m_pad, d, table_ptr, tiles, d_ff, gate_out)?;
            self.grouped_gemm(experts, block, Kind::Up, quants.1, qa, das, m_pad, d, table_ptr, tiles, d_ff, up_out)?;
            // Only this group's rows of the SwiGLU and the down are new,
            // but the whole buffers are cheap next to the copies.
            let plane = |buf| Plane { buf, offset: 0, pitch: d_ff };
            self.swiglu_planes(plane(gate_out), plane(up_out), h, m_pad, d_ff)?;
            let hact = self.quantize_act(h, m_pad, d_ff)?;
            let (hqa, hdas) = self.act_ptrs(hact)?;
            self.grouped_gemm(experts, block, Kind::Down, quants.2, hqa, hdas, m_pad, d_ff, table_ptr, tiles, d, down_out)?;
            // The table is read by launches already issued (eager), so the
            // buffer may go only once they have run.
            self.stream.synchronize()?;
            drop(table);
        }

        let (w_ptr, _) = {
            let b = &experts.blocks[block];
            (b.weights.as_device_ptr().as_raw(), ())
        };
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
        for buf in [xp, gate_out, up_out, h, down_out] {
            self.release(buf);
        }
        Ok(())
    }

    /// The group's experts into the block's slots, copies on the stream,
    /// returning each one's slot. Every miss crosses the bus once here too.
    fn bring_in(&self, experts: &mut Experts, block: usize, group: &[Segment]) -> Result<Vec<usize>> {
        let Experts { blocks, slabs, per_block, tick, stats, .. } = &mut *experts;
        *tick += 1;
        let (tick, per_block) = (*tick, *per_block);
        let slabs = slabs.as_ref().expect("laid out");
        let b = &mut blocks[block];
        for seg in group {
            let s = b.slot_of[seg.expert];
            if s != NONE {
                b.held[s as usize].1 = tick;
            }
        }
        let mut out = Vec::with_capacity(group.len());
        for seg in group {
            let e = seg.expert;
            let local = match b.slot_of[e] {
                s if s != NONE => {
                    stats.hits += seg.len as u64;
                    s as usize
                }
                _ => {
                    stats.misses += seg.len as u64;
                    let Some(victim) = (0..per_block).filter(|&s| b.held[s].1 != tick).min_by_key(|&s| b.held[s].1) else {
                        bail!("a group of {} experts does not fit the block's {per_block} slots", group.len())
                    };
                    let (old, _) = b.held[victim];
                    if old != NONE {
                        b.slot_of[old as usize] = NONE;
                    }
                    let src = b.mirror.expert(e);
                    for (k, kind) in KINDS.into_iter().enumerate() {
                        let slab = &slabs[&(kind, kind.stack(&b.set).quant())];
                        let (bytes_at, d_at) = slab.at(b.base[k] + victim);
                        for (dst, (ptr, len)) in [(bytes_at, src[k]), (d_at, src[3 + k])] {
                            // SAFETY: pinned source, a slot inside the slab.
                            cuda_ok(
                                unsafe { cust::sys::cuMemcpyHtoDAsync_v2(dst, ptr, len, self.stream.as_inner()) },
                                "copying an expert into its slot",
                            )?;
                            stats.bytes += len as u64;
                        }
                    }
                    b.held[victim] = (e as u32, tick);
                    b.slot_of[e] = victim as u32;
                    victim
                }
            };
            out.push(local);
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn grouped_gemm(
        &self,
        experts: &Experts,
        block: usize,
        kind: Kind,
        quant: Quant,
        qa: u64,
        das: u64,
        m_pad: usize,
        k: usize,
        table: u64,
        tiles: usize,
        n: usize,
        out: Buf,
    ) -> Result<()> {
        let slab = experts.slab(block, kind);
        let b = &experts.blocks[block];
        let (name, function) = match quant {
            Quant::Q4_K => ("q4k", "q4k_moe_qgemm"),
            Quant::Q5_K => ("q5k", "q5k_moe_qgemm"),
            Quant::Q6_K => ("q6k", "q6k_moe_qgemm"),
            other => bail!("no grouped GEMM for {} experts", other.name()),
        };
        let ns = ((experts.per_block + 1) * slab.n) as i64;
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
