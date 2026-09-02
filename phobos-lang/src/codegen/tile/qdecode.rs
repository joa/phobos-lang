// `<fmt>_qdecode_t`: a format's decode stored to the `[K, N]` scratch a
// batched matmul reads, where `<fmt>_qdot_t` contracts the same decode against
// one activation row. The decode itself is shared with `<fmt>_qdot.rs`; only
// the thread map differs, and the store sets it (see the doc comment on
// `qdecode_t_into`).

use super::iq1m::{IQ1M_BLOCK_BYTES, IQ1M_LANE, Iq1mBlock, Iq1mLane};
use super::iq1s::{IQ1S_BLOCK_BYTES, IQ1S_LANE, Iq1sBlock, Iq1sLane};
use super::iq2s::{IQ2S_BLOCK_BYTES, IQ2S_LANE, Iq2sBlock, Iq2sLane};
use super::iq2xs::{IQ2XS_BLOCK_BYTES, IQ2XS_LANE, Iq2xsBlock, Iq2xsLane};
use super::iq2xxs::{IQ2XXS_BLOCK_BYTES, IQ2XXS_LANE, Iq2xxsBlock, Iq2xxsLane};
use super::iq3s::{IQ3S_BLOCK_BYTES, IQ3S_HALF, Iq3sBlock, Iq3sLane};
use super::iq3xxs::{IQ3XXS_BLOCK_BYTES, IQ3XXS_HALF, Iq3xxsBlock, Iq3xxsLane};
use super::*;

/// A raw format with an expansion intrinsic, named by the callee.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::codegen) enum QFormat {
    Iq1s,
    Iq2xxs,
    Iq1m,
    Iq2s,
    Iq2xs,
    Iq3xxs,
    Iq3s,
}

/// A lane's loop-invariant geometry, per format.
enum LaneGeom<'c> {
    Iq1s(Iq1sLane<'c>),
    Iq2xxs(Iq2xxsLane<'c>),
    Iq1m(Iq1mLane<'c>),
    Iq2s(Iq2sLane<'c>),
    Iq2xs(Iq2xsLane<'c>),
    Iq3xxs(Iq3xxsLane<'c>),
    Iq3s(Iq3sLane<'c>),
}

/// A lane's per-block decode state, per format.
enum BlockDec<'c> {
    Iq1s(Iq1sBlock<'c>),
    Iq2xxs(Iq2xxsBlock<'c>),
    Iq1m(Iq1mBlock<'c>),
    Iq2s(Iq2sBlock<'c>),
    Iq2xs(Iq2xsBlock<'c>),
    Iq3xxs(Iq3xxsBlock<'c>),
    Iq3s(Iq3sBlock<'c>),
}

impl QFormat {
    /// The format an intrinsic name selects, if it is one of these.
    pub(in crate::codegen) fn from_intrinsic(callee: &str) -> Option<Self> {
        match callee {
            "iq1s_qdecode_t" => Some(Self::Iq1s),
            "iq2xxs_qdecode_t" => Some(Self::Iq2xxs),
            "iq1m_qdecode_t" => Some(Self::Iq1m),
            "iq2s_qdecode_t" => Some(Self::Iq2s),
            "iq2xs_qdecode_t" => Some(Self::Iq2xs),
            "iq3xxs_qdecode_t" => Some(Self::Iq3xxs),
            "iq3s_qdecode_t" => Some(Self::Iq3s),
            _ => None,
        }
    }

    pub(in crate::codegen) fn intrinsic(self) -> &'static str {
        match self {
            Self::Iq1s => "iq1s_qdecode_t",
            Self::Iq2xxs => "iq2xxs_qdecode_t",
            Self::Iq1m => "iq1m_qdecode_t",
            Self::Iq2s => "iq2s_qdecode_t",
            Self::Iq2xs => "iq2xs_qdecode_t",
            Self::Iq3xxs => "iq3xxs_qdecode_t",
            Self::Iq3s => "iq3s_qdecode_t",
        }
    }

    /// Table operands after `(qb, d)`: a magnitude grid, and a sign table for
    /// the formats that carry their signs separately.
    pub(in crate::codegen) fn tables(self) -> usize {
        match self {
            Self::Iq1s | Self::Iq1m => 1,
            Self::Iq2xxs | Self::Iq2s | Self::Iq2xs | Self::Iq3xxs | Self::Iq3s => 2,
        }
    }

    fn block_bytes(self) -> i64 {
        match self {
            Self::Iq1s => IQ1S_BLOCK_BYTES,
            Self::Iq2xxs => IQ2XXS_BLOCK_BYTES,
            Self::Iq1m => IQ1M_BLOCK_BYTES,
            Self::Iq2s => IQ2S_BLOCK_BYTES,
            Self::Iq2xs => IQ2XS_BLOCK_BYTES,
            Self::Iq3xxs => IQ3XXS_BLOCK_BYTES,
            Self::Iq3s => IQ3S_BLOCK_BYTES,
        }
    }

    fn lane_elems(self) -> i64 {
        match self {
            Self::Iq1s => IQ1S_LANE,
            Self::Iq2xxs => IQ2XXS_LANE,
            Self::Iq1m => IQ1M_LANE,
            Self::Iq2s => IQ2S_LANE,
            Self::Iq2xs => IQ2XS_LANE,
            Self::Iq3xxs => 2 * IQ3XXS_HALF,
            Self::Iq3s => 2 * IQ3S_HALF,
        }
    }

    /// Elements per grid entry. IQ3_XXS and IQ3_S split a lane's eight over
    /// two entries; the rest read one.
    fn half(self) -> i64 {
        match self {
            Self::Iq3xxs => IQ3XXS_HALF,
            Self::Iq3s => IQ3S_HALF,
            _ => self.lane_elems(),
        }
    }
}

impl<'c> LaneGeom<'c> {
    fn k_lane_off(&self) -> Value<'c, 'c> {
        match self {
            Self::Iq1s(g) => g.k_lane_off,
            Self::Iq2xxs(g) => g.k_lane_off,
            Self::Iq1m(g) => g.k_lane_off,
            Self::Iq2s(g) => g.k_lane_off,
            Self::Iq2xs(g) => g.k_lane_off,
            Self::Iq3xxs(g) => g.k_lane_off,
            Self::Iq3s(g) => g.k_lane_off,
        }
    }
}

impl<'c> Codegen<'c> {
    /// `out[kbase + lane*8 + y, j] = decode(..)` over the whole of `k`.
    ///
    /// Consecutive threads take consecutive output *columns*, so a warp's 32
    /// stores land in four fully covered 32-byte sectors. Nothing is staged and
    /// nothing synchronizes. `out` is a tensor slice, never a tile: the decoded
    /// values are already in registers at the rows they belong in, which is
    /// also why there is no value form.
    pub(in crate::codegen) fn qdecode_t_into(
        &mut self,
        block: &Block<'c>,
        fmt: QFormat,
        qb: &MemVal<'c>,
        d: &MemVal<'c>,
        tables: &[MemVal<'c>],
        out: &MemVal<'c>,
    ) -> Result<()> {
        let what = fmt.intrinsic();
        if tables.len() != fmt.tables() {
            bail!("{what} takes {} table operands", fmt.tables());
        }
        for (v, role) in [(qb, "qb"), (d, "d"), (out, "destination")]
            .into_iter()
            .chain(tables.iter().map(|t| (t, "table")))
        {
            if v.shape.len() != 2 {
                bail!("{what} {role} must be rank-2");
            }
            if v.is_masked() {
                bail!("{what} {role} must be a fully in-bounds slice");
            }
        }
        if qb.elem != self.i8_t {
            bail!("{what} decodes raw i8 block bytes");
        }
        if d.elem != self.f16_t {
            bail!("{what}'s block scale must be f16");
        }
        if tables.iter().any(|t| t.elem != self.i8_t) {
            bail!("{what}'s tables must hold packed i8 lanes");
        }
        // f16 is free where the reader is `stage_to_f16`, which truncates a
        // weight operand anyway. The caller decides; see `project_raw_dense`.
        if out.elem != self.f32_t && out.elem != self.f16_t {
            bail!("{what} produces f32 or f16 weights, not {}", out.elem);
        }
        if out.shared {
            bail!("{what} writes a tensor slice, not a shared tile");
        }
        if self.cta_threads % WARP != 0 {
            bail!("{what} needs a CTA that is a whole number of warps");
        }
        let cols = qb.shape[0];
        if cols == DYN {
            bail!("{what} needs a static output width");
        }
        self.check_shapes(&[cols], &[d.shape[0]], &format!("{what} d rows"))?;
        self.check_shapes(&[cols], &[out.shape[1]], &format!("{what} destination width"))?;
        if out.shape[0] != DYN && out.shape[0] % 256 != 0 {
            bail!("{what} needs a whole number of 256-element blocks");
        }

        let cols_w = self.const_index(block, cols)?;
        let total = self.const_index(block, cols * WARP)?;
        let tid = self.thread_id(block)?;
        let bdim = self.block_dim(block)?;

        let body = Block::new(&[(self.index_t, self.loc)]);
        let li = detach(body.argument(0)?.into());
        // Lane over columns, not columns over lanes: `j` is the fast axis.
        let lane = self.divui(&body, li, cols_w)?;
        let j = self.remui(&body, li, cols_w)?;

        let geom = match fmt {
            QFormat::Iq1s => LaneGeom::Iq1s(self.iq1s_lane(&body, lane)?),
            QFormat::Iq2xxs => LaneGeom::Iq2xxs(self.iq2xxs_lane(&body, lane)?),
            QFormat::Iq1m => LaneGeom::Iq1m(self.iq1m_lane(&body, lane)?),
            QFormat::Iq2s => LaneGeom::Iq2s(self.iq2s_lane(&body, lane)?),
            QFormat::Iq2xs => LaneGeom::Iq2xs(self.iq2xs_lane(&body, lane)?),
            QFormat::Iq3xxs => LaneGeom::Iq3xxs(self.iq3xxs_lane(&body, lane)?),
            QFormat::Iq3s => LaneGeom::Iq3s(self.iq3s_lane(&body, lane)?),
        };

        let step = self.const_index(&body, 256)?;
        let zero_k = self.const_index(&body, 0)?;
        let kd = if out.shape[0] == DYN {
            let zero = self.const_index(&body, 0)?;
            self.push(&body, memref::dim(out.mem, zero, self.loc))?
        } else {
            self.const_index(&body, out.shape[0])?
        };
        let blk_bytes = self.const_index(&body, fmt.block_bytes())?;

        let kb = Block::new(&[(self.index_t, self.loc)]);
        let kbase = detach(kb.argument(0)?.into());
        let blk = self.divui(&kb, kbase, step)?;
        let at = self.raw_block_at(&kb, j, blk, blk_bytes)?;
        // A one-table format never reads `signs`; aliasing it to the grid
        // keeps one shape for the call below.
        let two = QTables {
            grid: &tables[0],
            signs: tables.last().expect("checked non-empty above"),
        };
        let dec = match &geom {
            LaneGeom::Iq1s(g) => {
                BlockDec::Iq1s(self.iq1s_block(&kb, g, qb, d, &tables[0], &at)?)
            }
            LaneGeom::Iq2xxs(g) => BlockDec::Iq2xxs(
                self.iq2xxs_block(&kb, g, qb, d, &two, &at)?,
            ),
            LaneGeom::Iq1m(g) => {
                BlockDec::Iq1m(self.iq1m_block(&kb, g, qb, d, &tables[0], &at)?)
            }
            LaneGeom::Iq2s(g) => BlockDec::Iq2s(
                self.iq2s_block(&kb, g, qb, d, &two, &at)?,
            ),
            LaneGeom::Iq2xs(g) => BlockDec::Iq2xs(
                self.iq2xs_block(&kb, g, qb, d, &two, &at)?,
            ),
            LaneGeom::Iq3xxs(g) => BlockDec::Iq3xxs(
                self.iq3xxs_block(&kb, g, qb, d, &two, &at)?,
            ),
            LaneGeom::Iq3s(g) => BlockDec::Iq3s(
                self.iq3s_block(&kb, g, qb, d, &two, &at)?,
            ),
        };

        let row0 = self.addi(&kb, kbase, geom.k_lane_off())?;
        let half = fmt.half();
        for entry in 0..fmt.lane_elems() / half {
            // Both the entry's element offset and its offset into the sign
            // entry, which are the same number. Zero for a one-entry format.
            let entry_off = self.const_index(&kb, entry * half)?;
            let base = if entry == 0 {
                row0
            } else {
                self.addi(&kb, row0, entry_off)?
            };
            for y in 0..half {
                let yc = self.const_index(&kb, y)?;
                let decoded = match &dec {
                    BlockDec::Iq1s(b) => self.iq1s_decoded(&kb, b, y)?,
                    BlockDec::Iq2xxs(b) => self.iq2xxs_decoded(&kb, b, y)?,
                    BlockDec::Iq1m(b) => self.iq1m_decoded(&kb, b, y)?,
                    BlockDec::Iq2s(b) => self.iq2s_decoded(&kb, b, y)?,
                    BlockDec::Iq2xs(b) => self.iq2xs_decoded(&kb, b, y)?,
                    BlockDec::Iq3xxs(b) => self.iq3xxs_decoded(&kb, b, entry, y)?,
                    BlockDec::Iq3s(b) => self.iq3s_decoded(&kb, b, entry, y)?,
                };
                let decoded = self.numeric_cast(&kb, decoded, out.elem)?;
                let row = self.addi(&kb, base, yc)?;
                kb.append_operation(memref::store(decoded, out.mem, &[row, j], self.loc));
            }
        }
        kb.append_operation(scf::r#yield(&[], self.loc));

        let kr = Region::new();
        kr.append_block(kb);
        body.append_operation(scf::r#for(zero_k, kd, step, kr, self.loc));
        body.append_operation(scf::r#yield(&[], self.loc));

        let region = Region::new();
        region.append_block(body);
        block.append_operation(scf::r#for(tid, total, bdim, region, self.loc));
        Ok(())
    }
}
