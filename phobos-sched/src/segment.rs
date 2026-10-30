// Segmentation: cuts a node's instruction list into segments that each fit
// the memory budget.

use std::collections::HashMap;

use anyhow::{Result, bail};
use phobos_cluster::isa::{Instr, Op, Segment};
use phobos_cluster::tile::{DataType, TileId};

use crate::SegMem;

/// Splits one node's instruction list into segments whose incremental
/// working set stays within `budget`. Returns the segments, their
/// [`SegMem`], and the node's peak resident bytes.
///
/// The incremental working set is the bytes allocated since the segment
/// began that are still live at its peak. A cut falls before any ALLOC that
/// would go over budget. A single tile larger than the budget is an error.
/// Cutting never reorders, so deps across a cut stay valid (see [`crate::validate`]).
pub(super) fn segment(
    instrs: Vec<Instr>,
    budget: u64,
    seg_id: &mut u64,
) -> Result<(Vec<Segment>, Vec<SegMem>, u64)> {
    // tile -> bytes, learned from ALLOC ops (FREE carries no size).
    let mut bytes_of: HashMap<TileId, u64> = HashMap::new();
    let data_type_bytes = |d: DataType| d.bytes() as u64;
    for i in &instrs {
        if let Op::Alloc {
            tile,
            shape,
            data_type,
        } = &i.op
        {
            bytes_of.insert(
                *tile,
                shape.iter().product::<u64>() * data_type_bytes(*data_type),
            );
        }
    }

    let mut segs = Vec::new();
    let mut mems = Vec::new();
    let mut cur: Vec<Instr> = Vec::new();

    let mut resident: u64 = 0; // absolute, across the whole node program
    let mut seg_start: u64 = 0; // resident when the current segment began
    let mut seg_peak: u64 = 0; // absolute high-water within the current segment
    let mut node_peak: u64 = 0;

    let flush = |cur: &mut Vec<Instr>,
                 segs: &mut Vec<Segment>,
                 mems: &mut Vec<SegMem>,
                 seg_peak: u64,
                 seg_start: u64,
                 seg_id: &mut u64| {
        if cur.is_empty() {
            return;
        }
        segs.push(Segment {
            id: *seg_id,
            instructions: std::mem::take(cur),
        });
        mems.push(SegMem {
            peak: seg_peak,
            incremental: seg_peak.saturating_sub(seg_start),
        });
        *seg_id += 1;
    };

    for instr in instrs {
        if let Op::Alloc { tile, .. } = &instr.op {
            let b = bytes_of[tile];
            if b > budget {
                bail!("supertile of {b} bytes exceeds the memory budget of {budget} bytes",);
            }
            // Cut if this allocation takes the segment over budget. FREEs can
            // drop resident below `seg_start`, so the subtraction saturates.
            if !cur.is_empty() && (resident + b).saturating_sub(seg_start) > budget {
                flush(&mut cur, &mut segs, &mut mems, seg_peak, seg_start, seg_id);
                seg_start = resident;
                seg_peak = resident;
            }
            resident += b;
        } else if let Op::Free { tile, .. } = &instr.op {
            resident = resident.saturating_sub(bytes_of.get(tile).copied().unwrap_or(0));
        }
        seg_peak = seg_peak.max(resident);
        node_peak = node_peak.max(resident);
        cur.push(instr);
    }
    flush(&mut cur, &mut segs, &mut mems, seg_peak, seg_start, seg_id);

    Ok((segs, mems, node_peak))
}
