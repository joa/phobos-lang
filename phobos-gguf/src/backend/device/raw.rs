// The raw-format upload: a format's block bytes and scale planes as the
// device kernels read them.

use super::*;
use crate::quant::RawScales;

/// A device-resident raw-block weight: where its file bytes and `f16` header
/// plane(s) landed in the [`arena::Arena`] (`dmin` absent for a format with no
/// minimum term), output width, super-blocks per row, and which format's
/// kernel decodes it.
///
/// Device pointers rather than buffers because the arena owns the allocation:
/// see `arena.rs` for why the weights share a dozen of those rather than
/// taking one each.
pub(super) struct DeviceRaw {
    pub(super) bytes: u64,
    pub(super) d: u64,
    pub(super) dmin: Option<u64>,
    pub(super) n: usize,
    pub(super) nb: usize,
    pub(super) quant: Quant,
}

impl DeviceBackend {
    /// [`Backend::constant_raw`]: the payload and scale planes, grouped
    /// where the format wants it.
    pub(super) fn upload_raw(&self, key: &str, packed: &Packed) -> Result<RawBuf> {

        if let Some(&buf) = self.raw_constants.borrow().get(key) {
            return Ok(buf);
        }
        let (k, n) = (packed.k(), packed.n());
        let scales = packed.raw_scales()?;
        let block = packed.spec().block;
        let nb = k / block;
        let mut bytes = packed.device_blocks();
        let mut d = scales.d.clone();
        if packed.quant().grouped_rows() {
            let dev = packed.quant().device_block().1;
            bytes = group_rows(&bytes, n, nb, dev);
            d = group_rows(&d, n, nb, 1);
        }
        let scales = RawScales { d, dmin: scales.dmin };
        let uploaded = if self.arena_weights {
            let dmin = match scales.dmin.is_empty() {
                true => None,
                false => Some(self.arena.upload(&scales.dmin)?),
            };
            DeviceRaw {
                bytes: self.arena.upload(&bytes)?,
                d: self.arena.upload(&scales.d)?,
                dmin,
                n,
                nb,
                quant: packed.quant(),
            }
        } else {
            let owned = (
                DeviceBuffer::from_slice(&bytes)?,
                DeviceBuffer::from_slice(&scales.d)?,
                match scales.dmin.is_empty() {
                    true => None,
                    false => Some(DeviceBuffer::from_slice(&scales.dmin)?),
                },
            );
            let at = DeviceRaw {
                bytes: owned.0.as_device_ptr().as_raw(),
                d: owned.1.as_device_ptr().as_raw(),
                dmin: owned.2.as_ref().map(|m| m.as_device_ptr().as_raw()),
                n,
                nb,
                quant: packed.quant(),
            };
            self.owned_raw.borrow_mut().push(owned);
            at
        };
        let mut raws = self.raw_quants.borrow_mut();
        raws.push(uploaded);
        let buf = RawBuf(raws.len() - 1);
        drop(raws);
        self.raw_constants.borrow_mut().insert(key.to_string(), buf);
        Ok(buf)
    }
}

/// Columns a grouped upload is padded to: the widest decode tile.
pub(super) const RAW_GROUP_PAD: usize = 64;

/// `[n][nb][unit]` rows as `[n' / 8][nb][8][unit]`, `n'` the next multiple
/// of [`RAW_GROUP_PAD`], zero-padded.
fn group_rows<T: Copy + Default>(rows: &[T], n: usize, nb: usize, unit: usize) -> Vec<T> {
    const GROUP: usize = 8;
    let groups = n.div_ceil(RAW_GROUP_PAD) * (RAW_GROUP_PAD / GROUP);
    let mut out = vec![T::default(); groups * nb * GROUP * unit];
    for j in 0..n {
        let (group, in_group) = (j / GROUP, j % GROUP);
        for b in 0..nb {
            let src = (j * nb + b) * unit;
            let dst = ((group * nb + b) * GROUP + in_group) * unit;
            out[dst..dst + unit].copy_from_slice(&rows[src..src + unit]);
        }
    }
    out
}
