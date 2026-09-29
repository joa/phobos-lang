// Uploading a raw-format weight: its block bytes and scale planes, laid out
// as the device kernels read them.

use super::*;
use crate::quant::RawScales;
use crate::quant::grouped::group_rows;

/// A device-resident raw-block weight: the addresses of its block bytes and
/// `f16` scale planes, its output width `n`, super-blocks per row `nb`, and
/// its format. `dmin` is `None` for a format with no minimum term.
///
/// These are raw pointers because the [`arena::Arena`] owns the memory.
pub(super) struct DeviceRaw {
    pub(super) bytes: u64,
    pub(super) d: u64,
    pub(super) dmin: Option<u64>,
    pub(super) n: usize,
    pub(super) nb: usize,
    pub(super) quant: Quant,
}

impl DeviceBackend {
    /// [`Backend::constant_raw`]. Uploads the blocks and scale planes, with
    /// rows grouped for formats that need it.
    pub(super) fn upload_raw(&self, key: &str, packed: &Packed) -> Result<RawBuf> {

        if let Some(&buf) = self.raw_constants.borrow().get(key) {
            return Ok(buf);
        }
        let (k, n) = (packed.k(), packed.n());
        let scales = packed.raw_scales()?;
        let block = packed.spec().block;
        let nb = k / block;
        let mut bytes = packed.device_blocks();
        let mut d = scales.d;
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

