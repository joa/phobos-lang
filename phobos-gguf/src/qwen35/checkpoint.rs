// A point in the sequence a session can return to.
//
// Attention blocks rewind by position. A delta net's recurrent state
// summarises every token it has seen and has no prefix to keep, so its state
// at one position is copied out instead. A rewind to that position or later
// restores the copy and runs the rest again.
//
// The copy lives on the host: it is read back at most once per request, and
// device memory is the scarce resource.

use anyhow::Result;

use super::{LayerState, State};
use crate::backend::{Backend, Buf, read_vec};

pub(super) struct Checkpoint {
    pos: usize,
    /// A delta net block's carry and recurrent state; `None` for an
    /// attention block, whose cache needs nothing saved.
    layers: Vec<Option<[Option<Vec<f32>>; 2]>>,
}

impl State {
    /// Saves this position as the one [`State::truncate`] returns to,
    /// replacing any earlier save.
    pub fn checkpoint(&mut self, backend: &dyn Backend) -> Result<()> {
        let save = |held: &Option<(Buf, usize)>| held.map(|(buf, len)| read_vec(backend, buf, len)).transpose();
        let layers = self
            .layers
            .iter()
            .map(|layer| match layer {
                LayerState::Attention(_) => Ok(None),
                LayerState::DeltaNet { carry, recurrent } => Ok(Some([save(carry)?, save(recurrent)?])),
            })
            .collect::<Result<_>>()?;
        self.saved = Some(Checkpoint { pos: self.pos, layers });
        Ok(())
    }

    /// Forgets everything past `positions` and returns how many positions
    /// are kept. That is `positions` itself when nothing needs forgetting,
    /// else the saved checkpoint's position if it is at or before
    /// `positions`.
    ///
    /// Returns `None` when there is no such point, leaving the state
    /// unchanged. An error during the restore leaves the state fit only for
    /// release.
    pub fn truncate(&mut self, positions: usize, backend: &dyn Backend) -> Result<Option<usize>> {
        if positions == self.pos {
            return Ok(Some(positions));
        }
        let Some(saved) = self.saved.as_ref().filter(|saved| saved.pos <= positions.min(self.pos)) else {
            return Ok(None);
        };
        // Release the staging buffers only after every copy is queued. An
        // upload writes immediately, not in stream order, so an upload
        // reusing an earlier layer's staging buffer could overwrite it
        // before that layer's copy reads it.
        let mut staged = Vec::new();
        for (layer, held) in self.layers.iter_mut().zip(&saved.layers) {
            if let (LayerState::DeltaNet { carry, recurrent }, Some([saved_carry, saved_recurrent])) = (layer, held) {
                staged.extend(restore(backend, carry, saved_carry)?);
                staged.extend(restore(backend, recurrent, saved_recurrent)?);
            }
        }
        for buf in staged {
            backend.release(buf);
        }
        self.pos = saved.pos;
        Ok(Some(self.pos))
    }
}

/// Puts `saved` back into `live` and returns the staging buffer the copy
/// reads from, if any. Reuses the live buffer when its size still matches,
/// since a cached decode graph records its address.
fn restore(backend: &dyn Backend, live: &mut Option<(Buf, usize)>, saved: &Option<Vec<f32>>) -> Result<Option<Buf>> {
    match (*live, saved) {
        (Some((buf, len)), Some(data)) if len == data.len() => {
            let staged = backend.upload(data)?;
            backend.copy(staged, 0, buf, 0, len)?;
            Ok(Some(staged))
        }
        _ => {
            if let Some((buf, _)) = live.take() {
                backend.release(buf);
            }
            *live = saved.as_ref().map(|data| backend.upload(data).map(|buf| (buf, data.len()))).transpose()?;
            Ok(None)
        }
    }
}
