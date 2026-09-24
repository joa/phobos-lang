// A point in the sequence a session can return to.
//
// The attention blocks rewind by position, as llama's do, but a delta net's
// recurrent state summarises every token it has seen and has no prefix to
// keep. What it held at one position is instead copied out, and a rewind to
// anywhere at or past that position restores the copy and runs the rest
// again. The copy lives on the host: it is read back at most once a request,
// and the card is where room is short.

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
    /// Save this position as the one [`State::truncate`] returns to,
    /// replacing any saved before.
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

    /// Forget everything past `positions` and return how many positions are
    /// kept: `positions` itself when nothing is to be forgotten, else the
    /// saved checkpoint when it is at or before `positions`. `None` when
    /// there is no such point; the state is then unchanged, except after an
    /// error in the restore, which leaves it fit only for release.
    pub fn truncate(&mut self, positions: usize, backend: &dyn Backend) -> Result<Option<usize>> {
        if positions == self.pos {
            return Ok(Some(positions));
        }
        let Some(saved) = self.saved.as_ref().filter(|saved| saved.pos <= positions.min(self.pos)) else {
            return Ok(None);
        };
        // The staging buffers go back only once every copy is queued: an
        // upload writes at once, not in stream order, so one reusing a
        // buffer released a layer earlier could land before that layer's
        // copy has read it.
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

/// Put `saved` back into `live`, returning the staging buffer the copy reads
/// from, if there is one. Into the same buffer when it is still the right
/// size, since a cached decode graph records the buffer's address.
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
