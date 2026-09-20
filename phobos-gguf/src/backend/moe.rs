// The mixture-of-experts feed-forward as one backend operation.

use super::{Buf, QAct};

/// What a backend needs to run the next block's router on the residual as
/// it stands, ahead of that block: its post-attention norm, the router's
/// dense `[d_model, n_expert]` weight and its expert set. A backend that
/// prefetches uses it to start the next block's misses early; one that
/// does not ignores it.
#[derive(Clone, Copy, Debug)]
pub struct Lookahead {
    pub gain: Buf,
    pub eps: f32,
    pub router: Buf,
    pub experts: ExpertsBuf,
}

/// A handle to a block's expert set as a backend holds it. See
/// [`super::Backend::constant_experts`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertsBuf(pub usize);

/// A block's routed feed-forward: each row's top experts applied to it,
/// their outputs summed by the router's weights, the gated shared expert
/// added, and the whole thing added into `dest`.
///
/// One operation rather than a loop over experts, because which experts a
/// row chose exists only on the backend: nothing reads back inside a pass,
/// so the choice can neither reach the model code nor pick a weight there.
///
/// The router follows llama.cpp's `build_moe_ffn` for this architecture:
/// softmax over every expert's logit, the `n_used` largest probabilities
/// selected, and those renormalized to sum to one. See [`route`].
pub struct Moe {
    /// The block's normalized input, `[rows, d_model]`, and its quantized
    /// copy where the caller has one.
    pub x: Buf,
    pub act: Option<QAct>,
    pub rows: usize,
    pub d_model: usize,
    /// Width of one expert.
    pub d_ff: usize,
    /// `[rows, n_expert]` router logits.
    pub logits: Buf,
    pub n_expert: usize,
    pub n_used: usize,
    pub experts: ExpertsBuf,
    /// The shared expert's output, `[rows, d_model]`, and its gate's logit,
    /// `[rows, 1]`: the output is scaled by the sigmoid of the logit and
    /// added along with the routed experts.
    pub shared: Option<(Buf, Buf)>,
    /// The residual, `[rows, d_model]`, accumulated into.
    pub dest: Buf,
    /// The next block's router, for a backend that prefetches; see
    /// [`Lookahead`]. `None` for the last block or a caller without one.
    pub lookahead: Option<Lookahead>,
    /// Where to leave each row's chosen experts, `[rows, n_used]` ids held as
    /// f32 the way [`super::Backend::argmax`] carries an index, for a caller
    /// tracing the router. Read back after the pass, never inside it.
    pub routes: Option<Buf>,
}

/// The router's choice for one row: `(expert, weight)` pairs in descending
/// weight, `n_used` of them, the weights summing to one.
///
/// Softmax first over all `logits`, then the `n_used` largest probabilities,
/// then those renormalized; the order matters, since selecting first and
/// softmaxing after gives different weights. A tie keeps the lower index.
pub fn route(logits: &[f32], n_used: usize) -> Vec<(usize, f32)> {
    assert!(n_used <= logits.len(), "routing to {n_used} of {} experts", logits.len());
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let total: f32 = exps.iter().sum();
    let mut ranked: Vec<(usize, f32)> = exps.iter().map(|&e| e / total).enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked.truncate(n_used);
    let picked: f32 = ranked.iter().map(|&(_, p)| p).sum();
    for (_, p) in &mut ranked {
        *p /= picked;
    }
    ranked
}

#[cfg(test)]
mod tests {
    use super::route;

    #[test]
    fn routing_takes_the_softmax_top_and_renormalizes_it() {
        // Probabilities 1 : 2 : 4 : 8; the top two are 8/15 and 4/15, which
        // renormalize to 2/3 and 1/3.
        let logits = [0.0, 2f32.ln(), 4f32.ln(), 8f32.ln()];
        let picked = route(&logits, 2);
        assert_eq!(picked.len(), 2);
        assert_eq!((picked[0].0, picked[1].0), (3, 2));
        assert!((picked[0].1 - 2.0 / 3.0).abs() < 1e-6, "{picked:?}");
        assert!((picked[1].1 - 1.0 / 3.0).abs() < 1e-6, "{picked:?}");
    }

    #[test]
    fn routing_breaks_a_tie_toward_the_lower_index_and_survives_large_logits() {
        let picked = route(&[1000.0, 5.0, 1000.0], 2);
        assert_eq!((picked[0].0, picked[1].0), (0, 2));
        assert!((picked[0].1 - 0.5).abs() < 1e-6 && (picked[1].1 - 0.5).abs() < 1e-6);
    }
}
