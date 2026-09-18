// PrismML's Hadamard folding (`prism.hadamard.*`). A folded weight stores
// `W H S` for a blockwise normalized Sylvester-Walsh-Hadamard `H` over its
// input axis and a fixed +-1 sign vector `S`, so the projection only means
// `W` once its activation is carried through `H S` first: `y = (W H S)
// (S H x)`, both factors their own inverse. A lookup table is the other way
// round, rows stored as `H S h`, restored after the lookup as `h = S (H z)`.
//
// Nothing about a folded weight looks wrong, and a projection that skips the
// transform still produces fluent-looking garbage, so the loader refuses any
// file whose folding it does not understand, as the reference does.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};

use crate::backend::HeadPerm;
use crate::meta::Metadata;

const PREFIX: &str = "prism.hadamard";

/// The folding a file declares, validated.
#[derive(Debug)]
pub struct Folding {
    pub block: usize,
    /// One sign vector per input width.
    signs: HashMap<usize, Arc<Vec<f32>>>,
    weights: HashSet<String>,
    inverse: HashSet<String>,
    /// The delta net's output projection reads its heads regrouped from the
    /// tiled `[head_dim, groups, repeat]` order to `[head_dim, repeat,
    /// groups]` before the transform.
    pub gdn_v_grouped: bool,
}

impl Folding {
    /// The file's folding, or `None` for a file with none.
    pub fn from_metadata(meta: &Metadata) -> Result<Option<Folding>> {
        let key = |suffix: &str| format!("{PREFIX}.{suffix}");
        if !meta.contains(&key("version")) {
            return Ok(None);
        }
        let version = meta.int(&key("version"))?;
        ensure!(version == 1, "unsupported {PREFIX}.version {version}");
        let block = meta.count(&key("block_size"))?;
        ensure!(
            block.is_power_of_two() && block >= 32,
            "{PREFIX}.block_size {block} is not a power of two of at least 32"
        );
        for (suffix, want) in [
            ("transform", "normalized-sylvester-walsh-hadamard"),
            ("axis", "input-last-dimension"),
        ] {
            let got = meta.string(&key(suffix))?;
            ensure!(got == want, "unsupported {PREFIX}.{suffix} '{got}'");
        }

        let mut signs = HashMap::new();
        match meta.string(&key("sign_mode"))? {
            "identity" => {}
            "explicit" => {
                let widths = meta.ints(&key("sign_widths"))?;
                let values = meta.ints(&key("sign_values"))?;
                ensure!(!widths.is_empty(), "{PREFIX}.sign_widths is empty");
                let mut at = 0usize;
                for width in widths {
                    let width = usize::try_from(width)
                        .ok()
                        .filter(|w| *w > 0 && w.is_multiple_of(block))
                        .with_context(|| format!("{PREFIX} sign width {width} is not a whole number of blocks"))?;
                    let run = values
                        .get(at..at + width)
                        .with_context(|| format!("{PREFIX}.sign_values ends inside width {width}"))?;
                    ensure!(
                        run.iter().all(|&v| v == 1 || v == -1),
                        "{PREFIX} sign values must be +-1"
                    );
                    ensure!(
                        signs.insert(width, Arc::new(run.iter().map(|&v| v as f32).collect())).is_none(),
                        "{PREFIX} repeats sign width {width}"
                    );
                    at += width;
                }
                ensure!(at == values.len(), "{PREFIX}.sign_values has {} left over", values.len() - at);
            }
            other => bail!("unsupported {PREFIX}.sign_mode '{other}'"),
        }

        let weights: HashSet<String> = meta.strings(&key("weight_names"))?.iter().cloned().collect();
        let inverse: HashSet<String> = match meta.get(&key("inverse_weight_names")) {
            Some(_) => meta.strings(&key("inverse_weight_names"))?.iter().cloned().collect(),
            None => HashSet::new(),
        };
        ensure!(!weights.is_empty(), "{PREFIX}.weight_names is empty");
        for name in &inverse {
            ensure!(
                name == "token_embd.weight" && !weights.contains(name),
                "{PREFIX}: '{name}' is not a table this loader restores after lookup"
            );
        }
        let gdn_v_grouped = meta
            .get(&key("gdn_v_grouped"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Ok(Some(Folding { block, signs, weights, inverse, gdn_v_grouped }))
    }

    /// Whether `name` is stored folded, so its activation needs the transform.
    pub fn folds(&self, name: &str) -> bool {
        self.weights.contains(name)
    }

    /// Whether `name` is a table whose looked-up rows need the inverse.
    pub fn restores(&self, name: &str) -> bool {
        self.inverse.contains(name)
    }

    /// Every name the file declares, folded or restored, for the loader to
    /// check it consumed each one.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.weights.iter().chain(&self.inverse).map(String::as_str)
    }

    /// The sign vector for an input `width`: all ones where the file carries
    /// none.
    pub fn signs(&self, width: usize) -> Result<Arc<Vec<f32>>> {
        ensure!(
            width.is_multiple_of(self.block),
            "a {width}-wide input is not a whole number of {}-element Hadamard blocks",
            self.block
        );
        if self.signs.is_empty() {
            return Ok(Arc::new(vec![1.0; width]));
        }
        self.signs
            .get(&width)
            .cloned()
            .with_context(|| format!("{PREFIX} has no sign vector for width {width}"))
    }
}

/// The normalized Walsh-Hadamard transform of `x` in place, blockwise.
pub fn fwht(x: &mut [f32], block: usize) {
    let scale = (block as f32).sqrt().recip();
    for chunk in x.chunks_exact_mut(block) {
        let mut half = 1;
        while half < block {
            for base in (0..block).step_by(2 * half) {
                for i in base..base + half {
                    let (a, b) = (chunk[i], chunk[i + half]);
                    chunk[i] = a + b;
                    chunk[i + half] = a - b;
                }
            }
            half *= 2;
        }
        for v in chunk.iter_mut() {
            *v *= scale;
        }
    }
}

/// Where element `j` of a row regrouped by `perm` comes from: grouped head
/// `k * repeat + r` is tiled head `r * groups + k`.
pub fn perm_source(perm: HeadPerm, j: usize) -> usize {
    let (head, within) = (j / perm.head_dim, j % perm.head_dim);
    let (k, r) = (head / perm.repeat, head % perm.repeat);
    (r * perm.groups + k) * perm.head_dim + within
}

/// One row through the activation side: regroup, sign, transform.
pub fn rotate_row(src: &[f32], signs: &[f32], perm: Option<HeadPerm>, block: usize, out: &mut [f32]) {
    for (j, (o, &s)) in out.iter_mut().zip(signs).enumerate() {
        let from = perm.map_or(j, |p| perm_source(p, j));
        *o = src[from] * s;
    }
    fwht(out, block);
}

/// A looked-up table row restored in place: `h = S (H z)`.
pub fn restore_row(row: &mut [f32], signs: &[f32], block: usize) {
    fwht(row, block);
    for (v, &s) in row.iter_mut().zip(signs) {
        *v *= s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(len: usize) -> Vec<f32> {
        (0..len).map(|i| ((i * 37 % 101) as f32 - 50.0) / 17.0).collect()
    }

    #[test]
    fn the_transform_is_its_own_inverse() {
        let x = ramp(2048);
        let mut y = x.clone();
        fwht(&mut y, 1024);
        fwht(&mut y, 1024);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn the_transform_matches_the_sylvester_matrix() {
        let block = 64;
        let x = ramp(block);
        let mut y = x.clone();
        fwht(&mut y, block);
        for (row, &got) in y.iter().enumerate() {
            let want: f32 = x
                .iter()
                .enumerate()
                .map(|(col, &v)| if (row & col).count_ones() % 2 == 1 { -v } else { v })
                .sum::<f32>()
                / (block as f32).sqrt();
            assert!((got - want).abs() < 1e-4, "row {row}: {got} vs {want}");
        }
    }

    #[test]
    fn rotating_then_restoring_is_the_identity() {
        // A folded weight sees `H S x`, a restored row `S H z`: the two undo
        // each other, which is what makes a folded table and a folded
        // projection agree.
        let x = ramp(2048);
        let signs: Vec<f32> = (0..2048).map(|i| if i % 3 == 0 { -1.0 } else { 1.0 }).collect();
        let mut y = vec![0.0; 2048];
        rotate_row(&x, &signs, None, 1024, &mut y);
        restore_row(&mut y, &signs, 1024);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn regrouping_transposes_the_tiled_heads() {
        let perm = HeadPerm { head_dim: 2, groups: 3, repeat: 2 };
        // Tiled heads r * 3 + k: [k0r0, k1r0, k2r0, k0r1, k1r1, k2r1].
        let sources: Vec<usize> = (0..6).map(|h| perm_source(perm, 2 * h) / 2).collect();
        assert_eq!(sources, [0, 3, 1, 4, 2, 5]);
    }
}
