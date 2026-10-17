// Inputs for the GEMM checks and an f64 reference for them.
//
// Errors are normalized by the dot's condition, |alpha| * sum|a*b| +
// |beta * c|, which is what floating-point summation error scales with
// whatever the order. A plain relative error is also reported, over the
// elements whose result is not itself a cancellation.

use super::contract::{Elem, Kind};
use rand::prelude::*;

/// How a check's inputs are drawn.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Pattern {
    /// Uniform in [-1, 1].
    Uniform,
    /// Uniform in [-1, 1] times 2^e, e uniform in [-E, E], E per kind.
    Varied,
    /// The second half of k repeats the first with B negated and nudged, so
    /// every exact result is small next to sum|a*b|.
    Cancel,
}

pub(crate) const PATTERNS: &[Pattern] = &[Pattern::Uniform, Pattern::Varied, Pattern::Cancel];

impl Pattern {
    pub fn label(self) -> &'static str {
        match self {
            Pattern::Uniform => "uniform",
            Pattern::Varied => "varied-magnitude",
            Pattern::Cancel => "cancellation",
        }
    }
}

/// Host copies of A [m, k], B [k, n] and C [m, n], already rounded to their
/// storage types, so the device sees exactly these values.
pub(crate) struct Inputs {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Vec<f32>,
}

#[allow(non_snake_case)]
pub(crate) fn make_inputs(kind: Kind, pattern: Pattern, (M, N, K): (usize, usize, usize), seed: u64) -> Inputs {
    let mut rng = SmallRng::seed_from_u64(seed);
    // f16 accumulation overflows past 65504, so its magnitudes stay small.
    let exp: i32 = if kind.acc() == Elem::F16 { 2 } else { 6 };
    let draw = |rng: &mut SmallRng| -> f32 {
        let x = rng.gen_range(-1.0f32..=1.0);
        match pattern {
            Pattern::Varied => x * 2f32.powi(rng.gen_range(-exp..=exp)),
            _ => x,
        }
    };
    let mut a: Vec<f32> = (0..M * K).map(|_| draw(&mut rng)).collect();
    let mut b: Vec<f32> = (0..K * N).map(|_| draw(&mut rng)).collect();
    if pattern == Pattern::Cancel {
        let h = K / 2;
        for i in 0..M {
            for k in 0..h {
                a[i * K + h + k] = a[i * K + k];
            }
        }
        for k in 0..h {
            for j in 0..N {
                // A nudge f16 can still hold next to a value near 1.
                let nudge = rng.gen_range(-1.0f32..=1.0) * 2f32.powi(-8);
                b[(h + k) * N + j] = -b[k * N + j] + nudge;
            }
        }
    }
    let c: Vec<f32> = (0..M * N).map(|_| draw(&mut rng)).collect();
    let (ab, ce) = (kind.ab(), kind.c());
    Inputs {
        a: a.into_iter().map(|x| ab.round(x)).collect(),
        b: b.into_iter().map(|x| ab.round(x)).collect(),
        c: c.into_iter().map(|x| ce.round(x)).collect(),
    }
}

/// What one check found, over `checked` output elements.
#[derive(Clone, Debug, Default)]
pub(crate) struct Errors {
    pub checked: usize,
    pub full: bool,
    pub max_abs: f64,
    /// Over elements with |want| >= 1e-3 of their condition.
    pub max_rel: f64,
    pub max_norm: f64,
    pub rms_norm: f64,
    /// Elements whose result is not finite where the reference is.
    pub nonfinite: usize,
    pub worst: Option<(usize, usize, f64, f64)>,
    pub tol_norm: f64,
}

impl Errors {
    pub fn pass(&self) -> bool {
        self.nonfinite == 0 && self.max_norm <= self.tol_norm
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "checked": self.checked,
            "full": self.full,
            "max_abs": self.max_abs,
            "max_rel_wellcond": self.max_rel,
            "max_norm": self.max_norm,
            "rms_norm": self.rms_norm,
            "nonfinite": self.nonfinite,
            "worst": self.worst.map(|(i, j, got, want)| serde_json::json!({"i": i, "j": j, "got": got, "want": want})),
            "tol_norm": self.tol_norm,
            "pass": self.pass(),
        })
    }

    pub fn summary(&self) -> String {
        format!(
            "{} {} elements: max norm err {:.2e} (tol {:.2e}), rms {:.2e}, max abs {:.2e}, max rel {:.2e}, nonfinite {}",
            if self.full { "all" } else { "sampled" },
            self.checked,
            self.max_norm,
            self.tol_norm,
            self.rms_norm,
            self.max_abs,
            self.max_rel,
            self.nonfinite
        )
    }
}

/// The normalized error a correct kernel of this kind stays under.
///
/// With f32 accumulation the bound is the deterministic gamma_K = K*u, or
/// 16 sqrt(K) u, the random-walk size with a wide margin, whichever is
/// smaller; tensor cores get twice that, since Turing's f32 MMA accumulate
/// does not round to nearest. f16 accumulation takes 8 sqrt(K) u16: its
/// deterministic bound is past 1 at any useful K. Output rounding adds 2u
/// of the storage type.
pub(crate) fn tolerance(kind: Kind, k: usize) -> f64 {
    let k = k as f64;
    let u = kind.acc().unit_roundoff();
    let acc = if kind.acc() == Elem::F16 {
        8.0 * k.sqrt() * u
    } else {
        let tc = if kind.tensor_cores() { 2.0 } else { 1.0 };
        tc * (2.0 * k).min(16.0 * k.sqrt()) * u
    };
    acc + 2.0 * kind.c().unit_roundoff()
}

/// The output elements a check reads: all of them when the reference is
/// cheap enough, otherwise corners, rows and columns either side of every
/// 16-element boundary on the edges, and a random sample.
#[allow(non_snake_case)]
pub(crate) fn positions(M: usize, N: usize, K: usize) -> (Vec<(usize, usize)>, bool) {
    const FULL_MACS: usize = 1 << 28;
    if M * N * K <= FULL_MACS {
        return ((0..M).flat_map(|i| (0..N).map(move |j| (i, j))).collect(), true);
    }
    let mut at = vec![(0, 0), (0, N - 1), (M - 1, 0), (M - 1, N - 1)];
    for i in (0..M).step_by(16) {
        for d in [0, 15] {
            let r = (i + d).min(M - 1);
            at.push((r, 0));
            at.push((r, N - 1));
        }
    }
    for j in (0..N).step_by(16) {
        for d in [0, 15] {
            let c = (j + d).min(N - 1);
            at.push((0, c));
            at.push((M - 1, c));
        }
    }
    let mut rng = SmallRng::seed_from_u64(0x243F_6A88_85A3_08D3);
    at.extend((0..2048).map(|_| (rng.gen_range(0..M), rng.gen_range(0..N))));
    at.sort_unstable();
    at.dedup();
    (at, false)
}

/// Checks `got` against alpha * A*B + beta * C over `at`, with A and B
/// rounded to `mul` first, as the kernel's multiply sees them.
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)] // mirrors the BLAS gemm signature
pub(crate) fn check(
    got: &[f32],
    inp: &Inputs,
    mul: Elem,
    (alpha, beta): (f32, f32),
    (_, N, K): (usize, usize, usize),
    at: &[(usize, usize)],
    full: bool,
    tol_norm: f64,
) -> Errors {
    let (alpha, beta) = (f64::from(alpha), f64::from(beta));
    // The columns of B the check reads, transposed once, so each dot reads
    // two contiguous rows. Transposing all of B for a sampled check costs
    // more host time than the probe it gates.
    let mut cols: Vec<usize> = at.iter().map(|&(_, j)| j).collect();
    cols.sort_unstable();
    cols.dedup();
    let mut slot = vec![usize::MAX; N];
    let mut bt = vec![0f64; cols.len() * K];
    for (s, &j) in cols.iter().enumerate() {
        slot[j] = s;
        for k in 0..K {
            bt[s * K + k] = f64::from(mul.round(inp.b[k * N + j]));
        }
    }
    let mut row = vec![0f64; K];
    let mut row_of = usize::MAX;

    let mut e = Errors {
        checked: at.len(),
        full,
        tol_norm,
        ..Default::default()
    };
    let mut sum_sq = 0f64;
    for &(i, j) in at {
        if row_of != i {
            for (r, &x) in row.iter_mut().zip(&inp.a[i * K..(i + 1) * K]) {
                *r = f64::from(mul.round(x));
            }
            row_of = i;
        }
        let col = &bt[slot[j] * K..(slot[j] + 1) * K];
        let (mut dot, mut mag) = (0f64, 0f64);
        for (x, y) in row.iter().zip(col) {
            dot += x * y;
            mag += (x * y).abs();
        }
        let c = f64::from(inp.c[i * N + j]);
        let want = alpha * dot + beta * c;
        let cond = alpha.abs() * mag + (beta * c).abs();
        let g = f64::from(got[i * N + j]);
        if !g.is_finite() {
            e.nonfinite += 1;
            e.worst.get_or_insert((i, j, g, want));
            continue;
        }
        let abs = (g - want).abs();
        let norm = abs / cond.max(f64::MIN_POSITIVE);
        e.max_abs = e.max_abs.max(abs);
        if want.abs() >= 1e-3 * cond {
            e.max_rel = e.max_rel.max(abs / want.abs());
        }
        if norm > e.max_norm || e.worst.is_none() {
            e.worst = Some((i, j, g, want));
        }
        e.max_norm = e.max_norm.max(norm);
        sum_sq += norm * norm;
    }
    e.rms_norm = (sum_sq / at.len().max(1) as f64).sqrt();
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_product_passes_and_a_dropped_k_slice_fails() {
        let (m, n, k) = (8, 8, 64);
        let inp = make_inputs(Kind::F32, Pattern::Uniform, (m, n, k), 1);
        let reference = |kmax: usize| -> Vec<f32> {
            let mut c = vec![0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let dot: f64 = (0..kmax).map(|q| f64::from(inp.a[i * k + q]) * f64::from(inp.b[q * n + j])).sum();
                    c[i * n + j] = (1.5 * dot + 2.5 * f64::from(inp.c[i * n + j])) as f32;
                }
            }
            c
        };
        let (at, full) = positions(m, n, k);
        let tol = tolerance(Kind::F32, k);
        assert!(check(&reference(k), &inp, Elem::F32, (1.5, 2.5), (m, n, k), &at, full, tol).pass());
        assert!(!check(&reference(k - 8), &inp, Elem::F32, (1.5, 2.5), (m, n, k), &at, full, tol).pass());
    }

    #[test]
    fn nonfinite_output_fails() {
        let inp = make_inputs(Kind::F32, Pattern::Cancel, (4, 4, 8), 2);
        let got = vec![f32::NAN; 16];
        let (at, full) = positions(4, 4, 8);
        let e = check(&got, &inp, Elem::F32, (1.0, 0.0), (4, 4, 8), &at, full, 1.0);
        assert_eq!(e.nonfinite, 16);
        assert!(!e.pass());
    }
}
