// Pointwise, strided copy and swiglu kernel sources.

use phobos_kernels::launch::STATIC_SHARED_LIMIT;

/// Elements per block for the pointwise kernels.
pub(crate) const ELEM_TILE: usize = 128;

/// Pointwise elements a program takes when there are enough of them. A decode
/// step wants [`ELEM_TILE`] and any blocks at all; a prompt's 1.8-million-element
/// SwiGLU halves at this one. Shared memory is the ceiling, `swiglu` holding
/// about six tiles at once.
pub(crate) const ELEM_TILE_WIDE: usize = 1024;

/// Below this the narrow tile still wins: enough wide tiles to fill the card
/// several times over.
pub(crate) const WIDE_FLOOR: usize = 192 * ELEM_TILE_WIDE;

/// Pointwise kernels over a flat buffer viewed as one row, so the tail is a
/// masked tile.
pub(crate) const POINTWISE_SRC: &str = "\
@launch(256)
@autotune(TILE in [1024])
kernel add_into(A: tensor<f32>[M, N], B: tensor<f32>[M, N]) {
  let p = program_id(0)
  var a = A[0 :+ 1, p * TILE :+ TILE]
  var b = B[0 :+ 1, p * TILE :+ TILE]
  A[0 :+ 1, p * TILE :+ TILE] = a + b
}

@launch(256)
@autotune(TILE in [1024])
kernel swiglu(G: tensor<f32>[M, N], U: tensor<f32>[M, N], O: tensor<f32>[M, N]) {
  let p = program_id(0)
  var g = G[0 :+ 1, p * TILE :+ TILE]
  var u = U[0 :+ 1, p * TILE :+ TILE]
  var s = g / (1.0 + exp(-g))
  O[0 :+ 1, p * TILE :+ TILE] = s * u
}

@launch(256)
@autotune(TILE in [1024])
kernel copy(S: tensor<f32>[M, N], D: tensor<f32>[M, N]) {
  let p = program_id(0)
  D[0 :+ 1, p * TILE :+ TILE] = S[0 :+ 1, p * TILE :+ TILE]
}

@launch(256)
@autotune(TILE in [1024])
kernel gate_into(X: tensor<f32>[M, N], G: tensor<f32>[M, N]) {
  let p = program_id(0)
  var x = X[0 :+ 1, p * TILE :+ TILE]
  var g = G[0 :+ 1, p * TILE :+ TILE]
  X[0 :+ 1, p * TILE :+ TILE] = x / (1.0 + exp(-g))
}
";

/// What a strided copy moves between. The caches are f16 and everything else is
/// f32, so a copy into or out of one converts; see [`HBuf`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Strided {
    /// f32 either side, the shape every fused projection's split takes.
    Dense,
    /// Narrowing: keys and values landing in a cache.
    Store,
    /// Widening: a cached head gathered for the prompt's matmuls.
    Load,
}

impl Strided {
    pub(crate) fn kernel(self) -> &'static str {
        match self {
            Strided::Dense => "copy_2d",
            Strided::Store => "store_2d",
            Strided::Load => "load_2d",
        }
    }

    /// Source type, destination type, and the conversion between them, which
    /// the language wants written out rather than implied by the store.
    pub(crate) fn types(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Strided::Dense => ("f32", "f32", ""),
            Strided::Store => ("f32", "f16", "f16"),
            Strided::Load => ("f16", "f32", "f32"),
        }
    }
}

/// A strided block copy. The width is baked in rather than tiled, so a row is
/// one tile with no remainder to mask, and the pitches are the declared extents
/// with the starting corner a pointer offset.
pub(crate) fn copy_2d_src(kind: Strided, width: usize, aligned: bool) -> String {
    // Without the promise every element carries a bounds check and the copy does
    // not vectorize. It says both pitches are whole multiples of the width,
    // which a gather of one head satisfies and a slice of a fused projection
    // does not, so it compiles both ways.
    let claim = if aligned {
        "@aligned(SW = W, DW = W)"
    } else {
        ""
    };
    let (from, to, convert) = kind.types();
    let read = "S[r :+ 1, 0 :+ W]";
    let value = if convert.is_empty() {
        read.to_string()
    } else {
        format!("{convert}({read})")
    };

    format!(
        "@launch(256)
@autotune(W in [{width}])
{claim}
kernel {name}(S: tensor<{from}>[R, SW], D: tensor<{to}>[R, DW]) {{
  let r = program_id(0)
  D[r :+ 1, 0 :+ W] = {value}
}}
",
        name = kind.kernel(),
    )
}

/// Two independent [`Strided::Store`] copies in one launch: attention's value
/// and key, both narrowing f32 into the f16 cache at the same row. Both windows
/// share `width` (the two writes always do -- one query group's worth of key
/// or value), so one autotune constant covers both.
pub(crate) fn store_2d_pair_src(width: usize, aligned: bool) -> String {
    let claim = if aligned {
        "@aligned(SW0 = W, DW0 = W, SW1 = W, DW1 = W)"
    } else {
        ""
    };
    format!(
        "@launch(256)
@autotune(W in [{width}])
{claim}
kernel store_2d_pair(S0: tensor<f32>[R, SW0], S1: tensor<f32>[R, SW1], D0: tensor<f16>[R, DW0], D1: tensor<f16>[R, DW1]) {{
  let r = program_id(0)
  D0[r :+ 1, 0 :+ W] = f16(S0[r :+ 1, 0 :+ W])
  D1[r :+ 1, 0 :+ W] = f16(S1[r :+ 1, 0 :+ W])
}}
"
    )
}

/// A SwiGLU whose two operands are planes of a wider buffer. Shaped like
/// [`copy_2d_src`], but taking a column tile rather than a whole row since it
/// holds three at once; see [`swiglu_2d_tile`]. The promise is that every pitch
/// is a whole number of tiles, which a fused gate-and-up projection satisfies.
pub(crate) fn swiglu_2d_src(tile: usize) -> String {
    format!(
        "@launch(256)
@autotune(T in [{tile}])
@aligned(GW = T, UW = T, OW = T)
kernel swiglu_2d(G: tensor<f32>[R, GW], U: tensor<f32>[R, UW], O: tensor<f32>[R, OW]) {{
  let r = program_id(0)
  let c = program_id(1) * T
  var g = G[r :+ 1, c :+ T]
  O[r :+ 1, c :+ T] = (g / (1.0 + exp(-g))) * U[r :+ 1, c :+ T]
}}
"
    )
}

/// Columns one program of the strided SwiGLU covers. It holds three tiles of
/// this width in static shared memory, which caps at 48 KB whatever the card
/// has, so a row that does not fit splits into the widest tile that divides it
/// and does.
pub(crate) fn swiglu_2d_tile(width: usize) -> usize {
    const OPERANDS: usize = 3;
    let fits = STATIC_SHARED_LIMIT / (OPERANDS * size_of::<f32>());
    (1..=width.min(fits))
        .rev()
        .find(|t| width.is_multiple_of(*t))
        .unwrap_or(width)
}
