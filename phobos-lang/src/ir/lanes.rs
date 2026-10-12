// The vocabulary of the per-thread ops: which index a thread reads of
// itself, and what a `dp4a` side does to its bytes.

/// What one side of a [`OpKind::Dot4`](super::OpKind::Dot4) does to its four bytes before the
/// product. Every value stays under 128, so a byte reads the same signed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bytes {
    /// As loaded.
    Raw,
    /// `(byte >> shift) & mask`, done on the packed word.
    Masked { shift: u8, mask: u8 },
    /// Four ones, so the product sums the other side.
    Ones,
}

impl Bytes {
    /// The `i8` operands this side takes.
    pub fn operands(self) -> usize {
        if self == Bytes::Ones { 0 } else { 4 }
    }
}

/// Which index [`OpKind::Thread`](super::OpKind::Thread) reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadIndex {
    /// `ctx.thread`, in the CTA.
    Thread,
    /// `ctx.warp`, the warp's index in the CTA.
    Warp,
    /// `ctx.lane`, in the warp.
    Lane,
}

impl ThreadIndex {
    pub fn name(self) -> &'static str {
        match self {
            ThreadIndex::Thread => "thread_id",
            ThreadIndex::Warp => "warp_id",
            ThreadIndex::Lane => "lane_id",
        }
    }

    /// How far the index differs between threads.
    pub fn kind(self) -> super::uniform::Kind {
        match self {
            ThreadIndex::Warp => super::uniform::Kind::Warp,
            ThreadIndex::Thread | ThreadIndex::Lane => super::uniform::Kind::Varying,
        }
    }
}
