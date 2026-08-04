/// A node in the cluster.
/// One node = one GPU.
pub type NodeId = u16;

/// Globally Unique Deterministic Supertile Identity (GUDSI)
///
/// id = (tensor:u12 << 52) | (version:u16  << 36)| (linear supertile coord:u36)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct TileId(pub u64);

impl TileId {
    pub fn new(tensor: u16, version: u16, coord: u64) -> TileId {
        debug_assert!(tensor < (1 << 12), "tensor index exceeds 12 bits");
        debug_assert!(coord < (1 << 36), "supertile coord exceeds 36 bits");
        TileId(((tensor as u64) << 52) | ((version as u64) << 36) | coord)
    }

    pub fn tensor(self) -> u16 {
        (self.0 >> 52) as u16
    }

    pub fn version(self) -> u16 {
        ((self.0 >> 36) & 0xFFFF) as u16
    }

    /// Linear supertile coordinate (row-major).
    pub fn coord(self) -> u64 {
        self.0 & ((1 << 36) - 1)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DataType {
    F16,
    BF16,
    F32,
    F64,
    I8,
    I32,
    I64,
    Bool,
}

impl DataType {
    pub fn bytes(self) -> usize {
        match self {
            DataType::I8 | DataType::Bool => 1,
            DataType::F16 | DataType::BF16 => 2,
            DataType::F32 | DataType::I32 => 4,
            DataType::F64 | DataType::I64 => 8,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ScalarValue {
    F32(f32),
    F64(f64),
    I32(i32),
    I64(i64),
    Bool(bool),
}

impl ScalarValue {
    pub fn data_type(self) -> DataType {
        match self {
            ScalarValue::F32(_) => DataType::F32,
            ScalarValue::F64(_) => DataType::F64,
            ScalarValue::I32(_) => DataType::I32,
            ScalarValue::I64(_) => DataType::I64,
            ScalarValue::Bool(_) => DataType::Bool,
        }
    }

    /// little-endian
    pub fn to_bits(self) -> u64 {
        match self {
            ScalarValue::F32(x) => x.to_bits() as u64,
            ScalarValue::F64(x) => x.to_bits(),
            ScalarValue::I32(x) => x as u32 as u64,
            ScalarValue::I64(x) => x as u64,
            ScalarValue::Bool(b) => b as u64,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AccessMode {
    Read,
    Write,
    RMW, // consumes one ssa node and produces next
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TileState {
    Unmaterialized,
    Loading,
    Resident,
    InFlight,
    Freed,
}

#[derive(Clone, Debug)]
pub struct TileMeta {
    pub id: TileId,
    pub shape: Vec<u64>,
    pub data_type: DataType,
    pub state: TileState,
    pub location: NodeId,         // valid while Resident / InFlight
    pub remaining_consumers: u32, // can't free while > 0
}

#[cfg(test)]
mod tests {
    use super::TileId;

    #[test]
    fn tile_id_round_trips() {
        let id = TileId::new(7, 3, 123_456);
        assert_eq!(id.tensor(), 7);
        assert_eq!(id.version(), 3);
        assert_eq!(id.coord(), 123_456);

        let max = TileId::new((1 << 12) - 1, u16::MAX, (1 << 36) - 1);
        assert_eq!(max.tensor(), (1 << 12) - 1);
        assert_eq!(max.version(), u16::MAX);
        assert_eq!(max.coord(), (1 << 36) - 1);
    }
}
