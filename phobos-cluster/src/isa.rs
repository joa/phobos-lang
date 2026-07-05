use crate::tile::{AccessMode, DataType, NodeId, ScalarValue, TileId};

pub type InstrId = u64;

pub type KernelId = u32;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ScalarArg {
    pub pos: u32,
    pub value: ScalarValue,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Region {
    pub offset: Vec<u64>,
    pub shape: Vec<u64>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum StorageRef {
    Tensor { tensor: u32, region: Region },
}

#[derive(Clone, Debug)]
pub enum Op {
    /// Reserve a local buffer and register tile in the node-local table.
    Alloc {
        tile: TileId,
        shape: Vec<u64>,
        data_type: DataType,
    },
    /// Read a supertile from durable storage into the ALLOCd buffer.
    Load { tile: TileId, src: StorageRef },
    /// Pull a tile from another node (the from field) into the ALLOCd buffer.
    /// Waits server-side until the tile is resident there.
    Fetch { tile: TileId, from: NodeId },
    /// Launch kernel.
    Compute {
        kernel: KernelId,
        args: Vec<(TileId, AccessMode)>,
        scalars: Vec<ScalarArg>, // in param order
        grid: (u32, u32, u32),
        cta: (u32, u32, u32),
    },
    /// Write a supertile to durable storage.
    Store { tile: TileId, dst: StorageRef },
    /// Release the buffer once local dependencies are met and the tile server
    /// has served the tile expected_serves times (0 = no remote readers).
    Free { tile: TileId, expected_serves: u32 },
}

#[derive(Clone, Debug)]
pub struct Instr {
    pub iid: InstrId,
    pub deps: Vec<InstrId>,
    pub op: Op,
}

#[derive(Clone, Debug)]
pub struct Segment {
    pub id: u64,
    pub instructions: Vec<Instr>,
}
