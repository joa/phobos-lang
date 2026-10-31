use std::fmt;

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
    /// Reserve a local buffer and register the tile in the node-local table.
    Alloc {
        tile: TileId,
        shape: Vec<u64>,
        data_type: DataType,
    },
    /// Read a supertile from durable storage into its allocated buffer.
    Load { tile: TileId, src: StorageRef },
    /// Pull a tile from node `from` into its allocated buffer.
    /// The server waits until the tile is resident there.
    Fetch { tile: TileId, from: NodeId },
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
    /// has served the tile `expected_serves` times. Zero means no remote readers.
    Free { tile: TileId, expected_serves: u32 },
}

impl Op {
    /// The mnemonics, in the order of the variants.
    pub const NAMES: [&'static str; 6] = ["ALLOC", "LOAD", "FETCH", "COMPUTE", "STORE", "FREE"];

    /// The variant's position, which is its mnemonic's in [`Op::NAMES`].
    pub fn index(&self) -> usize {
        match self {
            Op::Alloc { .. } => 0,
            Op::Load { .. } => 1,
            Op::Fetch { .. } => 2,
            Op::Compute { .. } => 3,
            Op::Store { .. } => 4,
            Op::Free { .. } => 5,
        }
    }

    pub fn name(&self) -> &'static str {
        Op::NAMES[self.index()]
    }
}

/// The mnemonic and the operands, as a node logs what it issues.
impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())?;
        match self {
            Op::Alloc { tile, shape, .. } => write!(f, " tile={:#x} shape={shape:?}", tile.0),
            Op::Load { tile, .. } | Op::Store { tile, .. } => write!(f, " tile={:#x}", tile.0),
            Op::Fetch { tile, from } => write!(f, " tile={:#x} from=node{from}", tile.0),
            Op::Compute {
                kernel,
                args,
                scalars,
                grid,
                cta,
            } => {
                let tiles: Vec<String> = args.iter().map(|(t, m)| format!("{:#x}:{m:?}", t.0)).collect();
                write!(f, " k{kernel} grid={grid:?} cta={cta:?} tiles=[{}]", tiles.join(" "))?;
                if !scalars.is_empty() {
                    let scalars: Vec<String> = scalars.iter().map(|s| format!("@{}={:?}", s.pos, s.value)).collect();
                    write!(f, " scalars=[{}]", scalars.join(" "))?;
                }
                Ok(())
            }
            Op::Free {
                tile,
                expected_serves,
            } => write!(f, " tile={:#x} serves={expected_serves}", tile.0),
        }
    }
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

#[cfg(test)]
mod tests {
    use super::{Op, ScalarArg};
    use crate::tile::{AccessMode, ScalarValue, TileId};

    #[test]
    fn an_op_prints_its_mnemonic_then_its_operands() {
        let compute = Op::Compute {
            kernel: 1,
            args: vec![(TileId(0x10), AccessMode::Read), (TileId(0x20), AccessMode::RMW)],
            scalars: vec![ScalarArg {
                pos: 4,
                value: ScalarValue::F32(0.125),
            }],
            grid: (128, 128, 1),
            cta: (256, 1, 1),
        };
        assert_eq!(compute.name(), "COMPUTE");
        assert_eq!(
            compute.to_string(),
            "COMPUTE k1 grid=(128, 128, 1) cta=(256, 1, 1) tiles=[0x10:Read 0x20:RMW] scalars=[@4=F32(0.125)]"
        );

        let free = Op::Free {
            tile: TileId(0x10),
            expected_serves: 2,
        };
        assert_eq!(free.to_string(), "FREE tile=0x10 serves=2");
        assert_eq!(Op::NAMES[free.index()], free.name());
    }
}
