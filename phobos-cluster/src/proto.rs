use anyhow::{Context, Result, bail};

use crate::isa;
use crate::tile::{self, ScalarValue as Sv, TileId};

tonic::include_proto!("phobos.v1");

pub fn data_type_to_i32(d: tile::DataType) -> i32 {
    use tile::DataType::*;
    match d {
        F16 => 1,
        BF16 => 2,
        F32 => 3,
        F64 => 4,
        I8 => 5,
        I32 => 6,
        I64 => 7,
        Bool => 8,
    }
}

pub fn data_type_from_i32(v: i32) -> Result<tile::DataType> {
    use tile::DataType::*;
    Ok(match v {
        1 => F16,
        2 => BF16,
        3 => F32,
        4 => F64,
        5 => I8,
        6 => I32,
        7 => I64,
        8 => Bool,
        _ => bail!("unknown DataType {v}"),
    })
}

pub fn am_to_i32(m: tile::AccessMode) -> i32 {
    use tile::AccessMode::*;
    match m {
        Read => 1,
        Write => 2,
        RMW => 3,
    }
}

pub fn am_from_i32(v: i32) -> Result<tile::AccessMode> {
    use tile::AccessMode::*;
    Ok(match v {
        1 => Read,
        2 => Write,
        3 => RMW,
        _ => bail!("unknown AccessMode {v}"),
    })
}

pub fn scalar_value_to_proto(v: Sv) -> ScalarValue {
    use scalar_value::Value;
    let value = match v {
        Sv::F32(x) => Value::F32(x),
        Sv::F64(x) => Value::F64(x),
        Sv::I32(x) => Value::I32(x),
        Sv::I64(x) => Value::I64(x),
        Sv::Bool(b) => Value::Boolean(b),
    };
    ScalarValue { value: Some(value) }
}

pub fn scalar_value_from_proto(v: &ScalarValue) -> Result<Sv> {
    use scalar_value::Value;
    Ok(
        match v.value.as_ref().context("ScalarValue.value missing")? {
            Value::F32(x) => Sv::F32(*x),
            Value::F64(x) => Sv::F64(*x),
            Value::I32(x) => Sv::I32(*x),
            Value::I64(x) => Sv::I64(*x),
            Value::Boolean(b) => Sv::Bool(*b),
        },
    )
}

fn scalar_arg_to_proto(s: &isa::ScalarArg) -> ScalarArgument {
    ScalarArgument {
        position: s.pos,
        value: Some(scalar_value_to_proto(s.value)),
    }
}

fn scalar_arg_from_proto(s: &ScalarArgument) -> Result<isa::ScalarArg> {
    Ok(isa::ScalarArg {
        pos: s.position,
        value: scalar_value_from_proto(s.value.as_ref().context("ScalarArgument.value missing")?)?,
    })
}

// ISA <-> .proto

fn dim3(g: (u32, u32, u32)) -> Dim3 {
    Dim3 {
        x: g.0,
        y: g.1,
        z: g.2,
    }
}
fn dim3_from(d: &Dim3) -> (u32, u32, u32) {
    (d.x, d.y, d.z)
}

fn region_to_proto(r: &isa::Region) -> Region {
    Region {
        offset: r.offset.clone(),
        shape: r.shape.clone(),
    }
}

fn region_from_proto(r: &Region) -> isa::Region {
    isa::Region {
        offset: r.offset.clone(),
        shape: r.shape.clone(),
    }
}

fn storage_ref_to_proto(s: &isa::StorageRef) -> StorageReference {
    let isa::StorageRef::Tensor { tensor, region } = s;
    StorageReference {
        tensor: *tensor,
        region: Some(region_to_proto(region)),
    }
}

fn storage_ref_from_proto(s: &StorageReference) -> Result<isa::StorageRef> {
    Ok(isa::StorageRef::Tensor {
        tensor: s.tensor,
        region: region_from_proto(
            s.region
                .as_ref()
                .context("StorageReference.region missing")?,
        ),
    })
}

fn op_to_proto(op: &isa::Op) -> Operation {
    use isa::Op as I;
    let kind = match op {
        I::Alloc {
            tile,
            shape,
            data_type,
        } => operation::Kind::Allocate(Allocate {
            tile: tile.0,
            shape: shape.clone(),
            data_type: data_type_to_i32(*data_type),
        }),
        I::Load { tile, src } => operation::Kind::Load(Load {
            tile: tile.0,
            source: Some(storage_ref_to_proto(src)),
        }),
        I::Fetch { tile, from } => operation::Kind::Fetch(Fetch {
            tile: tile.0,
            from: *from as u32,
        }),
        I::Compute {
            kernel,
            args,
            scalars,
            grid,
            cta,
        } => operation::Kind::Compute(Compute {
            kernel: *kernel,
            arguments: args
                .iter()
                .map(|(t, m)| ComputeArgument {
                    tile: t.0,
                    mode: am_to_i32(*m),
                })
                .collect(),
            scalars: scalars.iter().map(scalar_arg_to_proto).collect(),
            grid: Some(dim3(*grid)),
            cta: Some(dim3(*cta)),
        }),
        I::Store { tile, dst } => operation::Kind::Store(Store {
            tile: tile.0,
            destination: Some(storage_ref_to_proto(dst)),
        }),
        I::Free {
            tile,
            expected_serves,
        } => operation::Kind::Free(Free {
            tile: tile.0,
            expected_serves: *expected_serves,
        }),
    };
    Operation { kind: Some(kind) }
}

fn op_from_proto(op: &Operation) -> Result<isa::Op> {
    let kind = op.kind.as_ref().context("Operation.kind missing")?;
    Ok(match kind {
        operation::Kind::Allocate(a) => isa::Op::Alloc {
            tile: TileId(a.tile),
            shape: a.shape.clone(),
            data_type: data_type_from_i32(a.data_type)?,
        },
        operation::Kind::Load(l) => isa::Op::Load {
            tile: TileId(l.tile),
            src: storage_ref_from_proto(l.source.as_ref().context("Load.source missing")?)?,
        },
        operation::Kind::Fetch(f) => isa::Op::Fetch {
            tile: TileId(f.tile),
            from: f.from as tile::NodeId,
        },
        operation::Kind::Compute(c) => isa::Op::Compute {
            kernel: c.kernel,
            args: c
                .arguments
                .iter()
                .map(|a| Ok((TileId(a.tile), am_from_i32(a.mode)?)))
                .collect::<Result<_>>()?,
            scalars: c
                .scalars
                .iter()
                .map(scalar_arg_from_proto)
                .collect::<Result<_>>()?,
            grid: dim3_from(c.grid.as_ref().context("Compute.grid missing")?),
            cta: dim3_from(c.cta.as_ref().context("Compute.cta missing")?),
        },
        operation::Kind::Store(s) => isa::Op::Store {
            tile: TileId(s.tile),
            dst: storage_ref_from_proto(
                s.destination
                    .as_ref()
                    .context("Store.destination missing")?,
            )?,
        },
        operation::Kind::Free(f) => isa::Op::Free {
            tile: TileId(f.tile),
            expected_serves: f.expected_serves,
        },
    })
}

pub fn instr_to_proto(i: &isa::Instr) -> Instruction {
    Instruction {
        iid: i.iid,
        dependencies: i.deps.clone(),
        operation: Some(op_to_proto(&i.op)),
    }
}
pub fn instr_from_proto(i: &Instruction) -> Result<isa::Instr> {
    Ok(isa::Instr {
        iid: i.iid,
        deps: i.dependencies.clone(),
        op: op_from_proto(
            i.operation
                .as_ref()
                .context("Instruction.operation missing")?,
        )?,
    })
}

pub fn segment_to_proto(s: &isa::Segment) -> Segment {
    Segment {
        id: s.id,
        instructions: s.instructions.iter().map(instr_to_proto).collect(),
    }
}
pub fn segment_from_proto(s: &Segment) -> Result<isa::Segment> {
    Ok(isa::Segment {
        id: s.id,
        instructions: s
            .instructions
            .iter()
            .map(instr_from_proto)
            .collect::<Result<_>>()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_round_trips() {
        let ops = [
            isa::Op::Alloc {
                tile: TileId::new(2, 0, 5),
                shape: vec![64, 32],
                data_type: tile::DataType::F32,
            },
            isa::Op::Fetch {
                tile: TileId::new(0, 0, 3),
                from: 7,
            },
            isa::Op::Compute {
                kernel: 1,
                args: vec![
                    (TileId::new(0, 0, 0), tile::AccessMode::Read),
                    (TileId::new(2, 0, 0), tile::AccessMode::RMW),
                ],
                scalars: vec![isa::ScalarArg {
                    pos: 4,
                    value: Sv::F32(0.125),
                }],
                grid: (128, 128, 1),
                cta: (256, 1, 1),
            },
            isa::Op::Free {
                tile: TileId::new(0, 0, 0),
                expected_serves: 3,
            },
        ];
        for op in ops {
            let back = op_from_proto(&op_to_proto(&op)).unwrap();
            assert_eq!(format!("{op:?}"), format!("{back:?}"));
        }
    }
}
