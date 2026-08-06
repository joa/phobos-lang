use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use prost::Message;

use crate::ir::{
    Attribute, DataType, Dim, Graph, Model, Node, Shape, Tensor, TensorData, ValueInfo,
};
use crate::proto;

/// Decode a serialized `ModelProto` into a [`Model`]. A faithful translation
/// only: unknown ops and attribute kinds are kept or dropped rather than
/// rejected, so a partial model still loads for inspection.
pub fn load_model(bytes: &[u8]) -> Result<Model> {
    let model = proto::ModelProto::decode(bytes).context("decoding ModelProto")?;
    model_from_proto(model)
}

fn model_from_proto(m: proto::ModelProto) -> Result<Model> {
    let graph = m.graph.context("model has no graph")?;
    let opset = m
        .opset_import
        .into_iter()
        .map(|o| (o.domain.unwrap_or_default(), o.version.unwrap_or_default()))
        .collect();

    Ok(Model {
        graph: graph_from_proto(graph)?,
        ir_version: m.ir_version.unwrap_or_default(),
        producer_name: m.producer_name.unwrap_or_default(),
        opset,
    })
}

fn graph_from_proto(g: proto::GraphProto) -> Result<Graph> {
    let inputs: Vec<ValueInfo> = g.input.into_iter().map(value_info_from_proto).collect();
    let outputs: Vec<ValueInfo> = g.output.into_iter().map(value_info_from_proto).collect();
    let value_info: Vec<ValueInfo> = g
        .value_info
        .into_iter()
        .map(value_info_from_proto)
        .collect();

    let mut values = HashMap::new();
    for vi in inputs.iter().chain(&outputs).chain(&value_info) {
        values.insert(vi.name.clone(), vi.clone());
    }

    let mut initializers = HashMap::new();
    for t in g.initializer {
        let name = t.name.clone().unwrap_or_default();
        if name.is_empty() {
            bail!("initializer without a name");
        }
        initializers.insert(name, Arc::new(tensor_from_proto(t)?));
    }

    let nodes = g
        .node
        .into_iter()
        .map(node_from_proto)
        .collect::<Result<Vec<_>>>()?;

    Ok(Graph {
        name: g.name.unwrap_or_default(),
        inputs,
        outputs,
        nodes,
        initializers,
        values,
    })
}

fn node_from_proto(n: proto::NodeProto) -> Result<Node> {
    let mut attrs = HashMap::new();
    for a in n.attribute {
        let name = a.name.clone().unwrap_or_default();
        if let Some(attr) = attribute_from_proto(a)? {
            attrs.insert(name, attr);
        }
    }
    Ok(Node {
        name: n.name.unwrap_or_default(),
        op_type: n.op_type.unwrap_or_default(),
        inputs: n.input,
        outputs: n.output,
        attrs,
    })
}

fn value_info_from_proto(vi: proto::ValueInfoProto) -> ValueInfo {
    let (data_type, shape) = match vi.r#type.and_then(|t| t.value) {
        Some(proto::type_proto::Value::TensorType(t)) => {
            let data_type = t.elem_type.map(data_type_from_code);
            (data_type, shape_from_proto(t.shape))
        }
        // Sequence / map / optional / sparse types are not modeled yet.
        _ => (None, Shape::default()),
    };
    ValueInfo {
        name: vi.name.unwrap_or_default(),
        data_type,
        shape,
    }
}

fn shape_from_proto(shape: Option<proto::TensorShapeProto>) -> Shape {
    let Some(shape) = shape else {
        return Shape(None);
    };
    let dims = shape
        .dim
        .into_iter()
        .map(|d| match d.value {
            Some(proto::tensor_shape_proto::dimension::Value::DimValue(v)) => Dim::Fixed(v),
            Some(proto::tensor_shape_proto::dimension::Value::DimParam(s)) => Dim::Symbol(s),
            None => Dim::Unknown,
        })
        .collect();
    Shape(Some(dims))
}

fn data_type_from_code(code: i32) -> DataType {
    use proto::tensor_proto::DataType as Dt;
    match code {
        c if c == Dt::Float as i32 => DataType::F32,
        c if c == Dt::Float16 as i32 => DataType::F16,
        c if c == Dt::Double as i32 => DataType::F64,
        c if c == Dt::Int8 as i32 => DataType::I8,
        c if c == Dt::Uint8 as i32 => DataType::U8,
        c if c == Dt::Int32 as i32 => DataType::I32,
        c if c == Dt::Int64 as i32 => DataType::I64,
        c if c == Dt::Bool as i32 => DataType::Bool,
        other => DataType::Other(other),
    }
}

fn tensor_from_proto(t: proto::TensorProto) -> Result<Tensor> {
    let data_type = data_type_from_code(t.data_type.unwrap_or_default());
    let dims = t.dims.clone();
    let raw = t.raw_data.as_deref().filter(|b| !b.is_empty());

    let data = match data_type {
        DataType::F32 => match raw {
            Some(b) => TensorData::F32(decode_le(b, f32::from_le_bytes)),
            None => TensorData::F32(t.float_data),
        },
        DataType::F16 => match raw {
            Some(b) => TensorData::F16(decode_le(b, u16::from_le_bytes)),
            // Per spec, f16 lives bit-wise in int32_data.
            None => TensorData::F16(t.int32_data.iter().map(|&v| v as u16).collect()),
        },
        DataType::F64 => match raw {
            Some(b) => TensorData::F64(decode_le(b, f64::from_le_bytes)),
            None => TensorData::F64(t.double_data),
        },
        DataType::I32 => match raw {
            Some(b) => TensorData::I32(decode_le(b, i32::from_le_bytes)),
            None => TensorData::I32(t.int32_data),
        },
        DataType::I64 => match raw {
            Some(b) => TensorData::I64(decode_le(b, i64::from_le_bytes)),
            None => TensorData::I64(t.int64_data),
        },
        DataType::Bool => match raw {
            Some(b) => TensorData::Bool(b.iter().map(|&v| v != 0).collect()),
            None => TensorData::Bool(t.int32_data.iter().map(|&v| v != 0).collect()),
        },
        // i8, u8 and unmodeled data types keep their bytes undecoded.
        _ => TensorData::Raw(t.raw_data.unwrap_or_default()),
    };

    Ok(Tensor {
        data_type,
        dims,
        data,
    })
}

/// Decode a little-endian byte buffer into a vector of fixed-width values.
fn decode_le<T, const N: usize>(bytes: &[u8], f: fn([u8; N]) -> T) -> Vec<T> {
    bytes
        .chunks_exact(N)
        .map(|c| f(c.try_into().expect("chunk_exact yields N bytes")))
        .collect()
}

fn attribute_from_proto(a: proto::AttributeProto) -> Result<Option<Attribute>> {
    use proto::attribute_proto::AttributeType as At;

    let kind = a.r#type.unwrap_or(At::Undefined as i32);

    let attr = if kind == At::Float as i32 {
        Some(Attribute::Float(a.f.unwrap_or_default()))
    } else if kind == At::Int as i32 {
        Some(Attribute::Int(a.i.unwrap_or_default()))
    } else if kind == At::String as i32 {
        Some(Attribute::String(bytes_to_string(a.s.unwrap_or_default())))
    } else if kind == At::Tensor as i32 {
        let t = a.t.context("tensor attribute without a tensor")?;
        Some(Attribute::Tensor(tensor_from_proto(t)?))
    } else if kind == At::Floats as i32 {
        Some(Attribute::Floats(a.floats))
    } else if kind == At::Ints as i32 {
        Some(Attribute::Ints(a.ints))
    } else if kind == At::Strings as i32 {
        Some(Attribute::Strings(
            a.strings.into_iter().map(bytes_to_string).collect(),
        ))
    } else if kind == At::Undefined as i32 {
        infer_untyped_attribute(a)?
    } else {
        // Graph, sparse and type-proto attributes are not needed yet.
        None
    };

    Ok(attr)
}

/// Pre-0.0.2 models leave the attribute `type` unset; recover it from whichever
/// field is populated.
fn infer_untyped_attribute(a: proto::AttributeProto) -> Result<Option<Attribute>> {
    Ok(if let Some(f) = a.f {
        Some(Attribute::Float(f))
    } else if let Some(i) = a.i {
        Some(Attribute::Int(i))
    } else if let Some(s) = a.s {
        Some(Attribute::String(bytes_to_string(s)))
    } else if let Some(t) = a.t {
        Some(Attribute::Tensor(tensor_from_proto(t)?))
    } else if !a.floats.is_empty() {
        Some(Attribute::Floats(a.floats))
    } else if !a.ints.is_empty() {
        Some(Attribute::Ints(a.ints))
    } else if !a.strings.is_empty() {
        Some(Attribute::Strings(
            a.strings.into_iter().map(bytes_to_string).collect(),
        ))
    } else {
        None
    })
}

fn bytes_to_string(b: Vec<u8>) -> String {
    String::from_utf8_lossy(&b).into_owned()
}
