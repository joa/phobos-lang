pub use phobos_kernels::abi;

pub mod backend;
pub mod eval;
pub mod ir;
pub mod layout;
pub mod load;
pub mod lower;
pub mod proto;
pub mod runtime;
pub mod shape;
pub mod tokenizer;
pub mod transform;

pub use backend::{HostBackend, MatmulBackend, Tensor};
pub use ir::{Graph, Model, Node};
pub use load::load_model;
pub use runtime::OnnxModel;
pub use tokenizer::Gpt2Tokenizer;

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    /// A minimal `Y = MatMul(X, W)` model, encoded and loaded back: the whole
    /// proto to IR path, edges, initializer, shapes and op.
    #[test]
    fn loads_a_matmul_graph() {
        use proto::tensor_proto::DataType;

        let tensor_type = |dims: &[proto::tensor_shape_proto::Dimension]| proto::TypeProto {
            value: Some(proto::type_proto::Value::TensorType(
                proto::type_proto::Tensor {
                    elem_type: Some(DataType::Float as i32),
                    shape: Some(proto::TensorShapeProto { dim: dims.to_vec() }),
                },
            )),
            denotation: None,
        };
        let fixed = |n: i64| proto::tensor_shape_proto::Dimension {
            value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(n)),
            denotation: None,
        };
        let sym = |s: &str| proto::tensor_shape_proto::Dimension {
            value: Some(proto::tensor_shape_proto::dimension::Value::DimParam(
                s.into(),
            )),
            denotation: None,
        };

        let x = proto::ValueInfoProto {
            name: Some("X".into()),
            r#type: Some(tensor_type(&[sym("M"), fixed(8)])),
            ..Default::default()
        };
        let y = proto::ValueInfoProto {
            name: Some("Y".into()),
            r#type: Some(tensor_type(&[sym("M"), fixed(4)])),
            ..Default::default()
        };
        let w = proto::TensorProto {
            name: Some("W".into()),
            data_type: Some(DataType::Float as i32),
            dims: vec![8, 4],
            float_data: vec![0.5; 32],
            ..Default::default()
        };
        let node = proto::NodeProto {
            input: vec!["X".into(), "W".into()],
            output: vec!["Y".into()],
            op_type: Some("MatMul".into()),
            name: Some("mm".into()),
            ..Default::default()
        };
        let graph = proto::GraphProto {
            name: Some("g".into()),
            node: vec![node],
            initializer: vec![w],
            input: vec![x],
            output: vec![y],
            ..Default::default()
        };
        let model = proto::ModelProto {
            ir_version: Some(9),
            producer_name: Some("test".into()),
            opset_import: vec![proto::OperatorSetIdProto {
                domain: Some(String::new()),
                version: Some(21),
            }],
            graph: Some(graph),
            ..Default::default()
        };

        let bytes = model.encode_to_vec();
        let loaded = load_model(&bytes).expect("load");

        assert_eq!(loaded.ir_version, 9);
        assert_eq!(loaded.opset.get(""), Some(&21));

        let g = &loaded.graph;
        assert_eq!(g.nodes.len(), 1);
        assert_eq!(g.nodes[0].op_type, "MatMul");
        assert_eq!(g.nodes[0].inputs, vec!["X", "W"]);

        // W is a constant fed by an initializer; X is a graph input.
        assert!(g.is_constant("W"));
        assert!(!g.is_constant("X"));
        let w = &g.initializers["W"];
        assert_eq!(w.dims, vec![8, 4]);
        assert_eq!(w.data_type, ir::DataType::F32);
        assert!(matches!(&w.data, ir::TensorData::F32(v) if v.len() == 32));

        // The input shape keeps its symbolic leading axis.
        let x_shape = &g.values["X"].shape.0.as_ref().unwrap();
        assert_eq!(x_shape[0], ir::Dim::Symbol("M".into()));
        assert_eq!(x_shape[1], ir::Dim::Fixed(8));
        assert_eq!(g.values["Y"].data_type, Some(ir::DataType::F32));
    }

    /// f16 initializers may arrive as little-endian `raw_data`, whose bit
    /// patterns have to be decoded rather than dropped.
    #[test]
    fn decodes_f16_raw_data() {
        use proto::tensor_proto::DataType;

        // Two half values: 1.0 (0x3C00) and 2.0 (0x4000), little-endian.
        let raw = vec![0x00, 0x3C, 0x00, 0x40];
        let w = proto::TensorProto {
            name: Some("H".into()),
            data_type: Some(DataType::Float16 as i32),
            dims: vec![2],
            raw_data: Some(raw),
            ..Default::default()
        };
        let graph = proto::GraphProto {
            initializer: vec![w],
            ..Default::default()
        };
        let model = proto::ModelProto {
            graph: Some(graph),
            ..Default::default()
        };

        let loaded = load_model(&model.encode_to_vec()).expect("load");
        let h = &loaded.graph.initializers["H"];
        assert_eq!(h.data_type, ir::DataType::F16);
        assert!(matches!(&h.data, ir::TensorData::F16(v) if v == &[0x3C00, 0x4000]));
    }
}
