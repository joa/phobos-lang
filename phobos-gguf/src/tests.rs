use super::*;

/// Minimal GGUF writer, enough to round-trip the reader.q
#[derive(Default)]
struct Builder {
    kv: Vec<u8>,
    kv_count: u64,
    tensors: Vec<u8>,
    tensor_count: u64,
    data: Vec<u8>,
}

fn push_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend((s.len() as u64).to_le_bytes());
    buf.extend(s.as_bytes());
}

impl Builder {
    fn kv_string(&mut self, key: &str, value: &str) -> &mut Self {
        push_str(&mut self.kv, key);
        self.kv.extend(8u32.to_le_bytes());
        push_str(&mut self.kv, value);
        self.kv_count += 1;
        self
    }

    fn kv_u32(&mut self, key: &str, value: u32) -> &mut Self {
        push_str(&mut self.kv, key);
        self.kv.extend(4u32.to_le_bytes());
        self.kv.extend(value.to_le_bytes());
        self.kv_count += 1;
        self
    }

    fn kv_string_array(&mut self, key: &str, values: &[&str]) -> &mut Self {
        push_str(&mut self.kv, key);
        self.kv.extend(9u32.to_le_bytes());
        self.kv.extend(8u32.to_le_bytes());
        self.kv.extend((values.len() as u64).to_le_bytes());
        for v in values {
            push_str(&mut self.kv, v);
        }
        self.kv_count += 1;
        self
    }

    fn tensor_f32(&mut self, name: &str, dims: &[u64], values: &[f32]) -> &mut Self {
        push_str(&mut self.tensors, name);
        self.tensors.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            self.tensors.extend(d.to_le_bytes());
        }
        self.tensors.extend(0u32.to_le_bytes()); // F32
        self.tensors.extend((self.data.len() as u64).to_le_bytes());
        self.tensor_count += 1;
        for v in values {
            self.data.extend(v.to_le_bytes());
        }
        self
    }

    fn build(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(b"GGUF");
        out.extend(3u32.to_le_bytes());
        out.extend(self.tensor_count.to_le_bytes());
        out.extend(self.kv_count.to_le_bytes());
        out.extend(&self.kv);
        out.extend(&self.tensors);
        out.resize(out.len().next_multiple_of(32), 0);
        out.extend(&self.data);
        out
    }
}

#[test]
fn round_trips_metadata_and_tensors() {
    let bytes = Builder::default()
        .kv_string("general.architecture", "llama")
        .kv_u32("llama.block_count", 12)
        .kv_string_array("tokenizer.ggml.tokens", &["a", "b"])
        .kv_string("tokenizer.ggml.model", "gpt2")
        .tensor_f32(
            "token_embd.weight",
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .tensor_f32("output_norm.weight", &[2], &[0.25, 0.5])
        .build();

    let gguf = Gguf::from_bytes(bytes).unwrap();
    assert_eq!(gguf.version(), 3);
    assert_eq!(gguf.architecture().unwrap(), "llama");
    assert_eq!(gguf.metadata().arch_count("block_count").unwrap(), 12);
    assert_eq!(gguf.tensors().len(), 2);

    let embd = gguf.tensor("token_embd.weight").unwrap();
    assert_eq!(embd.dims, vec![2, 3]);
    // ggml stores the fastest axis first, so this is 3 rows of 2.
    assert_eq!(embd.row_major_dims(), vec![3, 2]);
    assert_eq!(embd.numel(), 6);
    assert_eq!(
        gguf.dequantize("token_embd.weight").unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(
        gguf.dequantize("output_norm.weight").unwrap(),
        vec![0.25, 0.5]
    );
    assert_eq!(gguf.parameter_count(), 8);

    let vocab = gguf.vocab().unwrap();
    assert_eq!(vocab.tokens, vec!["a", "b"]);
}

#[test]
fn rejects_a_non_gguf_file() {
    let err = Gguf::from_bytes(b"ONNX and some padding bytes".to_vec()).unwrap_err();
    assert!(err.to_string().contains("not a GGUF file"), "{err}");
}

#[test]
fn rejects_a_truncated_header() {
    let mut bytes = Builder::default()
        .kv_string("general.architecture", "llama")
        .build();
    bytes.truncate(20);
    assert!(Gguf::from_bytes(bytes).is_err());
}

#[test]
fn rejects_a_tensor_running_past_the_file() {
    let mut bytes = Builder::default()
        .kv_string("general.architecture", "llama")
        .tensor_f32("w", &[4], &[1.0, 2.0, 3.0, 4.0])
        .build();
    // Drop half the payload; the directory still claims 16 bytes.
    bytes.truncate(bytes.len() - 8);

    let gguf = Gguf::from_bytes(bytes).unwrap();
    let err = gguf.dequantize("w").unwrap_err();
    assert!(format!("{err:#}").contains("but the file is"), "{err:#}");
}

#[test]
fn reports_unknown_tensors() {
    let bytes = Builder::default()
        .kv_string("general.architecture", "llama")
        .build();
    let gguf = Gguf::from_bytes(bytes).unwrap();
    assert!(gguf.tensor("missing").is_none());
    assert!(
        gguf.dequantize("missing")
            .unwrap_err()
            .to_string()
            .contains("no tensor")
    );
}
