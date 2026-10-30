use super::*;
use crate::quant::Quant;
use crate::tensor::GgmlType;

/// A deterministic stream of `u64`s from a seed.
pub(crate) fn xorshift(mut seed: u64) -> impl FnMut() -> u64 {
    move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    }
}

/// A deterministic stream of floats in [-1, 1) from a seed.
pub(crate) fn uniform(seed: u64) -> impl FnMut() -> f32 {
    let mut next = xorshift(seed);
    move || (next() >> 40) as f32 / 8388608.0 - 1.0
}

/// The ggml type a format is stored under.
pub(crate) fn ggml_type(quant: Quant) -> GgmlType {
    GgmlType::all().find(|t| t.quant() == Some(quant)).expect("a stored format")
}

/// Minimal GGUF writer, enough to round-trip the reader and to stand in for
/// a real file in the crate's other tests.
#[derive(Default)]
pub(crate) struct Builder {
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
    pub(crate) fn kv_string(&mut self, key: &str, value: &str) -> &mut Self {
        push_str(&mut self.kv, key);
        self.kv.extend(8u32.to_le_bytes());
        push_str(&mut self.kv, value);
        self.kv_count += 1;
        self
    }

    pub(crate) fn kv_u32(&mut self, key: &str, value: u32) -> &mut Self {
        push_str(&mut self.kv, key);
        self.kv.extend(4u32.to_le_bytes());
        self.kv.extend(value.to_le_bytes());
        self.kv_count += 1;
        self
    }

    pub(crate) fn kv_f32(&mut self, key: &str, value: f32) -> &mut Self {
        push_str(&mut self.kv, key);
        self.kv.extend(6u32.to_le_bytes());
        self.kv.extend(value.to_le_bytes());
        self.kv_count += 1;
        self
    }

    pub(crate) fn kv_string_array(&mut self, key: &str, values: &[&str]) -> &mut Self {
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

    pub(crate) fn tensor_f32(&mut self, name: &str, dims: &[u64], values: &[f32]) -> &mut Self {
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

    /// A tensor of any ggml type from its storage bytes.
    pub(crate) fn tensor_raw(&mut self, name: &str, dims: &[u64], ty: GgmlType, bytes: &[u8]) -> &mut Self {
        push_str(&mut self.tensors, name);
        self.tensors.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            self.tensors.extend(d.to_le_bytes());
        }
        self.tensors.extend(ty.code().to_le_bytes());
        self.tensors.extend((self.data.len() as u64).to_le_bytes());
        self.tensor_count += 1;
        self.data.extend(bytes);
        self
    }

    pub(crate) fn build(&self) -> Vec<u8> {
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

/// A header with no tensors and one key whose value bytes the caller writes,
/// starting at its value type.
fn header_with_value(value: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend(b"GGUF");
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.extend(1u64.to_le_bytes());
    push_str(&mut out, "k");
    out.extend(value);
    out
}

fn error_of(bytes: Vec<u8>) -> String {
    match Gguf::from_bytes(bytes) {
        Ok(_) => panic!("a malformed file parsed"),
        Err(err) => format!("{err:#}"),
    }
}

#[test]
fn deeply_nested_arrays_are_an_error_not_a_stack_overflow() {
    // Each level is an array of one array; 12 bytes a level.
    let levels = 400_000;
    let mut value = 9u32.to_le_bytes().to_vec();
    for _ in 0..levels {
        value.extend(9u32.to_le_bytes());
        value.extend(1u64.to_le_bytes());
    }
    let err = error_of(header_with_value(&value));
    assert!(err.contains("nest more than"), "{err}");
}

#[test]
fn a_count_the_file_cannot_hold_is_refused_before_allocating() {
    // A million strings in a few bytes: each needs at least its length.
    let mut value = 9u32.to_le_bytes().to_vec();
    value.extend(8u32.to_le_bytes());
    value.extend(1_000_000u64.to_le_bytes());
    value.extend([0u8; 64]);
    let err = error_of(header_with_value(&value));
    assert!(err.contains("elements of at least 8 bytes"), "{err}");

    // A tensor count past what the rest of the file can describe.
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(1_000u64.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.extend([0u8; 256]);
    let err = error_of(bytes);
    assert!(err.contains("elements of at least"), "{err}");
}

#[test]
fn a_tensor_with_no_dimensions_is_refused() {
    let bytes = Builder::default()
        .kv_string("general.architecture", "llama")
        .tensor_f32("token_embd.weight", &[], &[])
        .build();
    let err = error_of(bytes);
    assert!(err.contains("declares 0 dimensions"), "{err}");
}

#[test]
fn dimensions_whose_product_overflows_are_refused() {
    let bytes = Builder::default()
        .tensor_f32("w", &[1 << 32, 1 << 32, 2], &[])
        .build();
    let err = error_of(bytes);
    assert!(err.contains("overflows 64 bits"), "{err}");
}

#[test]
fn a_huge_tensor_fails_before_its_output_is_allocated() {
    // 2^62 elements, a product that fits, in a file of a few bytes.
    let bytes = Builder::default()
        .tensor_f32("w", &[1 << 31, 1 << 31], &[1.0])
        .build();
    let gguf = Gguf::from_bytes(bytes).unwrap();
    assert!(gguf.dequantize("w").is_err());
}
