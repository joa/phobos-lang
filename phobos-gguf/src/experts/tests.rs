use phobos_base::half::f32_to_f16;

use super::*;
use crate::quant::grouped::group_rows;
use crate::tests::Builder;

/// Q4_K's ggml type code.
pub(crate) const Q4_K: u32 = 12;

/// `count` experts of `[n, k]` Q4_K blocks with deterministic contents:
/// small finite headers, pseudo-random scales and quants.
pub(crate) fn q4k_stack(count: usize, n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut words = crate::tests::xorshift(seed);
    let mut next = move || (words() >> 56) as u8;
    let blocks = count * n * k / 256;
    let mut out = Vec::with_capacity(blocks * 144);
    for _ in 0..blocks {
        let d = f32_to_f16(0.001 + f32::from(next()) / 4096.0);
        let dmin = f32_to_f16(0.0005 + f32::from(next()) / 8192.0);
        out.extend(d.to_le_bytes());
        out.extend(dmin.to_le_bytes());
        out.extend((0..140).map(|_| next()));
    }
    out
}

/// A file holding one `[count, n, k]` Q4_K stack under `name`.
pub(crate) fn stack_file(name: &str, count: usize, n: usize, k: usize, seed: u64) -> (Gguf, Vec<u8>) {
    let bytes = q4k_stack(count, n, k, seed);
    let file = Builder::default()
        .kv_string("general.architecture", "qwen35moe")
        .tensor_raw(name, &[k as u64, n as u64, count as u64], Q4_K, &bytes)
        .build();
    (Gguf::from_bytes(file).unwrap(), bytes)
}

#[test]
fn a_stack_addresses_experts_by_stride() {
    let (count, n, k) = (3, 16, 512);
    let (gguf, bytes) = stack_file("blk.0.ffn_gate_exps.weight", count, n, k, 7);
    let stack = ExpertStack::load(&gguf, "blk.0.ffn_gate_exps.weight", count, n, k).unwrap();
    assert_eq!(stack.quant(), Quant::Q4_K);
    assert_eq!(stack.blocks_per_row(), 2);
    assert_eq!(stack.expert_bytes(), n * 2 * 144);
    assert_eq!(stack.byte_len(), bytes.len());
    for e in 0..count {
        let stride = stack.expert_bytes();
        assert_eq!(stack.expert(e), &bytes[e * stride..(e + 1) * stride]);
    }
}

#[test]
fn a_stack_outlives_the_file_it_came_from() {
    let (gguf, bytes) = stack_file("w", 2, 8, 256, 11);
    let stack = ExpertStack::load(&gguf, "w", 2, 8, 256).unwrap();
    drop(gguf);
    assert_eq!(stack.expert(1), &bytes[8 * 144..]);
}

#[test]
fn a_stack_decodes_an_expert_as_the_file_reader_does() {
    let (count, n, k) = (2, 8, 256);
    let (gguf, bytes) = stack_file("w", count, n, k, 3);
    let stack = ExpertStack::load(&gguf, "w", count, n, k).unwrap();
    let mut got = vec![0.0f32; n * k];
    stack.dequantize(1, &mut got).unwrap();
    let mut whole = vec![0.0f32; count * n * k];
    crate::dequantize_into(crate::GgmlType::Q4_K, &bytes, &mut whole).unwrap();
    assert_eq!(got, whole[n * k..]);
    assert!(got.iter().all(|v| v.is_finite()));
}

#[test]
fn a_stack_groups_an_expert_as_the_upload_does() {
    let (count, n, k) = (2, 24, 512);
    let (gguf, _) = stack_file("w", count, n, k, 5);
    let stack = ExpertStack::load(&gguf, "w", count, n, k).unwrap();
    let mut got = vec![0u8; stack.grouped_bytes()];
    stack.grouped_into(1, &mut got);
    // Q4_K keeps its whole block on the device, so the reference regroups
    // the file bytes as they are.
    assert_eq!(got, group_rows(stack.expert(1), n, 2, 144));
    // 24 rows pad to 64: the grouped buffer is 64 rows of 2 blocks.
    assert_eq!(got.len(), 64 * 2 * 144);
}

#[test]
fn a_stack_refuses_the_wrong_extents_and_a_dense_type() {
    let (gguf, _) = stack_file("w", 2, 8, 256, 1);
    let err = ExpertStack::load(&gguf, "w", 2, 16, 256).err().unwrap();
    assert!(err.to_string().contains("expected [256, 16, 2]"), "{err}");

    let file = Builder::default()
        .kv_string("general.architecture", "qwen35moe")
        .tensor_f32("w", &[4, 2, 1], &[0.0; 8])
        .build();
    let gguf = Gguf::from_bytes(file).unwrap();
    let err = ExpertStack::load(&gguf, "w", 1, 2, 4).err().unwrap();
    assert!(err.to_string().contains("not a quantized format"), "{err}");
}

#[test]
fn a_set_loads_its_three_stacks() {
    let (count, d_model, d_ff) = (2, 256, 512);
    let file = Builder::default()
        .kv_string("general.architecture", "qwen35moe")
        .tensor_raw(
            "blk.0.ffn_gate_exps.weight",
            &[d_model as u64, d_ff as u64, count as u64],
            Q4_K,
            &q4k_stack(count, d_ff, d_model, 1),
        )
        .tensor_raw(
            "blk.0.ffn_up_exps.weight",
            &[d_model as u64, d_ff as u64, count as u64],
            Q4_K,
            &q4k_stack(count, d_ff, d_model, 2),
        )
        .tensor_raw(
            "blk.0.ffn_down_exps.weight",
            &[d_ff as u64, d_model as u64, count as u64],
            Q4_K,
            &q4k_stack(count, d_model, d_ff, 3),
        )
        .build();
    let gguf = Gguf::from_bytes(file).unwrap();
    let set = ExpertSet::load(&gguf, "blk.0", count, d_model, d_ff).unwrap();
    assert_eq!(set.count(), count);
    assert_eq!((set.gate.n(), set.gate.k()), (d_ff, d_model));
    assert_eq!((set.down.n(), set.down.k()), (d_model, d_ff));
    assert_eq!(set.expert_bytes(), 3 * (d_ff * d_model / 256) * 144);
    assert_eq!(set.byte_len(), count * set.expert_bytes());
}
