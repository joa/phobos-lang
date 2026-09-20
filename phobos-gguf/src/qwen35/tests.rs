// A qwen35moe model small enough to build in memory: two blocks, one delta
// net and one attention, four experts of which two route, every trunk
// weight F32 and the expert stacks Q4_K. The stand-in for a small file of
// this architecture, which does not exist.

use crate::backend::HostBackend;
use crate::experts::tests::{Q4_K, q4k_stack};
use crate::tests::Builder;
use crate::{Decoder, Gguf};

const D: usize = 256;
const VOCAB: usize = 64;
const EXPERTS: usize = 4;
const USED: usize = 2;
const D_EXPERT: usize = 256;
const D_SHARED: usize = 256;
const HEAD_DIM: usize = 64;
const N_HEAD: usize = 2;
const N_KV: usize = 1;
const SSM_STATE: usize = 64;
const SSM_INNER: usize = 128;
const SSM_HEADS: usize = SSM_INNER / SSM_STATE;
const SSM_KV_HEADS: usize = 1;
const CONV: usize = 4;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / 8388608.0 - 1.0
    }

    fn weights(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| scale * self.next()).collect()
    }
}

/// The file, as bytes.
pub(crate) fn tiny_moe_file() -> Vec<u8> {
    let mut rng = Rng(0x5eed_1234_abcd_0001);
    let mut b = Builder::default();
    b.kv_string("general.architecture", "qwen35moe")
        .kv_u32("qwen35moe.block_count", 2)
        .kv_u32("qwen35moe.embedding_length", D as u32)
        .kv_u32("qwen35moe.expert_count", EXPERTS as u32)
        .kv_u32("qwen35moe.expert_used_count", USED as u32)
        .kv_u32("qwen35moe.expert_feed_forward_length", D_EXPERT as u32)
        .kv_u32("qwen35moe.expert_shared_feed_forward_length", D_SHARED as u32)
        .kv_u32("qwen35moe.context_length", 64)
        .kv_f32("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6)
        .kv_u32("qwen35moe.attention.head_count", N_HEAD as u32)
        .kv_u32("qwen35moe.attention.head_count_kv", N_KV as u32)
        .kv_u32("qwen35moe.attention.key_length", HEAD_DIM as u32)
        .kv_u32("qwen35moe.full_attention_interval", 2)
        .kv_u32("qwen35moe.rope.dimension_count", 32)
        .kv_f32("qwen35moe.rope.freq_base", 10000.0)
        .kv_u32("qwen35moe.ssm.group_count", SSM_KV_HEADS as u32)
        .kv_u32("qwen35moe.ssm.inner_size", SSM_INNER as u32)
        .kv_u32("qwen35moe.ssm.state_size", SSM_STATE as u32)
        .kv_u32("qwen35moe.ssm.time_step_rank", SSM_HEADS as u32)
        .kv_u32("qwen35moe.ssm.conv_kernel", CONV as u32);

    let scale = 0.05;
    let embd = rng.weights(D * VOCAB, 1.0);
    b.tensor_f32("token_embd.weight", &[D as u64, VOCAB as u64], &embd);
    b.tensor_f32("output_norm.weight", &[D as u64], &vec![1.0; D]);

    let channels = 2 * SSM_KV_HEADS * SSM_STATE + SSM_INNER;
    for block in 0..2 {
        let p = format!("blk.{block}");
        b.tensor_f32(&format!("{p}.attn_norm.weight"), &[D as u64], &vec![1.0; D]);
        b.tensor_f32(&format!("{p}.post_attention_norm.weight"), &[D as u64], &vec![1.0; D]);
        if block == 0 {
            b.tensor_f32(&format!("{p}.attn_qkv.weight"), &[D as u64, channels as u64], &rng.weights(D * channels, scale));
            b.tensor_f32(&format!("{p}.attn_gate.weight"), &[D as u64, SSM_INNER as u64], &rng.weights(D * SSM_INNER, scale));
            b.tensor_f32(&format!("{p}.ssm_alpha.weight"), &[D as u64, SSM_HEADS as u64], &rng.weights(D * SSM_HEADS, scale));
            b.tensor_f32(&format!("{p}.ssm_beta.weight"), &[D as u64, SSM_HEADS as u64], &rng.weights(D * SSM_HEADS, scale));
            b.tensor_f32(&format!("{p}.ssm_out.weight"), &[SSM_INNER as u64, D as u64], &rng.weights(SSM_INNER * D, scale));
            b.tensor_f32(&format!("{p}.ssm_conv1d.weight"), &[CONV as u64, channels as u64], &rng.weights(CONV * channels, scale));
            b.tensor_f32(&format!("{p}.ssm_a"), &[SSM_HEADS as u64], &[-0.5; SSM_HEADS]);
            b.tensor_f32(&format!("{p}.ssm_dt.bias"), &[SSM_HEADS as u64], &[0.1; SSM_HEADS]);
            b.tensor_f32(&format!("{p}.ssm_norm.weight"), &[SSM_STATE as u64], &vec![1.0; SSM_STATE]);
        } else {
            let q_out = N_HEAD * HEAD_DIM * 2;
            let kv = N_KV * HEAD_DIM;
            b.tensor_f32(&format!("{p}.attn_q.weight"), &[D as u64, q_out as u64], &rng.weights(D * q_out, scale));
            b.tensor_f32(&format!("{p}.attn_k.weight"), &[D as u64, kv as u64], &rng.weights(D * kv, scale));
            b.tensor_f32(&format!("{p}.attn_v.weight"), &[D as u64, kv as u64], &rng.weights(D * kv, scale));
            b.tensor_f32(&format!("{p}.attn_output.weight"), &[(N_HEAD * HEAD_DIM) as u64, D as u64], &rng.weights(N_HEAD * HEAD_DIM * D, scale));
            b.tensor_f32(&format!("{p}.attn_q_norm.weight"), &[HEAD_DIM as u64], &vec![1.0; HEAD_DIM]);
            b.tensor_f32(&format!("{p}.attn_k_norm.weight"), &[HEAD_DIM as u64], &vec![1.0; HEAD_DIM]);
        }
        b.tensor_f32(&format!("{p}.ffn_gate_inp.weight"), &[D as u64, EXPERTS as u64], &rng.weights(D * EXPERTS, 1.0));
        b.tensor_raw(&format!("{p}.ffn_gate_exps.weight"), &[D as u64, D_EXPERT as u64, EXPERTS as u64], Q4_K, &q4k_stack(EXPERTS, D_EXPERT, D, 100 + block as u64));
        b.tensor_raw(&format!("{p}.ffn_up_exps.weight"), &[D as u64, D_EXPERT as u64, EXPERTS as u64], Q4_K, &q4k_stack(EXPERTS, D_EXPERT, D, 200 + block as u64));
        b.tensor_raw(&format!("{p}.ffn_down_exps.weight"), &[D_EXPERT as u64, D as u64, EXPERTS as u64], Q4_K, &q4k_stack(EXPERTS, D, D_EXPERT, 300 + block as u64));
        b.tensor_f32(&format!("{p}.ffn_gate_inp_shexp.weight"), &[D as u64], &rng.weights(D, scale));
        b.tensor_f32(&format!("{p}.ffn_gate_shexp.weight"), &[D as u64, D_SHARED as u64], &rng.weights(D * D_SHARED, scale));
        b.tensor_f32(&format!("{p}.ffn_up_shexp.weight"), &[D as u64, D_SHARED as u64], &rng.weights(D * D_SHARED, scale));
        b.tensor_f32(&format!("{p}.ffn_down_shexp.weight"), &[D_SHARED as u64, D as u64], &rng.weights(D_SHARED * D, scale));
    }
    b.build()
}

#[test]
fn the_tiny_moe_model_loads_and_runs_on_the_host() {
    let gguf = Gguf::from_bytes(tiny_moe_file()).unwrap();
    let model = Decoder::load(&gguf).unwrap();
    assert_eq!(model.architecture(), "qwen35moe");
    let footprint = model.footprint(64);
    // Three stacks of four experts a block, two blocks, each matrix
    // 256 x 256 Q4_K.
    assert_eq!(footprint.streamed_bytes, 2 * 3 * EXPERTS * (D * D_EXPERT / 256) * 144);
    assert!(footprint.weight_bytes > 0 && footprint.weight_bytes < footprint.streamed_bytes * 4);

    let backend = HostBackend::new();
    let mut state = model.new_state();
    let logits = model.forward(&mut state, &[1, 2, 3], &backend).unwrap();
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()));
    let again = model.forward(&mut state, &[4], &backend).unwrap();
    assert!(again.iter().all(|v| v.is_finite()));
    assert_ne!(logits, again);
    assert_eq!(state.len(), 4);
    state.release(&backend);
}

#[test]
fn a_traced_pass_reports_routes_and_lookahead_per_block() {
    let gguf = Gguf::from_bytes(tiny_moe_file()).unwrap();
    let model = Decoder::load(&gguf).unwrap();
    let backend = HostBackend::new();
    let mut state = model.new_state();
    let rows = 3;
    let (logits, trace) = model.forward_traced(&mut state, &[5, 6, 7], &backend).unwrap();
    assert_eq!(logits.len(), VOCAB);
    assert_eq!(trace.n_used, USED);
    assert_eq!(trace.routes.len(), 2 * rows * USED);
    assert_eq!(trace.lookahead.len(), 2 * rows * USED);
    assert!(trace.routes.iter().all(|&e| (e as usize) < EXPERTS));
    // Two distinct experts a row.
    for pair in trace.routes.chunks_exact(USED) {
        assert_ne!(pair[0], pair[1]);
    }
    // Block 0 predicts block 1; block 1, the last, predicts nothing.
    assert!(trace.lookahead[..rows * USED].iter().all(|&e| (e as usize) < EXPERTS));
    assert!(trace.lookahead[rows * USED..].iter().all(|&e| e == 0));

    // The traced pass and the plain one agree on the logits.
    let mut plain = model.new_state();
    let expected = model.forward(&mut plain, &[5, 6, 7], &backend).unwrap();
    for (a, b) in logits.iter().zip(&expected) {
        assert!((a - b).abs() <= 1e-5 * (1.0 + b.abs()), "{a} vs {b}");
    }
    state.release(&backend);
    plain.release(&backend);
}
