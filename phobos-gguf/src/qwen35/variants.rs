/// Points where the tensor extents alone do not pin the architecture down.
/// [`Variants::REFERENCE`] is llama.cpp's `qwen35` impl and our default; the
/// rest are swept by `examples/sweep.rs` against a repeated-phrase probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Variants {
    /// `attn_qkv` groups query/key/value per head rather than as three
    /// contiguous blocks.
    pub qkv_interleaved: bool,
    /// `attn_q` emits all queries then all gates, rather than pairing them
    /// inside each head's slice.
    pub attn_gate_contiguous: bool,
    /// The gate precedes the query wherever they are split.
    pub attn_gate_first: bool,
    /// Normalize before applying the output gate rather than after.
    pub norm_before_gate: bool,
    /// `ssm_a` holds `log(A)`, as the HuggingFace checkpoint does. The GGUF
    /// converter instead stores `-exp(A_log)` and llama.cpp multiplies by it
    /// directly, so this is false for GGUF files.
    pub decay_from_log: bool,
    /// L2-normalize delta-rule queries and keys.
    pub l2_normalize_qk: bool,
    /// `ssm_alpha` supplies the delta-rule write strength and `ssm_beta` the
    /// decay, rather than the other way round.
    pub swap_alpha_beta: bool,
    /// The convolution's taps run newest-first.
    pub conv_reversed: bool,
    /// The delta-rule readout contracts the value axis instead of the key axis.
    pub query_contracts_value: bool,
}

impl Variants {
    pub const REFERENCE: Variants = Variants {
        qkv_interleaved: false,
        attn_gate_contiguous: false,
        attn_gate_first: false,
        norm_before_gate: true,
        decay_from_log: false,
        l2_normalize_qk: true,
        swap_alpha_beta: false,
        conv_reversed: false,
        query_contracts_value: false,
    };

    /// The combinations the sweep explores; the rest are settled and held
    /// fixed.
    pub fn all() -> Vec<Variants> {
        (0..16u32)
            .map(|bits| Variants {
                qkv_interleaved: bits & 1 != 0,
                norm_before_gate: bits & 2 != 0,
                conv_reversed: bits & 4 != 0,
                attn_gate_contiguous: false,
                attn_gate_first: false,
                decay_from_log: false,
                l2_normalize_qk: bits & 8 == 0,
                swap_alpha_beta: false,
                query_contracts_value: false,
            })
            .collect()
    }
}

impl Default for Variants {
    fn default() -> Variants {
        Variants::REFERENCE
    }
}
