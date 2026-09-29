use anyhow::Result;

pub struct ModelInfo {
    pub label: String,
    pub backend: &'static str,
    pub vocab_size: usize,
    pub context_limit: usize,
    pub chat_template: Option<String>,
}

/// What a loaded model occupies, in bytes. For display only.
#[derive(Clone, Copy, Debug)]
pub struct Footprint {
    /// Every weight, as held by the backend rather than as stored on disk.
    pub weight_bytes: u64,
    /// The part of [`Footprint::weight_bytes`] held widened because the
    /// backend cannot use the file's own format.
    pub dense_bytes: u64,
    /// Streamed expert weights, at their file size. Not counted in
    /// [`Footprint::weight_bytes`] since they are never all resident. Zero for
    /// a model without streamed experts.
    pub streamed_bytes: u64,
    /// What one more position of context costs across every cached layer.
    pub kv_bytes_per_token: u64,
}

/// The kind of one block of the network.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlockKind {
    /// Softmax attention, reading a key/value cache that grows with the
    /// sequence.
    Attention,
    /// A recurrent or linear-attention block. Its state has a fixed size
    /// however long the sequence gets.
    Recurrent,
}

impl BlockKind {
    pub fn label(self) -> &'static str {
        match self {
            BlockKind::Attention => "attention",
            BlockKind::Recurrent => "recurrent",
        }
    }
}

/// The shape of the network, for a caller that displays it.
#[derive(Clone, Debug)]
pub struct Architecture {
    pub d_model: usize,
    pub d_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    /// One entry per block, in the order they run.
    pub blocks: Vec<BlockKind>,
}

impl Architecture {
    pub fn count(&self, kind: BlockKind) -> usize {
        self.blocks.iter().filter(|&&b| b == kind).count()
    }
}

/// Hit and miss counts for the caches a backend keeps.
///
/// Counts rather than ratios, so a caller can show a rate over the whole run
/// or since it last looked. They level off as a model warms up.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Kernels found already compiled, and kernels that had to be compiled.
    /// The second should stop rising early.
    pub kernels_reused: u64,
    pub kernels_compiled: u64,
    /// Device buffers taken off the pool's free list, and buffers newly
    /// allocated because nothing of that size was free.
    pub buffers_reused: u64,
    pub buffers_allocated: u64,
    /// Device memory in pooled buffers, split into in use and idle on the
    /// free list.
    pub buffer_live_bytes: u64,
    pub buffer_idle_bytes: u64,
    /// Streamed experts only. A decode step's expert lookups that hit the
    /// device cache, those that missed, and the bytes copied over the bus by
    /// any pass. All zero for a model without experts.
    pub expert_hits: u64,
    pub expert_misses: u64,
    pub expert_bytes: u64,
    /// The same lookups by a prompt pass. These count experts, not tokens,
    /// since a prompt pass looks up each expert once for all its rows.
    pub expert_prompt_hits: u64,
    pub expert_prompt_misses: u64,
    /// Experts copied into side slots ahead of use, either by lookahead or
    /// as a host-computed miss kept for later tokens, and how many of those
    /// were then used.
    pub expert_prefetches: u64,
    pub expert_prefetch_hits: u64,
    /// Decode misses computed on the host instead of copied, and the host
    /// time they took. Zero unless that path is on.
    pub expert_cpu_misses: u64,
    pub expert_cpu_nanos: u64,
}

impl CacheStats {
    /// Share of kernel lookups that found one ready, or `None` before the
    /// first lookup.
    pub fn kernel_hit_rate(&self) -> Option<f64> {
        rate(self.kernels_reused, self.kernels_compiled)
    }

    /// Share of the experts a decode step wanted that were already on the
    /// device, or `None` for a model that streams none.
    pub fn expert_hit_rate(&self) -> Option<f64> {
        rate(self.expert_hits, self.expert_misses)
    }

    pub fn buffer_hit_rate(&self) -> Option<f64> {
        rate(self.buffers_reused, self.buffers_allocated)
    }
}

fn rate(hits: u64, misses: u64) -> Option<f64> {
    let total = hits + misses;
    (total > 0).then(|| hits as f64 / total as f64)
}

/// The card a model is running on, as the driver describes it.
///
/// Read once and fixed for the life of the process. The clocks are the card's
/// maxima, not live readings.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    /// Compute capability, major and minor.
    pub capability: (u32, u32),
    pub multiprocessors: u32,
    pub core_clock_khz: u32,
    pub memory_clock_khz: u32,
    pub memory_bus_bits: u32,
    /// The CUDA driver API version, major and minor. The driver's release
    /// version is [`DeviceInfo::driver`].
    pub cuda: (u32, u32),
    /// The display driver's own version, when it could be found.
    pub driver: Option<String>,
}

impl DeviceInfo {
    /// Peak memory bandwidth in bytes per second, from the specification:
    /// double data rate times bus width in bytes times the clock.
    ///
    /// Decode reads every weight once per token, so this is the ceiling to
    /// compare it against.
    pub fn peak_bandwidth(&self) -> u64 {
        2 * (self.memory_bus_bits as u64 / 8) * self.memory_clock_khz as u64 * 1_000
    }
}

/// A device's memory as the driver reports it, for the whole card rather than
/// this process.
#[derive(Clone, Copy, Debug)]
pub struct DeviceMemory {
    pub free_bytes: u64,
    pub total_bytes: u64,
}

impl DeviceMemory {
    pub fn used_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.free_bytes)
    }
}

pub trait Model {
    fn info(&self) -> &ModelInfo;

    fn tokenizer(&self) -> &dyn Tokenizer;

    fn session(&self) -> Result<Box<dyn Session + '_>>;

    /// What this model occupies, for display. `None` when the front end does
    /// not account for its weights.
    fn footprint(&self) -> Option<Footprint> {
        None
    }

    /// Free and total bytes on this model's device, read fresh on every call.
    /// A host backend reports nothing.
    fn device_memory(&self) -> Option<DeviceMemory> {
        None
    }

    /// The shape of the network, for a caller that displays it. Fixed at load.
    fn architecture(&self) -> Option<Architecture> {
        None
    }

    /// The card this model computes on. Fixed, so ask once: finding the
    /// display driver's version may spawn a subprocess.
    fn device_info(&self) -> Option<DeviceInfo> {
        None
    }

    /// The backend's cache counters so far, read fresh on every call.
    fn cache_stats(&self) -> Option<CacheStats> {
        None
    }

    /// Does up front what the first request would otherwise wait for, such as
    /// putting the weights on the device. Call once, right after loading.
    fn warm_up(&self) -> Result<()> {
        Ok(())
    }
}

pub trait Session {
    /// Run `ids` and return the logits for the position after the last one.
    ///
    /// Typically the prompt is one call and each generated token another. An
    /// implementation may split a long call into batches internally.
    fn extend(&mut self, ids: &[i64]) -> Result<Vec<f32>>;

    /// The batch size a long [`Session::extend`] is split into, if any.
    /// Feeding the prompt a batch at a time then costs nothing and allows
    /// progress reports. `None` means the prompt must arrive in one call.
    fn prompt_batch(&self) -> Option<usize> {
        None
    }

    /// [`Session::extend`] returning only the argmax token id, for greedy
    /// decoding. A backend can skip copying out the logits. The default is
    /// [`Session::extend`] plus a host-side argmax.
    fn extend_greedy(&mut self, ids: &[i64]) -> Result<i64> {
        Ok(crate::sampling::argmax(&self.extend(ids)?))
    }

    /// Tokens consumed so far, prompt included.
    fn len(&self) -> usize;

    /// Drop everything past `positions` and return how many positions were
    /// kept. The next [`Session::extend`] continues from there.
    ///
    /// The result may be fewer than asked, since a session can only go back
    /// to points it can return to. The caller runs the rest again. `None`
    /// means the session could not go back at all and should be dropped.
    ///
    /// A recurrent block's state cannot be rewound by position, only restored
    /// to a [`Session::checkpoint`]. The default rewinds nowhere, but still
    /// accepts a request that drops nothing, so the session can be extended.
    fn truncate(&mut self, positions: usize) -> Option<usize> {
        (positions == self.len()).then_some(positions)
    }

    /// Remember the current position as one [`Session::truncate`] can
    /// return to, replacing any earlier checkpoint.
    fn checkpoint(&mut self) -> Result<()> {
        Ok(())
    }

    /// Bytes of key/value cache this session has reserved right now, which
    /// can be well above what [`Session::len`] needs.
    fn cache_bytes(&self) -> Option<u64> {
        None
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<i64>>;

    fn decode_bytes(&self, ids: &[i64]) -> Vec<u8>;

    fn is_eog(&self, id: i64) -> bool;

    fn bos_text(&self) -> Option<&str> {
        None
    }

    fn decode(&self, ids: &[i64]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }
}
