use anyhow::Result;

pub struct ModelInfo {
    pub label: String,
    pub backend: &'static str,
    pub vocab_size: usize,
    pub context_limit: usize,
    pub chat_template: Option<String>,
}

/// What a loaded model occupies, in bytes, for a caller that reports rather
/// than decides: a front end that cannot say returns nothing rather than zero.
#[derive(Clone, Copy, Debug)]
pub struct Footprint {
    /// Every weight, as held by the backend rather than as stored on disk.
    pub weight_bytes: u64,
    /// Of [`Footprint::weight_bytes`], what the backend could not keep in the
    /// file's own format and holds widened instead.
    pub dense_bytes: u64,
    /// Weights not counted in [`Footprint::weight_bytes`] because they are
    /// never all resident: a mixture of experts the backend streams from the
    /// file, as the file holds them. Zero for a model without any.
    pub streamed_bytes: u64,
    /// What one more position of context costs across every cached layer.
    pub kv_bytes_per_token: u64,
}

/// What one block of the network does, as much as anything outside a front
/// end needs to know.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlockKind {
    /// Softmax attention, reading a key/value cache that grows with the
    /// sequence.
    Attention,
    /// A recurrent or linear-attention block, carrying state of a fixed size
    /// however long the sequence gets. It is why a model can interleave and
    /// still fit: only the attention blocks pay per position.
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
    /// One entry a block, in the order they run.
    pub blocks: Vec<BlockKind>,
}

impl Architecture {
    pub fn count(&self, kind: BlockKind) -> usize {
        self.blocks.iter().filter(|&&b| b == kind).count()
    }
}

/// What a backend got back out of the caches it keeps, against what it had to
/// make from scratch.
///
/// Counts rather than ratios, so a caller can show a rate over the whole run
/// or a change since it last looked. Every one of these rises towards a plateau
/// as a model warms up: the shapes a model uses are fixed once it is loaded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Kernels a launch found already compiled, and kernels it had to compile.
    /// Compiling is measured in seconds, so the second number should stop
    /// rising early and then stay still.
    pub kernels_reused: u64,
    pub kernels_compiled: u64,
    /// Device buffers taken off the pool's free list, and buffers that had to
    /// be allocated because nothing of that size was free.
    pub buffers_reused: u64,
    pub buffers_allocated: u64,
    /// Device memory in pooled buffers: handed out and in use, and released
    /// and waiting on the free list for a request of the same size.
    pub buffer_live_bytes: u64,
    pub buffer_idle_bytes: u64,
    /// For a model whose experts stream: experts a decode step wanted that
    /// were in the device cache, ones that were not and had to cross the
    /// bus, and the bytes that crossed for any pass. All zero for a model
    /// without experts.
    pub expert_hits: u64,
    pub expert_misses: u64,
    pub expert_bytes: u64,
    /// The same lookups by a prompt pass, which looks up an expert once for
    /// all the rows routed to it, so these count experts and not tokens.
    pub expert_prompt_hits: u64,
    pub expert_prompt_misses: u64,
    /// Experts a lookahead copied ahead of their block, and how many of
    /// those the block then wanted. Zero without a lookahead.
    pub expert_prefetches: u64,
    pub expert_prefetch_hits: u64,
    /// Misses computed on the host rather than copied, and the host time
    /// they took. Zero unless that path is on.
    pub expert_cpu_misses: u64,
    pub expert_cpu_nanos: u64,
}

impl CacheStats {
    /// Share of kernel lookups that found one ready, or `None` before the
    /// first lookup, which is not a rate of zero.
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
/// Everything here is fixed for the life of the process and read once. The
/// clocks are the card's maxima rather than what it is running at: an idle
/// card sits far below them, and nothing here samples a live one.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    /// Compute capability, major and minor.
    pub capability: (u32, u32),
    pub multiprocessors: u32,
    pub core_clock_khz: u32,
    pub memory_clock_khz: u32,
    pub memory_bus_bits: u32,
    /// The CUDA driver API version, major and minor. Not the version a driver
    /// release is known by, which is [`DeviceInfo::driver`].
    pub cuda: (u32, u32),
    /// The display driver's own version, when something could be asked for it.
    pub driver: Option<String>,
}

impl DeviceInfo {
    /// Peak memory bandwidth, in bytes a second: double data rate, times the
    /// bus width in bytes, times the clock.
    ///
    /// A ceiling from the card's specification rather than a measurement, and
    /// the number a decode is worth comparing against, since decoding reads
    /// every weight once per token and is bound by this long before it is
    /// bound by arithmetic.
    pub fn peak_bandwidth(&self) -> u64 {
        2 * (self.memory_bus_bits as u64 / 8) * self.memory_clock_khz as u64 * 1_000
    }
}

/// A device's memory as the driver reports it, which is the whole card and not
/// this process's share of it.
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

    /// What this model occupies, for a caller that only displays it. The
    /// default is `None`, so a front end that does not account for its own
    /// weights is not obliged to invent a number.
    fn footprint(&self) -> Option<Footprint> {
        None
    }

    /// Free and total bytes on the device this model computes on, read fresh
    /// on every call. A host backend reports nothing: it competes with the
    /// whole machine rather than with a fixed budget.
    fn device_memory(&self) -> Option<DeviceMemory> {
        None
    }

    /// The shape of the network, for a caller that displays it. Fixed at load.
    fn architecture(&self) -> Option<Architecture> {
        None
    }

    /// The card this model computes on. Fixed, and worth asking for once:
    /// finding the display driver's version may cost a subprocess.
    fn device_info(&self) -> Option<DeviceInfo> {
        None
    }

    /// What the backend's caches have returned so far, read fresh on every
    /// call. A backend that keeps no caches reports nothing.
    fn cache_stats(&self) -> Option<CacheStats> {
        None
    }

    /// Does now what the first request would otherwise wait for, such as
    /// putting the weights on the device. A caller runs it once, straight
    /// after the load; a backend with nothing to prepare does nothing.
    fn warm_up(&self) -> Result<()> {
        Ok(())
    }
}

pub trait Session {
    /// Run `ids` and return the logits for the position after the last one.
    ///
    /// The prompt is one call and each generated token another, so a caller
    /// never has to keep a model and a state in step by hand. An implementation
    /// is free to split a long call into batches; that is a property of the
    /// backend, not of the interface.
    fn extend(&mut self, ids: &[i64]) -> Result<Vec<f32>>;

    /// The batch a long [`Session::extend`] is split into, when the backend
    /// splits one. A caller can then hand the prompt over in pieces, a batch
    /// at a time to report progress between them at no cost, since the
    /// passes run are the same. `None`, the default, is a backend whose
    /// prompt has to arrive in one call.
    fn prompt_batch(&self) -> Option<usize> {
        None
    }

    /// [`Session::extend`] for a caller that only wants the winning token id,
    /// as greedy decoding does: `choose` over a one-element argmax is the
    /// identity, so skipping the full logits vector changes only how much a
    /// backend has to move, not the result. Purely additive, since the
    /// default is [`Session::extend`] plus a host-side argmax.
    fn extend_greedy(&mut self, ids: &[i64]) -> Result<i64> {
        Ok(crate::sampling::argmax(&self.extend(ids)?))
    }

    /// Tokens consumed so far, prompt included.
    fn len(&self) -> usize;

    /// Drop everything past `positions`, so the next [`Session::extend`]
    /// continues from there, and return how many positions the session
    /// kept. That may be fewer than asked: a session goes back only to
    /// points it can return to, and the caller runs the rest again. `None`
    /// is a session that could not go back at all, which the caller drops.
    ///
    /// Rewinding is what lets a session be kept and reused for a request that
    /// shares a prefix with the last one. Not every backend can: a recurrent
    /// block's state summarises every token it has seen rather than storing
    /// them by position, and there is nothing to subtract. It can only return
    /// to a [`Session::checkpoint`]. The default is a session with neither,
    /// and it still permits the case that asks for nothing to be dropped,
    /// which is how a session is extended rather than rewound.
    fn truncate(&mut self, positions: usize) -> Option<usize> {
        (positions == self.len()).then_some(positions)
    }

    /// Remember the current position as one [`Session::truncate`] can
    /// return to, in place of any remembered before. A session that rewinds
    /// to any position, or to none, has nothing to remember.
    fn checkpoint(&mut self) -> Result<()> {
        Ok(())
    }

    /// Bytes of key/value cache this session holds right now, which is what
    /// the backend reserved and not what [`Session::len`] would need: a cache
    /// that grows by doubling is mostly headroom just after it grows.
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
