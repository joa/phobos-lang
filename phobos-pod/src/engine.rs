use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::c_void;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use cust::event::{Event, EventFlags, EventStatus};
use cust::stream::{Stream, StreamFlags};
use tokio::runtime::Handle;
use tokio::sync::{mpsc, oneshot};

use phobos_base::{phdebug, phinfo};
use phobos_cluster::isa::{InstrId, Op, ScalarArg, Segment, StorageRef};
use phobos_cluster::proto::tile_server_client::TileServerClient;
use phobos_cluster::proto::{KernelHash, TileRequest};
use phobos_cluster::storage;
use phobos_cluster::tile::{AccessMode, NodeId, TileId};

use crate::ptx_cache::PtxCache;
use crate::tile_store::TileStore;

#[derive(Clone)]
pub struct TensorMeta {
    pub uri: String,
    pub shape: Vec<u64>,
    pub mode: AccessMode,
}

pub enum EngineMsg {
    SetTensors(Vec<TensorMeta>),
    SetKernels {
        table: Vec<(u32, String, String)>, // (id, hash, entry name)
        holders: Vec<(String, NodeId)>,
    },
    Kernel {
        hash: String,
        ptx: String,
    },
    NodeAddrs(Vec<(NodeId, String)>),
    FetchLocs(Vec<(TileId, String)>),
    Issue(Segment),
    Cancel(Vec<InstrId>), // Scheduler may retract pending instructions (e.g. FREE when serve count changes)
    IoDone {
        iid: InstrId,
        tile: TileId,
        data: Vec<f32>,
    },
    StoreDone {
        iid: InstrId,
    },
    IoFailed {
        iid: InstrId,
        error: String,
    },
    KernelFetched {
        hash: String,
        ptx: String,
    },
    Serve {
        tile: TileId,
        reply: oneshot::Sender<Option<Vec<f32>>>,
    },
    GetKernel {
        hash: String,
        reply: oneshot::Sender<Option<String>>,
    },
    Shutdown,
}

pub type EngineTx = mpsc::UnboundedSender<EngineMsg>;

pub type Completion = (InstrId, u32, u64);

struct Node {
    op: Op,
    remaining: usize,
    dependents: Vec<InstrId>,
}

struct Inflight {
    event: Event,
    iid: InstrId,
    wrote: Option<TileId>,
}

const STREAMS: usize = 4;
const CHUNK: usize = 1 << 20; // 1 MiB tile-server chunks

pub struct Engine {
    rt: Handle,
    to_self: EngineTx,
    node_id: NodeId,

    store: TileStore,
    ptx: PtxCache,
    streams: Vec<Stream>,
    next_stream: usize,

    tensors: Vec<TensorMeta>,
    node_addrs: HashMap<NodeId, String>,
    fetch_locs: HashMap<TileId, String>,

    table: HashMap<InstrId, Node>,
    ready: VecDeque<InstrId>,
    inflight: Vec<Inflight>,
    deferred_allocs: Vec<(InstrId, TileId, Vec<u64>)>, // parked ALLOCs because arena is full; retried after every FREE
    outstanding_io: usize,                             // in-flight LOAD/FETCH/STORE tasks
    deferred_frees: Vec<(InstrId, TileId)>,
    parked_serves: HashMap<TileId, Vec<oneshot::Sender<Option<Vec<f32>>>>>,
    kernel_waiters: HashMap<String, Vec<InstrId>>,
    fetching_kernel: HashSet<String>,

    completions: Vec<Completion>,
    complete_sink: Option<mpsc::UnboundedSender<Completion>>,
    fail_sink: Option<mpsc::UnboundedSender<(InstrId, String)>>,
    idle: Option<oneshot::Sender<()>>,
    had_work: bool,
}

impl Engine {
    pub fn spawn(
        rt: Handle,
        node_id: NodeId,
        arena_bytes: usize,
        complete_sink: Option<mpsc::UnboundedSender<Completion>>,
        fail_sink: Option<mpsc::UnboundedSender<(InstrId, String)>>,
        idle: oneshot::Sender<()>,
    ) -> Result<EngineTx> {
        let (tx, rx) = mpsc::unbounded_channel();
        let to_self = tx.clone();
        let (boot_tx, boot_rx) = std::sync::mpsc::channel::<Result<()>>();
        thread::spawn(move || {
            // this thread owns the CUDA context for its whole lifetime!
            let _ctx = match cust::quick_init() {
                Ok(c) => c,
                Err(e) => {
                    let _ = boot_tx.send(Err(e.into()));
                    return;
                }
            };
            let engine = match Engine::new(
                rt,
                to_self,
                node_id,
                arena_bytes,
                complete_sink,
                fail_sink,
                idle,
            ) {
                Ok(e) => e,
                Err(e) => {
                    let _ = boot_tx.send(Err(e));
                    return;
                }
            };
            let _ = boot_tx.send(Ok(()));
            engine.run(rx);
        });
        boot_rx.recv().expect("engine boot")?;
        Ok(tx)
    }

    fn new(
        rt: Handle,
        to_self: EngineTx,
        node_id: NodeId,
        arena_bytes: usize,
        complete_sink: Option<mpsc::UnboundedSender<Completion>>,
        fail_sink: Option<mpsc::UnboundedSender<(InstrId, String)>>,
        idle: oneshot::Sender<()>,
    ) -> Result<Engine> {
        let streams = (0..STREAMS)
            .map(|_| Stream::new(StreamFlags::NON_BLOCKING, None))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Engine {
            rt,
            to_self,
            node_id,
            store: TileStore::new(arena_bytes)?,
            ptx: PtxCache::new(),
            streams,
            next_stream: 0,
            tensors: Vec::new(),
            node_addrs: HashMap::new(),
            fetch_locs: HashMap::new(),
            table: HashMap::new(),
            ready: VecDeque::new(),
            inflight: Vec::new(),
            deferred_allocs: Vec::new(),
            outstanding_io: 0,
            deferred_frees: Vec::new(),
            parked_serves: HashMap::new(),
            kernel_waiters: HashMap::new(),
            fetching_kernel: HashSet::new(),
            completions: Vec::new(),
            complete_sink,
            fail_sink,
            idle: Some(idle),
            had_work: false,
        })
    }

    fn run(mut self, mut rx: mpsc::UnboundedReceiver<EngineMsg>) {
        loop {
            // drain all queued messages without blocking.
            let mut msg_recvd = false;

            while let Ok(msg) = rx.try_recv() {
                msg_recvd = true;

                if matches!(msg, EngineMsg::Shutdown) {
                    return;
                }

                if let Err(e) = self.handle(msg) {
                    eprintln!("node{} engine error: {e:#}", self.node_id);
                }
            }

            self.poll_inflight();
            self.try_issue();
            self.flush_completions();
            self.check_idle();

            if !msg_recvd && self.inflight.is_empty() && self.ready.is_empty() {
                self.check_alloc_stall();

                // nothing immediately runnable: yield briefly so we don't livelock
                thread::sleep(Duration::from_micros(100));
            }
        }
    }

    fn handle(&mut self, msg: EngineMsg) -> Result<()> {
        match msg {
            EngineMsg::SetTensors(t) => self.tensors = t,
            EngineMsg::SetKernels { table, holders } => {
                self.ptx.set_table(table);
                self.ptx.set_holders(holders);
            }
            EngineMsg::Kernel { hash, ptx } => self.ptx.insert(&hash, &ptx)?,
            EngineMsg::NodeAddrs(a) => self.node_addrs.extend(a),
            EngineMsg::FetchLocs(l) => self.fetch_locs.extend(l),
            EngineMsg::Issue(seg) => self.add_segment(seg),
            EngineMsg::Cancel(iids) => self.cancel(iids),
            EngineMsg::IoDone { iid, tile, data } => {
                self.outstanding_io = self.outstanding_io.saturating_sub(1);
                self.store.h2d(tile, &data)?;
                phinfo!(
                    "node{}: H2D tile={:#x} {} uploaded to VRAM (resident)",
                    self.node_id,
                    tile.0,
                    human(data.len() * 4),
                );
                self.on_resident(tile)?;
                self.complete(iid)?;
            }
            EngineMsg::StoreDone { iid } => {
                self.outstanding_io = self.outstanding_io.saturating_sub(1);
                self.complete(iid)?;
            }
            EngineMsg::IoFailed { iid, error } => {
                self.outstanding_io = self.outstanding_io.saturating_sub(1);
                self.fail(iid, error);
            }
            EngineMsg::KernelFetched { hash, ptx } => {
                self.ptx.insert(&hash, &ptx)?;
                self.fetching_kernel.remove(&hash);
                for iid in self.kernel_waiters.remove(&hash).unwrap_or_default() {
                    self.ready.push_back(iid);
                }
            }
            EngineMsg::Serve { tile, reply } => self.on_serve(tile, reply)?,
            EngineMsg::GetKernel { hash, reply } => {
                let _ = reply.send(self.ptx.text_of(&hash));
            }
            EngineMsg::Shutdown => {}
        }
        Ok(())
    }

    fn add_segment(&mut self, seg: Segment) {
        phinfo!(
            "node{}: received segment {} ({} instrs)",
            self.node_id,
            seg.id,
            seg.instructions.len()
        );

        self.had_work = true;

        // add nodes, then wire dep counters.
        //
        // all deps are node-local and earlier in topological order,
        // so they are present and unfinished.
        for instr in &seg.instructions {
            self.table.insert(
                instr.iid,
                Node {
                    op: instr.op.clone(),
                    remaining: 0,
                    dependents: Vec::new(),
                },
            );
        }

        for instr in &seg.instructions {
            let mut remaining = 0;

            for d in &instr.deps {
                if let Some(dep) = self.table.get_mut(d) {
                    dep.dependents.push(instr.iid);
                    remaining += 1;
                }
            }

            self.table.get_mut(&instr.iid).unwrap().remaining = remaining;

            if remaining == 0 {
                self.ready.push_back(instr.iid);
            }
        }
    }

    fn cancel(&mut self, iids: Vec<InstrId>) {
        let set: HashSet<InstrId> = iids.into_iter().collect();

        self.ready.retain(|i| !set.contains(i));
        self.deferred_frees.retain(|(i, ..)| !set.contains(i));
        self.deferred_allocs.retain(|(i, ..)| !set.contains(i));

        for iid in set {
            let droppable = matches!(self.table.get(&iid), Some(n) if n.dependents.is_empty());

            if droppable {
                self.table.remove(&iid);
                phdebug!("node{}: cancelled #{iid}", self.node_id);
            } else if self.table.contains_key(&iid) {
                phdebug!(
                    "node{}: ignoring cancel of #{iid} (has dependents)",
                    self.node_id
                );
            }
        }
    }

    fn try_issue(&mut self) {
        while let Some(iid) = self.ready.pop_front() {
            if let Err(e) = self.issue(iid) {
                self.fail(iid, format!("{e:#}"));
            }
        }
    }

    fn issue(&mut self, iid: InstrId) -> Result<()> {
        let op = match self.table.get(&iid) {
            Some(n) => n.op.clone(),
            None => return Ok(()), // already retired (e.g. requeued spuriously)
        };

        phinfo!("node{}: issue #{iid} {}", self.node_id, op2log(&op));

        match op {
            Op::Alloc { tile, shape, .. } => {
                // node-side memory backpressure: if the arena is full, park the
                // ALLOC and retry it once a FREE makes room.
                if !self.try_alloc(iid, tile, shape)? {
                    phdebug!(
                        "node{}: ALLOC #{iid} tile={:#x} deferred: arena full",
                        self.node_id,
                        tile.0
                    );
                }
            }
            Op::Load { tile, src } => self.spawn_load(iid, tile, &src)?,
            Op::Fetch { tile, from } => self.spawn_fetch(iid, tile, from)?,
            Op::Compute {
                kernel,
                args,
                scalars,
                grid,
                cta,
            } => {
                self.launch(iid, kernel, &args, &scalars, grid, cta)?;
            }
            Op::Store { tile, dst } => self.spawn_store(iid, tile, &dst)?,
            Op::Free {
                tile,
                expected_serves,
            } => {
                self.store.set_expected_serves(tile, expected_serves);
                if self.try_free(iid, tile)? {
                    self.retry_allocs();
                }
            }
        }
        Ok(())
    }

    fn try_alloc(&mut self, iid: InstrId, tile: TileId, shape: Vec<u64>) -> Result<bool> {
        if !self.store.can_alloc(&shape) {
            self.deferred_allocs.push((iid, tile, shape));
            return Ok(false);
        }

        self.store.alloc(tile, shape)?;
        self.complete(iid)?;

        Ok(true)
    }

    fn try_free(&mut self, iid: InstrId, tile: TileId) -> Result<bool> {
        if !self.store.serves_satisfied(tile) {
            self.deferred_frees.push((iid, tile));
            return Ok(false);
        }

        self.store.free(tile)?;
        self.complete(iid)?;

        Ok(true)
    }

    fn spawn_load(&mut self, iid: InstrId, tile: TileId, src: &StorageRef) -> Result<()> {
        let StorageRef::Tensor { tensor, region } = src.clone();
        let meta = self
            .tensors
            .get(tensor as usize)
            .context("LOAD of unknown tensor")?
            .clone();
        let to = self.to_self.clone();
        let node = self.node_id;
        let uri = meta.uri.clone();

        self.outstanding_io += 1;
        self.rt.spawn(async move {
            let res = tokio::task::spawn_blocking(move || {
                storage::load_f32(&meta.uri, &meta.shape, &region)
            })
            .await
            .expect("load join");

            match res {
                Ok(data) => {
                    phinfo!(
                        "node{node}: LOAD #{iid} tile={:#x} read {} ({} f32) from {uri}",
                        tile.0,
                        human(data.len() * 4),
                        data.len(),
                    );
                    let _ = to.send(EngineMsg::IoDone { iid, tile, data });
                }
                Err(e) => {
                    let _ = to.send(EngineMsg::IoFailed {
                        iid,
                        error: format!("LOAD: {e:#}"),
                    });
                }
            }
        });
        Ok(())
    }

    fn spawn_store(&mut self, iid: InstrId, tile: TileId, dst: &StorageRef) -> Result<()> {
        let StorageRef::Tensor { tensor, region } = dst.clone();
        let meta = self
            .tensors
            .get(tensor as usize)
            .context("STORE of unknown tensor")?
            .clone();

        // D2H happens on the engine thread; the file write is offloaded
        let data = self.store.d2h(tile)?;
        phinfo!(
            "node{}: D2H tile={:#x} {} read back from VRAM, writing to {}",
            self.node_id,
            tile.0,
            human(data.len() * 4),
            meta.uri,
        );

        let to = self.to_self.clone();
        let node = self.node_id;
        let uri = meta.uri.clone();
        let nbytes = data.len() * 4;

        self.outstanding_io += 1;
        self.rt.spawn(async move {
            let res = tokio::task::spawn_blocking(move || {
                storage::store_f32(&meta.uri, &meta.shape, &region, &data)
            })
            .await
            .expect("store join");

            match res {
                Ok(()) => {
                    phinfo!("node{node}: STORE #{iid} wrote {} to {uri}", human(nbytes));
                    let _ = to.send(EngineMsg::StoreDone { iid });
                }
                Err(e) => {
                    let _ = to.send(EngineMsg::IoFailed {
                        iid,
                        error: format!("STORE: {e:#}"),
                    });
                }
            }
        });
        Ok(())
    }

    fn spawn_fetch(&mut self, iid: InstrId, tile: TileId, from: NodeId) -> Result<()> {
        let addr = self
            .fetch_locs
            .get(&tile)
            .cloned()
            .or_else(|| self.node_addrs.get(&from).cloned())
            .with_context(|| format!("no peer address for FETCH of tile {:#x}", tile.0))?;
        let to = self.to_self.clone();
        let node = self.node_id;
        self.outstanding_io += 1;
        self.rt.spawn(async move {
            match fetch_tile(&addr, tile).await {
                Ok(data) => {
                    phinfo!(
                        "node{node}: FETCH #{iid} tile={:#x} received {} from peer {addr}",
                        tile.0,
                        human(data.len() * 4),
                    );
                    let _ = to.send(EngineMsg::IoDone { iid, tile, data });
                }
                Err(e) => {
                    let _ = to.send(EngineMsg::IoFailed {
                        iid,
                        error: format!("FETCH from {addr}: {e:#}"),
                    });
                }
            }
        });
        Ok(())
    }

    fn launch(
        &mut self,
        iid: InstrId,
        kernel: u32,
        args: &[(TileId, AccessMode)],
        scalars: &[ScalarArg],
        grid: (u32, u32, u32),
        cta: (u32, u32, u32),
    ) -> Result<()> {
        // PTX may not be cached yet; fill it from a peer, parking this iid.
        if self.ptx.function(kernel)?.is_none() {
            let hash = self.ptx.hash_of(kernel)?.to_string();
            self.kernel_waiters
                .entry(hash.clone())
                .or_default()
                .push(iid);
            if self.fetching_kernel.insert(hash.clone()) {
                let holder = self
                    .ptx
                    .holder_of(&hash)
                    .with_context(|| format!("kernel {hash} missing with no holder"))?;
                let addr = self
                    .node_addrs
                    .get(&holder)
                    .cloned()
                    .with_context(|| format!("no address for kernel holder node {holder}"))?;
                let to = self.to_self.clone();
                let h = hash.clone();
                self.rt.spawn(async move {
                    match get_kernel(&addr, &h).await {
                        Ok(ptx) => {
                            let _ = to.send(EngineMsg::KernelFetched { hash: h, ptx });
                        }
                        Err(e) => eprintln!("GetKernel {h} from {addr} failed: {e:#}"),
                    }
                });
            }
            return Ok(());
        }

        // marshal the exploded-memref ABI (phobos-mlir, index_bitwidth = 32):
        //   each f32 tile -> (alloc_ptr, align_ptr, 0i32, sizes..., strides...).
        let mut addrs: Vec<u64> = Vec::with_capacity(args.len());
        let mut ints: Vec<i32> = Vec::new();
        let mut wrote = None;

        for (tile, mode) in args {
            addrs.push(self.store.addr(*tile)?);
            let shape = self.store.shape(*tile)?;
            ints.push(0); // offset
            for &s in &shape {
                ints.push(s as i32);
            }
            let mut stride = 1i64;
            let mut strides = vec![0i32; shape.len()];
            for (i, &s) in shape.iter().enumerate().rev() {
                strides[i] = stride as i32;
                stride *= s as i64;
            }
            ints.extend_from_slice(&strides);
            if matches!(mode, AccessMode::Write | AccessMode::RMW) {
                wrote = Some(*tile);
            }
        }

        let scalar_bits: Vec<u64> = scalars.iter().map(|s| s.value.to_bits()).collect();
        let total = args.len() + scalars.len();
        let mut raw: Vec<*mut c_void> = Vec::new();
        let mut int_at = 0;
        let mut next_tile = 0;
        for pos in 0..total as u32 {
            if let Some(si) = scalars.iter().position(|s| s.pos == pos) {
                raw.push(&scalar_bits[si] as *const u64 as *mut c_void);
                continue;
            }
            let (tile, _) = &args[next_tile];
            let rank = self.store.shape(*tile)?.len();
            let p = &addrs[next_tile] as *const u64 as *mut c_void;

            raw.push(p); // alloc_ptr
            raw.push(p); // align_ptr

            for _ in 0..1 + 2 * rank {
                raw.push(&ints[int_at] as *const i32 as *mut c_void);
                int_at += 1;
            }

            next_tile += 1;
        }

        let stream = &self.streams[self.next_stream];
        self.next_stream = (self.next_stream + 1) % self.streams.len();
        let func = self
            .ptx
            .function(kernel)?
            .expect("function present (checked above)");
        let event = Event::new(EventFlags::DEFAULT)?;

        // SAFETY: raw points into addrs/ints/scalar_bits which outlive this
        // call; the ABI matches phobos-mlir's exploded memref marshalling, with
        // scalar params as plain value args interleaved by parameter position.
        unsafe {
            stream
                .launch(&func, grid, cta, 0, &raw)
                .context("kernel launch")?;
        }

        event.record(stream)?;
        self.inflight.push(Inflight { event, iid, wrote });

        Ok(())
    }

    fn poll_inflight(&mut self) {
        let mut done = Vec::new();
        let mut i = 0;

        while i < self.inflight.len() {
            match self.inflight[i].event.query() {
                Ok(EventStatus::Ready) => {
                    let f = self.inflight.swap_remove(i);
                    done.push((f.iid, f.wrote));
                }
                Ok(EventStatus::NotReady) => i += 1,
                Err(e) => {
                    eprintln!("node{} event query failed: {e}", self.node_id);
                    i += 1;
                }
            }
        }

        for (iid, wrote) in done {
            if let Some(tile) = wrote
                && let Err(e) = self
                    .store
                    .mark_resident(tile)
                    .and_then(|_| self.on_resident(tile))
            {
                eprintln!("node{} post-compute residency error: {e:#}", self.node_id);
            }

            if let Err(e) = self.complete(iid) {
                eprintln!("node{} complete {iid} error: {e:#}", self.node_id);
            }
        }
    }

    fn on_resident(&mut self, tile: TileId) -> Result<()> {
        if let Some(waiters) = self.parked_serves.remove(&tile) {
            for reply in waiters {
                self.serve_now(tile, reply)?;
            }
        }
        Ok(())
    }

    fn on_serve(&mut self, tile: TileId, reply: oneshot::Sender<Option<Vec<f32>>>) -> Result<()> {
        if self.store.is_resident(tile) {
            self.serve_now(tile, reply)?;
        } else {
            self.parked_serves.entry(tile).or_default().push(reply);
        }
        Ok(())
    }

    fn serve_now(&mut self, tile: TileId, reply: oneshot::Sender<Option<Vec<f32>>>) -> Result<()> {
        let data = self.store.d2h(tile)?;
        phinfo!(
            "node{}: SERVE tile={:#x} {} -> peer (D2H)",
            self.node_id,
            tile.0,
            human(data.len() * 4),
        );
        let _ = reply.send(Some(data));
        self.store.record_serve(tile)?;
        self.retry_frees();
        Ok(())
    }

    fn retry_frees(&mut self) {
        let mut freed_any = false;

        for (iid, tile) in std::mem::take(&mut self.deferred_frees) {
            match self.try_free(iid, tile) {
                Ok(freed) => freed_any |= freed,
                Err(e) => self.fail(iid, format!("deferred free: {e:#}")),
            }
        }

        if freed_any {
            self.retry_allocs();
        }
    }

    fn retry_allocs(&mut self) {
        for (iid, tile, shape) in std::mem::take(&mut self.deferred_allocs) {
            if let Err(e) = self.try_alloc(iid, tile, shape) {
                self.fail(iid, format!("alloc retry: {e:#}"));
            }
        }
    }

    fn fail(&mut self, iid: InstrId, msg: String) {
        eprintln!("node{} instr {iid} failed: {msg}", self.node_id);

        if let Some(s) = &self.fail_sink {
            let _ = s.send((iid, msg));
        }
    }

    fn check_alloc_stall(&mut self) {
        if self.deferred_allocs.is_empty()
            || self.outstanding_io != 0
            || !self.deferred_frees.is_empty()
            || !self.fetching_kernel.is_empty()
        {
            return;
        }

        let stuck = std::mem::take(&mut self.deferred_allocs);
        for (iid, tile, shape) in stuck {
            let bytes = shape.iter().product::<u64>() as usize * 4;
            self.fail(
                iid,
                format!(
                    "device arena exhausted: tile {:#x} needs {} but the resident working \
                     set already fills the arena; raise the node's --arena, or lower \
                     SUPER_*/the scheduler memory budget",
                    tile.0,
                    human(bytes),
                ),
            );
        }
    }

    fn complete(&mut self, iid: InstrId) -> Result<()> {
        let node = match self.table.remove(&iid) {
            Some(n) => n,
            None => return Ok(()),
        };
        for d in node.dependents {
            if let Some(dep) = self.table.get_mut(&d) {
                dep.remaining -= 1;
                if dep.remaining == 0 {
                    self.ready.push_back(d);
                }
            }
        }
        self.completions.push((iid, 0, 0));
        phdebug!("node{}: #{iid} complete", self.node_id);
        Ok(())
    }

    fn flush_completions(&mut self) {
        if self.completions.is_empty() {
            return;
        }
        if let Some(sink) = &self.complete_sink {
            for c in self.completions.drain(..) {
                let _ = sink.send(c);
            }
        } else {
            self.completions.clear();
        }
    }

    fn check_idle(&mut self) {
        if self.had_work
            && self.table.is_empty()
            && self.inflight.is_empty()
            && self.ready.is_empty()
            && self.deferred_frees.is_empty()
            && self.deferred_allocs.is_empty()
            && let Some(tx) = self.idle.take()
        {
            let _ = tx.send(());
        }
    }
}

async fn fetch_tile(addr: &str, tile: TileId) -> Result<Vec<f32>> {
    let mut client = TileServerClient::connect(format!("http://{addr}")).await?;
    let resp = client
        .fetch_tile(TileRequest {
            tile: tile.0,
            deadline_ms: 30_000,
        })
        .await?;
    let mut stream = resp.into_inner();
    let mut bytes: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.message().await? {
        let off = chunk.offset as usize;
        if bytes.len() < off + chunk.data.len() {
            bytes.resize(off + chunk.data.len(), 0);
        }
        bytes[off..off + chunk.data.len()].copy_from_slice(&chunk.data);
    }
    storage::le_bytes_to_f32(&bytes)
}

async fn get_kernel(addr: &str, hash: &str) -> Result<String> {
    let mut client = TileServerClient::connect(format!("http://{addr}")).await?;
    let resp = client
        .get_kernel(KernelHash {
            hash: hash.to_string(),
        })
        .await?;
    let blob = resp.into_inner();
    if blob.ptx.is_empty() {
        bail!("peer returned empty PTX for {hash}");
    }
    Ok(blob.ptx)
}

fn human(bytes: usize) -> String {
    const MIB: usize = 1 << 20;
    const KIB: usize = 1 << 10;
    if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

fn op2log(op: &Op) -> String {
    match op {
        Op::Alloc { tile, shape, .. } => format!("ALLOC tile={:#x} shape={shape:?}", tile.0),
        Op::Load { tile, .. } => format!("LOAD tile={:#x}", tile.0),
        Op::Fetch { tile, from } => format!("FETCH tile={:#x} from=node{from}", tile.0),
        Op::Compute {
            kernel,
            args,
            scalars,
            grid,
            cta,
        } => format!(
            "COMPUTE k{kernel} grid={grid:?} cta={cta:?} tiles=[{}]{}",
            args.iter()
                .map(|(t, m)| format!("{:#x}:{m:?}", t.0))
                .collect::<Vec<_>>()
                .join(" "),
            if scalars.is_empty() {
                String::new()
            } else {
                format!(
                    " scalars=[{}]",
                    scalars
                        .iter()
                        .map(|s| format!("@{}={:?}", s.pos, s.value))
                        .collect::<Vec<_>>()
                        .join(" ")
                )
            }
        ),
        Op::Store { tile, .. } => format!("STORE tile={:#x}", tile.0),
        Op::Free {
            tile,
            expected_serves,
        } => {
            format!("FREE tile={:#x} serves={expected_serves}", tile.0)
        }
    }
}

pub fn chunk_bytes(data: &[f32]) -> Vec<(u64, Vec<u8>)> {
    let bytes = storage::f32_to_le_bytes(data);
    bytes
        .chunks(CHUNK)
        .enumerate()
        .map(|(i, c)| ((i * CHUNK) as u64, c.to_vec()))
        .collect()
}
