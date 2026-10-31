// A node's instruction engine, on the thread that owns its CUDA context. It
// issues each instruction once its deps retire; the submodules carry the ops
// out: io LOAD, STORE and FETCH, launch COMPUTE, residency ALLOC and FREE.

mod io;
mod launch;
mod residency;

use std::collections::{HashMap, HashSet, VecDeque};
use std::thread;
use std::time::Duration;

use anyhow::Result;
use cust::stream::{Stream, StreamFlags};
use tokio::runtime::Handle;
use tokio::sync::{mpsc, oneshot};

use phobos_base::cli::format_size;
use phobos_base::{phdebug, phinfo};
use phobos_cluster::isa::{InstrId, Op, Segment};
use phobos_cluster::tile::{AccessMode, NodeId, TileId};

use crate::ptx_cache::PtxCache;
use crate::tile_store::TileStore;

pub use io::chunk_bytes;
use launch::Inflight;

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
    Cancel(Vec<InstrId>), // the scheduler retracts pending instructions, e.g. a FREE whose serve count changed
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
}

pub type EngineTx = mpsc::UnboundedSender<EngineMsg>;

pub type Completion = (InstrId, u32, u64);

struct Node {
    op: Op,
    remaining: usize,
    dependents: Vec<InstrId>,
}

const STREAMS: usize = 4;

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
    deferred_allocs: Vec<(InstrId, TileId, Vec<u64>)>, // ALLOCs parked on a full arena, retried after every FREE
    outstanding_io: usize,                             // in-flight LOAD/FETCH/STORE tasks
    deferred_frees: Vec<(InstrId, TileId)>,
    parked_serves: HashMap<TileId, Vec<oneshot::Sender<Option<Vec<f32>>>>>,
    kernel_waiters: HashMap<String, Vec<InstrId>>,
    fetching_kernel: HashSet<String>,

    completions: Vec<Completion>,
    complete_sink: mpsc::UnboundedSender<Completion>,
    fail_sink: mpsc::UnboundedSender<(InstrId, String)>,
}

impl Engine {
    pub fn spawn(
        rt: Handle,
        node_id: NodeId,
        arena_bytes: usize,
        complete_sink: mpsc::UnboundedSender<Completion>,
        fail_sink: mpsc::UnboundedSender<(InstrId, String)>,
    ) -> Result<EngineTx> {
        let (tx, rx) = mpsc::unbounded_channel();
        let to_self = tx.clone();
        let (boot_tx, boot_rx) = std::sync::mpsc::channel::<Result<()>>();
        thread::spawn(move || {
            // this thread owns the CUDA context for its whole lifetime
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
        complete_sink: mpsc::UnboundedSender<Completion>,
        fail_sink: mpsc::UnboundedSender<(InstrId, String)>,
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
        })
    }

    fn run(mut self, mut rx: mpsc::UnboundedReceiver<EngineMsg>) {
        loop {
            // Drain all queued messages without blocking.
            let mut msg_recvd = false;

            while let Ok(msg) = rx.try_recv() {
                msg_recvd = true;

                if let Err(e) = self.handle(msg) {
                    eprintln!("node{} engine error: {e:#}", self.node_id);
                }
            }

            self.poll_inflight();
            self.try_issue();
            self.flush_completions();

            if !msg_recvd && self.inflight.is_empty() && self.ready.is_empty() {
                self.check_alloc_stall();

                // nothing immediately runnable: yield briefly to avoid livelock
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
                    format_size(data.len() as u64 * 4),
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

        // Insert every node, then wire dep counters. Deps are node-local and
        // come earlier in topological order, so a dep in the table is unfinished.
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

        phinfo!("node{}: issue #{iid} {op}", self.node_id);

        match op {
            Op::Alloc { tile, shape, .. } => {
                // Park the ALLOC while the arena is full. A FREE that makes room retries it.
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

    fn fail(&mut self, iid: InstrId, msg: String) {
        eprintln!("node{} instr {iid} failed: {msg}", self.node_id);
        let _ = self.fail_sink.send((iid, msg));
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
        for c in self.completions.drain(..) {
            let _ = self.complete_sink.send(c);
        }
    }
}
