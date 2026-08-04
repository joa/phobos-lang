use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use tokio::sync::Notify;
use tokio_stream::Stream;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::{Request, Response, Status, Streaming};

use phobos_base::{phdebug, phinfo};
use phobos_cluster::isa::{InstrId, Op, Segment};
use phobos_cluster::proto::scheduler_server::{Scheduler as SchedulerSvc, SchedulerServer};
use phobos_cluster::proto::{
    self, IssueSegment, Job, JobDone, JobEvent, JobManifest, Kernel, KernelEntry, KernelHolder,
    Manifest, NodeMessage, SchedulerMessage, TensorEntry, TileLocation,
};
use phobos_cluster::tile::{NodeId, ScalarValue};

use crate::autotune::ClusterFingerprint;
use crate::{IngestPolicy, Plan, default_supers, plan_budgeted_with, recover_plan, validate};
use phobos_cluster::ir::ClusterProgram;

#[derive(Clone, Default)]
pub struct DispatchConfig {
    pub nodes: u16,

    pub withhold: Vec<(NodeId, u32)>,

    /// Per-node memory budget for segment sizing.
    ///
    /// Won't let in-flight segments exceed this many bytes.
    pub budget_bytes: Option<u64>,

    pub autotune: Option<ClusterFingerprint>,

    pub ingest: IngestPolicy,
}

struct NodeConn {
    tx: tokio::sync::mpsc::UnboundedSender<Result<SchedulerMessage, Status>>,
    addr: String,
}

struct ActivePlan {
    plan: Plan,
    next_seg: Vec<usize>,
    inflight: Vec<u64>,
    seg_remaining: Vec<Vec<usize>>,
}

impl ActivePlan {
    fn new(plan: Plan) -> ActivePlan {
        let n = plan.node_segments.len();
        let seg_remaining = plan
            .node_segments
            .iter()
            .map(|segs| segs.iter().map(|s| s.instructions.len()).collect())
            .collect();

        ActivePlan {
            next_seg: vec![0; n],
            inflight: vec![0; n],
            seg_remaining,
            plan,
        }
    }
}

fn register_plan(
    plan_inst: usize,
    plan: &Plan,
    iid_seg: &mut HashMap<InstrId, (usize, usize, usize)>,
    store_of: &mut HashMap<InstrId, (usize, u64)>,
) {
    for (node, segs) in plan.node_segments.iter().enumerate() {
        for (si, seg) in segs.iter().enumerate() {
            for instr in &seg.instructions {
                iid_seg.insert(instr.iid, (plan_inst, node, si));
            }
        }
    }

    for &(iid, _, t, lin) in &plan.stores {
        store_of.insert(iid, (t, lin));
    }
}

#[derive(Default)]
struct State {
    nodes: HashMap<NodeId, NodeConn>,
    last_seen: HashMap<NodeId, Instant>,
    completed_iids: Vec<InstrId>,
    failure: Option<String>,
    down: Vec<NodeId>,
}

struct Inner {
    state: Mutex<State>,
    completed: AtomicU64,
    registered: Notify,
    complete: Notify,
}

#[derive(Clone)]
pub struct Scheduler {
    inner: Arc<Inner>, //TODO(joa): fix me, this is ugly
}

impl Scheduler {
    pub fn new() -> Scheduler {
        Scheduler {
            inner: Arc::new(Inner {
                state: Mutex::new(State::default()),
                completed: AtomicU64::new(0),
                registered: Notify::new(),
                complete: Notify::new(),
            }),
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.inner.state.lock().unwrap()
    }

    pub fn into_server(self) -> SchedulerServer<Scheduler> {
        SchedulerServer::new(self)
    }

    fn node_count(&self) -> usize {
        self.state().nodes.len()
    }

    fn node_addrs(&self) -> HashMap<NodeId, String> {
        self.state()
            .nodes
            .iter()
            .map(|(id, c)| (*id, c.addr.clone()))
            .collect()
    }

    fn send(&self, node: NodeId, payload: proto::scheduler_message::Payload) -> Result<()> {
        let state = self.state();
        let conn = state
            .nodes
            .get(&node)
            .with_context(|| format!("node {node} not registered"))?;

        conn.tx
            .send(Ok(SchedulerMessage {
                payload: Some(payload),
            }))
            .map_err(|_| anyhow::anyhow!("node {node} stream closed"))?;

        Ok(())
    }

    async fn await_nodes(&self, n: usize) {
        loop {
            let fut = self.inner.registered.notified();

            if self.node_count() >= n {
                break;
            }

            fut.await;
        }
    }

    fn ship_ready(
        &self,
        node: usize,
        plan: &Plan,
        budget: u64,
        next_seg: &mut [usize],
        inflight: &mut [u64],
    ) -> Result<()> {
        let segs = &plan.node_segments[node];

        while next_seg[node] < segs.len() {
            let si = next_seg[node];
            let inc = plan.segment_mem[node][si].incremental;

            if inflight[node] != 0 && inflight[node] + inc > budget {
                phdebug!(
                    "sched: node{node} backpressure: holding segment {} (inflight {}MiB + {}MiB > budget {}MiB)",
                    segs[si].id,
                    inflight[node] >> 20,
                    inc >> 20,
                    budget >> 20,
                );
                break;
            }

            phinfo!(
                "sched: -> node{node} issue segment {} ({} instrs: {}; incr {}MiB)",
                segs[si].id,
                segs[si].instructions.len(),
                seg_summary(&segs[si]),
                inc >> 20,
            );

            self.send(
                node as NodeId,
                proto::scheduler_message::Payload::IssueSegment(IssueSegment {
                    segment: Some(proto::segment_to_proto(&segs[si])),
                }),
            )?;

            inflight[node] += inc;
            next_seg[node] += 1;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn recover(
        &self,
        program: &ClusterProgram,
        dims: &HashMap<String, i64>,
        supers: &HashMap<String, i64>,
        width: u16,
        budget: u64,
        down_set: &HashSet<NodeId>,
        durable: &HashSet<(usize, u64)>,
        recovering: &mut HashSet<(usize, u64)>,
        recovery_gen: &mut u16,
        iid_high: &mut InstrId,
        total: &mut u64,
        plans: &mut Vec<ActivePlan>,
        iid_seg: &mut HashMap<InstrId, (usize, usize, usize)>,
        store_of: &mut HashMap<InstrId, (usize, u64)>,
        scalars: &HashMap<String, ScalarValue>,
    ) -> Result<()> {
        if down_set.len() >= width as usize {
            bail!("dispatch aborted: every node failed, nothing left to recover onto");
        }

        let mut exclude = durable.clone();
        exclude.extend(recovering.iter().copied());

        let dead: Vec<NodeId> = down_set.iter().copied().collect();

        *recovery_gen += 1;

        let rec = recover_plan(
            program,
            dims,
            supers,
            width,
            &dead,
            &exclude,
            budget,
            IngestPolicy::DirectLoad,
            *recovery_gen,
            *iid_high,
            scalars,
        )?;

        validate(&rec)?;

        let extra = rec.total_instrs();
        if extra == 0 {
            phinfo!("sched: node failure left no outstanding work to recover");
            return Ok(());
        }

        let plan_inst = plans.len();

        phinfo!(
            "sched: recovery gen {}: re-running {} lost output chain(s) ({extra} instr) on survivors",
            *recovery_gen,
            rec.output_owner.len(),
        );

        register_plan(plan_inst, &rec, iid_seg, store_of);

        for key in rec.output_owner.keys() {
            recovering.insert(*key);
        }

        *iid_high = (*iid_high).max(rec.max_iid());
        *total += extra;
        let mut ap = ActivePlan::new(rec);

        for node in 0..width as usize {
            if down_set.contains(&(node as NodeId)) {
                continue; // a dead node's recovery segments are empty anyway
            }
            self.ship_ready(node, &ap.plan, budget, &mut ap.next_seg, &mut ap.inflight)?;
        }

        plans.push(ap);
        Ok(())
    }

    fn spawn_watchdog(&self) -> tokio::task::JoinHandle<()> {
        const DEADLINE: std::time::Duration = std::time::Duration::from_secs(8);

        let inner = self.inner.clone();

        tokio::spawn(async move {
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(1));
            loop {
                tick.tick().await;

                let now = Instant::now();
                let silent: Vec<NodeId> = {
                    let st = inner.state.lock().unwrap();
                    st.last_seen
                        .iter()
                        .filter(|&(_, &t)| now.duration_since(t) > DEADLINE)
                        .map(|(&n, _)| n)
                        .collect()
                };

                for n in &silent {
                    {
                        let mut st = inner.state.lock().unwrap();
                        st.nodes.remove(n);
                        st.last_seen.remove(n);
                        st.down.push(*n);
                    }
                    phinfo!("sched: node{n} missed the heartbeat deadline (marked down)");
                }

                if !silent.is_empty() {
                    inner.complete.notify_waiters();
                }
            }
        })
    }

    pub async fn dispatch(&self, job: Job, cfg: DispatchConfig) -> Result<Vec<String>> {
        let kernel = phobos_lang::parse(&job.source)?
            .into_iter()
            .next()
            .context("empty source")?;
        let program = phobos_cluster::compile(&kernel)?;
        let dims: HashMap<String, i64> = job
            .dimensions
            .iter()
            .map(|d| (d.name.clone(), d.value))
            .collect();
        let scalars: HashMap<String, ScalarValue> = job
            .scalars
            .iter()
            .map(|s| {
                let v = proto::scalar_value_from_proto(
                    s.value.as_ref().context("scalar binding missing value")?,
                )?;
                Ok((s.name.clone(), v))
            })
            .collect::<Result<_>>()?;

        let supers = match cfg.autotune {
            Some(mut fp) => {
                fp.nodes = cfg.nodes;
                crate::autotune::best(&program, &dims, fp)?
            }
            None => default_supers(&program),
        };

        let budget = cfg.budget_bytes.unwrap_or(u64::MAX);
        let plan = plan_budgeted_with(
            &program, &dims, &supers, cfg.nodes, budget, cfg.ingest, &scalars,
        )?;

        validate(&plan)?;

        let mut supers_sorted: Vec<_> = supers.iter().collect();

        supers_sorted.sort();

        let seg_counts: Vec<usize> = plan.node_segments.iter().map(|s| s.len()).collect();

        phinfo!(
            "sched: planned '{}' supers={:?} ingest={:?} budget={} nodes={} segments/node={:?} peak={}MiB fetch={}MiB",
            program.name,
            supers_sorted,
            cfg.ingest,
            if budget == u64::MAX {
                "none".into()
            } else {
                format!("{}MiB", budget >> 20)
            },
            cfg.nodes,
            seg_counts,
            plan.peak_resident >> 20,
            plan.fetch_bytes >> 20,
        );

        let leaves = compile_leaves(&program)?; // (hash, name, ptx) by kernel id

        phinfo!("sched: compiled {} leaf kernel(s)", leaves.len());

        for (i, (hash, name, _)) in leaves.iter().enumerate() {
            phdebug!(
                "sched:   leaf {i} '{name}' = {}",
                &hash[..16.min(hash.len())]
            );
        }

        let tensors = program
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let jt = job
                    .tensors
                    .iter()
                    .find(|x| x.name == t.name)
                    .with_context(|| format!("job is missing tensor '{}'", t.name))?;
                Ok(TensorEntry {
                    index: i as u32,
                    uri: jt.uri.clone(),
                    data_type: proto::data_type_to_i32(t.data_type),
                    shape: jt.shape.clone(),
                    mode: proto::am_to_i32(t.mode),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let kernels: Vec<KernelEntry> = leaves
            .iter()
            .enumerate()
            .map(|(i, (hash, name, _))| KernelEntry {
                index: i as u32,
                hash: hash.clone(),
                name: name.clone(),
            })
            .collect();

        let holders: Vec<KernelHolder> = cfg
            .withhold
            .iter()
            .map(|(_, kidx)| KernelHolder {
                hash: leaves[*kidx as usize].0.clone(),
                node_id: 0,
            })
            .collect();

        phinfo!("sched: waiting for {} node(s) to register", cfg.nodes);

        self.await_nodes(cfg.nodes as usize).await;

        let addrs = self.node_addrs();

        phinfo!("sched: {} node(s) registered: {:?}", addrs.len(), {
            let mut a: Vec<_> = addrs
                .iter()
                .map(|(n, ad)| format!("node{n}={ad}"))
                .collect();
            a.sort();
            a
        });

        self.inner.completed.store(0, Ordering::SeqCst);
        {
            let mut st = self.state();
            st.completed_iids.clear();
            st.failure = None;
            st.down.clear();
        }

        for node_id in 0..cfg.nodes as usize {
            let nid = node_id as NodeId;
            self.send(
                nid,
                proto::scheduler_message::Payload::JobManifest(JobManifest {
                    tensors: tensors.clone(),
                    kernels: kernels.clone(),
                    kernel_holders: holders.clone(),
                }),
            )?;

            for (i, (hash, _, ptx)) in leaves.iter().enumerate() {
                if cfg
                    .withhold
                    .iter()
                    .any(|(wn, wk)| *wn == nid && *wk == i as u32)
                {
                    continue;
                }
                self.send(
                    nid,
                    proto::scheduler_message::Payload::Kernel(Kernel {
                        hash: hash.clone(),
                        ptx: ptx.clone(),
                    }),
                )?;
            }

            let tiles = plan.fetches[node_id]
                .iter()
                .map(|(tile, home)| TileLocation {
                    tile: tile.0,
                    node_id: *home as u32,
                    address: addrs.get(home).cloned().unwrap_or_default(),
                })
                .collect();

            self.send(
                nid,
                proto::scheduler_message::Payload::Manifest(Manifest { tiles }),
            )?;

            phdebug!(
                "sched: -> node{node_id} setup: {} tensors, {} kernels, {} fetch manifest entries",
                tensors.len(),
                kernels.len(),
                plan.fetches[node_id].len(),
            );
        }

        let width = cfg.nodes;
        let mut plans: Vec<ActivePlan> = vec![ActivePlan::new(plan)];
        let mut iid_seg: HashMap<InstrId, (usize, usize, usize)> = HashMap::new();
        let mut store_of: HashMap<InstrId, (usize, u64)> = HashMap::new();

        register_plan(0, &plans[0].plan, &mut iid_seg, &mut store_of);

        let mut total = plans[0].plan.total_instrs();
        let mut iid_high = plans[0].plan.max_iid();

        for node in 0..width as usize {
            let ap = &mut plans[0];
            self.ship_ready(node, &ap.plan, budget, &mut ap.next_seg, &mut ap.inflight)?;
        }

        let mut down_set: HashSet<NodeId> = HashSet::new();
        let mut durable: HashSet<(usize, u64)> = HashSet::new();
        let mut recovering: HashSet<(usize, u64)> = HashSet::new();
        let mut recovery_gen: u16 = 0;
        let watchdog = self.spawn_watchdog();

        let mut retired: u64 = 0;
        let mut abandoned: u64 = 0;
        let result: Result<()> = 'drain: loop {
            if retired + abandoned >= total {
                break Ok(());
            }
            if let Some(err) = self.state().failure.take() {
                break Err(anyhow::anyhow!("dispatch aborted: {err}"));
            }

            let downs: Vec<NodeId> = std::mem::take(&mut self.state().down);
            let mut newly_down = false;

            for d in downs {
                if !down_set.insert(d) {
                    continue;
                }

                newly_down = true;

                for ap in &mut plans {
                    let segs = &ap.seg_remaining[d as usize];
                    abandoned += segs.iter().map(|&r| r as u64).sum::<u64>();

                    for r in &mut ap.seg_remaining[d as usize] {
                        *r = 0;
                    }

                    ap.next_seg[d as usize] = ap.plan.node_segments[d as usize].len();
                }

                phinfo!(
                    "sched: node{d} down; {} of {width} node(s) left",
                    width as usize - down_set.len()
                );
            }

            if newly_down {
                if let Err(e) = self.recover(
                    &program,
                    &dims,
                    &supers,
                    width,
                    budget,
                    &down_set,
                    &durable,
                    &mut recovering,
                    &mut recovery_gen,
                    &mut iid_high,
                    &mut total,
                    &mut plans,
                    &mut iid_seg,
                    &mut store_of,
                    &scalars,
                ) {
                    break Err(e);
                }
                continue; // re-check termination with the updated totals
            }

            let fut = self.inner.complete.notified();
            let done: Vec<InstrId> = std::mem::take(&mut self.state().completed_iids);

            if done.is_empty() {
                fut.await;
                continue;
            }

            for iid in done {
                let Some(&(pi, node, si)) = iid_seg.get(&iid) else {
                    continue;
                };

                if down_set.contains(&(node as NodeId)) {
                    continue; // already abandoned with the node
                }

                if let Some(key) = store_of.get(&iid) {
                    durable.insert(*key);
                    recovering.remove(key);
                }

                retired += 1;

                let ap = &mut plans[pi];
                let r = &mut ap.seg_remaining[node][si];

                if *r == 0 {
                    continue;
                }

                *r -= 1;

                if *r == 0 {
                    ap.inflight[node] -= ap.plan.segment_mem[node][si].incremental;

                    if let Err(e) =
                        self.ship_ready(node, &ap.plan, budget, &mut ap.next_seg, &mut ap.inflight)
                    {
                        break 'drain Err(e);
                    }
                }
            }
        };

        watchdog.abort();
        result?;

        phinfo!(
            "sched: all {total} instructions accounted ({retired} retired, {abandoned} abandoned)"
        );

        Ok(job
            .tensors
            .iter()
            .filter(|t| {
                matches!(
                    proto::am_from_i32(t.mode),
                    Ok(phobos_cluster::tile::AccessMode::Write
                        | phobos_cluster::tile::AccessMode::RMW)
                )
            })
            .map(|t| t.uri.clone())
            .collect())
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Scheduler::new()
    }
}

fn seg_summary(seg: &Segment) -> String {
    const NAMES: [&str; 6] = ["ALLOC", "LOAD", "FETCH", "COMPUTE", "STORE", "FREE"];
    let mut counts = [0u32; 6];
    for i in &seg.instructions {
        counts[match i.op {
            Op::Alloc { .. } => 0,
            Op::Load { .. } => 1,
            Op::Fetch { .. } => 2,
            Op::Compute { .. } => 3,
            Op::Store { .. } => 4,
            Op::Free { .. } => 5,
        }] += 1;
    }
    NAMES
        .iter()
        .zip(counts)
        .filter(|(_, n)| *n > 0)
        .map(|(name, n)| format!("{n} {name}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn compile_leaves(
    program: &phobos_cluster::ir::ClusterProgram,
) -> Result<Vec<(String, String, String)>> {
    let base = phobos_base::context::Context::default();
    program
        .leaves
        .iter()
        .map(|l| {
            let ptx = phobos_mlir::gen_ptx(&base, |b, c, m| {
                phobos_lang::codegen::emit(b, std::slice::from_ref(&l.kernel), c, m).map(|_| ())
            })?;
            let mut h = Sha256::new();
            h.update(ptx.as_bytes());
            let hash = format!("{:x}", h.finalize());
            Ok((hash, l.kernel.name.clone(), ptx)) // name -> entrypoint
        })
        .collect()
}

// gRPC service

type SchedulerMessageStream =
    Pin<Box<dyn Stream<Item = Result<SchedulerMessage, Status>> + Send + 'static>>;
type JobEventStream = Pin<Box<dyn Stream<Item = Result<JobEvent, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl SchedulerSvc for Scheduler {
    type AttachStream = SchedulerMessageStream;
    type SubmitStream = JobEventStream;

    async fn attach(
        &self,
        req: Request<Streaming<NodeMessage>>,
    ) -> Result<Response<Self::AttachStream>, Status> {
        let mut inbound = req.into_inner();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let inner = self.inner.clone();
        tokio::spawn(async move {
            let mut node: Option<NodeId> = None;
            while let Ok(Some(msg)) = inbound.message().await {
                let Some(payload) = msg.payload else { continue };
                if let Some(n) = node {
                    inner
                        .state
                        .lock()
                        .unwrap()
                        .last_seen
                        .insert(n, Instant::now());
                }
                match payload {
                    proto::node_message::Payload::Register(r) => {
                        phinfo!(
                            "sched: node{} registered (tile server {})",
                            r.node_id,
                            r.address
                        );
                        let nid = r.node_id as NodeId;
                        node = Some(nid);
                        {
                            let mut st = inner.state.lock().unwrap();
                            st.last_seen.insert(nid, Instant::now());
                            st.nodes.insert(
                                nid,
                                NodeConn {
                                    tx: tx.clone(),
                                    addr: r.address,
                                },
                            );
                        }
                        inner.registered.notify_waiters();
                    }
                    proto::node_message::Payload::Complete(c) => {
                        inner
                            .completed
                            .fetch_add(c.batch.len() as u64, Ordering::SeqCst);
                        inner
                            .state
                            .lock()
                            .unwrap()
                            .completed_iids
                            .extend(c.batch.iter().map(|item| item.iid));
                        inner.complete.notify_waiters();
                    }
                    proto::node_message::Payload::Failed(f) => {
                        phinfo!("sched: node reported instr {} FAILED: {}", f.iid, f.error);
                        inner.state.lock().unwrap().failure =
                            Some(format!("instr {} failed: {}", f.iid, f.error));
                        inner.complete.notify_waiters();
                    }
                    proto::node_message::Payload::Heartbeat(_) => {}
                    proto::node_message::Payload::MemoryWatermark(_) => {}
                }
            }

            if let Some(n) = node {
                {
                    let mut st = inner.state.lock().unwrap();
                    st.nodes.remove(&n);
                    st.last_seen.remove(&n);
                    st.down.push(n);
                }
                phinfo!("sched: node{n} control stream closed (marked down)");
                inner.complete.notify_waiters();
            }
        });
        Ok(Response::new(Box::pin(UnboundedReceiverStream::new(rx))))
    }

    async fn submit(&self, req: Request<Job>) -> Result<Response<Self::SubmitStream>, Status> {
        let job = req.into_inner();
        let this = self.clone();
        let nodes = self.node_count().max(1) as u16;
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            let cfg = DispatchConfig {
                nodes,
                ..Default::default()
            };
            let event = match this.dispatch(job, cfg).await {
                Ok(output_uris) => JobEvent {
                    kind: Some(proto::job_event::Kind::Done(JobDone { output_uris })),
                },
                Err(e) => JobEvent {
                    kind: Some(proto::job_event::Kind::Progress(format!("error: {e:#}"))),
                },
            };
            let _ = tx.send(Ok(event));
        });
        Ok(Response::new(Box::pin(UnboundedReceiverStream::new(rx))))
    }
}

pub fn make_job(source: &str, dims: &[(&str, i64)], tensors: Vec<proto::TensorInput>) -> Job {
    Job {
        source: source.to_string(),
        dimensions: dims
            .iter()
            .map(|(name, value)| proto::DimensionBinding {
                name: name.to_string(),
                value: *value,
            })
            .collect(),
        tensors,
        scalars: vec![],
    }
}

#[cfg(test)]
mod recovery_tests {
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use tokio::net::TcpListener;
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::{TcpListenerStream, UnboundedReceiverStream};

    use phobos_cluster::proto::scheduler_client::SchedulerClient;
    use phobos_cluster::proto::{self, CompleteItem, NodeMessage, Register, TensorInput};
    use phobos_cluster::tile::{AccessMode, DataType};

    use super::{DispatchConfig, Scheduler, make_job};

    const MATMUL: &str = r#"
@cluster(TILE_M in [512, 16384], TILE_N in [512, 16384], TILE_K in [512, 16384])
@autotune(TILE_M in [32, 256], TILE_N in [32, 256], TILE_K in [4, 32])
kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
    let pm = program_id(0)
    let pn = program_id(1)
    var acc: tile<f32>[TILE_M, TILE_N] = 0.0
    for kt in range(0, K, TILE_K) {
        let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
        let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
        acc += dot(a, b)
    }
    C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
}"#;

    fn tensor(name: &str, n: i64, mode: AccessMode) -> TensorInput {
        TensorInput {
            name: name.to_string(),
            data_type: proto::data_type_to_i32(DataType::F32),
            shape: vec![n as u64, n as u64],
            mode: proto::am_to_i32(mode),
            uri: format!("file:///tmp/{name}.bin"), // mocks never LOAD/STORE
        }
    }

    /// A node that speaks the wire protocol but executes nothing: it acks every
    /// instruction in each segment it's issued (recording the iids). A good
    /// node also heartbeats so the watchdog can't mistake the busy scheduler for
    /// a dead node; a failing node skips heartbeats and drops its stream the
    /// moment it's handed work, which the scheduler sees as a failure.
    async fn mock_node(sched_addr: String, node_id: u32, fail: bool, acked: Arc<Mutex<Vec<u64>>>) {
        let mut client = SchedulerClient::connect(format!("http://{sched_addr}"))
            .await
            .unwrap();
        let (tx, rx) = mpsc::unbounded_channel::<NodeMessage>();
        tx.send(NodeMessage {
            payload: Some(proto::node_message::Payload::Register(Register {
                node_id,
                address: "127.0.0.1:1".to_string(),
                sm_architecture: String::new(),
                vram: 0,
                link_bandwidth: 0.0,
            })),
        })
        .unwrap();
        if !fail {
            let htx = tx.clone();
            tokio::spawn(async move {
                let mut iv = tokio::time::interval(Duration::from_millis(500));
                loop {
                    iv.tick().await;
                    if htx
                        .send(NodeMessage {
                            payload: Some(proto::node_message::Payload::Heartbeat(
                                proto::Heartbeat { node_id },
                            )),
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            });
        }
        let resp = client
            .attach(UnboundedReceiverStream::new(rx))
            .await
            .unwrap();
        let mut inbound = resp.into_inner();
        while let Ok(Some(msg)) = inbound.message().await {
            if let Some(proto::scheduler_message::Payload::IssueSegment(is)) = msg.payload {
                if fail {
                    return; // drop the only sender -> stream closes -> node down
                }
                if let Some(seg) = is.segment {
                    let batch: Vec<CompleteItem> = seg
                        .instructions
                        .iter()
                        .map(|i| CompleteItem {
                            iid: i.iid,
                            status: 0,
                            elapsed_ns: 0,
                        })
                        .collect();
                    acked.lock().unwrap().extend(batch.iter().map(|c| c.iid));
                    let _ = tx.send(NodeMessage {
                        payload: Some(proto::node_message::Payload::Complete(proto::Complete {
                            batch,
                        })),
                    });
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn recovery_reruns_failed_node_work() {
        const N: i64 = 1024;
        let dims_v = [("M", N), ("N", N), ("K", N)];
        let dmap = dims_v.iter().map(|(k, v)| (k.to_string(), *v)).collect();

        // Analytic expectation: node 0 (the only survivor) acks its own base
        // instructions plus every recovery instruction (all of node 1's lost
        // chains land on it).
        let kernel = phobos_lang::parse(MATMUL).unwrap().remove(0);
        let program = phobos_cluster::compile(&kernel).unwrap();
        let supers = crate::default_supers(&program);
        let base = crate::plan(&program, &dmap, &supers, 2).unwrap();
        let node0_base = base.node_instrs(0).count() as u64;
        let base_max_iid = base.max_iid();
        let recovered = crate::recover_plan(
            &program,
            &dmap,
            &supers,
            2,
            &[1],
            &HashSet::new(),
            u64::MAX,
            crate::IngestPolicy::DirectLoad,
            1,
            base_max_iid,
            &std::collections::HashMap::new(),
        )
        .unwrap()
        .total_instrs();
        assert!(recovered > 0, "node 1 should own some output chains");

        // Scheduler on an OS-assigned port.
        let sched = Scheduler::new();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let sched_addr = listener.local_addr().unwrap().to_string();
        let incoming = TcpListenerStream::new(listener);
        let server = sched.clone().into_server();
        tokio::spawn(async move {
            let _ = tonic::transport::Server::builder()
                .add_service(server)
                .serve_with_incoming(incoming)
                .await;
        });
        tokio::time::sleep(Duration::from_millis(200)).await;

        let acks = Arc::new(Mutex::new(Vec::new()));
        tokio::spawn(mock_node(sched_addr.clone(), 0, false, acks.clone()));
        tokio::spawn(mock_node(
            sched_addr.clone(),
            1,
            true,
            Arc::new(Mutex::new(Vec::new())),
        ));

        let job = make_job(
            MATMUL,
            &dims_v,
            vec![
                tensor("A", N, AccessMode::Read),
                tensor("B", N, AccessMode::Read),
                tensor("C", N, AccessMode::Write),
            ],
        );
        let cfg = DispatchConfig {
            nodes: 2,
            ..Default::default()
        };
        let out = tokio::time::timeout(Duration::from_secs(120), sched.dispatch(job, cfg))
            .await
            .expect("dispatch hung; recovery never converged")
            .expect("dispatch errored");
        assert_eq!(out.len(), 1, "one Write tensor (C)");

        let acked = acks.lock().unwrap();
        assert_eq!(
            acked.len() as u64,
            node0_base + recovered,
            "survivor should ack its own work ({node0_base}) plus all recovery ({recovered})",
        );
        assert!(
            acked.iter().any(|&i| i > base_max_iid),
            "no recovery instruction (iid above the base max {base_max_iid}) ran on the survivor",
        );
    }
}
