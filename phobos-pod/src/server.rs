use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::Stream;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::{Request, Response, Status};

use phobos_base::{phdebug, phinfo};
use phobos_cluster::isa::InstrId;
use phobos_cluster::proto::scheduler_client::SchedulerClient;
use phobos_cluster::proto::tile_server_server::{TileServer, TileServerServer};
use phobos_cluster::proto::{
    self, Complete, CompleteItem, Failed, KernelBlob, KernelHash, NodeMessage, Register,
    SchedulerMessage, TileChunk, TileRequest,
};
use phobos_cluster::tile::{NodeId, TileId};

use crate::engine::{self, Completion, EngineMsg, EngineTx, TensorMeta};

pub static SERVED_BYTES: AtomicU64 = AtomicU64::new(0);
pub static GET_KERNEL_CALLS: AtomicU64 = AtomicU64::new(0);

pub struct TileService {
    engine: EngineTx,
}

impl TileService {
    pub fn into_server(engine: EngineTx) -> TileServerServer<TileService> {
        TileServerServer::new(TileService { engine })
    }
}

type ChunkStream = Pin<Box<dyn Stream<Item = Result<TileChunk, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl TileServer for TileService {
    type FetchTileStream = ChunkStream;

    #[allow(clippy::result_large_err)]
    async fn fetch_tile(
        &self,
        req: Request<TileRequest>,
    ) -> Result<Response<Self::FetchTileStream>, Status> {
        let tile = TileId(req.into_inner().tile);
        let (tx, rx) = oneshot::channel();
        self.engine
            .send(EngineMsg::Serve { tile, reply: tx })
            .map_err(|_| Status::internal("engine gone"))?;
        let data = rx
            .await
            .map_err(|_| Status::internal("engine dropped serve reply"))?
            .ok_or_else(|| Status::not_found("tile not resident"))?;
        SERVED_BYTES.fetch_add((data.len() * 4) as u64, Ordering::SeqCst);
        let chunks: Vec<Result<TileChunk, Status>> = engine::chunk_bytes(&data)
            .into_iter()
            .map(|(offset, data)| Ok(TileChunk { offset, data }))
            .collect();
        Ok(Response::new(Box::pin(tokio_stream::iter(chunks))))
    }

    async fn get_kernel(&self, req: Request<KernelHash>) -> Result<Response<KernelBlob>, Status> {
        let hash = req.into_inner().hash;
        GET_KERNEL_CALLS.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.engine
            .send(EngineMsg::GetKernel { hash, reply: tx })
            .map_err(|_| Status::internal("engine gone"))?;
        let ptx = rx
            .await
            .map_err(|_| Status::internal("engine dropped kernel reply"))?
            .unwrap_or_default();
        Ok(Response::new(KernelBlob { ptx }))
    }
}

/// Connects this client to a scheduler.
///
/// First we connect, register and then start the pump loop. Exits when the stream
/// is closed by the scheduler (job done).
pub async fn attach(
    sched_addr: &str,
    node_id: NodeId,
    listen_addr: &str,
    engine: EngineTx,
    complete_rx: mpsc::UnboundedReceiver<Completion>,
    fail_rx: mpsc::UnboundedReceiver<(InstrId, String)>,
) -> Result<()> {
    let mut client = SchedulerClient::connect(format!("http://{sched_addr}")).await?;

    let (out_tx, out_rx) = mpsc::unbounded_channel::<NodeMessage>();
    out_tx.send(NodeMessage {
        payload: Some(proto::node_message::Payload::Register(Register {
            node_id: node_id as u32,
            address: listen_addr.to_string(),
            sm_architecture: String::new(),
            vram: 0,
            link_bandwidth: 0.0,
        })),
    })?;

    let resp = client.attach(UnboundedReceiverStream::new(out_rx)).await?;
    let mut inbound = resp.into_inner();
    phinfo!("node{node_id}: attached to scheduler {sched_addr}, tile server {listen_addr}");

    tokio::spawn(batch_completions(complete_rx, out_tx.clone()));
    tokio::spawn(heartbeat(node_id, out_tx.clone()));
    tokio::spawn(forward_failures(fail_rx, out_tx));

    while let Some(msg) = inbound.message().await? {
        dispatch(msg, &engine);
    }
    phinfo!("node{node_id}: scheduler closed the control stream (done)");
    Ok(())
}

async fn batch_completions(
    mut rx: mpsc::UnboundedReceiver<Completion>,
    out: mpsc::UnboundedSender<NodeMessage>,
) {
    while let Some(first) = rx.recv().await {
        let mut batch = vec![first];
        while let Ok(c) = rx.try_recv() {
            batch.push(c);
            if batch.len() >= 100 {
                break;
            }
        }
        let items = batch
            .into_iter()
            .map(|(iid, status, elapsed_ns)| CompleteItem {
                iid,
                status,
                elapsed_ns,
            })
            .collect();
        if out
            .send(NodeMessage {
                payload: Some(proto::node_message::Payload::Complete(Complete {
                    batch: items,
                })),
            })
            .is_err()
        {
            break;
        }
    }
}

async fn heartbeat(node_id: NodeId, out: mpsc::UnboundedSender<NodeMessage>) {
    const PERIOD: std::time::Duration = std::time::Duration::from_secs(1);
    let mut tick = tokio::time::interval(PERIOD);
    loop {
        tick.tick().await;
        if out
            .send(NodeMessage {
                payload: Some(proto::node_message::Payload::Heartbeat(proto::Heartbeat {
                    node_id: node_id as u32,
                })),
            })
            .is_err()
        {
            break;
        }
    }
}

async fn forward_failures(
    mut rx: mpsc::UnboundedReceiver<(InstrId, String)>,
    out: mpsc::UnboundedSender<NodeMessage>,
) {
    while let Some((iid, error)) = rx.recv().await {
        if out
            .send(NodeMessage {
                payload: Some(proto::node_message::Payload::Failed(Failed { iid, error })),
            })
            .is_err()
        {
            break;
        }
    }
}

fn dispatch(msg: SchedulerMessage, engine: &EngineTx) {
    use proto::scheduler_message::Payload;
    let Some(payload) = msg.payload else { return };
    match payload {
        Payload::JobManifest(jm) => {
            phdebug!(
                "node: job manifest: {} tensors, {} kernels, {} holders",
                jm.tensors.len(),
                jm.kernels.len(),
                jm.kernel_holders.len()
            );
            // tensors, placed by their declared index
            let max = jm.tensors.iter().map(|t| t.index).max().unwrap_or(0) as usize;
            let mut tensors = vec![
                TensorMeta {
                    uri: String::new(),
                    shape: Vec::new(),
                    mode: phobos_cluster::tile::AccessMode::Read,
                };
                max + 1
            ];
            for t in jm.tensors {
                let mode =
                    proto::am_from_i32(t.mode).unwrap_or(phobos_cluster::tile::AccessMode::Read);
                tensors[t.index as usize] = TensorMeta {
                    uri: t.uri,
                    shape: t.shape,
                    mode,
                };
            }
            let _ = engine.send(EngineMsg::SetTensors(tensors));
            let table = jm
                .kernels
                .into_iter()
                .map(|k| (k.index, k.hash, k.name))
                .collect();
            let holders = jm
                .kernel_holders
                .into_iter()
                .map(|h| (h.hash, h.node_id as NodeId))
                .collect();
            let _ = engine.send(EngineMsg::SetKernels { table, holders });
        }
        Payload::Kernel(k) => {
            phdebug!(
                "node: received kernel PTX {} ({} bytes)",
                &k.hash[..16.min(k.hash.len())],
                k.ptx.len()
            );
            let _ = engine.send(EngineMsg::Kernel {
                hash: k.hash,
                ptx: k.ptx,
            });
        }
        Payload::IssueSegment(is) => {
            if let Some(seg) = is.segment {
                match proto::segment_from_proto(&seg) {
                    Ok(seg) => {
                        let _ = engine.send(EngineMsg::Issue(seg));
                    }
                    Err(e) => eprintln!("bad segment: {e:#}"),
                }
            }
        }
        Payload::Manifest(m) => {
            phdebug!("node: fetch manifest: {} tile location(s)", m.tiles.len());
            let mut locs = Vec::new();
            let mut addrs = Vec::new();
            for t in m.tiles {
                locs.push((TileId(t.tile), t.address.clone()));
                addrs.push((t.node_id as NodeId, t.address));
            }
            let _ = engine.send(EngineMsg::FetchLocs(locs));
            let _ = engine.send(EngineMsg::NodeAddrs(addrs));
        }
        Payload::Cancel(c) => {
            phdebug!("node: cancel {} instruction(s)", c.iids.len());
            let _ = engine.send(EngineMsg::Cancel(c.iids));
        }
    }
}
