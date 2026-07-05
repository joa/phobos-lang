pub mod engine;
pub mod ptx_cache;
pub mod server;
pub mod tile_store;

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use anyhow::{Context, Result};
use tokio::net::TcpListener;
use tokio::runtime::Handle;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::TcpListenerStream;

use phobos_cluster::tile::NodeId;

use crate::engine::Engine;

/// Default device arena per node (one node = one GPU currently)
pub const DEFAULT_ARENA_BYTES: usize = 512 << 20;

pub async fn serve(
    node_id: NodeId,
    scheduler_addr: String,
    listen_addr: String,
    advertise_addr: Option<String>,
    arena_bytes: usize,
) -> Result<()> {
    let rt = Handle::current();
    let (complete_tx, complete_rx) = mpsc::unbounded_channel();
    let (fail_tx, fail_rx) = mpsc::unbounded_channel();
    let (idle_tx, _idle_rx) = oneshot::channel();
    let engine = Engine::spawn(
        rt,
        node_id,
        arena_bytes,
        Some(complete_tx),
        Some(fail_tx),
        idle_tx,
    )?;

    let addr: SocketAddr = listen_addr.parse().context("invalid listen address")?;
    let listener = TcpListener::bind(addr).await?;
    let bound = match advertise_addr {
        // for 0.0.0.0:0 shenanigans
        Some(a) => a,
        None => connectable(listener.local_addr()?),
    };
    let incoming = TcpListenerStream::new(listener);
    let svc = server::TileService::into_server(engine.clone());
    let server_task = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(svc)
            .serve_with_incoming(incoming)
            .await
    });

    let res = server::attach(
        &scheduler_addr,
        node_id,
        &bound,
        engine,
        complete_rx,
        fail_rx,
    )
    .await;
    server_task.abort();
    res
}

fn connectable(bound: SocketAddr) -> String {
    let ip = match bound.ip() {
        IpAddr::V4(v4) if v4.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V6(v6) if v6.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
        other => other,
    };
    SocketAddr::new(ip, bound.port()).to_string()
}
