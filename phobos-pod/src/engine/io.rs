// The engine's I/O tasks: LOAD and STORE against storage, FETCH from a
// peer's tile server, and that server's client and chunking.

use anyhow::{Context, Result, bail};

use phobos_base::cli::format_size;
use phobos_base::phinfo;
use phobos_cluster::isa::{InstrId, StorageRef};
use phobos_cluster::proto::tile_server_client::TileServerClient;
use phobos_cluster::proto::{KernelHash, TileRequest};
use phobos_cluster::storage;
use phobos_cluster::tile::{NodeId, TileId};

use super::{Engine, EngineMsg};

const CHUNK: usize = 1 << 20; // 1 MiB tile-server chunks

impl Engine {
    pub(super) fn spawn_load(&mut self, iid: InstrId, tile: TileId, src: &StorageRef) -> Result<()> {
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
                        format_size(data.len() as u64 * 4),
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

    pub(super) fn spawn_store(&mut self, iid: InstrId, tile: TileId, dst: &StorageRef) -> Result<()> {
        let StorageRef::Tensor { tensor, region } = dst.clone();
        let meta = self
            .tensors
            .get(tensor as usize)
            .context("STORE of unknown tensor")?
            .clone();

        // D2H runs on the engine thread, the file write on a blocking task.
        let data = self.store.d2h(tile)?;
        phinfo!(
            "node{}: D2H tile={:#x} {} read back from VRAM, writing to {}",
            self.node_id,
            tile.0,
            format_size(data.len() as u64 * 4),
            meta.uri,
        );

        let to = self.to_self.clone();
        let node = self.node_id;
        let uri = meta.uri.clone();
        let nbytes = data.len() as u64 * 4;

        self.outstanding_io += 1;
        self.rt.spawn(async move {
            let res = tokio::task::spawn_blocking(move || {
                storage::store_f32(&meta.uri, &meta.shape, &region, &data)
            })
            .await
            .expect("store join");

            match res {
                Ok(()) => {
                    phinfo!("node{node}: STORE #{iid} wrote {} to {uri}", format_size(nbytes));
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

    pub(super) fn spawn_fetch(&mut self, iid: InstrId, tile: TileId, from: NodeId) -> Result<()> {
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
                        format_size(data.len() as u64 * 4),
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

pub(super) async fn get_kernel(addr: &str, hash: &str) -> Result<String> {
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

pub fn chunk_bytes(data: &[f32]) -> Vec<(u64, Vec<u8>)> {
    let bytes = storage::f32_to_le_bytes(data);
    bytes
        .chunks(CHUNK)
        .enumerate()
        .map(|(i, c)| ((i * CHUNK) as u64, c.to_vec()))
        .collect()
}
