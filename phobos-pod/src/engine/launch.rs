// COMPUTE on the device: a launch marshals its tiles into phobos-mlir's ABI,
// waits on a peer for PTX the node lacks, and retires when its event fires.

use std::ffi::c_void;

use anyhow::{Context, Result};
use cust::event::{Event, EventFlags, EventStatus};

use phobos_cluster::isa::{InstrId, ScalarArg};
use phobos_cluster::tile::{AccessMode, TileId};

use super::io::get_kernel;
use super::{Engine, EngineMsg};

pub(super) struct Inflight {
    event: Event,
    iid: InstrId,
    wrote: Option<TileId>,
}

impl Engine {
    pub(super) fn launch(
        &mut self,
        iid: InstrId,
        kernel: u32,
        args: &[(TileId, AccessMode)],
        scalars: &[ScalarArg],
        grid: (u32, u32, u32),
        cta: (u32, u32, u32),
    ) -> Result<()> {
        // Without cached PTX, park this iid and fetch the PTX from a peer.
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

        // Marshal phobos-mlir's exploded-memref ABI with 32-bit indices. Each
        // f32 tile becomes (alloc_ptr, align_ptr, 0i32, sizes..., strides...).
        let mut addrs: Vec<u64> = Vec::with_capacity(args.len());
        let mut ints: Vec<i32> = Vec::new();
        let mut wrote = None;

        for (tile, mode) in args {
            addrs.push(self.store.addr(*tile)?);
            ints.extend(crate::marshal::memref_words(&self.store.shape(*tile)?)?);
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

        // SAFETY: raw points into addrs, ints and scalar_bits, which outlive
        // this call. The layout is phobos-mlir's exploded memref ABI, with
        // scalar params passed by value at their parameter positions.
        unsafe {
            stream
                .launch(&func, grid, cta, 0, &raw)
                .context("kernel launch")?;
        }

        event.record(stream)?;
        self.inflight.push(Inflight { event, iid, wrote });

        Ok(())
    }

    pub(super) fn poll_inflight(&mut self) {
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
}
