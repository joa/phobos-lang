// CUDA graph capture: what a recorded launch is, and how a pass is
// replayed and reported.

use super::*;
impl DeviceBackend {
    /// Puts one launch on the stream.
    pub(super) fn issue(&self, launch: &Recorded, name: &'static str) -> Result<()> {
        let mut argv: Vec<*mut c_void> = launch
            .slots
            .iter()
            .map(|s| s as *const u64 as *mut u64 as *mut c_void)
            .collect();
        // SAFETY: argv points into launch.slots, which outlives the call; the
        // layout matches phobos-mlir's exploded-memref ABI, and every buffer
        // stays alive because the caller holds its handle.
        cuda_ok(
            unsafe {
                cust::sys::cuLaunchKernel(
                    launch.func,
                    launch.grid.0,
                    launch.grid.1,
                    launch.grid.2,
                    launch.threads,
                    1,
                    1,
                    launch.shared,
                    self.stream.as_inner(),
                    argv.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            name,
        )
    }

    /// Issues whatever has been recorded so far and stops recording this pass.
    /// Growing a scratch arena frees the buffer the recorded launches point at,
    /// so the recording has to be spent before the old pointer dies. The arenas
    /// grow only until they have seen the widest projection, so after the
    /// warmup no pass flushes and every pass is one graph.
    pub(super) fn flush_pending(&self) -> Result<()> {
        if !self.recording.get() {
            return Ok(());
        }
        self.recording.set(false);
        self.flushed.set(true);
        self.issue_recorded("flushing a recorded launch")
    }

    /// Issues the recording as plain launches and empties it.
    pub(super) fn issue_recorded(&self, what: &'static str) -> Result<()> {
        let pending = self.pending.borrow();
        for launch in &pending[..self.recorded_len.replace(0)] {
            self.issue(launch, what)?;
        }
        Ok(())
    }

    /// What the pass costs in launches, and what each kernel costs in registers
    /// and shared memory.
    pub(super) fn print_report(&self) -> Result<()> {
        let ops = self.report.borrow();
        let sms = cust::device::Device::get_device(0)?
            .get_attribute(cust::device::DeviceAttribute::MultiprocessorCount)?;

        // Keyed by function, not by name: `with_kernel` compiles a module per
        // tile, so one name can cover several shapes with quite different
        // shared-memory footprints, and collapsing them hides exactly the thing
        // this report is for.
        type Key = (&'static str, usize);
        let mut order: Vec<Key> = Vec::new();
        let mut per: HashMap<Key, (usize, usize, &PassOp)> = HashMap::new();
        for op in ops.iter() {
            let key = (op.name, op.func as usize);
            match per.get_mut(&key) {
                Some(e) => {
                    e.0 += 1;
                    e.1 += op.blocks as usize;
                }
                None => {
                    order.push(key);
                    per.insert(key, (1, op.blocks as usize, op));
                }
            }
        }
        order.sort_by_key(|k| usize::MAX - per[k].0);

        // Which replay this was matters: a prefill replays too, and a prompt
        // pass takes different kernels than a decode step (swiglu_2d against
        // swiglu_q, q8_mma against q8_qdot). Naming the replay is what stops the
        // two being confused.
        eprintln!(
            "=== pass report, replay {}: {} launches, {sms} SMs ===",
            self.reported.get() + 1,
            ops.len()
        );
        eprintln!(
            "{:<18}{:>5}{:>8}{:>7}{:>6}{:>7}{:>7}{:>8}",
            "kernel", "n", "blocks", "thr", "regs", "shared", "blk/SM", "waves"
        );
        for key in &order {
            let (count, blocks, op) = per[key];
            let name = key.0;
            let attr = |a| -> Result<i32> {
                let mut v = 0i32;
                // SAFETY: op.func came from cuModuleGetFunction on a module the
                // backend holds for its lifetime.
                cuda_ok(
                    unsafe { cust::sys::cuFuncGetAttribute(&mut v, a, op.func) },
                    "querying a kernel attribute",
                )?;
                Ok(v)
            };
            let regs = attr(cust::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS)?;
            let stat = attr(cust::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?;
            let mut per_sm = 0i32;
            // SAFETY: same function handle; the dynamic shared size is the one
            // the launch used.
            cuda_ok(
                unsafe {
                    cust::sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(
                        &mut per_sm,
                        op.func,
                        op.threads as i32,
                        op.shared as usize,
                    )
                },
                "querying occupancy",
            )?;
            let per_launch = blocks as f64 / count as f64;
            eprintln!(
                "{name:<18}{count:>5}{blocks:>8}{:>7}{regs:>6}{:>7}{per_sm:>7}{:>8.2}",
                op.threads,
                stat as u32 + op.shared,
                per_launch / f64::from(sms.max(1)),
            );
        }
        let total: usize = order.iter().map(|n| per[n].1).sum();
        eprintln!(
            "total {} launches, {total} blocks, {:.1} blocks per launch",
            ops.len(),
            total as f64 / ops.len().max(1) as f64
        );
        Ok(())
    }

    /// Replays the recorded pass, building or patching the graph first. A
    /// rebuild is only needed when the pass's shape changes, which in practice
    /// means the first decode step after a prefill and the reverse. Otherwise
    /// the topology is identical and the only nodes that moved are the ones
    /// reading the key/value cache, whose length grew by a token.
    pub(super) fn replay(&self) -> Result<()> {
        if self.report_pass.get() != 0 {
            let left = self.report_pass.get() - 1;
            self.report_pass.set(left);
            if left == 0 {
                self.print_report()?;
            }
            self.reported.set(self.reported.get() + 1);
        }
        let pending = self.pending.borrow();
        let recorded = &pending[..self.recorded_len.replace(0)];
        let mut cached = self.pass.borrow_mut();
        let reusable = cached.as_ref().is_some_and(|p| {
            p.recorded.len() == recorded.len()
                && p.recorded
                    .iter()
                    .zip(recorded)
                    .all(|(a, b)| a.func == b.func)
        });
        if !reusable {
            *cached = Some(Self::build_graph(recorded)?);
        } else {
            let pass = cached.as_mut().expect("reusable implies present");
            let mut argv = Vec::new();
            for (i, (was, now)) in pass.recorded.iter_mut().zip(recorded).enumerate() {
                if was.same(now) {
                    continue;
                }
                let params = now.params(&mut argv);
                // SAFETY: the node belongs to this exec and the parameter
                // block matches the function it was built with.
                cuda_ok(
                    unsafe {
                        cust::sys::cuGraphExecKernelNodeSetParams(pass.exec, pass.nodes[i], &params)
                    },
                    "updating a graph node",
                )?;
                was.func = now.func;
                was.grid = now.grid;
                was.slots.clear();
                was.slots.extend_from_slice(&now.slots);
            }
        }

        let exec = cached.as_ref().expect("built above").exec;
        // SAFETY: the exec outlives the launch, held by self.pass.
        cuda_ok(
            unsafe { cust::sys::cuGraphLaunch(exec, self.stream.as_inner()) },
            "launching the pass graph",
        )?;
        Ok(())
    }

    /// Instantiates a recorded pass as a chain of kernel nodes. A chain rather
    /// than a dependency analysis: these launches shared one stream, so serial
    /// order is the ordering they already relied on.
    pub(super) fn build_graph(recorded: &[Recorded]) -> Result<PassGraph> {
        let mut graph: cust::sys::CUgraph = std::ptr::null_mut();
        // SAFETY: graph is written on success and destroyed by PassGraph.
        cuda_ok(
            unsafe { cust::sys::cuGraphCreate(&mut graph, 0) },
            "creating the pass graph",
        )?;

        let mut nodes: Vec<cust::sys::CUgraphNode> = Vec::with_capacity(recorded.len());
        let mut argv = Vec::new();
        for launch in recorded {
            let params = launch.params(&mut argv);
            let deps = nodes.last().copied();
            let mut node: cust::sys::CUgraphNode = std::ptr::null_mut();
            // SAFETY: deps points at the previous node, which belongs to graph.
            cuda_ok(
                unsafe {
                    cust::sys::cuGraphAddKernelNode(
                        &mut node,
                        graph,
                        deps.as_ref().map_or(std::ptr::null(), |d| d as *const _),
                        usize::from(deps.is_some()),
                        &params,
                    )
                },
                "adding a graph node",
            )?;
            nodes.push(node);
        }

        let mut exec: cust::sys::CUgraphExec = std::ptr::null_mut();
        // SAFETY: exec is written on success and destroyed by PassGraph.
        cuda_ok(
            unsafe {
                cust::sys::cuGraphInstantiate_v2(
                    &mut exec,
                    graph,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                )
            },
            "instantiating the pass graph",
        )?;

        Ok(PassGraph {
            graph,
            exec,
            nodes,
            recorded: recorded
                .iter()
                .map(|r| Recorded {
                    func: r.func,
                    grid: r.grid,
                    shared: r.shared,
                    threads: r.threads,
                    slots: r.slots.clone(),
                })
                .collect(),
        })
    }
}
