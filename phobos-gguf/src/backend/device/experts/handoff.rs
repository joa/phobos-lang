// A decode row's misses computed on the host and handed to the device
// without the host waiting to hand them over.
//
// A kernel copies the row into mapped memory before the block's sync point;
// after it the host starts the misses on the team and goes on recording
// the next block. The kernel here, recorded in between, holds the stream
// until the team raises the block's flag, then adds the row the team wrote
// beside it and lowers the flag again. It is written in PTX because
// the wait needs a system-scope acquire load, which the kernel language has
// no spelling for: a device-scope load or atomic can be answered from the
// L2's copy of the line and never see the host's store.
//
// Its parameters are the launch ABI's descriptors, `FLAG[1, 1]`, `Y[1, d]`
// and `DEST[1, d]`: seven words each, of which it reads the aligned
// pointers and `DEST`'s width.

use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{Context, Result};
use cust::module::Module;

use super::super::DeviceBackend;
use super::mapped::Mapped;
use crate::backend::Buf;
use crate::simd::{self, RowOut, Source, StartedRow};

const HOST_ADD_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry host_add(
    .param .u64 f_alloc, .param .u64 flag, .param .u32 f_off, .param .u32 f_s0, .param .u32 f_s1, .param .u32 f_t0, .param .u32 f_t1,
    .param .u64 y_alloc, .param .u64 y, .param .u32 y_off, .param .u32 y_s0, .param .u32 y_s1, .param .u32 y_t0, .param .u32 y_t1,
    .param .u64 d_alloc, .param .u64 dest, .param .u32 d_off, .param .u32 d_s0, .param .u32 len, .param .u32 d_t0, .param .u32 d_t1
)
.maxntid 256, 1, 1
{
    .reg .pred %p<4>;
    .reg .b32 %r<6>;
    .reg .f32 %f<4>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [flag];
    ld.param.u64 %rd2, [y];
    ld.param.u64 %rd3, [dest];
    ld.param.u32 %r1, [len];
    mov.u32 %r2, %tid.x;
    mov.u32 %r5, %ntid.x;
    setp.ne.u32 %p1, %r2, 0;
    @%p1 bra $SYNC;
$SPIN:
    ld.acquire.sys.global.u32 %r3, [%rd1];
    setp.eq.u32 %p2, %r3, 0;
    @%p2 bra $SPIN;
$SYNC:
    bar.sync 0;
    fence.acq_rel.sys;
    mov.u32 %r4, %r2;
$LOOP:
    setp.ge.u32 %p3, %r4, %r1;
    @%p3 bra $DONE;
    mul.wide.u32 %rd4, %r4, 4;
    add.s64 %rd5, %rd2, %rd4;
    ld.relaxed.sys.global.f32 %f1, [%rd5];
    add.s64 %rd6, %rd3, %rd4;
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f2, %f1;
    st.global.f32 [%rd6], %f3;
    add.u32 %r4, %r4, %r5;
    bra $LOOP;
$DONE:
    bar.sync 0;
    @%p1 bra $END;
    st.relaxed.sys.global.u32 [%rd1], 0;
$END:
    ret;
}
"#;

/// A block's decode row on its way to the host, the host's share of the
/// output on its way back, and the flag between them: raised by the host
/// once the share is written, lowered by the device once it has added it.
pub(super) struct Handoff {
    x: Mapped<f32>,
    y: Mapped<f32>,
    ready: Mapped<u32>,
}

impl Handoff {
    pub(super) fn new(d: usize) -> Result<Handoff> {
        Ok(Handoff { x: Mapped::new(d)?, y: Mapped::new(d)?, ready: Mapped::new(1)? })
    }

    /// The share of `misses`, each an expert and its router weight, started
    /// on the team over the row the device sent. With none, or when the
    /// start fails, a zero row goes back at once: the device waits for the
    /// flag either way.
    pub(super) fn start(&mut self, source: impl Source + Send + 'static, misses: Vec<(usize, f32)>) -> Result<Option<StartedRow>> {
        if misses.is_empty() {
            self.raise_zeros();
            return Ok(None);
        }
        let to = RowOut { out: self.y.host_ptr(), len: self.y.host().len(), ready: self.ready.host_ptr() as *const AtomicU32 };
        // SAFETY: the row and the flag are this hand-off's, and nothing
        // touches them before the device lowers the flag, a sync point
        // later.
        match unsafe { simd::start_experts_row(source, misses, self.x.host(), to) } {
            Ok(started) => Ok(Some(started)),
            Err(e) => {
                self.raise_zeros();
                Err(e)
            }
        }
    }

    fn raise_zeros(&mut self) {
        self.y.host_mut().fill(0.0);
        // SAFETY: the flag is mapped for the hand-off's lifetime.
        unsafe { &*(self.ready.host_ptr() as *const AtomicU32) }.store(1, Ordering::Release);
    }
}

impl DeviceBackend {
    /// `x`'s first `len` sent to the host through `handoff`, in stream
    /// order, by a kernel's stores rather than a copy.
    pub(super) fn send_row(&self, x: Buf, handoff: &Handoff, len: usize) -> Result<()> {
        self.pointwise_raw("copy", &[self.ptr(x, 0)?, handoff.x.dev()], len)
    }

    /// The host's share, once `handoff`'s flag is raised, added into the
    /// first `len` of `dest`, in stream order; the flag is lowered after.
    pub(super) fn host_add(&self, handoff: &Handoff, dest: Buf, len: usize) -> Result<()> {
        if self.host_add.borrow().is_none() {
            let module = Module::from_ptx(HOST_ADD_PTX, &[]).context("loading the host hand-off kernel")?;
            *self.host_add.borrow_mut() = Some(module);
        }
        let module = self.host_add.borrow();
        self.launch(
            module.as_ref().expect("loaded above"),
            "host_add",
            &[(handoff.ready.dev(), [1, 1]), (handoff.y.dev(), [1, len as i64]), (self.ptr(dest, 0)?, [1, len as i64])],
            (1, 1, 1),
        )
    }
}
