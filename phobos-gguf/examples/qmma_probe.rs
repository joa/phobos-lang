// A prompt projection on synthetic data, on its own:
//
//   cargo run --release --features cuda -p phobos-gguf --example qmma_probe
//   cargo run --release --features cuda -p phobos-gguf --example qmma_probe -- \
//       --intrinsic iq1s_qgemm_t --check
//
// One prompt-projection kernel at the model's own shapes, compiled and
// launched directly. `--check` runs the production intrinsic beside the one
// asked for and compares the outputs bit for bit.

use anyhow::{Result, ensure};
use cust::prelude::*;
use phobos_kernels::{compile, cuda_ok, push_descriptor};

const REPS: usize = 10;

/// The 27B's IQ1_S projections: the FFN's up and down shapes.
const D_MODEL: usize = 5120;
const D_FF: usize = 17408;
const M: usize = 128;

/// IQ1_S on the device: 32 `qs` bytes then 16 of `qh`, no scale in the block.
const BLOCK_BYTES: usize = 48;
/// The grid: 2048 entries of eight ternary lanes.
const GRID_ENTRIES: usize = 2048;
/// The signed grid: every entry folded both ways, eight bytes apiece.
const SIGNED_GRID_LEN: usize = GRID_ENTRIES * 2 * 8;
/// The two-bit grid: two bytes an entry.
const GRID2_LEN: usize = GRID_ENTRIES * 2;

/// Which table an intrinsic reads.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Table {
    Signed,
    TwoBit,
}

/// What one variant of the projection is: the intrinsic it calls, the tile
/// it is compiled at, and the table it takes.
#[derive(Clone)]
struct Variant {
    intrinsic: String,
    tm: usize,
    tn: usize,
    launch: String,
}

impl Variant {
    fn production() -> Variant {
        Variant {
            intrinsic: "iq1s_qmma_staged_t".to_string(),
            tm: 128,
            tn: 64,
            launch: "64".to_string(),
        }
    }

    fn named(intrinsic: &str) -> Variant {
        match intrinsic {
            "iq1s_qgemm_t" => Variant {
                intrinsic: intrinsic.to_string(),
                tm: 128,
                tn: 64,
                launch: "256, 2".to_string(),
            },
            _ => Variant {
                intrinsic: intrinsic.to_string(),
                ..Variant::production()
            },
        }
    }

    fn table(&self) -> Table {
        if self.intrinsic == "iq1s_qgemm_t" {
            Table::TwoBit
        } else {
            Table::Signed
        }
    }

    fn threads(&self) -> u32 {
        self.launch
            .split(',')
            .next()
            .and_then(|t| t.trim().parse().ok())
            .expect("a launch spec starts with the thread count")
    }

    fn source(&self) -> String {
        let table_len = match self.table() {
            Table::Signed => SIGNED_GRID_LEN,
            Table::TwoBit => GRID2_LEN,
        };
        format!(
            "@launch({launch})
@autotune(TM in [{tm}], TN in [{tn}])
@aligned(M = TM, N = TN, K = 256)
kernel iq1s_qmma(A: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                 QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                 GRID: tensor<i8>[1, {table_len}],
                 C: tensor<f32>[M, N]) {{
  let pm = program_id(0)
  let pn = program_id(1)
  C[pm * TM :+ TM, pn * TN :+ TN] = {intrinsic}(A[pm * TM :+ TM, :], AS[pm * TM :+ TM, :],
                                                QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :],
                                                GRID[0 :+ 1, :])
}}
",
            launch = self.launch,
            tm = self.tm,
            tn = self.tn,
            intrinsic = self.intrinsic,
        )
    }

    fn label(&self) -> String {
        format!("{} {}x{}x{}", self.intrinsic, self.tm, self.tn, self.launch)
    }
}

/// The operands one shape needs, uploaded once and shared by every variant so
/// the outputs are comparable: both tables are built from one ternary grid.
struct Operands {
    m: usize,
    k: usize,
    n: usize,
    a: DeviceBuffer<i8>,
    a_scales: DeviceBuffer<f32>,
    qb: DeviceBuffer<i8>,
    d: DeviceBuffer<u16>,
    signed: DeviceBuffer<i8>,
    two_bit: DeviceBuffer<i8>,
    c: DeviceBuffer<f32>,
}

impl Operands {
    fn new(m: usize, k: usize, n: usize) -> Result<Operands> {
        let nb = k / 256;
        let rb = nb * BLOCK_BYTES;
        // Values do not reach the timing, and the check compares two kernels
        // against each other, so any spread of bytes serves; they vary only
        // so that a constant page cannot stand in for the traffic.
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 24) as u8
        };
        let qb: Vec<i8> = (0..n * rb).map(|_| next() as i8).collect();
        let a: Vec<i8> = (0..m * k).map(|_| next() as i8).collect();
        let a_scales: Vec<f32> = (0..m * (k / 32)).map(|i| 0.01 + (i % 7) as f32 * 1e-3).collect();
        // f16 0.25 throughout: a block scale of the size a real file carries.
        let d: Vec<u16> = vec![0x3400u16; n * nb];

        // A random ternary grid, laid out both ways the kernels read it.
        let grid: Vec<[i8; 8]> = (0..GRID_ENTRIES)
            .map(|_| std::array::from_fn(|_| (next() % 3) as i8 - 1))
            .collect();
        let mut signed = Vec::with_capacity(SIGNED_GRID_LEN);
        let mut two_bit = Vec::with_capacity(GRID2_LEN);
        for entry in &grid {
            for delta in [1i8, -1] {
                signed.extend(entry.iter().map(|&g| 8 * g + delta));
            }
            let packed = entry
                .iter()
                .enumerate()
                .fold(0u16, |acc, (j, &g)| acc | (u16::from((g as u8) & 3) << (2 * j)));
            two_bit.extend(packed.to_le_bytes().map(|b| b as i8));
        }
        Ok(Operands {
            m,
            k,
            n,
            a: DeviceBuffer::from_slice(&a)?,
            a_scales: DeviceBuffer::from_slice(&a_scales)?,
            qb: DeviceBuffer::from_slice(&qb)?,
            d: DeviceBuffer::from_slice(&d)?,
            signed: DeviceBuffer::from_slice(&signed)?,
            two_bit: DeviceBuffer::from_slice(&two_bit)?,
            c: DeviceBuffer::from_slice(&vec![0.0f32; m * n])?,
        })
    }

    fn slots(&self, table: Table) -> Vec<u64> {
        let (m, k, n) = (self.m as i64, self.k as i64, self.n as i64);
        let nb = k / 256;
        let mut slots = Vec::new();
        push_descriptor(&mut slots, self.a.as_device_ptr().as_raw(), [m, k]);
        push_descriptor(&mut slots, self.a_scales.as_device_ptr().as_raw(), [m, k / 32]);
        push_descriptor(&mut slots, self.qb.as_device_ptr().as_raw(), [n, nb * BLOCK_BYTES as i64]);
        push_descriptor(&mut slots, self.d.as_device_ptr().as_raw(), [n, nb]);
        match table {
            Table::Signed => push_descriptor(
                &mut slots,
                self.signed.as_device_ptr().as_raw(),
                [1, SIGNED_GRID_LEN as i64],
            ),
            Table::TwoBit => push_descriptor(
                &mut slots,
                self.two_bit.as_device_ptr().as_raw(),
                [1, GRID2_LEN as i64],
            ),
        }
        push_descriptor(&mut slots, self.c.as_device_ptr().as_raw(), [m, n]);
        slots
    }
}

struct Args {
    variant: Variant,
    check: bool,
    shapes: Vec<(usize, usize)>,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        variant: Variant::production(),
        check: false,
        shapes: vec![(D_MODEL, D_FF), (D_FF, D_MODEL)],
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--intrinsic" => args.variant = Variant::named(&it.next().unwrap_or_default()),
            "--tile" => {
                let spec = it.next().unwrap_or_default();
                let mut parts = spec.split('x').map(|p| p.parse::<usize>());
                match (parts.next(), parts.next(), parts.next()) {
                    (Some(Ok(tm)), Some(Ok(tn)), Some(Ok(cta))) => {
                        args.variant.tm = tm;
                        args.variant.tn = tn;
                        args.variant.launch = cta.to_string();
                    }
                    _ => anyhow::bail!("--tile wants TMxTNxCTA"),
                }
            }
            "--check" => args.check = true,
            "--up" => args.shapes = vec![(D_MODEL, D_FF)],
            "--small" => args.shapes = vec![(256, 64)],
            "--shape" => {
                let spec = it.next().unwrap_or_default();
                let (k, n) = spec.split_once('x').unwrap_or_default();
                args.shapes = vec![(k.parse()?, n.parse()?)];
            }
            "--down" => args.shapes = vec![(D_FF, D_MODEL)],
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }
    Ok(args)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let _ctx = cust::quick_init()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    println!(
        "{:>8} {:>8} {:>9} {:>8} {:>8}  variant",
        "k", "n", "ms", "TOPS", "GB/s"
    );
    for &(k, n) in &args.shapes {
        let ops = Operands::new(M, k, n)?;
        let got = run(&stream, &args.variant, &ops)?;
        if args.check && args.variant.intrinsic != Variant::production().intrinsic {
            let want = run(&stream, &Variant::production(), &ops)?;
            let differing = want.iter().zip(&got).filter(|(w, g)| w.to_bits() != g.to_bits()).count();
            let worst = want
                .iter()
                .zip(&got)
                .map(|(w, g)| (w - g).abs())
                .fold(0.0f32, f32::max);
            let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
            println!(
                "  {differing} of {} elements differ, worst {worst:.3e} against a spread of {scale:.3e}",
                want.len()
            );
            if differing > 0 {
                // Where the disagreement sits, as a map of the first rows: a
                // layout defect has a shape, rounding does not.
                for row in 0..ops.m.min(20) {
                    let line: String = (0..ops.n.min(96))
                        .map(|col| {
                            let at = row * ops.n + col;
                            if want[at].to_bits() == got[at].to_bits() { '.' } else { 'X' }
                        })
                        .collect();
                    println!("  row {row:>3} {line}");
                }
                for (name, key) in [
                    ("row % 8", Box::new(|r: usize, _c: usize| r % 8) as Box<dyn Fn(usize, usize) -> usize>),
                    ("row / 8 % 4", Box::new(|r, _c| r / 8 % 4)),
                    ("row / 32", Box::new(|r, _c| r / 32)),
                    ("col % 8", Box::new(|_r, c| c % 8)),
                    ("col / 8 % 8", Box::new(|_r, c| c / 8 % 8)),
                ] {
                    let mut counts = [0usize; 16];
                    for row in 0..ops.m {
                        for col in 0..ops.n {
                            let at = row * ops.n + col;
                            if want[at].to_bits() != got[at].to_bits() {
                                counts[key(row, col)] += 1;
                            }
                        }
                    }
                    println!("  differing by {name}: {:?}", &counts[..counts.iter().rposition(|&c| c > 0).map_or(1, |p| p + 1)]);
                }
                let show: Vec<String> = (0..8).map(|c| format!("{:.3}/{:.3}", want[c], got[c])).collect();
                println!("  row 0, want/got: {}", show.join(" "));
            }
            ensure!(differing == 0, "the variant does not reproduce the production kernel");
        }
    }
    Ok(())
}

fn run(stream: &Stream, variant: &Variant, ops: &Operands) -> Result<Vec<f32>> {
    let module = compile(
        &variant.source(),
        &[("TM", variant.tm), ("TN", variant.tn)],
        "iq1s_qmma",
    )?;
    let function = module.get_function("iq1s_qmma")?.to_raw();
    let mut slots = ops.slots(variant.table());
    let grid = ((ops.m / variant.tm) as u32, (ops.n / variant.tn) as u32);
    let millis = time(stream, function, grid, variant.threads(), &mut slots)?;
    let mut got = vec![0.0f32; ops.m * ops.n];
    ops.c.copy_to(&mut got)?;
    let secs = millis / 1e3;
    let macs = (ops.m * ops.n * ops.k) as f64;
    let weight_bytes = (ops.n * (ops.k / 256) * BLOCK_BYTES) as f64;
    println!(
        "{:>8} {:>8} {millis:>9.3} {:>8.2} {:>8.1}  {}",
        ops.k,
        ops.n,
        2.0 * macs / secs / 1e12,
        weight_bytes / secs / 1e9,
        variant.label()
    );
    Ok(got)
}

fn time(
    stream: &Stream,
    function: cust::sys::CUfunction,
    grid: (u32, u32),
    threads: u32,
    slots: &mut [u64],
) -> Result<f64> {
    let mut params: Vec<*mut std::ffi::c_void> =
        slots.iter_mut().map(|s| (s as *mut u64).cast()).collect();
    // SAFETY: the function comes from a module held by the caller, and the
    // descriptor slots outlive the launch, which is synchronized below.
    let mut launch = || -> Result<()> {
        unsafe {
            cuda_ok(
                cust::sys::cuLaunchKernel(
                    function,
                    grid.0,
                    grid.1,
                    1,
                    threads,
                    1,
                    1,
                    0,
                    stream.as_inner(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "probe launch",
            )?;
        }
        Ok(())
    };
    launch()?;
    stream.synchronize()?;

    use cust::event::{Event, EventFlags};
    let (begin, end) = (Event::new(EventFlags::DEFAULT)?, Event::new(EventFlags::DEFAULT)?);
    let mut total = 0.0f32;
    for _ in 0..REPS {
        begin.record(stream)?;
        launch()?;
        end.record(stream)?;
        end.synchronize()?;
        total += end.elapsed_time_f32(&begin)?;
    }
    Ok(f64::from(total) / REPS as f64)
}
