use cust::prelude::*;
use phobos_base::phinfo;

// Masking only turns on for compile-time-constant extents that an aligned tile
// cannot tile evenly. The phobos-bench kernels pass their shapes as runtime
// extents, which the row-pitch ABI assumes aligned, so none of them reach it.
const CODE_COPY: &str = "kernel copy(A: tensor<f32>[100], B: tensor<f32>[100]) {
    let p = program_id(0)
    B[p * 32 :+ 32] = A[p * 32 :+ 32]
}";

const CODE_MATMUL: &str = "@autotune(TILE_M in [32], TILE_N in [32], TILE_K in [32])
kernel matmul(A: tensor<f32>[96, 64], B: tensor<f32>[64, 100], C: tensor<f32>[96, 100]) {
    let pm = program_id(0)
    let pn = program_id(1)
    var acc: tile<f32>[TILE_M, TILE_N] = 0.0
    for kt in range(0, 64, TILE_K) {
        var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
        var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
        acc += dot(a, b)
    }
    C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
}";

const SENTINEL: f32 = -12345.0;

const GUARD: usize = 64;

fn compile(code: &str, config: &[(String, i64)]) -> anyhow::Result<String> {
    let ctx = phobos_base::context::Context {
        shape_overrides: config.iter().cloned().collect(),
        ..Default::default()
    };
    phobos_lang::compile(&ctx, code)
}

fn cta_threads(code: &str) -> anyhow::Result<u32> {
    let kernels = phobos_lang::parse(code)?;
    Ok(kernels[0].cta_threads().map_err(anyhow::Error::msg)? as u32)
}

fn check_guard(out: &[f32], len: usize, what: &str) -> anyhow::Result<()> {
    for (i, &v) in out[len..].iter().enumerate() {
        anyhow::ensure!(
            v == SENTINEL,
            "{what}: out-of-bounds store at element {} past the extent: {v} != {SENTINEL}",
            len + i
        );
    }
    Ok(())
}

fn run_copy(stream: &Stream) -> anyhow::Result<()> {
    const LEN: usize = 100;
    const TILE: usize = 32;

    let ptx = compile(CODE_COPY, &[])?;
    let module = Module::from_ptx(ptx.as_str(), &[])?;
    let func = module.get_function("copy")?;
    let block = cta_threads(CODE_COPY)?;

    let a: Vec<f32> = (0..LEN).map(|i| i as f32 + 1.0).collect();
    let mut b = vec![SENTINEL; LEN + GUARD];

    let a_dev = a.as_slice().as_dbuf()?;
    let b_dev = b.as_slice().as_dbuf()?;
    let (a_ptr, b_ptr) = (a_dev.as_device_ptr(), b_dev.as_device_ptr());
    let (ap, bp) = (a_ptr.as_raw(), b_ptr.as_raw());

    // memref<100xf32> descriptor: allocated ptr, aligned ptr, offset, size, stride.
    let len_i = LEN as i32;
    let grid = LEN.div_ceil(TILE) as u32;
    unsafe {
        launch!(func<<<grid, block, 0, stream>>>(
            ap, ap, 0i32, len_i, 1i32,
            bp, bp, 0i32, len_i, 1i32
        ))?;
    }
    stream.synchronize()?;

    b_dev.copy_to(&mut b)?;

    for i in 0..LEN {
        anyhow::ensure!(b[i] == a[i], "copy: element {i} is {} not {}", b[i], a[i]);
    }
    check_guard(&b, LEN, "copy")?;
    phinfo!("copy[100] tile 32: {LEN} elements correct, guard intact");
    Ok(())
}

fn run_matmul(stream: &Stream) -> anyhow::Result<()> {
    const M: usize = 96;
    const K: usize = 64;
    const N: usize = 100;
    const TILE: usize = 32;

    let config: Vec<(String, i64)> = vec![
        ("TILE_M".to_string(), TILE as i64),
        ("TILE_N".to_string(), TILE as i64),
        ("TILE_K".to_string(), TILE as i64),
    ];

    let ptx = compile(CODE_MATMUL, &config)?;
    let module = Module::from_ptx(ptx.as_str(), &[])?;
    let func = module.get_function("matmul")?;
    let block = cta_threads(CODE_MATMUL)?;

    // Exactly representable values keep the reference sum exact.
    let a: Vec<f32> = (0..M * K).map(|i| ((i % 7) as f32) - 3.0).collect();
    let b: Vec<f32> = (0..K * N).map(|i| ((i % 5) as f32) - 2.0).collect();
    let mut c = vec![SENTINEL; M * N + GUARD];

    let a_dev = a.as_slice().as_dbuf()?;
    let b_dev = b.as_slice().as_dbuf()?;
    let c_dev = c.as_slice().as_dbuf()?;
    let (ap, bp, cp) = (
        a_dev.as_device_ptr().as_raw(),
        b_dev.as_device_ptr().as_raw(),
        c_dev.as_device_ptr().as_raw(),
    );

    let (m, k, n) = (M as i32, K as i32, N as i32);
    let grid = (M.div_ceil(TILE) as u32, N.div_ceil(TILE) as u32);
    unsafe {
        launch!(func<<<grid, block, 0, stream>>>(
            ap, ap, 0i32, m, k, k, 1i32,
            bp, bp, 0i32, k, n, n, 1i32,
            cp, cp, 0i32, m, n, n, 1i32
        ))?;
    }
    stream.synchronize()?;

    c_dev.copy_to(&mut c)?;

    for i in 0..M {
        for j in 0..N {
            let mut want = 0.0f32;
            for p in 0..K {
                want += a[i * K + p] * b[p * N + j];
            }
            let got = c[i * N + j];
            anyhow::ensure!(
                (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                "matmul: C[{i},{j}] is {got} not {want}"
            );
        }
    }
    check_guard(&c, M * N, "matmul")?;
    phinfo!(
        "matmul[96,64]x[64,100] tile 32: {} elements correct, guard intact",
        M * N
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let _ctx = cust::quick_init()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    run_copy(&stream)?;
    run_matmul(&stream)?;

    phinfo!("masking: all checks passed");
    Ok(())
}
