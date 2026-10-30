mod common;

use anyhow::{Result, bail, ensure};
use phobos_cluster::proto::scheduler_client::SchedulerClient;
use phobos_cluster::proto::{self, Job};
use phobos_cluster::storage;
use phobos_cluster::tile::AccessMode;
use phobos_sched::server::make_job;

use common::{check_matmul, f32_tensor, matmul_job, matmul_src, randoms, seed_matmul, start_cluster, temp_uris};

const ADD: &str = r#"
@cluster(BLOCK in [512, 16384])
@autotune(BLOCK in [16, 4096])
kernel add(a: tensor<f32>[N], b: tensor<f32>[N], c: tensor<f32>[N]) {
    let base = ctx.block.x * BLOCK
    c[base :+ BLOCK] = a[base :+ BLOCK] + b[base :+ BLOCK]
}"#;

async fn submit(client: &mut SchedulerClient<tonic::transport::Channel>, job: Job) -> Result<()> {
    let resp = client.submit(job).await?;
    let mut events = resp.into_inner();
    while let Some(ev) = events.message().await? {
        match ev.kind {
            Some(proto::job_event::Kind::Done(_)) => return Ok(()),
            Some(proto::job_event::Kind::Progress(p)) => bail!("job failed: {p}"),
            None => {}
        }
    }
    bail!("submit stream ended without a Done event")
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<()> {
    let uri = temp_uris("phobos_m1_loopback")?;
    let (_, sched_addr) = start_cluster(1, 256 << 20).await?;
    let mut client = SchedulerClient::connect(format!("http://{sched_addr}")).await?;

    const N: usize = 1024;
    let (a, b) = seed_matmul(&uri, N)?;
    submit(&mut client, matmul_job(&matmul_src(512, 16384), N, &uri)).await?;
    let c = storage::read_tensor_f32(&uri("C.bin"), N * N)?;
    check_matmul(&c, &a, &b, N, &[(0, 0), (1, 513), (511, 512), (1023, 1023)])?;
    println!("matmul loopback: 2x2x2 supertile grid OK");

    const M: usize = 1024;
    let x = randoms(M, 3);
    let y = randoms(M, 4);
    storage::write_tensor_f32(&uri("a.bin"), &x)?;
    storage::write_tensor_f32(&uri("b.bin"), &y)?;
    storage::write_tensor_f32(&uri("c.bin"), &vec![0.0; M])?;
    submit(
        &mut client,
        make_job(
            ADD,
            &[("N", M as i64)],
            vec![
                f32_tensor("a", &[M], AccessMode::Read, uri("a.bin")),
                f32_tensor("b", &[M], AccessMode::Read, uri("b.bin")),
                f32_tensor("c", &[M], AccessMode::Write, uri("c.bin")),
            ],
        ),
    )
    .await?;
    let cc = storage::read_tensor_f32(&uri("c.bin"), M)?;
    for i in 0..M {
        let want = x[i] + y[i];
        ensure!(
            cc[i].to_bits() == want.to_bits(),
            "c[{i}] = {} != {want} (must be bit-exact)",
            cc[i]
        );
    }
    println!("add loopback: bit-exact OK");
    println!("loopback PASS");
    Ok(())
}
