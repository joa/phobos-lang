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
                        payload: Some(proto::node_message::Payload::Heartbeat(proto::Heartbeat {
                            node_id,
                        })),
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
